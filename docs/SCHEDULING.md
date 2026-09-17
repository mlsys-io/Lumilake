# Scheduling for Dynamic Jobs

Design and roadmap for follow-up scheduling of dynamic (chain) jobs in Lumilake.
This document describes the current scheduler, why dynamic jobs break its model,
the distinct causes of scheduling bubbles, the fairness property we can honestly
claim, and a phased plan that ends in fair-weighted index ordering. It is a
design + roadmap doc; S10 records the decisions that are settled and the
questions that remain.

For the control-plane topology, package map, and storage model, see
[`docs/ARCHITECTURE.md`](ARCHITECTURE.md). For the operation classes a workload
is built from, see [`docs/OPS.md`](OPS.md).

## 1. Purpose & scope

This document covers:

- how the scheduler orders and dispatches work today,
- why dynamic jobs (chains) are structurally invisible to it,
- the distinct causes of scheduling bubbles and which phase fixes each,
- what fairness we can measure and claim, and why,
- where the cost model sits and where it needs to move,
- why the chain-length prior decides whether chain-aware ordering helps at all,
- a phased roadmap, the settled decisions, and what remains open.

It does not re-litigate the decisions recorded in S10: the fairness principal is
**`user_id`**, attained service is **exponentially-decayed resource-area**,
**preemption is not allowed**, and the cost model is **analytic and
hyperparameter-first** rather than fitted. The primary objective is **p95
end-to-end chain latency**.

## 2. Today's scheduler

The scheduler is a single loop in `LumilakeServer._scheduler_loop`
(`src/lumilake_server/runtime/server.py:744`). It waits for work, waits for batch
accumulation, reserves a batch, waits for a free worker group, commits the
reservation, and dispatches. Ordering is computed by `PriorityJobManager`
(`src/lumilake_server/runtime/job_manager/priority_queue.py`).

**Three priority classes with quantums.** `Priority` is `high | medium | low`
(`src/lumilake_server/runtime/protocol.py:9`). Each priority has a per-round
quantum — the number of items that priority may contribute to a batch — from
`DEFAULT_QUANTUMS` (`priority_queue.py:56`), backed by
`LUMILAKE_QUEUE_QUANTUM_{HIGH,MEDIUM,LOW}` (`packages/sdk/src/lumilake/envs.py:91`).

**Per-user round-robin.** Within a priority, items are held in per-user queues
(`priority_queue.py:87`), and `_peek_round_robin_for_partition_locked`
(`priority_queue.py:571`) walks users in round-robin order, taking up to the
priority quantum from each. The per-priority user pointer is rotated at commit
time (`_advance_user_round_robin_locked`, `priority_queue.py:549`).

**`miss_count` starvation pinning.** Every item carries a `miss_count`
(`job_manager/base.py:32`). Items whose `miss_count` reaches
`LUMILAKE_STARVATION_LIMIT` (default 3, `envs.py:100`) are collected as
`starved_global` during selection (`priority_queue.py:269`), pinned into the
batch ahead of everything else (`_apply_starvation_policy`,
`priority_queue.py:634`), and their partition is forced as the anchor
(`priority_queue.py:278`). Non-selected candidates get `miss_count += 1` at
commit (`priority_queue.py:439`).

**Partition round-robin.** A partition key is
`(principal_id, dispatch_token, optimizer_type, hardware_signature, requires_gpu)`
(`priority_queue.py:39`). Two requests share a FlowMesh dispatch only if their
partition keys are equal. `reserve_batch` picks one partition per round via
`_pick_partition_round_robin_locked` (`priority_queue.py:524`) and rotates past
it at commit (`_advance_partition_round_robin_locked`, `priority_queue.py:542`).
The hardware signature is the 4-tuple `(cpu, memory, gpu, gpu_memory)` from
`HardwareRequirements` (`protocol.py:15`, `priority_queue.py:32`). The fifth
`requires_gpu` element splits CPU-only and GPU-requiring items into separate
partitions so a busy GPU group never suppresses CPU-only items in the same
principal/token/optimizer/hardware class.

**Affinity clustering.** Within the candidate pool, `select_affinity_batch_ids`
(`priority_queue.py:350`, implemented in
`job_manager/cluster_algo/clustering.py:32`) groups workflows by agglomerative
clustering under average linkage, up to `batch_size`. Pairwise distance is
`1 - (alpha*S_model + beta*S_data + gamma*S_size)`, where `S_model` is the Jaccard
similarity of base model sets, `S_data` the Jaccard similarity of tokenized system
prompts, and `S_size` a node-count similarity. Enqueue time breaks ties (older
first), and `pinned_ids` are always included before affinity fills the remaining
slots. Model affinity dominates the weights, so a batch tends to share model
weights and prompt prefixes.

**Two-phase reserve/commit.** `reserve_batch` (`priority_queue.py:238`) computes
the next batch without mutating queue state and returns a `BatchReservation`
(`job_manager/base.py:45`). The caller either commits (`commit_reservation`,
`priority_queue.py:430`, which removes selected items and bumps `miss_count`) or
aborts (`abort_reservation`, `priority_queue.py:496`, which leaves the queue
untouched). The interface is declared on `BaseJobManager.reserve_batch`
(`job_manager/base.py:97`).

**One scheduler loop.** There is exactly one `_scheduler_loop` task
(`server.py:744`). It selects a batch, then blocks in
`_wait_for_available_worker_group` (`server.py:828`) — a `while True:` poll that
re-fetches workers, filters by hardware, and sleeps `LUMILAKE_POLL_INTERVAL_SECONDS`
(default 5, `envs.py:114`) until a group of the required size is free, bounded by
`LUMILAKE_POLL_TIMEOUT_SECONDS` (`server.py:905`).

## 3. Problem statement

A dynamic workflow (`_run_dynamic_job`, `src/lumilake_server/routes/jobs.py:1668`)
turns a job into a **chain**: each round is planned by an LLM, submitted as a
child job via `_submit_dynamic_child` (`jobs.py:1517`), awaited, and the next
round is planned from the result. The loop runs `while round_index < max_rounds`
(`jobs.py:1721`) and stops when the planner returns `StopPlan` or the round cap is
hit — so **chain length is not known up front**; the planner decides `STOP` at
runtime.

Each round re-enters the scheduler as a **fresh child job**: a new `JobRecord`
with `parent_job_id` set (`jobs.py:1562`), a new `enqueued_at`, and `miss_count=0`.
`LumilakeRequestConfig` (`protocol.py:40`) has **no lineage field** — the job
manager cannot tell round 7 of 8 from a cold arrival. The inter-round gap is
queueing, not planner think-time: the rounds are awaited back-to-back with no
sleep, so a chain pays *R* independent queueing delays, one per round, and each
delay is measured against a cold-arrival item.

The ordering machinery is built around **item counts** — equal turns, item-count
quantums, `miss_count` per item — while the resource is consumed in **area**
(demand × duration). A chain is invisible to both the count-based fairness and
the partition round-robin, and its rounds are individually starvable and
individually bubble-prone.

## 4. Bubble taxonomy

A scheduling bubble is idle-worker time while the queue is non-empty and that
worker could have served a queued item. The distinct causes, with evidence:

**(a) Head-of-line blocking: selection before capacity, one blocking loop.**
`_scheduler_loop` selects a batch (`server.py:750`) *before* it knows whether
capacity exists, then blocks in `_wait_for_available_worker_group`
(`server.py:828`). Because there is one loop, a GPU batch with no free GPU worker
stalls every dispatch — including CPU-only work with idle CPU workers. The GPU
partition is selected, the loop polls for a GPU group, and nothing else is
dispatched meanwhile. **Phase 1 fixes this** by making selection capacity-aware
and dispatch non-blocking.

**(b) `miss_count` inflation on capacity denial → thrash loop.** When the worker
wait fails, the loop does `continue` → `abort_reservation` (`server.py:754`,
`server.py:769`), and the next `reserve_batch` re-selects the same partition. On
commit, non-selected candidates get `miss_count += 1` (`priority_queue.py:439`).
Under sustained capacity pressure, the same partition is repeatedly re-selected
and its items' `miss_count` climbs until starvation pinning (`priority_queue.py:269`)
re-forces it — capacity pressure corrupts the starvation signal into a thrash
loop. **Phase 1 fixes this** by distinguishing a *policy* denial from a
*capacity* denial and only incrementing `miss_count` on the former.

**(c) Binary whole-worker exclusivity despite a 4-D demand vector.**
`_busy_workers` is a `set[str]` (`server.py:354`): a worker is wholly busy or
wholly free. Capacity is accounted per-whole-worker, in fixed groups sized by
`LUMILAKE_CPU_WORKER_GROUP_SIZE` / `LUMILAKE_GPU_WORKER_GROUP_SIZE` (static env,
`envs.py:104`). `HardwareRequirements` is a 4-D vector (`protocol.py:15`), but a
job that needs 2 of a worker's 8 GPUs still claims the whole worker, and a group
of size 4 cannot be assembled from 3 free workers even if their combined capacity
suffices. **Deferred** — replacing the binary set with vector capacity and
demand-sized groups is a real utilization win but is larger and wants harness
evidence first.

**(d) Static batch size and an accumulation window that delays idle dispatch.**
`LUMILAKE_OPTIMIZER_BATCH_SIZE` is static (`envs.py:88`). `_wait_for_batch_accumulation`
(`server.py:807`) sleeps up to `LUMILAKE_BATCH_ACCUMULATION_SECONDS` (default 0,
`envs.py:101`) even when workers sit idle — accumulating while workers idle is a
pure bubble. **Phase 1 fixes this** by skipping the accumulation window when free
capacity exceeds what the queue can consume.

## 5. Fairness: what we measure and what we can claim

Today fairness is counted in **items**: equal turns, item-count quantums,
`miss_count` per item. The resource is consumed in **area** (demand x duration).
A user whose graphs are 100x larger gets ~100x the resource-time at nominally
equal fairness - the count-based mechanism is fair in turns, not in resource.

The target is **fair share in resource-area**. Because demand is a 4-D vector
(`protocol.py:15`), area is collapsed to a scalar as the **dominant-resource
share** - the largest of the request's per-resource fractions of cluster capacity
- which keeps the accounting coherent with Dominant Resource Fairness (DRF).

**Decided: the fairness principal is `user_id`.** The partition key uses
`principal_id` (`priority_queue.py:34`) and the per-user queues key on `user_id`
(`priority_queue.py:130`); fair share is computed at the `user_id` level, which is
what the code already calls the fairness key. Chain-level accounting exists (the
lineage from S8.1) and is reported by the harness, but chains are not a separate
share tier in this design.

**Decided: attained service is exponentially decayed.** Per principal,

```
A <- A * exp(-dt / tau) + area_of_round
```

where `area_of_round` is in dominant-resource share and `tau` is a half-life
hyperparameter. The decay is what makes the accounting stable across
heterogeneous rounds: a chain that converges (rounds shrinking as it narrows) is
not penalised forever for one expensive early round, and a chain cannot bank
credit by idling. A cumulative (undecayed) sum has both failure modes. Default
`tau` is a few multiples of a typical round duration - a knob, not a fit.

DRF carries the named properties a paper can claim - **sharing-incentive**,
**strategy-proofness**, **envy-freeness**, and **Pareto-efficiency**. In a
cooperative single-team deployment strategy-proofness matters less (principals are
not adversarial), but the other three still hold and are the honest claims.

The reported fairness index is Jain's, computed over users and over chains, in
resource-area - not item counts.

## 6. Cost model: where it is and where it needs to be

HALO already has a cost model, but it sits **inside** the optimizer, downstream
of selection. `HaloOptimizer._exec_cost` (`src/lumilake_server/runtime/optimizer/halo.py:955`)
delegates to `compute_gpu_exec_cost` (`multimodal_cost.py:51`) with
`MultimodalCostCoefficients` (`multimodal_cost.py:16`), and the coefficients are
**hand-tuned constants** - `_model_init_sec_per_b = 0.75`, `_llm_base_sec_per_b =
0.25`, `_llm_input_sec = 0.15`, `_db_input_sec = 0.05` (`halo.py:110`). There is
**no feedback from measured durations**: nothing feeds observed op/job times back
to calibrate these coefficients. HALO minimizes max-GPU-makespan **within a
batch**; nothing optimizes **across** batches.

The direction is to lift cost estimation up into **selection**, reusing
`multimodal_cost.py` and `DataProfileCostEstimate`
(`src/lumilake_server/data_profile_models.py:6`) rather than inventing a second,
divergent cost model.

**Decided: the design is hyperparameter-first, not empirical.** There is no trace
corpus yet. Every estimate is an analytic function of graph shape and declared
demand, parameterised by named hyperparameters with reasoned defaults and env
overrides. Calibrating those hyperparameters against measured durations is
deliberately later work, gated on a corpus existing. Two consequences follow:

- the harness (S8.3) is **synthetic-only** for now - the replay-from-`JobSummary`
  workload source is specified but not built, because with zero recorded chains it
  would have nothing to replay;
- no part of the policy may depend on a fitted distribution. See S7.

## 7. Scheduling under unknown chain length

Chain length is not known up front: the planner decides `STOP` at runtime, so a
chain's remaining service is a random variable. Classical size-based policies -
**SRPT** (shortest remaining processing time) and **WSPT / Smith's rule** (order by
weight/processing-time) - both need `p`, the service requirement, and so do not
apply directly.

**The prior's shape decides whether chain-aware ordering can help at all.** This
is the load-bearing caveat for any later claim:

| Chain-length prior | Hazard rate | What size-aware ordering buys |
|---|---|---|
| Geometric (independent STOP per round) | constant - memoryless | **Nothing.** Attained service carries no information about remaining service, so the Gittins index is constant and the policy degenerates to the size-blind baseline. |
| Decreasing hazard (long chains likely to continue) | decreasing | LAS-style ordering wins - prefer least-attained-service. |
| Increasing hazard (chains converge to a cap) | increasing | SRPT-style ordering wins - prefer most-attained-service. |

Geometric is the natural no-data prior, because the planner makes an apparently
independent `STOP` decision each round. If that prior is right, chain-aware
ordering is provably worthless and the honest result is a negative one. Any claim
that it helps rests on the distribution being **non-memoryless**, and that is
exactly the assumption trace data would later confirm or kill.

Given the hyperparameter-first decision in S6, the implemented policy is therefore
the estimate-driven index, not a fitted bandit policy:

- **A fair-weighted index** `w(user) / p_hat(item)`, where `p_hat` is the analytic
  area estimate from the cost model and `w` decays with attained service per S5.
  This is Smith's rule with a fairness weight, and it unifies fairness and
  efficiency in one objective rather than a weighted-sum heuristic.
- **Least-attained-service** (rank by decayed attained area) as the estimate-free
  fallback for items the cost model cannot estimate - e.g. agent-mode retrieval,
  whose duration is not predictable from graph shape.
- **The Gittins index** remains the principled form *if* a non-memoryless
  distribution is ever established from traces. It is **not** implemented and is
  not claimed.

## 8. Phase 1 design

Three concrete changes, in order: chain lineage, capacity-aware non-blocking
dispatch, and the discrete-event simulation + replay harness.

### 8.1 Chain lineage in the scheduling config

Add to `LumilakeRequestConfig` (`protocol.py:40`):

- `chain_id: str | None = None` — the parent job id for a dynamic round, `None`
  for a standalone job.
- `chain_round: int = 0` — round index within the chain.

Populate in `_submit_dynamic_child` (`jobs.py:1517`) from `parent_job_id` /
`round_index`, which are already parameters there, and thread through `_run_job`
(`jobs.py:1903`) → `server.execute` (`server.py:3086`) → `Job` → `WorkflowItem`
the same way `user_id` already flows. Nothing consumes these for *ordering* in
this phase — they exist so the harness can compute chain latency, and so the
later policy has its principal. Keep them on `WorkflowItem` so `_format_item`
(`priority_queue.py:121`) can log them.

### 8.2 Capacity-aware, non-blocking dispatch

Restructure `_scheduler_loop` (`server.py:744`) so selection sees capacity and
dispatch never blocks:

```
wait_for_work()
free = snapshot_free_capacity()          # by class + hardware signature
if free is empty: await capacity_changed; continue
maybe_wait_for_batch_accumulation(free)  # skip the wait when capacity is idle
reservation = reserve_batch(batch_size, capacity=free)
if reservation is None: await capacity_changed | new_work; continue
claim workers atomically; commit; create_task(_run_batch)
```

Concretely:

- **`BaseJobManager.reserve_batch` gains a capacity argument**
  (`job_manager/base.py:97`, implemented in `priority_queue.py:238`). A partition
  whose hardware signature cannot be satisfied by current free capacity is
  excluded from `present_partitions` before the round-robin pick. This is the
  direct bubble fix: the GPU partition is simply not selected while no GPU is
  free, so the CPU partition is. Keep the existing starvation-pin behaviour, but
  only over *eligible* partitions.
- **Do not increment `miss_count` on capacity denial.** Distinguish
  `abort_reservation(reason=policy)` from `abort_reservation(reason=capacity)` so
  the starvation signal stays a fairness signal. This removes the thrash loop
  from §4(b).
- **Wake on capacity release.** Add an `asyncio.Event` set where workers are
  released (`server.py:1806`, `self._busy_workers.difference_update(workers)`),
  and await it instead of sleeping `LUMILAKE_POLL_INTERVAL_SECONDS`. Removes up
  to one poll interval of latency per dispatch.
- **Skip the accumulation window when capacity is idle.**
  `_wait_for_batch_accumulation` (`server.py:807`) should return immediately if
  free capacity exceeds what the current queue can consume.

Explicitly **not** in this change: fractional/vector worker capacity (replacing
the binary `_busy_workers` set) and dynamic worker-group sizing. Both are real
utilization wins but are larger and want harness evidence first.

### 8.3 Discrete-event simulation + replay harness

New package `tests/support/schedsim/` (test-support, not shipped in the server
image). It must drive the **real** `PriorityJobManager` and the **real** selection
path, so results characterise the shipped policy rather than a model of it.

Components:

- **Virtual clock.** A small injectable clock; the job manager and scheduler loop
  take it instead of calling `time.time()` directly. Today tests monkeypatch
  `time.time` / `asyncio.sleep` ad hoc
  (`tests/runtime/runtime_manager/flowmesh/test_progress.py:30`,
  `tests/utils/test_sqlite_job_storage.py:299`) — replace that with one helper
  the harness and tests share.
- **Simulated worker pool.** Honours the binary-busy model and group sizes so the
  simulator reproduces today's behaviour before the fix, and the fix's effect
  after.
- **Workload sources.** (a) synthetic generator — arrival process, chain-length
  distribution, op mix (GPU vs DB vs CPU per `docs/OPS.md`); (b) replay from
  recorded `JobSummary` rows (`job_storage.py:105`), reconstructing chains via
  `parent_job_id` (`job_storage.py:117`) and durations via `started_at` /
  `finished_at` (`job_storage.py:111`).
- **Metrics.** p50/p95 **chain** end-to-end latency (the target objective),
  per-chain slowdown, worker utilisation, **bubble-seconds** (idle-worker-seconds
  while the queue is non-empty and that worker could have served a queued item),
  and a fairness index (Jain) computed over both chains and users, in
  **resource-area** not item counts.

Model it on `tests/runtime/server/test_scheduler_outer_loop_persistence.py`,
which already drives `server._scheduler_loop()` directly with fake collaborators —
that is the established pattern for this code.

## 9. Roadmap

| Phase | Goal | Metric that proves it | Depends on |
|-------|------|----------------------|------------|
| **1** | Remove structural bubbles; build the synthetic harness | p95 chain latency, utilisation, bubble-seconds, Jain-in-area on a reproducible synthetic baseline | - |
| **2** | Analytic cost model lifted into selection, hyperparameterised | `p_hat` monotonicity and sanity vs graph shape; selection quality in the harness | Phase 1 harness |
| **3** | Per-user DRF over decayed resource-area | Jain-in-area over users and chains; sharing-incentive / envy-freeness / Pareto-efficiency | Phase 1 lineage + area accounting; Phase 2 `p_hat` |
| **4** | Fair-weighted index ordering (`w/p_hat`) with LAS fallback | p95 chain latency vs phase-1 baseline; per-chain slowdown | Phase 2 `p_hat`; Phase 3 fairness frame |
| **5** | Vector worker capacity + demand-sized groups | utilisation; bubble-seconds | Phase 1 harness evidence that the binary model is the bottleneck |
| **later** | Calibrate hyperparameters from a real trace corpus; test the chain-length prior | `p_hat` error vs measured durations; measured hazard rate | A deployed corpus that does not exist yet |

Phases 2-4 are implemented against hyperparameters with reasoned defaults. The
final row is explicitly gated on data, and the S7 table is what it would settle.

**Implementation status.** Phases 1-4 are implemented on branch
`feat/dynamic-scheduling`: chain lineage, capacity-aware non-blocking dispatch,
the synthetic harness (`tests/support/schedsim/`), the analytic cost model
(`job_manager/cost.py`), decayed attained service (`job_manager/attained.py`),
and the fair-weighted index. The index policy is **opt-in**:
`LUMILAKE_SCHEDULER_POLICY` defaults to `legacy`, which preserves the selection
**ordering** described in S2. Capacity-aware selection and non-blocking
dispatch are unconditional (they are bubble fixes, not policy); the flag gates
only the ordering policy. Phase 5 and the calibration row are not started.

**What `legacy` preserves, and what changed.** The `legacy` policy is not
byte-identical to the pre-capacity scheduler. It preserves the ordering
machinery: priority quantums, per-user round-robin, `miss_count` starvation
pinning, and affinity clustering *within* a partition. What changed is the
partition key itself: the fifth `requires_gpu` element means CPU-only and
GPU-requiring items now occupy separate partitions, so they no longer co-batch
even when they share principal/token/optimizer/hardware. Dispatch is
capacity-aware under every policy, including `legacy`. The trade-off: splitting
by `requires_gpu` costs mixed CPU/GPU co-batching (an affinity loss) in
exchange for correct capacity eligibility — a GPU-requiring partition can be
skipped while its GPU group is busy without suppressing runnable CPU-only work
in the same class.

## 10. Decisions and open questions

**Settled.**

| # | Question | Decision |
|---|---|---|
| 1 | Fairness principal: user or org/principal? | **`user_id`** - the level the code already treats as the fairness key. |
| 2 | Attributing chain area across heterogeneous rounds | **Exponentially-decayed attained area** in dominant-resource share, half-life `tau` (S5). |
| 3 | Is preemption acceptable? | **No.** A running batch holds whole workers (`server.py:354`); preempting wastes partial work and complicates the two-phase reserve/commit protocol. Not revisited before phase 5. |
| 4 | Empirical vs analytic cost model | **Analytic, hyperparameter-first** (S6). Calibration deferred until a trace corpus exists. |

**Still open.**

- **Is the chain-length prior memoryless?** Per S7 this decides whether
  chain-aware ordering has any value. Unanswerable without traces, and it
  determines whether phase 4's result is positive or negative.
- **What is a sensible default for `tau`?** Expressed as a multiple of typical
  round duration, but "typical" is itself unmeasured today.
- **Dominant-resource share against which denominator?** Total cluster capacity
  shifts as workers join and leave; the share must be defined against a snapshot
  or a smoothed capacity estimate.
- **Do CPU-only and GPU chains compete in one share pool, or two?** DRF handles
  mixed demand in principle, but a cluster whose GPU workers are the scarce
  resource may want GPU-denominated fairness specifically.
