# Scheduling for Dynamic Jobs

How Lumilake orders and dispatches work, and why dynamic (chain) jobs need more
than the item-count round-robin the scheduler started with. This is a design
document: it describes the mechanism and the reasoning behind it, not a delivery
plan.

For control-plane topology, the package map and the storage model, see
[`docs/ARCHITECTURE.md`](ARCHITECTURE.md). For the operation classes a workload
is built from, see [`docs/OPS.md`](OPS.md). For the settings named here, see
[`docs/ENV.md`](ENV.md).

## 1. Scope

Covered: the chain model and why it is hard, how dispatch decides what runs,
what causes idle workers, how fair share is measured, where cost estimation
belongs, and what is decided versus still open.

Not covered: intra-graph placement. That is HALO's job and runs *after* a batch
is chosen — see `runtime/optimizer/`.

## 2. The chain model

A dynamic workflow turns a job into a **chain**. The parent job owns the run;
each round is planned by an LLM, submitted as a child job, awaited, and the next
round is planned from the result. The loop ends when the planner returns a stop
plan or the round cap is reached, so **chain length is not known up front** — the
planner decides at runtime.

```text
 t ----------------------------------------------------------------->

 round 0   [ queue wait ][ execute + plan ]
 round 1                                   [ queue wait ][ exec + plan ]
 round 2                                                              [ q ][ ...
                                           ^
                                           round k+1 is not created - and so
                                           cannot be scheduled, batched, or
                                           anticipated - until round k finishes
```

Two consequences drive the rest of this document.

**A chain pays one queueing delay per round.** Rounds are awaited back to back
with no sleep, so the gap between them is queueing, not planner think-time. A
chain of *R* rounds queues *R* times, and end-to-end chain latency is the sum of
those waits plus execution. Latency variance compounds with chain length.

**Each round looks like a cold arrival.** A round enters the queue as a fresh
child job with its own enqueue timestamp and a zeroed miss counter.
`LumilakeRequestConfig` carries `chain_id` and `chain_round` so the scheduler can
tell round 7 of 8 from a first-time submission; without lineage the two are
indistinguishable.

The unit the user cares about is the **chain**, not the round. The objective this
design optimises for is p95 end-to-end chain latency.

## 3. Dispatch: capacity first, then selection

The scheduler is a single loop. The ordering decision lives in
`PriorityJobManager` (`runtime/job_manager/priority_queue.py`); the dispatch
decision lives in `LumilakeServer._scheduler_loop` (`runtime/server.py`).

The load-bearing property is that **selection sees capacity before it chooses**,
and that dispatch never blocks. Choosing a batch and only then discovering no
worker can run it parks the one loop that could have dispatched something else.

```text
BEFORE - selection before capacity, one blocking loop

  queue:  [ GPU item ][ CPU item ]      workers:  gpu-0 BUSY   cpu-0 IDLE
             |
             v
     select the GPU batch --> wait for a free GPU --> (loop parked)
                                                          |
     the CPU item is runnable right now, but the only loop that could
     dispatch it is asleep.  cpu-0 idles for the whole wait.  <- bubble

AFTER - capacity first, selection second

  queue:  [ GPU item ][ CPU item ]      workers:  gpu-0 BUSY   cpu-0 IDLE
             |
             v
     snapshot free capacity  ->  { cpu: [cpu-0], gpu: [] }
             |
             v
     GPU partition ineligible  -> skipped
     CPU partition eligible    -> selected --> claim cpu-0 --> dispatch
             |
             v
     loop returns immediately and keeps dispatching
```

The loop in outline:

```text
wait for work
snapshot free capacity                    -> FreeCapacity
if nothing idle:            wait for a capacity release, retry
skip batch accumulation if capacity already exceeds the queue
reserve a batch, passing capacity         -> only eligible partitions
if nothing eligible:        wait for a capacity release, retry
claim workers (one non-blocking attempt)
if the claim lost a race:   abort as a capacity denial, retry
commit the reservation, dispatch on a task, continue
```

Three details matter.

**Waiting is edge-triggered on capacity.** `_wait_capacity` waits on a capacity
release signal with a bounded fallback, never on "the queue is non-empty". The
queue being non-empty is exactly the state the loop is already in, so waiting on
it returns instantly and spins. Worker release is centralised in
`_release_workers`, which sets that signal.

**Claiming is a single non-blocking attempt.** `_try_claim_workers` takes the
capacity snapshot and either claims a group or returns nothing. It never polls.

**Capacity denial is not policy denial.** `abort_reservation` takes an
`AbortReason`. Only a *policy* denial advances an item's miss counter. If
capacity pressure inflated that counter, sustained pressure would trip starvation
pinning and re-force the same unrunnable partition — capacity pressure would
corrupt the fairness signal into a thrash loop.

## 4. Partitioning and eligibility

A FlowMesh dispatch cannot span principals, bearer tokens, optimizer types or
hardware shapes, so the queue is keyed by a partition tuple. GPU-ness is part of
that key.

```text
queue, keyed by (principal, token, optimizer, hardware, requires_gpu)

  +-------------------------------------------------+
  | (acme, tok..., halo, (8, 16Gi, 1, 24Gi), gpu=yes) |   item A   item B
  +-------------------------------------------------+
  +-------------------------------------------------+
  | (acme, tok..., halo, (8, 16Gi, 1, 24Gi), gpu=no ) |   item C
  +-------------------------------------------------+

  free capacity:   cpu = [cpu-0]        gpu = []

    gpu=yes  needs a GPU group  ->  INELIGIBLE, skipped this round
    gpu=no   needs a CPU group  ->  eligible,   selected
```

`requires_gpu` is computed once, where the graphs are already in hand, and
carried on the work item. The job manager never imports the runtime manager's
GPU predicate: duplicating it would let the scheduler's view drift from the
dispatcher's, and importing it would couple `BaseJobManager` to a concrete
runtime.

Including it in the key is what keeps a GPU item from making a CPU-only sibling
unschedulable. Eligibility is computed per partition, so without the split one
GPU item would suppress every CPU item sharing its principal, token, optimizer
and hardware — reintroducing head-of-line blocking inside the partition.

The split is conditional on the capacity-aware-selection lever
(`LUMILAKE_CAPACITY_AWARE_SELECTION`). With the lever on (default) the key
carries `requires_gpu`, so CPU and GPU items in the same class land in separate
partitions. With the lever off the key drops `requires_gpu`, so those items
share a partition and head-of-line blocking applies within it — the lever
controls both the partition key and the capacity filter.

That has a cost, stated plainly: CPU-only and GPU-requiring items that would
otherwise share a partition can no longer co-batch, which loses some affinity.
Correct eligibility was judged worth more than that co-batching.

Eligibility also accounts for hardware, not just class. `FreeCapacity`
(`runtime/capacity.py`) searches the whole free set for workers meeting the
request. Selection and claim share one helper — if selection counted only the
first few workers while claim searched all of them, selection would declare a
partition unrunnable while a usable worker sat idle.

Starvation pinning still applies, but only over partitions that are eligible.
Pinning an item that cannot run would produce an undispatchable anchor.

## 5. What still causes idle workers

A bubble is idle worker time while the queue holds an item that worker could
have served. Idle time with only GPU work queued and no GPU free is not a bubble.

| Cause | Status |
|---|---|
| Selection before capacity, blocking the single loop | addressed (§3) |
| Miss-count inflation under capacity pressure | addressed (§3) |
| One GPU item suppressing CPU siblings | addressed (§4) |
| Accumulation window delaying dispatch while workers idle | addressed — skipped when free capacity already exceeds the queue |
| **Whole-worker exclusivity** | **open** |
| **Static worker-group sizes** | **open** |

The two open ones share a root. A worker is wholly busy or wholly free, and
groups are fixed size, while `HardwareRequirements` is a four-dimensional vector.
A job needing two of a worker's eight GPUs still claims the whole worker, and a
group of four cannot be assembled from three free workers whose combined capacity
would suffice. Vector capacity and demand-sized groups would recover that, at the
cost of a substantially more complex claim path.

## 6. Scheduling policies

`LUMILAKE_SCHEDULER_POLICY` selects the policy. Policies are registered in
`SCHEDULING_POLICIES` (mirroring the optimizer's `OPTIMIZER_TYPES`): a new
policy is a `BaseSchedulingPolicy` subclass plus one registry entry, resolved
by `create_scheduling_policy`. The interface takes the candidate *set* and
returns the batch subset, so a policy can score sets rather than individual
items — batch execution cost is not separable over its members (a model swap
is charged once per distinct model in the batch, not per item).

The policy owns batch selection and nothing else. The job manager picks the
anchor partition, builds its candidate pool round-robin across users, and
forces starved candidates (those at `LUMILAKE_STARVATION_LIMIT` misses) into
the batch first, oldest first. It then hands the policy the remaining
candidates and the room left in the batch, and appends the policy's pick. When
starved candidates fill the batch, the policy is not consulted. Commit, miss
counting, and everything downstream (HALO, dispatch) are the same under every
policy.

Each policy returns at most `batch_size` ids, a subset of the candidates, and
may use only the queued items, their graphs, and what has already run — never
the future. The policies:

- **`default`**: affinity clustering with per-user round-robin fairness. It
  clusters the candidates by a distance over base models, system-prompt tokens
  and graph size, and picks the batch-sized subset with the lowest average
  intra-batch distance. It then keeps one item per user in round-robin order
  (that user's affinity pick when it has one), and fills the rest of the batch
  in affinity order, then candidate order.
- **`fifo`**: enqueue order (`enqueued_at`, then `workflow_id`).
- **`spt`**: smallest `estimate_area` first; items without an estimate after
  all estimated ones; ties by enqueue order.
- **`lpt`**: largest `estimate_area` first; items without an estimate after
  all estimated ones; ties by enqueue order.
- **`plas`**: program-level least attained service (Autellix). Keyed by the
  item's chain (`chain_id`, else the request id), lowest attained service
  first; ties by enqueue order. Each committed item charges its `estimate_area`
  to the chain.
- **`cohort_pack`**: dispatches units of chains cut from template-pure
  cohorts. Slices are grouped by chain, round and template (the request's
  template hash: the workflow without its inputs), so a group never mixes
  templates. The *cohort* of a ready template and round `r` is the chains ready
  on it plus every chain whose last committed round is `r - 1`, whose committed
  template history matches theirs, and which is not itself ready on another
  template or round (a round-0 cohort has no mates). A mate counts with the
  size of the largest batch its last committed round went out in (a round
  dispatched in several batches counts as its widest batch). There is no time
  bound, and Lumilake has no end-of-chain signal, so a chain that has ended keeps
  counting as a mate: a unit waiting for it goes after more than `patience`
  misses, or at once when no complete unit is ready (the anchor then falls back
  to any unit).
  On every selection the cohort is cut into balanced *units* of whole chains,
  in chain order, aiming at `ceil(S / B)` units for `S` cohort slices and batch
  size `B`; the cut never divides a chain between units, and a chain wider than
  `B` is a unit alone. A unit is complete when every chain cut into it is ready.
  The anchor unit is template-pure. Units are recomputed, not remembered:
  chains batched together usually arrive together and so are cut together
  again, but a chain that joins the cohort later can change the cut. A dispatch
  never exceeds the batch size, so an anchor unit with more than `B` slices
  goes out over several dispatches: this one takes its first `B` slices (its
  members in cut order, each group's slices by enqueue time, slice index and
  workflow id) and the rest stays queued for later selections.
  The anchor is the complete unit whose slices have the highest miss count,
  oldest first among ties; an incomplete unit passed over more than `patience`
  dispatches goes first instead: the one passed over most, then the lexically
  smallest chain id (a longer id ranks ahead of an id it extends), then the
  oldest. The global `LUMILAKE_STARVATION_LIMIT` overrides every policy once a
  candidate reaches it, so the effective release threshold is the smaller of
  `patience + 1` policy-level misses and that limit; set the limit above
  `patience` to let units wait for their mates as configured.
  Unless the unit is spread over lanes (below), the remaining room is filled by
  considering up to three whole ready rounds of any unit, of other templates
  too, and keeping the set with the lowest estimated time per slice, if it is
  strictly lower than the unit alone (ties: break the fewest other units, then
  take the rounds with the lowest total enqueue-order rank). With no fixed or
  mixing cost and `beta` `>= 1`, merging never lowers the time per slice, so the
  batch stays template-pure; with `beta < 1` the economy of scale alone can make
  merging pay. The batch-cost model (the `fixed_seconds`, `mix_seconds`, `beta`,
  `area_to_seconds` and `batch_size` args) converts the fixed and mixing costs
  to slice units with the per-slice `estimate_area` of the anchor unit's first
  round (an item the cost model cannot estimate counts as area `1`), then
  compares candidate batches by slice count, so slices count as equally costly.
  When the free-lane count is known, the unit's batch may be cut so the rest
  goes out on lanes that would otherwise idle (the free lanes minus one per
  other ready unit), while the model says each extra batch saves at least
  `gamma` times the lane-time it costs; that batch is not filled. The
  free-lane count is the number of whole dispatches the free workers can serve
  for the anchor partition: workers (those meeting its hardware, when the
  capacity snapshot carries worker profiles) divided by the worker group size
  of each class it claims (a CPU-only partition claims a CPU group; a GPU
  partition a GPU and a CPU group; a configured CPU group of zero still claims
  one CPU worker when the work has a `data_retrieval`, `api` or `python` node),
  taking the smallest; `0` when no class is claimed. When the candidates need a
  capability-gated task type, the CPU side is also bounded by the eligible CPU
  workers advertising every such type (the union over the anchor partition's
  candidates, a conservative bound), and the count is `0` without worker
  profiles. `reserve_batch` passes it only when it receives the free capacity,
  which the server does while `LUMILAKE_CAPACITY_AWARE_SELECTION` is on (its
  default); with it off the policy does not spread.

  `cohort_pack` takes these keys in `LUMILAKE_SCHEDULER_CUSTOM_ARGS` (a JSON
  object, e.g. `{"patience": 3, "gamma": 0.5}`); an unknown key or a value
  outside its constraint fails server startup:

  | Key | Meaning | Constraint | Default |
  |-----|---------|------------|---------|
  | `fixed_seconds` | Fixed cost per batch, in seconds. | finite, `>= 0` | `0.0` |
  | `mix_seconds` | Cost per extra distinct template in a batch, in seconds. | finite, `>= 0` | `0.0` |
  | `beta` | Economy-of-scale exponent: a `k`-slice batch scales its per-slice work by `(k / batch_size) ** (beta - 1)`. | finite, `> 0` | `1.0` |
  | `area_to_seconds` | Seconds of unit-speed work per unit of `estimate_area`. The fixed and mixing costs are divided by this times the per-slice `estimate_area` of the anchor unit's first round to get slice units; an item the cost model cannot estimate counts as area `1`. Candidate batches are then compared by slice count. | finite, `> 0` | `1.0` |
  | `gamma` | Spreading break-even factor: a unit's batch is cut for one more idle lane while the time saved is at least this multiple of the lane-time it costs. | finite, `>= 0` | `1.0` |
  | `batch_size` | Reference batch size of the economy-of-scale factor and of the spreading rule (distinct from the dispatch batch size `B`). | `> 0` | `10` |
  | `patience` | An incomplete unit whose slices have been passed over more than this many dispatches stops waiting for its missing mates and goes first. A chain that has ended keeps counting as a mate, so a unit waiting for it goes after more than this many misses, or at once when no complete unit is ready. | `>= 0` | `8` |

The size-based policies (`spt`, `lpt`) and the attained-service policy
(`plas`) use the analytic cost model of §7. `estimate_area` is a predicted
critical-path estimate charged at commit time; it is never reconciled against
observed duration, so a persistently wrong estimate distorts the ordering.
The ordering is **partition-local**: it orders items within a partition the
outer round-robin already selected, so it cannot rebalance across principals
or lanes.

## 7. Where cost estimation belongs

HALO already estimates execution cost, but it does so *inside* the optimizer,
downstream of selection, and it minimises makespan **within** a batch. Nothing
optimised **across** batches. Selection needs a different quantity — how much
resource-area an item will consume — to rank items before a batch exists.

`runtime/job_manager/cost.py` computes that, and it calls HALO's own GPU cost
function rather than introducing a second estimator. Two cost models that
disagree would be worse than one imperfect one.

```text
   graph ops --> per-op duration        --+
                   GPU: HALO's own cost   |
                   DB / CPU: coefficients |
                                          +-> critical path (not the sum;
   hardware request --> dominant share ---+     the graph runs in parallel)
                                                      |
                                                      v
                                            area = path * share
```

Estimates are **analytic and hyperparameter-first**. There is no trace corpus, so
nothing here fits or learns; every coefficient is a named setting with a reasoned
default. That is a deliberate trade: the estimates are coarser than HALO's own —
which derives input counts per node where this uses a configured default — in
exchange for having no dependency on data that does not exist yet. Calibrating
the coefficients against measured durations is the obvious later step, and
requires a corpus first.

## 8. Scheduling under unknown chain length

Classical size-based policies need *p*, the service requirement. SRPT and WSPT
both assume it is known. For a chain it is not: remaining service depends on how
many more rounds the planner will request, which is decided at runtime.

**Whether chain-aware ordering can help at all depends on the shape of the
chain-length prior**, not on the scheduler:

| Prior | Hazard rate | What size-aware ordering buys |
|---|---|---|
| Geometric (independent stop decision each round) | constant — memoryless | **Nothing.** Attained service carries no information about remaining service; the Gittins index is constant and the policy degenerates to a size-blind ordering. |
| Decreasing hazard (long chains tend to continue) | decreasing | Least-attained-service ordering wins. |
| Increasing hazard (chains converge toward a cap) | increasing | SRPT-like ordering wins. |

Geometric is the natural no-data prior, because the planner appears to make an
independent stop decision each round. If that prior holds, chain-aware ordering
is provably worthless and the honest result is a negative one. Any claim that it
helps rests on the distribution being non-memoryless — which only measurement can
establish.

This is why none of the policies of §6 is a bandit policy. A Gittins-index
policy is the principled form *when a distribution is known*; none is, so none
is claimed.

## 9. Design decisions

| Question | Decision | Why |
|---|---|---|
| Area from a vector demand | Dominant-resource share | Keeps the scalar consistent with DRF. |
| Preemption | Not allowed | A running batch holds whole workers; preempting wastes partial work and complicates the two-phase reserve/commit protocol. |
| Cost model | Analytic, hyperparameter-first | No trace corpus exists; nothing may depend on a fitted distribution. |
| GPU-ness in the partition key | Yes | Prevents a GPU item suppressing CPU siblings, at the cost of mixed co-batching. |
| Policy default | `default` | Co-batches similar workflows by affinity clustering and needs no cost estimate, so it orders every workload, including ones `estimate_area` cannot estimate. |

## 10. Open questions

- **Is the chain-length prior memoryless?** Per §8 this decides whether
  chain-aware ordering has any value at all. Unanswerable without measurement.
- **Dominant share against which denominator?** Cluster capacity shifts as
  workers join and leave; the share needs a defined snapshot or a smoothed
  estimate.
- **One share pool or two?** DRF handles mixed demand in principle, but a cluster
  whose GPU workers are the scarce resource may want GPU-denominated fairness
  specifically.
- **Whole-worker exclusivity** (§5) — worth the complexity of vector capacity?
