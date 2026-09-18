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

## 6. Fairness: area, not item counts

The original mechanism counted **items** — equal turns per user, item-count
quantums, one miss counter per item. The resource is consumed in **area**:
demand × duration. A user whose graphs are a hundred times larger gets a hundred
times the resource-time at nominally equal fairness. That is fair in turns and
unfair in resource.

Fair share is therefore measured in area. Because demand is a vector, area is
collapsed to a scalar by **dominant-resource share** — the largest of the
request's per-resource fractions — which keeps the accounting consistent with
Dominant Resource Fairness. DRF carries the properties worth claiming:
sharing-incentive, envy-freeness and Pareto-efficiency. Strategy-proofness also
holds, though it matters less in a cooperative deployment where principals are
not adversarial.

**The principal is `user_id`** — the level the queue already treats as the
fairness key.

Attained service **decays exponentially**. On each commit a user is charged the
area of the batch; between charges the value halves every `tau`
(`AttainedService` in `runtime/job_manager/attained.py`, decayed lazily on read).

```text
 A(user)
  2.0 |*
      | *
  1.5 |  *                    *
      |   *                  * *
  1.0 |    *  *             *   *
      |     **  *          *     *
  0.5 |          *  *  *  *       *  *  *
  0.0 +-------------------------------------------> t
      ^                     ^
      charge 2.0            charge 1.0
      |<----- tau ----->|
        value halves every tau with no new charge
```

Decay is what makes the accounting stable across heterogeneous rounds. A chain
that converges — rounds shrinking as it narrows — is not penalised forever for
one expensive early round, and a chain cannot bank credit by idling. A plain
cumulative sum has both failure modes.

Ordering then follows from a single index rather than a weighted-sum heuristic:

```text
  index(item) = w(user) / p_hat(item)
  w(user)     = 1 / (1 + attained(user) / fair_share_target)
```

Higher index first. This is Smith's rule with a fairness weight: prefer cheap,
under-served work. For items the cost model cannot estimate, ranking falls back
to least-attained-service rather than inventing a number.

The index policy is opt-in. `LUMILAKE_SCHEDULER_POLICY` defaults to `legacy`,
which keeps priority quantums, per-user round-robin, starvation pinning and
affinity selection within a partition.

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
| Geometric (independent stop decision each round) | constant — memoryless | **Nothing.** Attained service carries no information about remaining service; the Gittins index is constant and the policy degenerates to the size-blind baseline. |
| Decreasing hazard (long chains tend to continue) | decreasing | Least-attained-service ordering wins. |
| Increasing hazard (chains converge toward a cap) | increasing | SRPT-like ordering wins. |

Geometric is the natural no-data prior, because the planner appears to make an
independent stop decision each round. If that prior holds, chain-aware ordering
is provably worthless and the honest result is a negative one. Any claim that it
helps rests on the distribution being non-memoryless — which only measurement can
establish.

This is why the implemented policy is the estimate-driven index of §6 rather than
a bandit policy. A Gittins-index policy is the principled form *when a
distribution is known*; none is, so none is claimed.

## 9. Design decisions

| Question | Decision | Why |
|---|---|---|
| Fairness principal | `user_id` | The level the queue already keys on. |
| Attained service | Exponentially-decayed area, half-life `tau` | Stable across heterogeneous rounds; no permanent penalty, no credit for idling. |
| Area from a vector demand | Dominant-resource share | Keeps the scalar consistent with DRF. |
| Preemption | Not allowed | A running batch holds whole workers; preempting wastes partial work and complicates the two-phase reserve/commit protocol. |
| Cost model | Analytic, hyperparameter-first | No trace corpus exists; nothing may depend on a fitted distribution. |
| GPU-ness in the partition key | Yes | Prevents a GPU item suppressing CPU siblings, at the cost of mixed co-batching. |
| Index policy default | `legacy` | The new ordering must be switchable to be evaluable. |

## 10. Open questions

- **Is the chain-length prior memoryless?** Per §8 this decides whether
  chain-aware ordering has any value at all. Unanswerable without measurement.
- **What is a defensible default for `tau`?** Expressed as a multiple of a
  typical round duration, which is itself unmeasured.
- **Dominant share against which denominator?** Cluster capacity shifts as
  workers join and leave; the share needs a defined snapshot or a smoothed
  estimate.
- **One share pool or two?** DRF handles mixed demand in principle, but a cluster
  whose GPU workers are the scarce resource may want GPU-denominated fairness
  specifically.
- **Whole-worker exclusivity** (§5) — worth the complexity of vector capacity?
