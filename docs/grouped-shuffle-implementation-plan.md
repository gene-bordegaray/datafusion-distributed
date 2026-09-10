# Grouped shuffle implementation plan

## Decision

Implement a grouped one-level hash shuffle directly from current main. Hash each row once into the
final `M * K` logical partition space, combine each consumer's `K` logical partitions into one
physical shuffle frame, and recover the `K` final lanes from frame metadata at the consumer.

Use a temporary, crate-private `GroupedBatchPartitioner` in this repository while proving the
design. Once its required API is stable, contribute the grouped output API to DataFusion and replace
the local implementation with the upstream version.

PR #717 is a performance comparison point, not an implementation dependency. Do not merge its
two-level repartition into the grouped shuffle branch. Do not add a local benchmark harness that
carries all shuffle implementations. Compare branches with the existing remote benchmark service.

## Scope

The first version supports bounded, unordered hash shuffles.

Given:

- `P`: producer tasks
- `M`: consumer tasks
- `K`: output lanes per consumer task
- `M * K`: final logical hash partitions

Route each row as follows:

```text
logical_partition = hash(keys) % (M * K)
destination       = logical_partition / K
lane              = logical_partition % K
```

The producer exposes `M` physical destinations. Each consumer receives frames for one destination
and exposes `K` ordinary `RecordBatch` output streams.

```text
Producer task
-------------

  input RecordBatch
         |
         v
  GroupedBatchPartitioner
  hash once into M*K logical partitions
         |
         v
  one reordered RecordBatch
  + logical partition ranges
         |
         +--------------+--------------+------------------+
         v              v              v                  v
   consumer 0      consumer 1      consumer 2      consumer M-1
   ShuffleFrame    ShuffleFrame    ShuffleFrame    ShuffleFrame
   K lane ranges   K lane ranges   K lane ranges   K lane ranges
                        |
                        v
                 one Flight encoder
                 one producer-consumer RPC
                        |
                        v
Consumer task     one Flight decoder
-------------            |
                        v
                 validate lane ranges
                 zero-copy batch slices
                        |
               +--------+--------+
               v        v        v
             lane 0   lane 1   lane K-1
```

## Non-goals

- Do not introduce the PR #717 producer/consumer salted two-level repartition.
- Do not add a benchmark-only implementation selector containing legacy, PR #717, and grouped modes.
- Do not add a per-row lane column to the wire format.
- Do not generalize DataFusion's `ExecutionPlan` stream item beyond `RecordBatch`.
- Do not support ordered, range, round-robin, or unbounded repartitioning in the first version.
- Do not enable grouped shuffle by default until remote benchmarks and compatibility tests pass.

## Runtime invariants

1. Every input row is delivered exactly once.
2. Final lane assignment is identical to current one-level `RepartitionExec(Hash(..., M * K))`
   behavior.
3. All rows in a `ShuffleFrame` belong to the frame's physical destination.
4. Lane ranges are ordered, non-overlapping, in bounds, and cover the frame batch.
5. Empty lanes do not require metadata entries.
6. Producer queues, reservations, spill streams, and coalescers scale with `M`, not `M * K`.
7. Each input partition has an independent partitioning task.
8. Shared Arrow buffers are charged once and released after the last retaining frame is dropped.
9. A slow or unpolled destination cannot deadlock other destinations.
10. Dropping one consumer lane does not cancel its siblings. Dropping all lanes cancels the RPC and
    releases producer resources.
11. Old workers never receive a grouped producer-head request. New workers continue to support
    legacy requests.

## Target data structures

Keep the grouped partitioning types generic and free of distributed concepts:

```rust
pub(crate) struct PartitionRange {
    pub partition: usize,
    pub offset: usize,
    pub length: usize,
}

pub(crate) struct GroupedPartitionedBatch {
    pub batch: RecordBatch,
    pub ranges: Vec<PartitionRange>,
}

pub(crate) struct GroupedBatchPartitioner {
    exprs: Vec<Arc<dyn PhysicalExpr>>,
    partition_count: usize,
    hash_buffer: Vec<u64>,
    indices: Vec<Vec<u32>>,
    reducer: StrengthReducedU64,
}
```

The distributed layer converts grouped partition output into transport frames:

```rust
pub(crate) struct LaneRange {
    pub lane: u32,
    pub offset: u32,
    pub length: u32,
}

pub(crate) struct ShuffleFrame {
    pub batch: RecordBatch,
    pub lanes: Vec<LaneRange>,
    pub lease: FrameMemoryLease,
}
```

The destination is implicit in the physical producer stream and is repeated in wire metadata only
for validation.

## PR sequence

The PRs below are stacked in dependency order. Keep each PR focused and preserve the repository's
existing architecture until the next layer needs an explicit extension.

The dependency chain is:

```text
PR 1: shared Flight encoder
   |
   +----------------------------------------------+
   v                                              |
PR 2: local grouped partitioner                   |
   |                                              |
   v                                              |
PR 3: grouped producer core                       |
   |                                              |
   +--------------> upstream DataFusion PR        |
   |                     |                        |
   v                     |                        |
PR 4: runtime producer-head preparation           |
   |                     |                        |
   v                     |                        |
PR 5: grouped Flight transport <------------------+
   |                     |
   v                     |
PR 6: capability negotiation and opt-in planning
   |                     |
   |                     v
   |                 DataFusion release
   |                     |
   v                     v
PR 7: replace local partitioner with upstream API
   |
   v
remote benchmark qualification
   |
   v
PR 8: enable automatic selection
```

PRs 1 through 3 establish the data path without changing planner selection. PRs 4 through 6 connect
that path to distributed execution. The upstream DataFusion work can proceed once PR 3 proves the
required grouped result shape. PRs 7 and 8 remove temporary code and complete rollout.

The observable system state after each PR is:

| After | Producer routing                                            | Network representation               | Planner behavior                                   |
| ----- | ----------------------------------------------------------- | ------------------------------------ | -------------------------------------------------- |
| PR 1  | Existing `RepartitionExec(M * K)`                           | One encoder and decoder per RPC      | Legacy only                                        |
| PR 2  | Same; grouped partitioner exists only as a tested primitive | Same as PR 1                         | Legacy only                                        |
| PR 3  | Grouped producer has `M` physical outputs in direct tests   | Internal `ShuffleFrame` only         | Legacy only                                        |
| PR 4  | Worker runtime can prepare grouped producers                | Internal task output supports frames | Legacy only                                        |
| PR 5  | Grouped producer-to-consumer path is complete               | Versioned grouped Flight frames      | Legacy only                                        |
| PR 6  | Complete grouped shuffle                                    | Same as PR 5                         | Explicit `grouped`; guarded `auto`; default legacy |
| PR 7  | Same grouped shuffle using upstream partitioner             | Same as PR 5                         | Same as PR 6                                       |
| PR 8  | Same data path                                              | Same as PR 5                         | Default changes to guarded `auto`                  |

### PR 1: Share Flight encoding across shuffle lanes

Suggested title:

```text
perf: share Flight encoding across shuffle lanes
```

Goal: finish the current multiplexed Flight work as an independent transport improvement and
establish the encoder that grouped frames will reuse.

Technical architecture:

Current main already groups the `K` logical partitions into one producer-consumer RPC, but it
constructs an independent Arrow Flight encoder for each lane. Each encoder emits its own schema,
maintains its own dictionary state, and produces its own FlightData stream. The server merges those
encoded streams afterward. The client then creates one decoder per lane.

```text
Before PR 1
-----------

lane 0 RecordBatch stream --> Flight encoder 0 --+
lane 1 RecordBatch stream --> Flight encoder 1 --+--> select_all --> one RPC
...                                              |
lane K RecordBatch stream --> Flight encoder K --+

one RPC --> partition FlightData --> K Flight decoders --> K output streams
```

PR 1 moves multiplexing before IPC encoding. Each input batch is paired with its legacy partition
ID, then all tagged batches pass through one encoder. One decoder reads the RPC and routes decoded
batches into `K` queues.

```text
After PR 1
----------

lane 0 RecordBatch stream --+
lane 1 RecordBatch stream --+--> tagged batch mux --> one Flight encoder --> one RPC
...                          |
lane K RecordBatch stream --+

one RPC --> one Flight decoder --> inspect partition metadata --> K output streams
```

This PR does not change producer repartitioning. The producer still has `M * K` logical
`RepartitionExec` outputs. It removes duplicated IPC state and creates the transport seam that PR 5
will extend from a single partition ID to a list of lane ranges.

Why it is separate:

- It improves and simplifies the existing shuffle without depending on grouped routing.
- Dictionary, schema, compression, and decoder cancellation bugs can be resolved before introducing
  a new wire format.
- PR 5 can add grouped frame metadata to one encoder instead of modifying both the old and new
  encoder paths simultaneously.

Changes:

- Complete `src/protocol/grpc/multiplexed_flight.rs`.
- Use one schema and dictionary tracker per producer-consumer RPC.
- Use one decoder per RPC and demultiplex decoded batches into the requested output streams.
- Preserve the existing `WorkerChannel::execute_task` return type.
- Keep legacy partition metadata behavior unchanged.
- Preserve LZ4, Zstd, and uncompressed modes.
- Preserve cancellation when one or all output streams are dropped.
- Preserve the current connection-wide receive-buffer byte budget.

Tests:

- Multiple interleaved partitions.
- Dictionary and nested dictionary batches.
- Nested columns and zero-column batches.
- LZ4 and Zstd.
- Malformed and out-of-range partition metadata.
- Drop one lane while consuming siblings.
- Drop all lanes before first poll and during decoding.
- Errors after partial output.

Completion gate:

- Legacy shuffle results and partitioning are unchanged.
- One RPC uses one encoder and one decoder regardless of `K`.
- No grouped-shuffle protocol fields are introduced in this PR.

### PR 2: Add a local grouped hash partitioner

Suggested title:

```text
feat: add grouped hash partition output
```

Goal: provide the exact grouped batch and range representation needed by the producer without
waiting for a new DataFusion release.

Technical architecture:

DataFusion's `BatchPartitioner` already hashes a batch into logical partitions, concatenates every
partition's row indices, performs one Arrow `take_arrays` call, and slices the reordered parent
batch. Its public iterator exposes only the final slices. The parent batch and the offsets that
describe its layout are private implementation details.

```text
DataFusion public result today
------------------------------

input batch
    |
    v
hash rows and build M*K index vectors
    |
    v
one grouped Arrow take
    |
    v
reordered parent batch
    |
    +--> partition 0 slice
    +--> partition 1 slice       public partition_iter result
    +--> ...
    +--> partition M*K-1 slice

The parent batch and slice ranges are no longer available to the caller.
```

The local partitioner stops one step earlier and returns the parent batch with its range table:

```text
Local grouped result
--------------------

GroupedPartitionedBatch
  batch:  [ partition 0 | partition 1 | ... | partition M*K-1 ]
  ranges: [ (0, start, len), (1, start, len), ... ]
```

Because partitions are arranged in ascending order, every consumer owns a consecutive span of `K`
partitions. The distributed router can slice that entire span as one physical batch without
concatenating `K` separate partition batches.

This is algorithm reuse, not `RepartitionExec` reuse. The local type reproduces only the
hash/index/grouped-take portion. It deliberately excludes output channels, coalescing, spill, and
`ExecutionPlan` behavior.

Why it is separate:

- Differential tests can prove exact compatibility with DataFusion before network and concurrency
  code depend on it.
- The API shape can change cheaply while it remains crate-private.
- The resulting implementation and tests provide a concrete basis for the upstream DataFusion PR.

Changes:

- Add `src/execution_plans/grouped_batch_partitioner.rs`.
- Support hash partitioning only.
- Reuse DataFusion's public expression evaluation, hash creation, and `REPARTITION_RANDOM_STATE`.
- Reuse Arrow's public `take_arrays` implementation.
- Copy DataFusion's small strength-reduced remainder implementation so local measurements use the
  same partition-assignment mechanism.
- Reuse `hash_buffer` and per-partition index vectors across calls.
- Concatenate logical partition indices in ascending partition order.
- Perform one `take_arrays` call per input batch.
- Return the parent reordered batch and its partition ranges instead of immediately returning one
  slice per partition.
- Keep all new types crate-private.
- Add a source comment explaining that the implementation is temporary and mirrors DataFusion's
  grouped-take path.

Differential tests:

- Run DataFusion `BatchPartitioner::partition_iter` and the local grouped partitioner on the same
  input.
- Slice the local grouped result and compare every partition with DataFusion's output.
- Cover power-of-two and non-power-of-two partition counts.
- Cover `M * K` greater than the input row count.
- Cover empty batches, zero-column batches, null keys, multiple hash expressions, dictionary
  payloads, and nested payloads.
- Call the partitioner repeatedly to verify buffer clearing and reuse.

Completion gate:

- Every sliced local partition is identical to DataFusion's corresponding output.
- The module contains no consumer, lane, Flight, or worker concepts.
- No production plan selects the new partitioner yet.

### PR 3: Add the grouped producer core

Suggested title:

```text
feat: add grouped shuffle producer
```

Goal: replace `M * K` persistent producer outputs with `M` physical frame outputs while retaining
one logical `M * K` hash assignment.

Technical architecture:

This PR introduces the central performance change. Current `RepartitionExec(Hash(..., M * K))`
treats every final lane as a physical output. It therefore creates output machinery for every
consumer-lane pair.

```text
Current producer
----------------

input partition 0 --> BatchPartitioner --+--> output channel consumer 0, lane 0
input partition 1 --> BatchPartitioner --+--> output channel consumer 0, lane 1
...                                      +--> ...
                                         +--> output channel consumer M-1, lane K-1
                                         +--> M*K channels/coalescers/reservations
```

The grouped producer preserves `M * K` only as a transient logical index space. After one grouped
take, it immediately folds each consecutive set of `K` ranges into one physical destination frame.

```text
Grouped producer
----------------

input partition 0 --> GroupedBatchPartitioner --+
input partition 1 --> GroupedBatchPartitioner --+--> grouped frame router
...                                              |          |
input partition N --> GroupedBatchPartitioner --+          |
                                                            +--> destination 0 queue
                                                            +--> destination 1 queue
                                                            +--> ...
                                                            +--> destination M-1 queue

Persistent output state: M
Transient partition-index vectors per routing task: M*K
```

Each input partition retains its own routing task and partitioner, matching the parallelism model
used by `RepartitionExec`. The design does not merge producer inputs through one partitioner.

A frame queue is more than a Tokio channel. Consumers may request producer destinations at different
times. If destination 0 starts first while destination 1 is absent, routing must still process rows
for both. The destination mailbox therefore owns in-memory buffering, spill, end-of-stream
coordination, and error delivery.

```text
                 shared producer memory budget
                            |
             +--------------+--------------+
             v              v              v
       destination 0  destination 1  destination M-1
       memory queue   memory queue   memory queue
             |              |              |
       spill if needed spill if needed spill if needed
             |              |              |
          consumer 0     consumer 1     consumer M-1
```

Frame coalescing happens at the physical-destination level. This maintains `M` residual buffers
rather than `M * K`. Lane ranges are offset-adjusted when several frames are combined, so the
receiver can still recover every lane.

Why it is separate:

- It proves the core state reduction without involving protobuf or worker-version compatibility.
- Memory, spill, cancellation, and late-consumer behavior can be tested directly.
- The producer remains unreachable from the distributed planner until its lifecycle is complete.

Changes:

- Add `src/execution_plans/grouped_shuffle.rs`.
- Add a crate-private `GroupedShuffleProducer` that owns an input `ExecutionPlan`, final logical
  hash partitioning, destination count, and lanes per destination.
- Initialize the producer state once when the first destination is requested.
- Open every input partition and launch one routing task per input partition.
- Give every routing task an independent `GroupedBatchPartitioner`.
- Convert each consumer's consecutive logical partition range into one zero-copy destination frame.
- Maintain one physical output queue per destination.
- Maintain one coalescer per physical destination, never one per lane.
- Count active input tasks and let the final task flush residual frames and close every destination.
- Broadcast input errors to all active destinations.

Destination frame coalescing:

- Coalesce complete physical frames to the configured row or byte target.
- Concatenate physical frame batches at most once per flush.
- Adjust lane offsets after concatenation.
- Allow a lane to appear in more than one range when several frames are combined.
- Skip concatenation when a single frame already meets the target.

Memory and spill:

- Charge the globally reordered batch once even when several destination slices retain it.
- Attach a shared memory lease to destination frames.
- Add one spillable frame queue per physical destination.
- Store both the Arrow batch and lane-range sidecar for spilled frames.
- Preserve FIFO delivery within each physical destination.
- Release memory reservations and spill files on normal completion, error, or cancellation.
- Verify that an unpolled destination spills rather than blocking other destinations indefinitely.

Metrics:

- `rows_routed`
- `input_batches`
- `logical_partition_count`
- `physical_destination_count`
- `shuffle_frames`
- `frame_rows`
- `frame_bytes`
- `partition_time`
- `frame_coalesce_time`
- `blocked_send_time`
- `spilled_frames`
- `spilled_bytes`
- `max_buffered_bytes`

Tests:

- Exact row and lane routing for several `P`, `M`, and `K` shapes.
- Empty logical partitions and empty physical destinations.
- More logical partitions than rows.
- Multiple input partitions running concurrently.
- Slow and initially unpolled destinations.
- Producer input error after partial output.
- Drop one destination and continue the others.
- Drop all destinations and verify task, reservation, and spill cleanup.

Completion gate:

- The producer creates `M` physical queues, reservations, spill paths, and coalescers.
- Each row is hashed and reordered once.
- The producer works through a direct unit/in-process fixture but is not selected by the distributed
  planner.

### PR 4: Add grouped producer-head runtime preparation

Suggested title:

```text
refactor: separate logical and runtime producer heads
```

Goal: let planning retain ordinary executable DataFusion plans while workers prepare a grouped frame
producer at runtime.

Technical architecture:

The planner and the worker need different representations of the producer head.

During planning, every physical plan must remain a valid DataFusion `ExecutionPlan`. A grouped frame
stream cannot be placed directly in that plan because `ExecutionPlan::execute` returns
`RecordBatch`, not `ShuffleFrame`. The logical plan therefore continues to use the equivalent
`RepartitionExec(Hash(..., M * K))`.

At worker execution time, the producer head is a runtime instruction. The worker can remove the
logical `RepartitionExec` and replace it with `GroupedShuffleProducer`, whose output is consumed
directly by the worker protocol rather than by another DataFusion operator.

```text
Planning representation
-----------------------

producer input
     |
     v
RepartitionExec Hash(..., M*K)
     |
     v
NetworkShuffleExec boundary

This remains a normal, executable DataFusion plan.


Worker runtime representation
-----------------------------

decoded producer plan
     |
     +-- remove logical head RepartitionExec
     v
GroupedShuffleProducer Hash(..., M*K), physical outputs M
     |
     v
TaskOutput::GroupedFrames
```

Separating `insert_logical_plan` from `prepare_runtime` prevents the grouped frame type from leaking
into DataFusion's plan interface. It also makes the replacement explicit instead of relying on a
transport-specific downcast in the gRPC service.

`TaskData` initializes the prepared producer once. Every consumer request for that producer task
must supply the same producer-head configuration. Later requests take different destination streams
from the shared producer state rather than rebuilding or re-executing the producer input.

Why it is separate:

- It is an architectural refactor of producer preparation, independent of protobuf encoding.
- Existing broadcast, coalesce, and legacy repartition heads can be regression-tested before the
  gRPC branch is added.
- Both gRPC and in-process worker channels can consume the same `TaskOutput` abstraction in PR 5.

Changes:

- Add `ProducerHead::GroupedRepartition` in `src/distributed_planner/network_boundary.rs` with
  encoded `Partitioning` and `lanes_per_destination`.
- Split the current producer-head insertion behavior into logical-plan and runtime preparation
  paths.
- For logical planning, insert an ordinary `RepartitionExec(Hash(..., M * K))`. This keeps
  intermediate plans valid and executable.
- For worker runtime preparation, remove the logical head and create `GroupedShuffleProducer`.
- Change `TaskData` to cache a prepared producer enum rather than only `Arc<dyn ExecutionPlan>`.
- Change worker task execution to return either ordinary partition streams or one grouped frame
  stream.

Target internal types:

```rust
pub(crate) enum PreparedProducer {
    RecordBatches(Arc<dyn ExecutionPlan>),
    Grouped(Arc<GroupedShuffleProducer>),
}

pub(crate) enum TaskOutput {
    PartitionStreams(Vec<SendableRecordBatchStream>),
    GroupedFrames(SendableShuffleFrameStream),
}
```

Request semantics:

- Keep the consumer request expressed as its final global partition range:
  `consumer * K..(consumer + 1) * K`.
- Validate that grouped ranges are aligned to `K` and contain exactly `K` partitions.
- Derive the physical destination as `range.start / K`.

Tests:

- Logical producer plans remain executable.
- Legacy producer-head insertion is unchanged.
- Runtime grouped preparation strips the logical `RepartitionExec` exactly once.
- Conflicting producer heads for the same cached task return an explicit error.
- Invalid or unaligned grouped ranges return an explicit error.

Completion gate:

- Planner-facing types still satisfy normal `ExecutionPlan` contracts.
- Grouped runtime output never masquerades as an ordinary `RecordBatch` execution stream.

### PR 5: Add grouped Flight frames and consumer demultiplexing

Suggested title:

```text
feat: transport grouped shuffle frames over Flight
```

Goal: transport one physical destination stream per producer-consumer pair and recover `K` final
streams without receiver-side hashing.

Technical architecture:

PR 3 creates frames and PR 4 makes them available at worker execution. PR 5 defines how those frames
cross the `WorkerChannel` boundary.

```text
Producer worker                                      Consumer worker
---------------                                      ---------------

GroupedShuffleProducer
        |
        v
ShuffleFrame { batch, lane ranges }
        |
        v
one shared Flight IPC encoder
        |
        +-- schema message       no routing metadata
        +-- dictionary message   no routing metadata
        +-- record batch         grouped lane-range metadata
                                |
                                +-------- one gRPC stream -------->
                                                                  |
                                                                  v
                                                         one Flight decoder
                                                                  |
                                                                  v
                                                         validate all ranges
                                                                  |
                                                    +-------------+-------------+
                                                    v             v             v
                                                  lane 0        lane 1        lane K-1
                                                  slice         slice         slice
```

The decoder must validate the complete frame before publishing any lane. Otherwise a malformed later
range could produce partial, externally visible output before the RPC fails.

The client returns the same `Vec<RecordBatchStream>` abstraction used today. `NetworkShuffleExec`
and downstream DataFusion operators do not see `ShuffleFrame`; they only see their normal `K`
partitions. This keeps the new wire representation behind `WorkerChannel`.

IPC schema and dictionary messages describe the whole RPC schema, not an individual lane. Only
record-batch messages carry lane ranges. When a batch is split to respect a Flight message-size
limit, the encoder must intersect every lane range with each split and rebase its offsets.

Why it is separate:

- It isolates wire-format and Arrow IPC risk from producer routing risk.
- The grouped producer can be tested without serialization before this PR.
- In-process and gRPC transports can be required to produce identical public output streams.

Protocol changes:

- Add a new `GroupedRepartitionExecHead` variant to `ExecuteTaskRequest.producer_head`.
- Add versioned grouped frame metadata containing the physical destination and repeated lane ranges.
- Keep the legacy partition metadata field unchanged.
- Regenerate protobuf code with:

```bash
cargo run --manifest-path codegen/Cargo.toml
```

Encoder changes:

- Generalize the multiplexed encoder input to legacy partition batches or grouped frames.
- Reuse one schema and dictionary tracker per RPC.
- Attach routing metadata only to record-batch Flight messages.
- Keep dictionary and schema messages untagged.
- Split oversized frames before encoding, or adjust lane ranges precisely if Arrow splits an encoded
  record batch.
- Preserve configured IPC compression.

Decoder changes:

- Decode each Flight record batch once.
- Validate frame version, destination, lane count, and all ranges before publishing any slice.
- Create zero-copy lane slices and send them to the existing `K` consumer streams.
- Charge decoded memory once and release it after all retaining lane slices are consumed or dropped.
- Keep sibling lanes alive when one lane is dropped.

In-process transport:

- Implement the same grouped-frame demultiplexing without Flight serialization.
- Keep `WorkerChannel::execute_task` returning `Vec<RecordBatchStream>` so frame types remain
  internal.

Tests:

- Round-trip legacy and grouped metadata.
- Reject unknown frame versions.
- Reject duplicate, overlapping, unsorted, incomplete, and out-of-bounds ranges.
- Reject the wrong destination or lane count.
- Exercise dictionary messages, nested dictionaries, LZ4, Zstd, and uncompressed IPC.
- Exercise frame splitting across lane boundaries and within one lane.
- Exercise cancellation before first poll and during decode.
- Compare in-process and gRPC results.

Completion gate:

- The consumer receives `K` ordinary streams with the same rows and lane assignments as current
  main.
- No consumer-side hash evaluation or Arrow take is performed.
- One producer-consumer RPC uses one encoder and one decoder.

### PR 6: Add capability negotiation and opt-in planner selection

Suggested title:

```text
feat: select grouped transport for eligible hash shuffles
```

Goal: enable end-to-end grouped shuffle safely without requiring all deployments to upgrade
atomically.

Technical architecture:

The new producer head and grouped frame metadata are additive, but old workers do not know how to
execute them. The coordinator must choose the transport before any producer begins executing.
Falling back after receiving part of a grouped stream could duplicate rows, so there is no mid-query
retry in another mode.

```text
Coordinator resolves workers
           |
           v
collect worker capabilities and frame versions
           |
           +-- all support grouped + eligible hash boundary
           |              |
           |              v
           |      GroupedRepartition producer head
           |
           +-- any worker lacks support or boundary is ineligible
                          |
                          v
                  legacy RepartitionExec head
```

The planner-visible output remains `K` partitions per consumer in either case. Grouped selection
changes how bytes cross the network, not the final partitioning contract. Because
`logical_partition = hash % (M * K)` is unchanged, two sides of a partitioned join remain aligned
even if grouped transport is selected independently at each boundary.

The explicit modes serve different purposes:

- `legacy` is the rollback path.
- `grouped` is a development and qualification mode that fails early if unsupported.
- `auto` is the rollout mode that selects grouped only when capability and semantic checks pass.

Why it is separate:

- Planner selection should not be mixed with the transport implementation that it enables.
- Mixed-version behavior can be reviewed as an explicit compatibility policy.
- The grouped implementation can remain dormant until every safety condition is enforced.

Changes:

- Add a grouped-shuffle capability and frame-format version to worker information.
- Resolve capability support before dispatching producer tasks.
- Add `legacy`, `grouped`, and `auto` shuffle strategies.
- Keep `legacy` as the default in this PR.
- Make `grouped` fail during planning if any selected worker lacks the capability.
- Make `auto` select grouped only when all involved workers support it and the boundary is eligible.
- Do not retry from grouped to legacy after execution begins.

Initial eligibility:

- `Partitioning::Hash`.
- Unordered output.
- Bounded input.
- `M > 1` and `K > 1`.
- Supported frame version on every involved worker.

Planner behavior:

- `NetworkShuffleExec` continues to advertise `K` final output partitions per consumer.
- The logical producer partitioning remains `Hash(..., M * K)`.
- No whole-join-region rewrite is required because the final partition assignment is identical to
  current main.
- Existing single-task boundary elision remains unchanged.

Compatibility tests:

- New coordinator and new workers use grouped when requested.
- New coordinator and any old worker use legacy in `auto` mode.
- Explicit grouped mode with an old worker fails before execution.
- Old coordinator and new workers continue using legacy.
- Unsupported grouped producer-head and frame versions return explicit errors.
- Legacy wire fixtures remain decodable.

End-to-end tests:

- Multi-worker joins and aggregations match single-node results.
- Both sides of partitioned joins have aligned final lanes.
- TPC-H and TPC-DS correctness tests pass with grouped forced.
- Worker failure, coordinator cancellation, and early `LIMIT` release grouped resources.
- Metrics collection and explain output remain valid.

Completion gate:

- Grouped shuffle is usable through an explicit configuration.
- Legacy remains the default and rollback path.
- Mixed-version clusters never dispatch unsupported grouped requests.

### Upstream DataFusion PR: Expose grouped `BatchPartitioner` output

Open this PR after PR 3 proves the local API shape. It can proceed in parallel with PRs 4 through 6.

Suggested title:

```text
feat: expose grouped batch partition output
```

Goal: expose an intermediate representation DataFusion already constructs, without adding
distributed behavior.

Technical architecture:

The upstream change exposes the boundary already proven by the local prototype. It does not add a
new partitioning algorithm and does not teach DataFusion about workers or shuffle frames.

```text
Before upstream change
----------------------

BatchPartitioner::partition_iter(batch)
    |
    +-- private grouped take creates parent batch + ranges
    +-- private code immediately slices them
             |
             v
       iterator of partition batches


After upstream change
---------------------

BatchPartitioner::partition_grouped(batch)
    |
    v
GroupedPartitionedBatch { parent batch, ranges }
    |
    +-- partition_iter slices ranges for existing callers
    +-- datafusion-distributed groups ranges into destinations
```

`partition_iter` remains the compatibility API and should be implemented on top of the new grouped
result. Existing `RepartitionExec` behavior therefore remains unchanged. The only new capability is
that specialized consumers can retain the grouped parent and decide when to slice it.

Why it follows PR 3:

- PR 3 demonstrates that the parent batch and range table are sufficient for a real consumer.
- It reveals required ownership and accessor semantics before DataFusion commits to a public API.
- The upstream proposal stays small because distributed lifecycle code remains in this repository.

Changes in DataFusion:

- Add public `PartitionRange` and `GroupedPartitionedBatch` types or equivalent accessor-based
  types.
- Add `BatchPartitioner::partition_grouped`.
- Refactor `partition_iter` to slice the grouped result.
- Preserve all existing `partition_iter` behavior and metrics.
- Document that slices of the grouped parent can retain its complete backing allocation.
- Keep worker, destination, lane, transport, and Flight concepts out of DataFusion.

Upstream tests:

- Move or reproduce the local differential tests.
- Prove that `partition_iter` output is unchanged.
- Cover hash, range, and round-robin behavior as required by the public API.
- Cover empty partitions and partition counts greater than row counts.
- Cover nested and dictionary payloads.

The datafusion-distributed prototype is evidence for the API shape and a concrete downstream
consumer. Performance evidence is not required to justify the upstream change because it should not
alter existing `partition_iter` behavior.

### PR 7: Replace the local partitioner with upstream DataFusion

Suggested title:

```text
refactor: use DataFusion grouped batch partition output
```

Goal: remove the temporary duplicated hash partitioning code after the upstream API is released.

Technical architecture:

This PR changes the source of grouped partition results without changing the distributed router or
wire protocol.

```text
Before PR 7                         After PR 7
-----------                         ----------

local GroupedBatchPartitioner       DataFusion BatchPartitioner
             |                                  |
             v                                  v
GroupedPartitionedBatch             GroupedPartitionedBatch
             |                                  |
             +----------> same GroupedShuffleProducer
                                      |
                                      v
                               same ShuffleFrame protocol
```

The local compatibility tests become upstream unit tests. This repository retains an end-to-end
routing test because that test owns the `M * K` to `(destination, lane)` mapping, which is not
DataFusion's responsibility.

Why it is separate:

- A dependency update and deletion of temporary code can be reviewed without changing runtime
  behavior.
- Remote results before and after the swap can confirm that the upstream implementation is
  mechanically equivalent.
- The distributed code stops carrying a copy of DataFusion's hash partitioner.

Changes:

- Update the DataFusion dependency in an isolated commit or prerequisite PR.
- Replace the local partitioner with `BatchPartitioner::partition_grouped`.
- Delete `grouped_batch_partitioner.rs`, including the copied strength reducer.
- Retain distributed frame construction and routing unchanged.
- Keep a focused integration test proving final consumer/lane assignment.

Completion gate:

- Grouped shuffle results are unchanged.
- No duplicated DataFusion hash or grouped-take implementation remains.
- Existing remote benchmark comparisons are still valid after the dependency update.

### PR 8: Enable automatic grouped shuffle selection

Suggested title:

```text
perf: enable automatic grouped hash shuffle
```

Goal: change the default only after correctness, compatibility, resource, and remote performance
evidence is complete.

Technical architecture:

No data-path implementation changes should be necessary in this PR. It changes policy from an
explicit opt-in to capability-driven automatic selection.

```text
Before PR 8                         After PR 8
-----------                         ----------

default = legacy                    default = auto
explicit grouped required           grouped selected when:
                                    - hash boundary is eligible
                                    - every worker supports the frame version
                                    - topology passes deterministic guards

legacy remains available            legacy remains the rollback path
```

Why it is separate:

- The performance claim and the default behavior change are reviewed together.
- Rolling the feature back requires only configuration, not a binary downgrade.
- Implementation PRs can merge without changing existing users' execution paths.

Changes:

- Change the default strategy from `legacy` to `auto`.
- Keep explicit `legacy` as the rollback control for at least one release.
- Document eligibility, mixed-version fallback, metrics, and rollback configuration.
- Add an upgrade-guide entry if the public configuration or rollout behavior requires one.

Completion gate:

- The remote benchmark acceptance criteria below pass.
- Mixed-version and rollback tests pass.
- No unresolved correctness, memory-growth, spill, or cancellation defects remain.

## Remote benchmark plan

Do not add a PR that embeds all implementations in one benchmark binary. Use the existing remote
benchmark service to compare independent branches.

### Candidates

Run remote benchmarks for:

1. Current main at a recorded base SHA.
2. Exact PR #717 at a recorded head SHA.
3. The grouped shuffle branch with grouped mode forced.
4. The grouped shuffle branch after replacing the local partitioner with upstream DataFusion.

Keep the base revision, benchmark harness revision, dataset, worker type, worker count, thread
count, compression, and benchmark options identical. Record every candidate SHA in the result
summary.

### Suites

Use the same representative suites used to qualify PR #717:

```text
benchmark run tpch/sf100
benchmark run tpcds/sf1
```

Run repeated remote jobs on the same SHA when suite noise prevents a conclusion. Do not select
individual favorable runs.

### Evidence to report

- Full-suite wall time and distribution across repeated runs.
- Per-query distribution, with repeated confirmation for material regressions.
- Peak and steady worker memory where available.
- Network bytes and FlightData message count.
- Producer `partition_time` and consumer decode time.
- Maximum grouped producer and consumer buffered bytes.
- Spill bytes and spill frame count.
- Rows and bytes per shuffle frame.
- Number of physical destination queues, reservations, and coalescers.
- Count of grouped versus legacy boundaries under `auto`.

### Acceptance criteria

Before PR 8 changes the default:

- Grouped shuffle improves full-suite wall time over PR #717 on the high-fanout workloads, or
  provides a clearly better aggregate result without material repeatable regressions.
- Producer state scales with `M`, not `M * K`.
- Consumer repartition work is absent.
- Each row goes through one hash assignment and one grouped take.
- Network framing uses one encoder and decoder per producer-consumer RPC.
- Peak memory remains bounded as `K` increases at fixed data volume, `P`, and `M`.
- Slow consumers and spill do not introduce deadlocks or unbounded retained memory.
- Low-fanout regressions are handled by deterministic `auto` eligibility, not by a broad cost model.
- Results remain correct with compression and representative complex Arrow schemas.

## Verification commands

Run the smallest relevant command during each PR, then broaden before merging the end-to-end
feature:

```bash
cargo test grouped_batch_partitioner
cargo test grouped_shuffle
cargo test multiplexed_flight
cargo test worker_client
cargo test network_shuffle
cargo test --features integration
cargo test --features tpch
cargo test --features tpcds
cargo test --features clickbench
```

Run the repository's formatting, lint, and documentation checks required by CI after focused tests
pass.

## Final ownership boundaries

DataFusion owns:

- Physical expression evaluation.
- Hash creation and partition assignment.
- Partition index-buffer reuse.
- One grouped Arrow take.
- The grouped parent batch and logical partition ranges.

DataFusion Distributed owns:

- Mapping `M * K` logical partitions onto `M` consumer tasks.
- Physical destination queues, coalescing, spill, and cancellation.
- `ShuffleFrame` and lane-range validation.
- Producer-head runtime preparation.
- Capability negotiation and planner selection.
- Flight encoding, decoding, and consumer-lane demultiplexing.

Arrow Flight owns the IPC and transport representation. A separate Arrow contribution for
per-record-batch application metadata may later remove local encoder code, but it is not on the
critical path for grouped shuffle.

## Primary implementation risk

The hash and grouped-take algorithm are already understood. The main risk is reproducing
`RepartitionExec`'s multi-output lifecycle for `M` physical frame destinations: late consumers,
bounded memory, shared-buffer accounting, spill, error propagation, and cancellation. Treat those
behaviors as merge requirements for the grouped producer rather than post-implementation
optimizations.
