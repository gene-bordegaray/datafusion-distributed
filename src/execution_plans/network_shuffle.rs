use crate::common::require_one_child;
use crate::execution_plans::common::scale_partitioning;
use crate::stage::{LocalStage, Stage};
use crate::worker::WorkerConnectionPool;
use crate::{DistributedTaskContext, NetworkBoundary};
use datafusion::common::tree_node::{Transformed, TreeNodeRecursion};
use datafusion::common::{Result, exec_err, not_impl_err, plan_err};
use datafusion::error::DataFusionError;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::Partitioning;
use datafusion::physical_expr_common::metrics::MetricsSet;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, EmptyRecordBatchStream, ExecutionPlan, PlanProperties,
};
use std::any::Any;
use std::fmt::Formatter;
use std::ops::Range;
use std::sync::Arc;
use uuid::Uuid;

/// [ExecutionPlan] implementation that shuffles data across the network in a distributed context.
///
/// The easiest way of thinking about this node is as a plan [RepartitionExec] node that is
/// capable of fanning out the different produced partitions to different tasks.
/// This allows redistributing data across different tasks in different stages, so that different
/// physical machines can make progress on different non-overlapping sets of data.
///
/// This node allows fanning out of data from N tasks to M tasks, with N and M being arbitrary non-zero
/// positive numbers. Here are some examples of how data can be shuffled in different scenarios:
///
/// # 1 to many
///
/// ```text
/// ┌───────────────────────────┐  ┌───────────────────────────┐ ┌───────────────────────────┐     ■
/// │    NetworkShuffleExec     │  │    NetworkShuffleExec     │ │    NetworkShuffleExec     │     │
/// │         (task 1)          │  │         (task 2)          │ │         (task 3)          │     │
/// └┬─┬┬─┬┬─┬──────────────────┘  └─────────┬─┬┬─┬┬─┬─────────┘ └──────────────────┬─┬┬─┬┬─┬┘  Stage N+1
///  │1││2││3│                               │4││5││6│                              │7││8││9│      │
///  └─┘└─┘└─┘                               └─┘└─┘└─┘                              └─┘└─┘└─┘      │
///   ▲  ▲  ▲                                 ▲  ▲  ▲                                ▲  ▲  ▲       ■
///   └──┴──┴────────────────────────┬──┬──┐  │  │  │  ┌──┬──┬───────────────────────┴──┴──┘
///                                  │  │  │  │  │  │  │  │  │                                     ■
///                                 ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐                                    │
///                                 │1││2││3││4││5││6││7││8││9│                                    │
///                                ┌┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴┐                                Stage N
///                                │      RepartitionExec      │                                   │
///                                │         (task 1)          │                                   │
///                                └───────────────────────────┘                                   ■
/// ```
///
/// # many to 1
///
/// ```text
///                                ┌───────────────────────────┐                                   ■
///                                │    NetworkShuffleExec     │                                   │
///                                │         (task 1)          │                                   │
///                                └┬─┬┬─┬┬─┬┬─┬┬─┬┬─┬┬─┬┬─┬┬─┬┘                                Stage N+1
///                                 │1││2││3││4││5││6││7││8││9│                                    │
///                                 └─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘                                    │
///                                 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲                                    ■
///   ┌──┬──┬──┬──┬──┬──┬──┬──┬─────┴┼┴┴┼┴┴┼┴┴┼┴┴┼┴┴┼┴┴┼┴┴┼┴┴┼┴────┬──┬──┬──┬──┬──┬──┬──┬──┐
///   │  │  │  │  │  │  │  │  │      │  │  │  │  │  │  │  │  │     │  │  │  │  │  │  │  │  │       ■
///  ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐    ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐   ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐      │
///  │1││2││3││4││5││6││7││8││9│    │1││2││3││4││5││6││7││8││9│   │1││2││3││4││5││6││7││8││9│      │
/// ┌┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴┐  ┌┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴┐ ┌┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴┐  Stage N
/// │      RepartitionExec      │  │      RepartitionExec      │ │      RepartitionExec      │     │
/// │         (task 1)          │  │         (task 2)          │ │         (task 3)          │     │
/// └───────────────────────────┘  └───────────────────────────┘ └───────────────────────────┘     ■
/// ```
///
/// # many to many
///
/// ```text
///                    ┌───────────────────────────┐  ┌───────────────────────────┐                ■
///                    │    NetworkShuffleExec     │  │    NetworkShuffleExec     │                │
///                    │         (task 1)          │  │         (task 2)          │                │
///                    └┬─┬┬─┬┬─┬┬─┬───────────────┘  └───────────────┬─┬┬─┬┬─┬┬─┬┘             Stage N+1
///                     │1││2││3││4│                                  │5││6││7││8│                 │
///                     └─┘└─┘└─┘└─┘                                  └─┘└─┘└─┘└─┘                 │
///                     ▲▲▲▲▲▲▲▲▲▲▲▲                                  ▲▲▲▲▲▲▲▲▲▲▲▲                 ■
///     ┌──┬──┬──┬──┬──┬┴┴┼┴┴┼┴┴┴┴┴┴───┬──┬──┬──┬──┬──┬──┬──┬────────┬┴┴┼┴┴┼┴┴┼┴┴┼──┬──┬──┐
///     │  │  │  │  │  │  │  │         │  │  │  │  │  │  │  │        │  │  │  │  │  │  │  │        ■
///    ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐       ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐      ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐       │
///    │1││2││3││4││5││6││7││8│       │1││2││3││4││5││6││7││8│      │1││2││3││4││5││6││7││8│       │
/// ┌──┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴─┐  ┌──┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴─┐ ┌──┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴─┐  Stage N
/// │      RepartitionExec      │  │      RepartitionExec      │ │      RepartitionExec      │     │
/// │         (task 1)          │  │         (task 2)          │ │         (task 3)          │     │
/// └───────────────────────────┘  └───────────────────────────┘ └───────────────────────────┘     ■
/// ```
///
/// The communication between two stages across a [NetworkShuffleExec] has two implications:
///
/// - Each task in Stage N+1 gathers data from all tasks in Stage N
/// - The total number of partitions across all tasks in Stage N+1 is equal to the
///   number of partitions in a single task in Stage N. (e.g. (1,2,3,4)+(5,6,7,8) = (1,2,3,4,5,6,7,8) )
///
/// This node has two variants.
/// 1. Pending: acts as a placeholder for the distributed optimization step to mark it as ready.
/// 2. Ready: runs within a distributed stage and queries the next input stage over the network
///    using Arrow Flight.
#[derive(Debug, Clone)]
pub struct NetworkShuffleExec {
    /// the properties we advertise for this execution plan
    pub(crate) properties: Arc<PlanProperties>,
    pub(crate) input_stage: Stage,
    pub(crate) worker_connections: WorkerConnectionPool,
}

impl NetworkShuffleExec {
    pub(crate) fn scale_input(
        plan: Arc<dyn ExecutionPlan>,
        consumer_partitions: usize,
        _consumer_task_count: usize,
    ) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
        let Some(repartition_exec) = plan.as_any().downcast_ref::<RepartitionExec>() else {
            return Ok(Transformed::no(plan));
        };

        let child = require_one_child(repartition_exec.children())?;
        let partitioning =
            scale_partitioning(repartition_exec.partitioning(), |_| consumer_partitions);

        // Keep one logical hash partition space across all consumer tasks. Each consumer task
        // fetches only the subset of those logical partitions that it owns.
        let scaled = Arc::new(RepartitionExec::try_new(child, partitioning)?);
        Ok(Transformed::new(scaled, true, TreeNodeRecursion::Stop))
    }

    pub(crate) fn from_stage(input_stage: LocalStage) -> Self {
        Self {
            properties: input_stage.plan.properties().clone(),
            worker_connections: WorkerConnectionPool::new(0),
            input_stage: Stage::Local(input_stage),
        }
    }

    /// Creates a new [NetworkShuffleExec] fed by the provided [RepartitionExec]. The input plan
    /// will be executed in a remote worker in `producer_tasks` number of tasks.
    pub fn try_new(input: Arc<dyn ExecutionPlan>, producer_tasks: usize) -> Result<Self> {
        let Some(r_exec) = input.as_any().downcast_ref::<RepartitionExec>() else {
            return plan_err!("The input of a NetworkShuffleExec can only be a RepartitionExec");
        };
        if !matches!(r_exec.partitioning(), Partitioning::Hash(_, _)) {
            return plan_err!("The input of a NetworkShuffleExec must be hash partitioned");
        }

        Ok(Self::from_stage(LocalStage {
            // At this point, query_id and num are just placeholders that will be filled by
            // prepare_network_boundaries.rs. Users are not expected to provide valid values for
            // these two parameters.
            query_id: Uuid::nil(),
            num: 0,
            plan: input,
            tasks: producer_tasks,
        }))
    }
}

fn consumer_partition_range(
    logical_partition_count: usize,
    consumer_task_count: usize,
    consumer_task_index: usize,
) -> Option<Range<usize>> {
    if consumer_task_count == 0 || consumer_task_index >= consumer_task_count {
        return None;
    }

    let base = logical_partition_count / consumer_task_count;
    let extra = logical_partition_count % consumer_task_count;
    let start = consumer_task_index * base + consumer_task_index.min(extra);
    let len = base + usize::from(consumer_task_index < extra);

    Some(start..start + len)
}

fn consumer_owns_partition(partition_range: &Range<usize>, partition: usize) -> bool {
    partition >= partition_range.start && partition < partition_range.end
}

impl NetworkBoundary for NetworkShuffleExec {
    fn input_stage(&self) -> &Stage {
        &self.input_stage
    }

    fn with_input_stage(&self, input_stage: Stage) -> Result<Arc<dyn ExecutionPlan>> {
        let mut self_clone = self.clone();
        self_clone.worker_connections = WorkerConnectionPool::new(input_stage.task_count());
        self_clone.input_stage = input_stage;
        Ok(Arc::new(self_clone))
    }
}

impl DisplayAs for NetworkShuffleExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        let input_tasks = self.input_stage.task_count();
        let partitions = self.properties.partitioning.partition_count();
        let stage = self.input_stage.num();
        write!(
            f,
            "[Stage {stage}] => NetworkShuffleExec: output_partitions={partitions}, input_tasks={input_tasks}",
        )
    }
}

impl ExecutionPlan for NetworkShuffleExec {
    fn name(&self) -> &str {
        "NetworkShuffleExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        match &self.input_stage.local_plan() {
            Some(v) => vec![v],
            None => vec![],
        }
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        let mut self_clone = self.as_ref().clone();
        match &mut self_clone.input_stage {
            Stage::Local(local) => {
                local.plan = require_one_child(children)?;
            }
            Stage::Remote(_) => not_impl_err!("NetworkBoundary cannot accept children")?,
        }
        Ok(Arc::new(self_clone))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream, DataFusionError> {
        let remote_stage = match &self.input_stage {
            Stage::Local(local) => return local.execute(partition, context),
            Stage::Remote(remote_stage) => remote_stage,
        };

        let task_context = DistributedTaskContext::from_ctx(&context);
        let Some(partition_range) = consumer_partition_range(
            self.properties.partitioning.partition_count(),
            task_context.task_count,
            task_context.task_index,
        ) else {
            return exec_err!(
                "NetworkShuffleExec invalid task context: task_index={} task_count={}",
                task_context.task_index,
                task_context.task_count
            );
        };

        if !consumer_owns_partition(&partition_range, partition) {
            return Ok(Box::pin(EmptyRecordBatchStream::new(self.schema())));
        }

        let mut streams = Vec::with_capacity(remote_stage.workers.len());
        for input_task_index in 0..remote_stage.workers.len() {
            let worker_connection = self.worker_connections.get_or_init_worker_connection(
                remote_stage,
                partition_range.clone(),
                input_task_index,
                &context,
            )?;

            let stream = worker_connection.stream_partition(partition, |_meta| {})?;
            streams.push(stream);
        }

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            futures::stream::select_all(streams),
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.worker_connections.metrics.clone_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::{consumer_owns_partition, consumer_partition_range};

    #[test]
    fn consumer_partition_range_splits_partitions_evenly() {
        assert_eq!(consumer_partition_range(8, 2, 0), Some(0..4));
        assert_eq!(consumer_partition_range(8, 2, 1), Some(4..8));

        assert_eq!(consumer_partition_range(10, 3, 0), Some(0..4));
        assert_eq!(consumer_partition_range(10, 3, 1), Some(4..7));
        assert_eq!(consumer_partition_range(10, 3, 2), Some(7..10));
    }

    #[test]
    fn consumer_partition_range_allows_more_consumers_than_partitions() {
        assert_eq!(consumer_partition_range(2, 4, 0), Some(0..1));
        assert_eq!(consumer_partition_range(2, 4, 1), Some(1..2));
        assert_eq!(consumer_partition_range(2, 4, 2), Some(2..2));
        assert_eq!(consumer_partition_range(2, 4, 3), Some(2..2));
    }

    #[test]
    fn consumer_partition_range_rejects_invalid_consumers() {
        assert_eq!(consumer_partition_range(4, 0, 0), None);
        assert_eq!(consumer_partition_range(4, 2, 2), None);
    }

    #[test]
    fn consumer_owns_partition_checks_half_open_range() {
        assert!(!consumer_owns_partition(&(2..5), 1));
        assert!(consumer_owns_partition(&(2..5), 2));
        assert!(consumer_owns_partition(&(2..5), 4));
        assert!(!consumer_owns_partition(&(2..5), 5));
        assert!(!consumer_owns_partition(&(2..2), 2));
    }
}
