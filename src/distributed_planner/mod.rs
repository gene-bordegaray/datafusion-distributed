mod distributed_config;
mod distributed_query_planner;
mod inject_network_boundaries;
mod insert_broadcast;
mod insert_local_exchange_split;
mod network_boundary;
mod partial_reduce_below_network_shuffles;
mod prepare_network_boundaries;
mod session_state_builder_ext;
mod task_estimator;

pub use distributed_config::{
    DistributedConfig, LOCAL_EXCHANGE_SPLIT_MODE_FINAL_AGG,
    LOCAL_EXCHANGE_SPLIT_MODE_FINAL_AGG_AND_JOIN, LOCAL_EXCHANGE_SPLIT_MODE_OFF,
};
pub use network_boundary::{NetworkBoundary, NetworkBoundaryExt};
pub use session_state_builder_ext::SessionStateBuilderExt;
pub use task_estimator::{TaskCountAnnotation, TaskEstimation, TaskEstimator, TaskRoutingContext};
pub(crate) use task_estimator::{get_distributed_task_estimator, set_distributed_task_estimator};

#[cfg(test)]
pub(crate) async fn insert_network_boundaries_for_test(
    plan: std::sync::Arc<dyn datafusion::physical_plan::ExecutionPlan>,
    cfg: &datafusion::config::ConfigOptions,
) -> datafusion::common::Result<Option<std::sync::Arc<dyn datafusion::physical_plan::ExecutionPlan>>>
{
    use datafusion::common::tree_node::TreeNode;

    let plan = inject_network_boundaries::inject_network_boundaries(plan, cfg).await?;
    let plan = prepare_network_boundaries::prepare_network_boundaries(plan)?;
    if plan.exists(|plan| Ok(plan.is_network_boundary()))? {
        Ok(Some(plan))
    } else {
        Ok(None)
    }
}
