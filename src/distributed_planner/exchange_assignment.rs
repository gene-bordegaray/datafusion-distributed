use datafusion::common::{Result, plan_err};
use std::ops::Range;
use std::sync::Arc;

/// Assignment rules for network exchange boundaries.
///
/// The planner has two decisions to make when a query plan is crossing a stage / network
/// boundary:
///
/// 1. Decide that a boundary is needed and how many producer/consumer tasks it has.
/// 2. Decide which upstream task and partition each consumer-local output partition should read.
///
/// This module is only responsible for the second decision. Once the planner know the boundary type,
/// they create an [`ExchangeLayout`] and the execution plans use it to resolve concrete reads at
/// runtime.
///
/// Keeping this mapping in one place and explicit makes the network boundary shapes easier to plan.
/// The planner builds an [`ExchangeLayout`], the boundary itself receives the layout, ask it to
/// resolve a local slot, and then perform the network reads the layout describes.
///
/// Upstream read target for one consumer-local output slot.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SlotReadPlan {
    /// Read the same partition from each producer task.
    Fanout {
        producer_tasks: Range<usize>,
        producer_partition: usize,
    },
    /// Read one partition from one producer task.
    Single {
        producer_task: usize,
        producer_partition: usize,
    },
}

/// Assignment for hash repartitioned data crossing a network boundary.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ShuffleExchangeLayout {
    producer_task_count: usize,
    consumer_task_count: usize,
    partitions_per_consumer: usize,
}

impl ShuffleExchangeLayout {
    fn producer_task_range(&self, _consumer_task_idx: usize) -> Range<usize> {
        0..self.producer_task_count
    }

    fn resolve_slot(
        &self,
        consumer_task_idx: usize,
        local_partition_idx: usize,
    ) -> Option<SlotReadPlan> {
        if consumer_task_idx >= self.consumer_task_count
            || local_partition_idx >= self.partitions_per_consumer
        {
            return None;
        }

        Some(SlotReadPlan::Fanout {
            producer_tasks: self.producer_task_range(consumer_task_idx),
            producer_partition: consumer_task_idx * self.partitions_per_consumer
                + local_partition_idx,
        })
    }
}

/// Assignment for preserving partitioning while merging producer task groups.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CoalesceExchangeLayout {
    producer_task_count: usize,
    consumer_task_count: usize,
    partitions_per_producer_task: usize,
    producer_task_ranges: Vec<Range<usize>>,
}

impl CoalesceExchangeLayout {
    fn max_input_task_count_per_consumer(&self) -> usize {
        self.producer_task_ranges
            .iter()
            .map(|range| range.len())
            .max()
            .unwrap_or(0)
    }

    fn max_partition_count_per_consumer(&self) -> usize {
        self.max_input_task_count_per_consumer() * self.partitions_per_producer_task
    }

    fn resolve_slot(
        &self,
        consumer_task_idx: usize,
        local_partition_idx: usize,
    ) -> Option<SlotReadPlan> {
        let producer_task_range = self.producer_task_ranges.get(consumer_task_idx)?;
        let producer_task_offset = local_partition_idx / self.partitions_per_producer_task;
        let producer_partition = local_partition_idx % self.partitions_per_producer_task;
        let producer_task = producer_task_range.clone().nth(producer_task_offset)?;
        Some(SlotReadPlan::Single {
            producer_task,
            producer_partition,
        })
    }
}

/// Assignment for broadcast data crossing a network boundary.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BroadcastExchangeLayout {
    producer_task_count: usize,
    consumer_task_count: usize,
    partitions_per_consumer: usize,
}

impl BroadcastExchangeLayout {
    fn producer_task_range(&self, _consumer_task_idx: usize) -> Range<usize> {
        0..self.producer_task_count
    }

    fn resolve_slot(
        &self,
        consumer_task_idx: usize,
        local_partition_idx: usize,
    ) -> Option<SlotReadPlan> {
        if consumer_task_idx >= self.consumer_task_count
            || local_partition_idx >= self.partitions_per_consumer
        {
            return None;
        }

        Some(SlotReadPlan::Fanout {
            producer_tasks: self.producer_task_range(consumer_task_idx),
            producer_partition: consumer_task_idx * self.partitions_per_consumer
                + local_partition_idx,
        })
    }
}

/// Network exchange slot assignment for a distributed boundary.
///
/// The enum keeps each exchange kind's arithmetic separate while giving execution plans one
/// runtime API: [`ExchangeLayout::resolve_slot`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ExchangeLayout {
    Shuffle(ShuffleExchangeLayout),
    Coalesce(CoalesceExchangeLayout),
    Broadcast(BroadcastExchangeLayout),
}

impl ExchangeLayout {
    pub fn try_shuffle(
        producer_task_count: usize,
        consumer_task_count: usize,
        partitions_per_consumer: usize,
    ) -> Result<Arc<Self>> {
        if producer_task_count == 0 {
            return plan_err!("shuffle exchange requires producer_task_count > 0");
        }
        if consumer_task_count == 0 {
            return plan_err!("shuffle exchange requires consumer_task_count > 0");
        }
        if partitions_per_consumer == 0 {
            return plan_err!("shuffle exchange requires partitions_per_consumer > 0");
        }

        Ok(Arc::new(Self::Shuffle(ShuffleExchangeLayout {
            producer_task_count,
            consumer_task_count,
            partitions_per_consumer,
        })))
    }

    pub fn try_coalesce(
        producer_task_count: usize,
        consumer_task_count: usize,
        partitions_per_producer_task: usize,
    ) -> Result<Arc<Self>> {
        if consumer_task_count == 0 {
            return plan_err!("coalesce exchange requires consumer_task_count > 0");
        }
        if partitions_per_producer_task == 0 {
            return plan_err!("coalesce exchange requires partitions_per_producer_task > 0");
        }

        Ok(Arc::new(Self::Coalesce(CoalesceExchangeLayout {
            producer_task_count,
            consumer_task_count,
            partitions_per_producer_task,
            producer_task_ranges: split_ranges(producer_task_count, consumer_task_count),
        })))
    }

    pub fn try_broadcast(
        producer_task_count: usize,
        consumer_task_count: usize,
        partitions_per_consumer: usize,
    ) -> Result<Arc<Self>> {
        if producer_task_count == 0 {
            return plan_err!("broadcast exchange requires producer_task_count > 0");
        }
        if consumer_task_count == 0 {
            return plan_err!("broadcast exchange requires consumer_task_count > 0");
        }
        if partitions_per_consumer == 0 {
            return plan_err!("broadcast exchange requires partitions_per_consumer > 0");
        }

        Ok(Arc::new(Self::Broadcast(BroadcastExchangeLayout {
            producer_task_count,
            consumer_task_count,
            partitions_per_consumer,
        })))
    }

    pub fn producer_task_count(&self) -> usize {
        match self {
            Self::Shuffle(layout) => layout.producer_task_count,
            Self::Coalesce(layout) => layout.producer_task_count,
            Self::Broadcast(layout) => layout.producer_task_count,
        }
    }

    pub fn consumer_task_count(&self) -> usize {
        match self {
            Self::Shuffle(layout) => layout.consumer_task_count,
            Self::Coalesce(layout) => layout.consumer_task_count,
            Self::Broadcast(layout) => layout.consumer_task_count,
        }
    }

    pub fn max_partition_count_per_consumer(&self) -> usize {
        match self {
            Self::Shuffle(layout) => layout.partitions_per_consumer,
            Self::Coalesce(layout) => layout.max_partition_count_per_consumer(),
            Self::Broadcast(layout) => layout.partitions_per_consumer,
        }
    }

    pub fn partitions_per_producer_task(&self) -> usize {
        match self {
            Self::Shuffle(layout) => layout.partitions_per_consumer * layout.consumer_task_count,
            Self::Coalesce(layout) => layout.partitions_per_producer_task,
            Self::Broadcast(layout) => layout.partitions_per_consumer * layout.consumer_task_count,
        }
    }

    pub fn max_input_task_count_per_consumer(&self) -> Option<usize> {
        match self {
            Self::Coalesce(layout) => Some(layout.max_input_task_count_per_consumer()),
            Self::Shuffle(_) | Self::Broadcast(_) => None,
        }
    }

    pub(crate) fn resolve_slot(
        &self,
        consumer_task_idx: usize,
        local_partition_idx: usize,
    ) -> Option<SlotReadPlan> {
        match self {
            Self::Shuffle(layout) => layout.resolve_slot(consumer_task_idx, local_partition_idx),
            Self::Coalesce(layout) => layout.resolve_slot(consumer_task_idx, local_partition_idx),
            Self::Broadcast(layout) => layout.resolve_slot(consumer_task_idx, local_partition_idx),
        }
    }
}

fn split_ranges(total: usize, groups: usize) -> Vec<Range<usize>> {
    if groups == 0 {
        return Vec::new();
    }

    let base = total / groups;
    let extra = total % groups;
    let mut ranges = Vec::with_capacity(groups);
    let mut start = 0;
    for idx in 0..groups {
        let len = base + usize::from(idx < extra);
        ranges.push(start..start + len);
        start += len;
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_layout_preserves_scaled_fanout_assignment() {
        let layout = ExchangeLayout::try_shuffle(3, 2, 4).unwrap();

        assert_eq!(layout.max_partition_count_per_consumer(), 4);
        assert_eq!(layout.partitions_per_producer_task(), 8);
        assert_eq!(
            layout.resolve_slot(1, 2),
            Some(SlotReadPlan::Fanout {
                producer_tasks: 0..3,
                producer_partition: 6,
            })
        );
        assert_eq!(layout.resolve_slot(1, 4), None);
    }

    #[test]
    fn coalesce_layout_assigns_task_groups() {
        let layout = ExchangeLayout::try_coalesce(3, 2, 4).unwrap();

        assert_eq!(layout.max_input_task_count_per_consumer(), Some(2));
        assert_eq!(layout.max_partition_count_per_consumer(), 8);
        assert_eq!(
            layout.resolve_slot(0, 4),
            Some(SlotReadPlan::Single {
                producer_task: 1,
                producer_partition: 0,
            })
        );
        assert_eq!(
            layout.resolve_slot(1, 3),
            Some(SlotReadPlan::Single {
                producer_task: 2,
                producer_partition: 3,
            })
        );
        assert_eq!(layout.resolve_slot(1, 4), None);
    }

    #[test]
    fn broadcast_layout_preserves_per_consumer_partition_assignment() {
        let layout = ExchangeLayout::try_broadcast(2, 3, 4).unwrap();

        assert_eq!(layout.max_partition_count_per_consumer(), 4);
        assert_eq!(layout.partitions_per_producer_task(), 12);
        assert_eq!(
            layout.resolve_slot(2, 1),
            Some(SlotReadPlan::Fanout {
                producer_tasks: 0..2,
                producer_partition: 9,
            })
        );
        assert_eq!(layout.resolve_slot(2, 4), None);
    }
}
