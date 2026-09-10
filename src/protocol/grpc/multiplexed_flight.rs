use super::generated::worker as pb;
use super::spawn_select_all::MemoryFootPrint;
use crate::common::now_ns;
use arrow_flight::error::{FlightError, Result};
use arrow_flight::{FlightData, SchemaAsIpc};
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::{DataType, Field, FieldRef, Fields, Schema, SchemaRef};
use datafusion::arrow::ipc::writer::{
    DictionaryTracker, IpcDataGenerator, IpcWriteContext, IpcWriteOptions,
};
use futures::stream::BoxStream;
use futures::{Stream, StreamExt};
use prost::Message;
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};

pub(super) struct PartitionedBatch {
    pub partition: usize,
    pub batch: RecordBatch,
}

impl MemoryFootPrint for PartitionedBatch {
    fn get_memory_size(&self) -> usize {
        self.batch.get_array_memory_size()
    }
}

/// Encodes partitioned batches with one schema and one shared IPC dictionary state.
/// Partition identity is attached only to record batch messages; schema and dictionary
/// messages remain in-band for the single decoder on the receiving side.
pub(super) struct MultiplexedFlightDataEncoder {
    inner: BoxStream<'static, Result<PartitionedBatch>>,
    options: IpcWriteOptions,
    data_gen: IpcDataGenerator,
    dictionary_tracker: DictionaryTracker,
    ipc_write_context: IpcWriteContext,
    queue: VecDeque<FlightData>,
    done: bool,
}

impl MultiplexedFlightDataEncoder {
    pub fn new<S>(inner: S, schema: SchemaRef, options: IpcWriteOptions) -> Self
    where
        S: Stream<Item = Result<PartitionedBatch>> + Send + 'static,
    {
        let mut dictionary_tracker = DictionaryTracker::new(false);
        let schema = prepare_schema_for_flight(&schema, &mut dictionary_tracker);
        let schema_message = SchemaAsIpc::new(&schema, &options).into();

        Self {
            inner: inner.boxed(),
            options,
            data_gen: IpcDataGenerator::default(),
            dictionary_tracker,
            ipc_write_context: IpcWriteContext::default(),
            queue: VecDeque::from([schema_message]),
            done: false,
        }
    }

    fn encode_batch(&mut self, partitioned: PartitionedBatch) -> Result<()> {
        self.ipc_write_context.set_reserve_scratch(false);
        let (dictionaries, encoded_batch) = self.data_gen.encode(
            &partitioned.batch,
            &mut self.dictionary_tracker,
            &self.options,
            &mut self.ipc_write_context,
        )?;

        self.queue.extend(dictionaries.into_iter().map(Into::into));

        let metadata = pb::FlightAppMetadata {
            partition: partitioned.partition as u64,
            created_timestamp_unix_nanos: now_ns::<u64>(),
        };
        let batch: FlightData = encoded_batch.into();
        self.queue
            .push_back(batch.with_app_metadata(metadata.encode_to_vec()));
        Ok(())
    }
}

impl Stream for MultiplexedFlightDataEncoder {
    type Item = Result<FlightData>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(message) = self.queue.pop_front() {
                return Poll::Ready(Some(Ok(message)));
            }
            if self.done {
                return Poll::Ready(None);
            }

            match ready!(self.inner.poll_next_unpin(cx)) {
                None => self.done = true,
                Some(Err(error)) => {
                    self.done = true;
                    return Poll::Ready(Some(Err(error)));
                }
                Some(Ok(batch)) => {
                    if let Err(error) = self.encode_batch(batch) {
                        self.done = true;
                        self.queue.clear();
                        return Poll::Ready(Some(Err(FlightError::from(error))));
                    }
                }
            }
        }
    }
}

fn prepare_schema_for_flight(
    schema: &Schema,
    dictionary_tracker: &mut DictionaryTracker,
) -> Schema {
    let fields: Fields = schema
        .fields()
        .iter()
        .map(|field| prepare_field_for_flight(field, dictionary_tracker))
        .collect();
    Schema::new(fields).with_metadata(schema.metadata().clone())
}

// Dictionary IDs must be assigned in the same depth-first order used by IpcDataGenerator.
fn prepare_field_for_flight(field: &FieldRef, dictionary_tracker: &mut DictionaryTracker) -> Field {
    let data_type = match field.data_type() {
        DataType::List(inner) => {
            DataType::List(prepare_field_for_flight(inner, dictionary_tracker).into())
        }
        DataType::LargeList(inner) => {
            DataType::LargeList(prepare_field_for_flight(inner, dictionary_tracker).into())
        }
        DataType::ListView(inner) => {
            DataType::ListView(prepare_field_for_flight(inner, dictionary_tracker).into())
        }
        DataType::LargeListView(inner) => {
            DataType::LargeListView(prepare_field_for_flight(inner, dictionary_tracker).into())
        }
        DataType::FixedSizeList(inner, size) => DataType::FixedSizeList(
            prepare_field_for_flight(inner, dictionary_tracker).into(),
            *size,
        ),
        DataType::Struct(fields) => DataType::Struct(
            fields
                .iter()
                .map(|field| prepare_field_for_flight(field, dictionary_tracker))
                .collect::<Vec<_>>()
                .into(),
        ),
        DataType::Union(fields, mode) => DataType::Union(
            fields
                .iter()
                .map(|(id, field)| {
                    (
                        id,
                        Arc::new(prepare_field_for_flight(field, dictionary_tracker)),
                    )
                })
                .collect(),
            *mode,
        ),
        DataType::Dictionary(_, value_type) => {
            let values = Arc::new(Field::new("values", value_type.as_ref().clone(), true));
            prepare_field_for_flight(&values, dictionary_tracker);
            dictionary_tracker.next_dict_id();
            field.data_type().clone()
        }
        DataType::RunEndEncoded(run_ends, values) => DataType::RunEndEncoded(
            run_ends.clone(),
            prepare_field_for_flight(values, dictionary_tracker).into(),
        ),
        DataType::Map(inner, sorted) => DataType::Map(
            prepare_field_for_flight(inner, dictionary_tracker).into(),
            *sorted,
        ),
        data_type => data_type.clone(),
    };

    #[allow(deprecated)]
    let prepared = if matches!(field.data_type(), DataType::Dictionary(_, _)) {
        Field::new_dict(
            field.name(),
            data_type,
            field.is_nullable(),
            0,
            field.dict_is_ordered().unwrap_or_default(),
        )
    } else {
        Field::new(field.name(), data_type, field.is_nullable())
    };
    prepared.with_metadata(field.metadata().clone())
}
