#include "lsh/lsh_index_physical_create.hpp"

#include "duckdb/catalog/catalog_entry/duck_index_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/exception/transaction_exception.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/index_storage_info.hpp"
#include "duckdb/storage/storage_manager.hpp"
#include "duckdb/storage/table_io_manager.hpp"
#include "duckdb/storage/table/column_data.hpp"
#include "duckdb/parallel/base_pipeline_event.hpp"
#include "duckdb/parallel/executor_task.hpp"

#include "lsh/lsh_index.hpp"

namespace duckdb {

PhysicalCreateLSHIndex::PhysicalCreateLSHIndex(PhysicalPlan &physical_plan, const vector<LogicalType> &types_p,
                                               TableCatalogEntry &table_p, const vector<column_t> &column_ids,
                                               unique_ptr<CreateIndexInfo> info,
                                               vector<unique_ptr<Expression>> unbound_expressions,
                                               idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, types_p, estimated_cardinality),
      table(table_p.Cast<DuckTableEntry>()), info(std::move(info)), unbound_expressions(std::move(unbound_expressions)),
      sorted(false) {
	for (auto &column_id : column_ids) {
		storage_ids.push_back(table.GetColumns().LogicalToPhysical(LogicalIndex(column_id)).index);
	}
}

//-------------------------------------------------------------
// Global State
//-------------------------------------------------------------
class CreateLSHIndexGlobalState final : public GlobalSinkState {
public:
	CreateLSHIndexGlobalState(const PhysicalOperator &op_p) : op(op_p) {
	}

	const PhysicalOperator &op;
	unique_ptr<LSHIndex> global_index;

	mutex glock;
	unique_ptr<ColumnDataCollection> collection;
	shared_ptr<ClientContext> context;

	ColumnDataParallelScanState scan_state;

	atomic<bool> is_building = {false};
	atomic<idx_t> loaded_count = {0};
	atomic<idx_t> built_count = {0};
};

unique_ptr<GlobalSinkState> PhysicalCreateLSHIndex::GetGlobalSinkState(ClientContext &context) const {
	auto gstate = make_uniq<CreateLSHIndexGlobalState>(*this);

	vector<LogicalType> data_types = {unbound_expressions[0]->return_type, LogicalType::ROW_TYPE};
	gstate->collection = make_uniq<ColumnDataCollection>(BufferManager::GetBufferManager(context), data_types);
	gstate->context = context.shared_from_this();

	auto &storage = table.GetStorage();
	auto &table_manager = TableIOManager::Get(storage);
	auto &constraint_type = info->constraint_type;
	auto &db = storage.db;

	gstate->global_index = make_uniq<LSHIndex>(info->index_name, constraint_type, storage_ids, table_manager,
	                                           unbound_expressions, db, info->options, IndexStorageInfo(),
	                                           estimated_cardinality);

	return std::move(gstate);
}

//-------------------------------------------------------------
// Local State
//-------------------------------------------------------------
class CreateLSHIndexLocalState final : public LocalSinkState {
public:
	unique_ptr<ColumnDataCollection> collection;
	ColumnDataAppendState append_state;
};

unique_ptr<LocalSinkState> PhysicalCreateLSHIndex::GetLocalSinkState(ExecutionContext &context) const {
	auto state = make_uniq<CreateLSHIndexLocalState>();

	vector<LogicalType> data_types = {unbound_expressions[0]->return_type, LogicalType::ROW_TYPE};
	state->collection = make_uniq<ColumnDataCollection>(BufferManager::GetBufferManager(context.client), data_types);
	state->collection->InitializeAppend(state->append_state);
	return std::move(state);
}

//-------------------------------------------------------------
// Sink
//-------------------------------------------------------------
SinkResultType PhysicalCreateLSHIndex::Sink(ExecutionContext &context, DataChunk &chunk,
                                            OperatorSinkInput &input) const {
	auto &lstate = input.local_state.Cast<CreateLSHIndexLocalState>();
	auto &gstate = input.global_state.Cast<CreateLSHIndexGlobalState>();

	lstate.collection->Append(lstate.append_state, chunk);
	gstate.loaded_count += chunk.size();
	return SinkResultType::NEED_MORE_INPUT;
}

//-------------------------------------------------------------
// Combine
//-------------------------------------------------------------
SinkCombineResultType PhysicalCreateLSHIndex::Combine(ExecutionContext &context,
                                                      OperatorSinkCombineInput &input) const {
	auto &gstate = input.global_state.Cast<CreateLSHIndexGlobalState>();
	auto &lstate = input.local_state.Cast<CreateLSHIndexLocalState>();

	if (lstate.collection->Count() == 0) {
		return SinkCombineResultType::FINISHED;
	}

	lock_guard<mutex> l(gstate.glock);
	if (!gstate.collection) {
		gstate.collection = std::move(lstate.collection);
	} else {
		gstate.collection->Combine(*lstate.collection);
	}

	return SinkCombineResultType::FINISHED;
}

//-------------------------------------------------------------
// Finalize
//-------------------------------------------------------------
class LSHIndexConstructTask final : public ExecutorTask {
public:
	LSHIndexConstructTask(shared_ptr<Event> event_p, ClientContext &context, CreateLSHIndexGlobalState &gstate_p,
	                      size_t thread_id_p, const PhysicalCreateLSHIndex &op_p)
	    : ExecutorTask(context, std::move(event_p), op_p), gstate(gstate_p), thread_id(thread_id_p),
	      local_scan_state() {
		gstate.collection->InitializeScanChunk(scan_chunk);
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		auto &collection = gstate.collection;

		while (collection->Scan(gstate.scan_state, local_scan_state, scan_chunk)) {
			const auto count = scan_chunk.size();
			auto &vec_vec = scan_chunk.data[0];
			auto &data_vec = ArrayVector::GetEntry(vec_vec);
			auto &rowid_vec = scan_chunk.data[1];

			const auto array_size = ArrayType::GetSize(vec_vec.GetType());

			UnifiedVectorFormat vec_format;
			UnifiedVectorFormat data_format;
			UnifiedVectorFormat rowid_format;

			vec_vec.ToUnifiedFormat(count, vec_format);
			data_vec.ToUnifiedFormat(count * array_size, data_format);
			rowid_vec.ToUnifiedFormat(count, rowid_format);

			const auto row_ptr = UnifiedVectorFormat::GetData<row_t>(rowid_format);
			const auto data_ptr = UnifiedVectorFormat::GetData<float>(data_format);

			for (idx_t i = 0; i < count; i++) {
				const auto vec_idx = vec_format.sel->get_index(i);
				const auto row_idx = rowid_format.sel->get_index(i);

				const auto vec_valid = vec_format.validity.RowIsValid(vec_idx);
				const auto rowid_valid = rowid_format.validity.RowIsValid(row_idx);
				if (!vec_valid || !rowid_valid) {
					executor.PushError(
					    ErrorData("Invalid data in LSH index construction: Cannot construct index with NULL values."));
					return TaskExecutionResult::TASK_ERROR;
				}

				std::vector<float> vec(array_size);
				for (idx_t d = 0; d < array_size; d++) {
					vec[d] = data_ptr[vec_idx * array_size + d];
				}

				gstate.global_index->AddVector(row_ptr[row_idx], vec);
			}

			gstate.built_count += count;

			if (mode == TaskExecutionMode::PROCESS_PARTIAL) {
				return TaskExecutionResult::TASK_NOT_FINISHED;
			}
		}

		event->FinishTask();
		return TaskExecutionResult::TASK_FINISHED;
	}

private:
	CreateLSHIndexGlobalState &gstate;
	size_t thread_id;

	DataChunk scan_chunk;
	ColumnDataLocalScanState local_scan_state;
};

class LSHIndexConstructionEvent final : public BasePipelineEvent {
public:
	LSHIndexConstructionEvent(const PhysicalCreateLSHIndex &op_p, CreateLSHIndexGlobalState &gstate_p,
	                          Pipeline &pipeline_p, CreateIndexInfo &info_p, const vector<column_t> &storage_ids_p,
	                          DuckTableEntry &table_p)
	    : BasePipelineEvent(pipeline_p), op(op_p), gstate(gstate_p), info(info_p), storage_ids(storage_ids_p),
	      table(table_p) {
	}

	const PhysicalCreateLSHIndex &op;
	CreateLSHIndexGlobalState &gstate;
	CreateIndexInfo &info;
	const vector<column_t> &storage_ids;
	DuckTableEntry &table;

public:
	void Schedule() override {
		auto &context = pipeline->GetClientContext();

		vector<shared_ptr<Task>> construct_tasks;
		construct_tasks.push_back(make_uniq<LSHIndexConstructTask>(shared_from_this(), context, gstate, 0, op));
		SetTasks(std::move(construct_tasks));
	}

	void FinishEvent() override {
		gstate.global_index->SetDirty();
		gstate.global_index->SyncSize();

		auto &storage = table.GetStorage();

		if (!storage.IsRoot()) {
			throw TransactionException("Cannot create index on non-root transaction");
		}

		auto &schema = table.schema;
		info.column_ids = storage_ids;

		if (schema.GetEntry(schema.GetCatalogTransaction(*gstate.context), CatalogType::INDEX_ENTRY, info.index_name)) {
			if (info.on_conflict != OnCreateConflict::IGNORE_ON_CONFLICT) {
				throw CatalogException("Index with name \"%s\" already exists", info.index_name);
			}
		}

		const auto index_entry = schema.CreateIndex(schema.GetCatalogTransaction(*gstate.context), info, table).get();
		D_ASSERT(index_entry);

		auto &duck_index = index_entry->Cast<DuckIndexEntry>();
		duck_index.initial_index_size = 0;

		storage.AddIndex(std::move(gstate.global_index));
	}
};

SinkFinalizeType PhysicalCreateLSHIndex::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                  OperatorSinkFinalizeInput &input) const {
	auto &gstate = input.global_state.Cast<CreateLSHIndexGlobalState>();
	auto &collection = gstate.collection;

	gstate.is_building = true;
	collection->InitializeScan(gstate.scan_state, ColumnDataScanProperties::ALLOW_ZERO_COPY);

	auto new_event = make_shared_ptr<LSHIndexConstructionEvent>(*this, gstate, pipeline, *info, storage_ids, table);
	event.InsertEvent(std::move(new_event));

	return SinkFinalizeType::READY;
}

ProgressData PhysicalCreateLSHIndex::GetSinkProgress(ClientContext &context, GlobalSinkState &gstate,
                                                     ProgressData source_progress) const {
	ProgressData res;
	const auto &state = gstate.Cast<CreateLSHIndexGlobalState>();

	if (!state.is_building) {
		res.done = state.loaded_count + 0.0;
		res.total = estimated_cardinality + estimated_cardinality;
	} else {
		res.done = state.loaded_count + state.built_count;
		res.total = state.loaded_count + state.loaded_count;
	}
	return res;
}

} // namespace duckdb