#pragma once

#include "duckdb/common/assert.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/execution/index/bound_index.hpp"
#include "duckdb/optimizer/matcher/expression_matcher.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/storage/index_storage_info.hpp"
#include "duckdb/storage/storage_lock.hpp"
#include "duckdb/storage/table/scan_state.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace duckdb {


struct LSHIndexStats {
	idx_t count = 0;
	idx_t capacity = 0;
	idx_t approx_size = 0;
};

struct LSHBucketTable {
	std::unordered_map<uint64_t, std::vector<row_t>> buckets;
};

class LSHIndex : public BoundIndex {
public:
	static constexpr const char *TYPE_NAME = "LSH";

public:
	LSHIndex(const string &name, IndexConstraintType index_constraint_type, const vector<column_t> &column_ids,
	         TableIOManager &table_io_manager, const vector<unique_ptr<Expression>> &unbound_expressions,
	         AttachedDatabase &db, const case_insensitive_map_t<Value> &options,
	         const IndexStorageInfo &info = IndexStorageInfo(), idx_t estimated_cardinality = 0);

	static PhysicalOperator &CreatePlan(PlanIndexInput &input);

	unique_ptr<IndexScanState> InitializeScan(float *query_vector, idx_t limit, ClientContext &context);
	idx_t Scan(IndexScanState &state, Vector &result, idx_t result_offset = 0);

	unique_ptr<IndexScanState> InitializeMultiScan(ClientContext &context);
	idx_t ExecuteMultiScan(IndexScanState &state, float *query_vector, idx_t limit);
	const Vector &GetMultiScanResult(IndexScanState &state);
	void ResetMultiScan(IndexScanState &state);

	idx_t GetVectorSize() const;
	string GetMetric() const;

	void Construct(DataChunk &input, Vector &row_ids, idx_t thread_idx);
	void Compact();

	unique_ptr<LSHIndexStats> GetStats();

	void VerifyBuffers(IndexLock &lock) override;

	static const case_insensitive_map_t<string> METRIC_KIND_MAP;
	static const unordered_map<uint8_t, string> SCALAR_KIND_MAP;

	bool TryMatchDistanceFunction(const unique_ptr<Expression> &expr, vector<reference<Expression>> &bindings) const;
	bool TryBindIndexExpression(LogicalGet &get, unique_ptr<Expression> &result) const;

	ErrorData Append(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) override;
	void CommitDrop(IndexLock &index_lock) override;
	void Delete(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) override;
	ErrorData Insert(IndexLock &lock, DataChunk &data, Vector &row_ids) override;

	IndexStorageInfo SerializeToDisk(QueryContext context, const case_insensitive_map_t<Value> &options) override;
	IndexStorageInfo SerializeToWAL(const case_insensitive_map_t<Value> &options) override;

	idx_t GetInMemorySize(IndexLock &state) override;
	bool MergeIndexes(IndexLock &state, BoundIndex &other_index) override;
	void Vacuum(IndexLock &state) override;
	string VerifyAndToString(IndexLock &state, const bool only_verify) override;
	void VerifyAllocations(IndexLock &state) override;

	string GetConstraintViolationMessage(VerifyExistenceType verify_type, idx_t failed_index,
	                                     DataChunk &input) override {
		return "Constraint violation in LSH index";
	}

	void SetDirty() {
		is_dirty = true;
	}
	void SyncSize() {
		index_size = row_vectors.size();
	}

public:
	idx_t num_tables = 8;
	idx_t num_bits = 16;
	idx_t dim = 0;
	uint64_t seed = 42;
	string metric = "l2sq";

	std::vector<std::vector<std::vector<float>>> random_planes;
	std::vector<LSHBucketTable> tables;
	std::unordered_map<row_t, std::vector<float>> row_vectors;

	void AddVector(row_t row_id, const std::vector<float> &vec);
	std::vector<row_t> Search(float *query_vector, idx_t limit) const;

private:
	bool is_dirty = false;
	StorageLock rwlock;
	std::atomic<idx_t> index_size {0};
	unique_ptr<ExpressionMatcher> function_matcher;

	unique_ptr<ExpressionMatcher> MakeFunctionMatcher() const;
};

} // namespace duckdb