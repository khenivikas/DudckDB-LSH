#include "lsh/lsh_index.hpp"
#include "lsh/lsh.hpp"

#include "duckdb/common/assert.hpp"
#include "duckdb/common/column_index.hpp"
#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/common/enums/index_constraint_type.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector_size.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/optimizer/matcher/expression_type_matcher.hpp"
#include "duckdb/optimizer/matcher/function_matcher.hpp"
#include "duckdb/optimizer/matcher/set_matcher.hpp"
#include "duckdb/optimizer/matcher/type_matcher.hpp"
#include "duckdb/planner/column_binding.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/index_storage_info.hpp"
#include "duckdb/storage/table_io_manager.hpp"
#include "lsh/lsh_index_physical_create.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace duckdb {

namespace {
static inline void AddProbeHash(std::vector<uint64_t> &probes, std::unordered_set<uint64_t> &seen, uint64_t hash,
                                idx_t max_probes) {
	if (probes.size() >= max_probes) {
		return;
	}
	if (seen.insert(hash).second) {
		probes.push_back(hash);
	}
}

static std::vector<uint64_t> GenerateProbeHashes(uint64_t exact_hash, idx_t bit_count, idx_t max_probes) {
	std::vector<uint64_t> probes;
	probes.reserve(max_probes);

	std::unordered_set<uint64_t> seen;
	seen.reserve(max_probes * 2);

	AddProbeHash(probes, seen, exact_hash, max_probes);

	// 1-bit neighbors
	for (idx_t i = 0; i < bit_count && probes.size() < max_probes; i++) {
		AddProbeHash(probes, seen, exact_hash ^ (1ULL << i), max_probes);
	}

	// 2-bit neighbors
	for (idx_t i = 0; i < bit_count && probes.size() < max_probes; i++) {
		for (idx_t j = i + 1; j < bit_count && probes.size() < max_probes; j++) {
			AddProbeHash(probes, seen, exact_hash ^ (1ULL << i) ^ (1ULL << j), max_probes);
		}
	}

	return probes;
}

static inline float DotProduct(const float *a, const float *b, idx_t dim) {
	float sum = 0.0f;
	for (idx_t i = 0; i < dim; i++) {
		sum += a[i] * b[i];
	}
	return sum;
}

static inline float L2Squared(const float *a, const float *b, idx_t dim) {
	float sum = 0.0f;
	for (idx_t i = 0; i < dim; i++) {
		float diff = a[i] - b[i];
		sum += diff * diff;
	}
	return sum;
}

static inline float InnerProductDistance(const float *a, const float *b, idx_t dim) {
	return -DotProduct(a, b, dim);
}

static inline float CosineDistance(const float *a, const float *b, idx_t dim) {
	float dot = 0.0f;
	float na = 0.0f;
	float nb = 0.0f;
	for (idx_t i = 0; i < dim; i++) {
		dot += a[i] * b[i];
		na += a[i] * a[i];
		nb += b[i] * b[i];
	}
	if (na == 0.0f || nb == 0.0f) {
		return 1.0f;
	}
	return 1.0f - (dot / (std::sqrt(na) * std::sqrt(nb)));
}

static inline float ComputeDistance(const float *a, const float *b, idx_t dim, const string &metric) {
	if (metric == "l2sq") {
		return L2Squared(a, b, dim);
	}
	if (metric == "cosine") {
		return CosineDistance(a, b, dim);
	}
	if (metric == "ip") {
		return InnerProductDistance(a, b, dim);
	}
	throw NotImplementedException("Unsupported metric for LSH: %s", metric);
}

static inline uint64_t HashVector(const float *vec, idx_t dim, const std::vector<std::vector<float>> &planes) {
	uint64_t hash = 0;
	const idx_t bits = MinValue<idx_t>(planes.size(), 64);
	for (idx_t i = 0; i < bits; i++) {
		float dot = DotProduct(vec, planes[i].data(), dim);
		if (dot >= 0.0f) {
			hash |= (1ULL << i);
		}
	}
	return hash;
}

struct CandidateEntry {
	row_t row_id;
	float distance;
};

static inline bool CandidateLess(const CandidateEntry &a, const CandidateEntry &b) {
	if (a.distance != b.distance) {
		return a.distance < b.distance;
	}
	return a.row_id < b.row_id;
}

static void TryBindIndexExpressionInternal(Expression &expr, idx_t table_idx, const vector<column_t> &index_columns,
                                           const vector<ColumnIndex> &table_columns, bool &success, bool &found) {
	if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
		found = true;
		auto &ref = expr.Cast<BoundColumnRefExpression>();

		ref.binding.table_index = table_idx;

		const auto referenced_column = index_columns[ref.binding.column_index];
		for (idx_t i = 0; i < table_columns.size(); i++) {
			if (table_columns[i].GetPrimaryIndex() == referenced_column) {
				ref.binding.column_index = i;
				return;
			}
		}
		success = false;
	}

	ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) {
		TryBindIndexExpressionInternal(child, table_idx, index_columns, table_columns, success, found);
	});
}

} // namespace

//------------------------------------------------------------------------------
// Scan states
//------------------------------------------------------------------------------

struct LSHScanState : public IndexScanState {
	idx_t current_row = 0;
	idx_t total_rows = 0;
	std::unique_ptr<row_t[]> row_ids;
};

struct LSHMultiScanState : public IndexScanState {
	Vector vec;
	std::vector<row_t> row_ids;
	idx_t current_row = 0;

	LSHMultiScanState() : vec(LogicalType::ROW_TYPE) {
	}
};

//------------------------------------------------------------------------------
// LSHIndex methods
//------------------------------------------------------------------------------

LSHIndex::LSHIndex(const string &name, IndexConstraintType index_constraint_type, const vector<column_t> &column_ids,
                   TableIOManager &table_io_manager, const vector<unique_ptr<Expression>> &unbound_expressions,
                   AttachedDatabase &db, const case_insensitive_map_t<Value> &options, const IndexStorageInfo &info,
                   idx_t estimated_cardinality)
    : BoundIndex(name, TYPE_NAME, index_constraint_type, column_ids, table_io_manager, unbound_expressions, db) {

	if (index_constraint_type != IndexConstraintType::NONE) {
		throw NotImplementedException("LSH indexes do not support unique or primary key constraints");
	}

	if (logical_types.size() != 1) {
		throw BinderException("LSH indexes can only be created over a single column");
	}
	if (logical_types[0].id() != LogicalTypeId::ARRAY) {
		throw BinderException("LSH index keys must be of type FLOAT[N]");
	}
	auto &child_type = ArrayType::GetChildType(logical_types[0]);
	if (child_type.id() != LogicalTypeId::FLOAT) {
		throw BinderException("LSH index keys must be FLOAT arrays");
	}

	dim = ArrayType::GetSize(logical_types[0]);

	auto metric_opt = options.find("metric");
	if (metric_opt != options.end() && !metric_opt->second.IsNull()) {
		metric = StringUtil::Lower(metric_opt->second.ToString());
		if (metric != "l2sq" && metric != "cosine" && metric != "ip") {
			throw BinderException("LSH index metric must be one of: l2sq, cosine, ip");
		}
	}

	auto tables_opt = options.find("lsh_tables");
	if (tables_opt != options.end() && !tables_opt->second.IsNull()) {
		num_tables = MaxValue<idx_t>(1, tables_opt->second.GetValue<idx_t>());
	}

	auto bits_opt = options.find("lsh_bits");
	if (bits_opt != options.end() && !bits_opt->second.IsNull()) {
		num_bits = MaxValue<idx_t>(1, MinValue<idx_t>(64, bits_opt->second.GetValue<idx_t>()));
	}

	auto seed_opt = options.find("lsh_seed");
	if (seed_opt != options.end() && !seed_opt->second.IsNull()) {
		seed = seed_opt->second.GetValue<uint64_t>();
	}

	tables.resize(num_tables);
	random_planes.resize(num_tables);

	std::mt19937_64 rng(seed);
	std::normal_distribution<float> normal_dist(0.0f, 1.0f);

	for (idx_t t = 0; t < num_tables; t++) {
		random_planes[t].resize(num_bits);
		for (idx_t b = 0; b < num_bits; b++) {
			random_planes[t][b].resize(dim);
			for (idx_t d = 0; d < dim; d++) {
				random_planes[t][b][d] = normal_dist(rng);
			}
		}
	}

	if (info.IsValid()) {
		throw NotImplementedException("LSH persistence/loading is not implemented yet");
	}

	index_size = 0;
	function_matcher = MakeFunctionMatcher();
}

idx_t LSHIndex::GetVectorSize() const {
	return dim;
}

string LSHIndex::GetMetric() const {
	return metric;
}

const case_insensitive_map_t<string> LSHIndex::METRIC_KIND_MAP = {
    {"l2sq", "l2sq"},
    {"cosine", "cosine"},
    {"ip", "ip"},
};

const unordered_map<uint8_t, string> LSHIndex::SCALAR_KIND_MAP = {
    {static_cast<uint8_t>(LogicalTypeId::FLOAT), "f32"},
};

unique_ptr<LSHIndexStats> LSHIndex::GetStats() {
	auto result = make_uniq<LSHIndexStats>();
	result->count = index_size.load();
	result->capacity = index_size.load();
	result->approx_size = static_cast<idx_t>(sizeof(*this) + row_vectors.size() * (sizeof(row_t) + dim * sizeof(float)));
	return result;
}

unique_ptr<IndexScanState> LSHIndex::InitializeScan(float *query_vector, idx_t limit, ClientContext &context) {
	auto state = make_uniq<LSHScanState>();
	auto top = Search(query_vector, limit);

	state->total_rows = top.size();
	state->current_row = 0;
	state->row_ids = make_uniq_array<row_t>(top.size());

	for (idx_t i = 0; i < top.size(); i++) {
		state->row_ids[i] = top[i];
	}
	return std::move(state);
}

idx_t LSHIndex::Scan(IndexScanState &state, Vector &result, idx_t result_offset) {
	auto &scan_state = state.Cast<LSHScanState>();
	idx_t count = 0;
	auto row_ids = FlatVector::GetData<row_t>(result) + result_offset;

	while (count < STANDARD_VECTOR_SIZE && scan_state.current_row < scan_state.total_rows) {
		row_ids[count++] = scan_state.row_ids[scan_state.current_row++];
	}
	return count;
}

unique_ptr<IndexScanState> LSHIndex::InitializeMultiScan(ClientContext &context) {
	return make_uniq<LSHMultiScanState>();
}

idx_t LSHIndex::ExecuteMultiScan(IndexScanState &state_p, float *query_vector, idx_t limit) {
	auto &state = state_p.Cast<LSHMultiScanState>();
	auto top = Search(query_vector, limit);

	const auto offset = state.row_ids.size();
	state.row_ids.resize(offset + top.size());
	for (idx_t i = 0; i < top.size(); i++) {
		state.row_ids[offset + i] = top[i];
	}
	return top.size();
}

const Vector &LSHIndex::GetMultiScanResult(IndexScanState &state) {
	auto &scan_state = state.Cast<LSHMultiScanState>();
	FlatVector::SetData(scan_state.vec, reinterpret_cast<data_ptr_t>(scan_state.row_ids.data()));
	return scan_state.vec;
}

void LSHIndex::ResetMultiScan(IndexScanState &state) {
	auto &scan_state = state.Cast<LSHMultiScanState>();
	scan_state.row_ids.clear();
	scan_state.current_row = 0;
}

void LSHIndex::CommitDrop(IndexLock &index_lock) {
	auto lock = rwlock.GetExclusiveLock();
	row_vectors.clear();
	for (auto &table : tables) {
		table.buckets.clear();
	}
	index_size = 0;
}

void LSHIndex::Construct(DataChunk &input, Vector &row_ids, idx_t thread_idx) {
	D_ASSERT(input.ColumnCount() == 1);
	D_ASSERT(row_ids.GetType().InternalType() == ROW_TYPE);

	auto count = input.size();
	input.Flatten();

	auto &vec_vec = input.data[0];
	auto &vec_child_vec = ArrayVector::GetEntry(vec_vec);
	auto array_size = ArrayType::GetSize(vec_vec.GetType());

	auto vec_child_data = FlatVector::GetData<float>(vec_child_vec);
	auto rowid_data = FlatVector::GetData<row_t>(row_ids);

	auto to_add_count = FlatVector::Validity(vec_vec).CountValid(count);

	{
		auto lock = rwlock.GetExclusiveLock();
		index_size += to_add_count;
	}

	for (idx_t i = 0; i < count; i++) {
		if (FlatVector::IsNull(vec_vec, i)) {
			continue;
		}
		std::vector<float> vec(array_size);
		for (idx_t d = 0; d < array_size; d++) {
			vec[d] = vec_child_data[i * array_size + d];
		}
		AddVector(rowid_data[i], vec);
	}
}

void LSHIndex::Compact() {
	// No-op for the in-memory LSH index.
}

void LSHIndex::Delete(IndexLock &lock, DataChunk &input, Vector &rowid_vec) {
	auto count = input.size();
	rowid_vec.Flatten(count);
	auto row_id_data = FlatVector::GetData<row_t>(rowid_vec);

	auto _lock = rwlock.GetExclusiveLock();
	for (idx_t i = 0; i < count; i++) {
		row_vectors.erase(row_id_data[i]);
	}
	index_size = row_vectors.size();
}

ErrorData LSHIndex::Insert(IndexLock &lock, DataChunk &input, Vector &rowid_vec) {
	Construct(input, rowid_vec, 0);
	return ErrorData {};
}

ErrorData LSHIndex::Append(IndexLock &lock, DataChunk &appended_data, Vector &row_identifiers) {
	DataChunk expression_result;
	expression_result.Initialize(Allocator::DefaultAllocator(), logical_types);
	ExecuteExpressions(appended_data, expression_result);
	Construct(expression_result, row_identifiers, 0);
	return ErrorData {};
}

IndexStorageInfo LSHIndex::SerializeToDisk(QueryContext context, const case_insensitive_map_t<Value> &options) {
	IndexStorageInfo info;
	info.name = name;
	return info;
}

IndexStorageInfo LSHIndex::SerializeToWAL(const case_insensitive_map_t<Value> &options) {
	IndexStorageInfo info;
	info.name = name;
	return info;
}

idx_t LSHIndex::GetInMemorySize(IndexLock &state) {
	idx_t size = sizeof(*this);
	size += row_vectors.size() * (sizeof(row_t) + sizeof(std::vector<float>));
	for (auto &table : tables) {
		size += table.buckets.size() * (sizeof(uint64_t) + sizeof(std::vector<row_t>));
	}
	return size;
}

bool LSHIndex::MergeIndexes(IndexLock &state, BoundIndex &other_index) {
	throw NotImplementedException("LSHIndex::MergeIndexes() not implemented");
}

void LSHIndex::Vacuum(IndexLock &state) {
}

string LSHIndex::VerifyAndToString(IndexLock &state, const bool only_verify) {
	return only_verify ? string() : "LSH index";
}

void LSHIndex::VerifyAllocations(IndexLock &state) {
}

bool LSHIndex::TryBindIndexExpression(LogicalGet &get, unique_ptr<Expression> &result) const {
	auto expr_ptr = unbound_expressions.back()->Copy();

	auto &expr = *expr_ptr;
	auto &index_columns = GetColumnIds();
	auto &table_columns = get.GetColumnIds();

	auto success = true;
	auto found = false;

	TryBindIndexExpressionInternal(expr, get.table_index, index_columns, table_columns, success, found);

	if (success && found) {
		result = std::move(expr_ptr);
		return true;
	}
	return false;
}

bool LSHIndex::TryMatchDistanceFunction(const unique_ptr<Expression> &expr,
                                        vector<reference<Expression>> &bindings) const {
	return function_matcher->Match(*expr, bindings);
}

unique_ptr<ExpressionMatcher> LSHIndex::MakeFunctionMatcher() const {
	unordered_set<string> distance_functions;
	if (metric == "l2sq") {
		distance_functions = {"array_distance", "<->"};
	} else if (metric == "cosine") {
		distance_functions = {"array_cosine_distance", "<=>"};
	} else if (metric == "ip") {
		distance_functions = {"array_negative_inner_product", "<#>"};
	} else {
		throw NotImplementedException("Unknown metric for LSH");
	}

	auto matcher = make_uniq<FunctionExpressionMatcher>();
	matcher->function = make_uniq<ManyFunctionMatcher>(distance_functions);
	matcher->expr_type = make_uniq<SpecificExpressionTypeMatcher>(ExpressionType::BOUND_FUNCTION);
	matcher->policy = SetMatcher::Policy::UNORDERED;

	auto lhs_matcher = make_uniq<ExpressionMatcher>();
	lhs_matcher->type = make_uniq<SpecificTypeMatcher>(LogicalType::ARRAY(LogicalType::FLOAT, GetVectorSize()));
	matcher->matchers.push_back(std::move(lhs_matcher));

	auto rhs_matcher = make_uniq<ExpressionMatcher>();
	rhs_matcher->type = make_uniq<SpecificTypeMatcher>(LogicalType::ARRAY(LogicalType::FLOAT, GetVectorSize()));
	matcher->matchers.push_back(std::move(rhs_matcher));

	return std::move(matcher);
}

void LSHIndex::VerifyBuffers(IndexLock &lock) {
}



//------------------------------------------------------------------------------
// Register index type
//------------------------------------------------------------------------------

void LSHModule::RegisterIndex(DatabaseInstance &db) {
	IndexType index_type;
	index_type.name = LSHIndex::TYPE_NAME;

	index_type.create_instance = [](CreateIndexInput &input) -> unique_ptr<BoundIndex> {
		return make_uniq<LSHIndex>(input.name, input.constraint_type, input.column_ids, input.table_io_manager,
                           input.unbound_expressions, input.db, input.options, input.storage_info);
	};

	index_type.create_plan = LSHIndex::CreatePlan;
	db.config.GetIndexTypes().RegisterIndexType(index_type);

	db.config.AddExtensionOption("lsh_tables", "number of LSH hash tables", LogicalType::BIGINT, Value::BIGINT(8));
	db.config.AddExtensionOption("lsh_bits", "number of bits per LSH table", LogicalType::BIGINT, Value::BIGINT(16));
	db.config.AddExtensionOption("lsh_seed", "random seed for LSH plane generation", LogicalType::UBIGINT,
	                             Value::UBIGINT(42));
	db.config.AddExtensionOption("lsh_metric", "LSH distance metric", LogicalType::VARCHAR, Value("l2sq"));
}

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------

void LSHIndex::AddVector(row_t row_id, const std::vector<float> &vec) {
	row_vectors[row_id] = vec;

	for (idx_t t = 0; t < num_tables; t++) {
		uint64_t hash = HashVector(vec.data(), dim, random_planes[t]);
		tables[t].buckets[hash].push_back(row_id);
	}
}

std::vector<row_t> LSHIndex::Search(float *query_vector, idx_t limit) const {
	if (limit == 0 || num_tables == 0) {
		return {};
	}

	// We only probe up to 64 bits because the hash is uint64_t.
	const idx_t bit_count = MinValue<idx_t>(num_bits, 64);

	// Small, bounded multi-probe:
	// exact bucket + 1-bit flips + 2-bit flips, capped for speed.
	const idx_t max_probes = MinValue<idx_t>(128, 1 + bit_count + (bit_count * (bit_count - 1)) / 2);

	std::unordered_set<row_t> candidates;
	candidates.reserve(MinValue<idx_t>(64 + limit * 32, 1 << 20));
	candidates.max_load_factor(0.7f);

	for (idx_t t = 0; t < num_tables; t++) {
		const uint64_t exact_hash = HashVector(query_vector, dim, random_planes[t]);
		const auto probe_hashes = GenerateProbeHashes(exact_hash, bit_count, max_probes);

		for (uint64_t probe_hash : probe_hashes) {
			auto it = tables[t].buckets.find(probe_hash);
			if (it == tables[t].buckets.end()) {
				continue;
			}
			for (row_t row_id : it->second) {
				candidates.insert(row_id);
			}
		}
	}

	if (candidates.empty()) {
		return {};
	}

	std::vector<CandidateEntry> scored;
	scored.reserve(candidates.size());

	for (row_t row_id : candidates) {
		auto found = row_vectors.find(row_id);
		if (found == row_vectors.end()) {
			continue;
		}
		const float dist = ComputeDistance(query_vector, found->second.data(), dim, metric);
		scored.push_back({row_id, dist});
	}

	if (scored.empty()) {
		return {};
	}

	// Faster than sorting the whole candidate list.
	if (scored.size() > limit) {
		auto nth = scored.begin() + limit;
		std::nth_element(scored.begin(), nth, scored.end(), CandidateLess);
		scored.resize(limit);
	}

	std::sort(scored.begin(), scored.end(), CandidateLess);

	std::vector<row_t> result;
	result.reserve(scored.size());
	for (const auto &entry : scored) {
		result.push_back(entry.row_id);
	}
	return result;

}

} // namespace duckdb