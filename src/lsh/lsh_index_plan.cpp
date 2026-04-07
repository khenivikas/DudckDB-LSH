#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/parser/parsed_data/create_index_info.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_create_index.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/storage_manager.hpp"

#include "lsh/lsh_index.hpp"
#include "lsh/lsh_index_physical_create.hpp"

namespace duckdb {

PhysicalOperator &LSHIndex::CreatePlan(PlanIndexInput &input) {
	auto &op = input.op;
	auto &context = input.context;
	auto &planner = input.planner;

	auto ParseIntegerOption = [&](const Value &v, const string &name) -> int64_t {
		if (v.type() == LogicalType::VARCHAR) {
			return std::stoll(v.GetValue<string>());
		}
		if (v.type() == LogicalType::BIGINT || v.type() == LogicalType::INTEGER || v.type() == LogicalType::SMALLINT ||
		    v.type() == LogicalType::TINYINT || v.type() == LogicalType::UBIGINT || v.type() == LogicalType::UINTEGER ||
		    v.type() == LogicalType::USMALLINT || v.type() == LogicalType::UTINYINT) {
			return v.GetValue<int64_t>();
		}
		throw BinderException("LSH index '%s' must be an integer", name);
	};

	Value enable_persistence;
	const bool has_persistence_setting =
	    context.TryGetCurrentSetting("lsh_enable_experimental_persistence", enable_persistence);
	const bool persistence_enabled = has_persistence_setting && !enable_persistence.IsNull() &&
	                                 enable_persistence.GetValue<bool>();

	const auto is_disk_db = !op.table.GetStorage().db.GetStorageManager().InMemory();
	if (is_disk_db && !persistence_enabled) {
		throw BinderException("LSH indexes can only be created in in-memory databases, or when the configuration "
		                      "option 'lsh_enable_experimental_persistence' is set to true.");
	}

	// Validate index options.
	for (auto &option : op.info->options) {
		auto &k = option.first;
		auto &v = option.second;

		if (StringUtil::CIEquals(k, "metric")) {
			if (v.type() != LogicalType::VARCHAR) {
				throw BinderException("LSH index 'metric' must be a string");
			}
			auto metric = StringUtil::Lower(v.GetValue<string>());
			if (LSHIndex::METRIC_KIND_MAP.find(metric) == LSHIndex::METRIC_KIND_MAP.end()) {
				vector<string> allowed_metrics;
				for (auto &entry : LSHIndex::METRIC_KIND_MAP) {
					allowed_metrics.push_back(StringUtil::Format("'%s'", entry.first));
				}
				throw BinderException("LSH index 'metric' must be one of: %s",
				                      StringUtil::Join(allowed_metrics, ", "));
			}
		} else if (StringUtil::CIEquals(k, "lsh_tables")) {
			auto tables = ParseIntegerOption(v, "lsh_tables");
			if (tables < 1) {
				throw BinderException("LSH index 'lsh_tables' must be at least 1");
			}
		} else if (StringUtil::CIEquals(k, "lsh_bits")) {
			auto bits = ParseIntegerOption(v, "lsh_bits");
			if (bits < 1) {
				throw BinderException("LSH index 'lsh_bits' must be at least 1");
			}
			if (bits > 64) {
				throw BinderException("LSH index 'lsh_bits' must be at most 64");
			}
		} else if (StringUtil::CIEquals(k, "lsh_seed")) {
			auto seed = ParseIntegerOption(v, "lsh_seed");
			if (seed < 0) {
				throw BinderException("LSH index 'lsh_seed' must be a non-negative integer");
			}
		} else {
			throw BinderException("Unknown option for LSH index: '%s'", k);
		}
	}

	// Validate the indexed expression.
	if (op.expressions.size() != 1) {
		throw BinderException("LSH indexes can only be created over a single column of keys.");
	}

	auto &arr_type = op.expressions[0]->return_type;
	if (arr_type.id() != LogicalTypeId::ARRAY) {
		throw BinderException("LSH index keys must be of type FLOAT[N]");
	}

	auto &child_type = ArrayType::GetChildType(arr_type);
	if (child_type.id() != LogicalTypeId::FLOAT) {
		throw BinderException("LSH index keys must be FLOAT[N]");
	}

	auto child_type_val = LSHIndex::SCALAR_KIND_MAP.find(static_cast<uint8_t>(child_type.id()));
	if (child_type_val == LSHIndex::SCALAR_KIND_MAP.end()) {
		vector<string> allowed_types;
		for (auto &entry : LSHIndex::SCALAR_KIND_MAP) {
			auto id = static_cast<LogicalTypeId>(entry.first);
			allowed_types.push_back(StringUtil::Format("'%s[N]'", LogicalType(id).ToString()));
		}
		throw BinderException("LSH index key type must be one of: %s", StringUtil::Join(allowed_types, ", "));
	}

	// Projection to execute expressions on the key columns.
	vector<LogicalType> new_column_types;
	vector<unique_ptr<Expression>> select_list;
	for (auto &expression : op.expressions) {
		new_column_types.push_back(expression->return_type);
		select_list.push_back(std::move(expression));
	}
	new_column_types.emplace_back(LogicalType::ROW_TYPE);

	auto row_id_index = op.info->scan_types.size() - 1;
	select_list.push_back(make_uniq<BoundReferenceExpression>(LogicalType::ROW_TYPE, row_id_index));

	auto &projection =
	    planner.Make<PhysicalProjection>(new_column_types, std::move(select_list), op.estimated_cardinality);
	projection.children.push_back(input.table_scan);

	// Filter operator for IS_NOT_NULL on each key column.
	vector<LogicalType> filter_types;
	vector<unique_ptr<Expression>> filter_select_list;

	for (idx_t i = 0; i < new_column_types.size() - 1; i++) {
		filter_types.push_back(new_column_types[i]);
		auto is_not_null_expr =
		    make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_IS_NOT_NULL, LogicalType::BOOLEAN);
		auto bound_ref = make_uniq<BoundReferenceExpression>(new_column_types[i], i);
		is_not_null_expr->children.push_back(std::move(bound_ref));
		filter_select_list.push_back(std::move(is_not_null_expr));
	}

	auto &null_filter =
	    planner.Make<PhysicalFilter>(std::move(filter_types), std::move(filter_select_list), op.estimated_cardinality);
	null_filter.types.emplace_back(LogicalType::ROW_TYPE);
	null_filter.children.push_back(projection);

	// Save values before moving unique_ptrs.
	auto column_ids = op.info->column_ids;
	auto index_info = std::move(op.info);
	auto unbound_expressions = std::move(op.unbound_expressions);

	auto &physical_create_index = planner.Make<PhysicalCreateLSHIndex>(
	    op.types, op.table, column_ids, std::move(index_info), std::move(unbound_expressions),
	    op.estimated_cardinality);

	physical_create_index.children.push_back(null_filter);
	return physical_create_index;
}

} // namespace duckdb