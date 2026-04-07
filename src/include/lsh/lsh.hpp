#pragma once

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
namespace duckdb {

struct LSHModule {
public:
	static void Register(ExtensionLoader &loader) {
	auto &db = loader.GetDatabaseInstance();
	RegisterIndex(db);
	RegisterIndexScan(loader);
	RegisterScanOptimizer(db);
}

private:
	static void RegisterIndex(DatabaseInstance &ldb);
	static void RegisterIndexScan(ExtensionLoader &loader);
	static void RegisterScanOptimizer(DatabaseInstance &ldb);
};

} // namespace duckdb
