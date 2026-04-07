set(EXTENSION_SOURCES
    ${EXTENSION_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/lsh_index.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lsh_index_plan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lsh_index_physical_create.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lsh_index_scan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lsh_optimize_scan.cpp
    PARENT_SCOPE)