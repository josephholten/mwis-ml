cmake_minimum_required(VERSION 2.8.9)

include(CheckCXXCompilerFlag)
project(mwis-ml C CXX)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release) 
endif (NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fno-omit-frame-pointer -march=native -DNDEBUG -fPIC -O3 -Wpedantic")

CHECK_CXX_COMPILER_FLAG(-Wno-unused-value COMPILER_SUPPORTS_NOUNUSED)
if(COMPILER_SUPPORTS_NOUNUSED)
add_definitions(-Wno-unused-value)
endif()

CHECK_CXX_COMPILER_FLAG(-Wno-unused-value COMPILER_SUPPORTS_NOUNUSEDRES)
if(COMPILER_SUPPORTS_NOUNUSEDRES)
add_definitions(-Wno-unused-result)
endif()

CHECK_CXX_COMPILER_FLAG(-fno-stack-limit COMPILER_SUPPORTS_FNOSTACKLIMITS)
if(COMPILER_SUPPORTS_FNOSTACKLIMITS)
  add_definitions(-fno-stack-limit)
endif()

        CHECK_CXX_COMPILER_FLAG(-funroll-loops COMPILER_SUPPORTS_FUNROLL_LOOPS)
if(COMPILER_SUPPORTS_FUNROLL_LOOPS)
  add_definitions(-funroll-loops)
endif()


include_directories(
        app
        lib
        extern/xgboost/include

        extern/wmis/lib
        extern/wmis/lib/data_structure
        extern/wmis/lib/data_structure/priority_queues
        extern/wmis/lib/mis
        extern/wmis/lib/mis/ils
        extern/wmis/lib/mis/initial_mis
        extern/wmis/lib/mis/kernel
        extern/wmis/lib/tools
        extern/wmis/extern/KaHIP
        extern/wmis/extern/KaHIP/interface
        extern/wmis/extern/KaHIP/lib
        extern/wmis/extern/KaHIP/lib/algorithms
        extern/wmis/extern/KaHIP/lib/data_structure
        extern/wmis/extern/KaHIP/lib/data_structure/matrix
        extern/wmis/extern/KaHIP/lib/data_structure/priority_queues
        extern/wmis/extern/KaHIP/lib/io
        extern/wmis/extern/KaHIP/lib/parallel_mh
        extern/wmis/extern/KaHIP/lib/parallel_mh/exchange
        extern/wmis/extern/KaHIP/lib/parallel_mh/galinier_combine
        extern/wmis/extern/KaHIP/lib/partition
        extern/wmis/extern/KaHIP/lib/tools
        extern/wmis/extern/argtable3-3.0.3
)

set(SOURCES
        extern/wmis/lib/tools/mis_log.cpp
        extern/wmis/lib/mis/ils/ils.cpp
        extern/wmis/lib/mis/ils/local_search.cpp
        extern/wmis/lib/mis/initial_mis/greedy_mis.cpp
        extern/wmis/lib/mis/initial_mis/greedy_vertex.cpp
        extern/wmis/lib/mis/initial_mis/random_mis.cpp
        extern/wmis/lib/mis/initial_mis/initial_mis.cpp
        extern/wmis/lib/data_structure/mis_permutation.cpp
        extern/wmis/lib/data_structure/candidate_list.cpp
        extern/wmis/lib/data_structure/operation_log.cpp
        extern/wmis/lib/data_structure/priority_queues/bucket_array.cpp
        extern/wmis/lib/mis/kernel/branch_and_reduce_algorithm.cpp
        extern/wmis/lib/mis/kernel/reductions.cpp

        extern/wmis/extern/KaHIP/lib/io/graph_io.cpp
        extern/wmis/extern/KaHIP/lib/tools/random_functions.cpp
        extern/wmis/extern/KaHIP/lib/tools/graph_extractor.cpp
        extern/wmis/extern/KaHIP/lib/algorithms/strongly_connected_components.cpp
        extern/wmis/extern/KaHIP/lib/algorithms/push_relabel.cpp

        extern/wmis/extern/argtable3-3.0.3/argtable3.c
        )


add_library(libsources2 OBJECT ${SOURCES})

set(MWIS-ML_SOURCES
        lib/wmis_interface/weighted_ls.cpp
        lib/ml_reducer.cpp)

add_library(libmwis-ml OBJECT ${MWIS-ML_SOURCES})

add_subdirectory(extern/xgboost)

add_executable(ml-reduce "app/ml-reduce.cpp" $<TARGET_OBJECTS:libsources2> $<TARGET_OBJECTS:libmwis-ml>)
target_link_libraries(ml-reduce xgboost)

add_executable(ml-train "app/ml-train.cpp" $<TARGET_OBJECTS:libsources2> $<TARGET_OBJECTS:libmwis-ml>)
target_link_libraries(ml-train xgboost)

