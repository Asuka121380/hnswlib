cmake_minimum_required(VERSION 3.0...3.26)

project(hnswlib
    LANGUAGES CXX)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

add_library(hnswlib INTERFACE)
add_library(hnswlib::hnswlib ALIAS hnswlib)

option(HNSWLIB_ENABLE_BASELINE_TRACE "Enable baseline HNSW search tracing." OFF)
if(HNSWLIB_ENABLE_BASELINE_TRACE)
    target_compile_definitions(hnswlib INTERFACE HNSWLIB_ENABLE_BASELINE_TRACE=1)
endif()

option(HNSWLIB_ENABLE_EDGE_QUANT_V0 "Enable V0 conservative edge-quantised pruning support." OFF)
option(HNSWLIB_ENABLE_V0_SHADOW_VALIDATION "Enable exact shadow validation for V0 pruning decisions." OFF)
option(HNSWLIB_ENABLE_V0_APPROX_SHADOW "Enable hypothetical margin/retry shadow instrumentation." OFF)
option(HNSWLIB_ENABLE_V0_REAL_PRUNING "Enable opt-in V0 pruning that skips proven-far exact distances." OFF)
option(HNSWLIB_BUILD_FAISS_PQ_TOOLS "Build Faiss-based offline V0 PQ tools." OFF)
set(FAISS_INCLUDE_DIR "" CACHE PATH "Faiss source/install root containing faiss/impl/ProductQuantizer.h")
set(FAISS_LIBRARY "" CACHE FILEPATH "Path to the Faiss CPU shared/static library")

if(HNSWLIB_ENABLE_V0_SHADOW_VALIDATION AND NOT HNSWLIB_ENABLE_EDGE_QUANT_V0)
    message(FATAL_ERROR
        "HNSWLIB_ENABLE_V0_SHADOW_VALIDATION requires HNSWLIB_ENABLE_EDGE_QUANT_V0=ON")
endif()

if(HNSWLIB_ENABLE_V0_REAL_PRUNING AND NOT HNSWLIB_ENABLE_EDGE_QUANT_V0)
    message(FATAL_ERROR
        "HNSWLIB_ENABLE_V0_REAL_PRUNING requires HNSWLIB_ENABLE_EDGE_QUANT_V0=ON")
endif()

if(HNSWLIB_ENABLE_V0_APPROX_SHADOW AND NOT HNSWLIB_ENABLE_V0_SHADOW_VALIDATION)
    message(FATAL_ERROR
        "HNSWLIB_ENABLE_V0_APPROX_SHADOW requires HNSWLIB_ENABLE_V0_SHADOW_VALIDATION=ON")
endif()

if(HNSWLIB_BUILD_FAISS_PQ_TOOLS AND NOT HNSWLIB_ENABLE_EDGE_QUANT_V0)
    message(FATAL_ERROR
        "HNSWLIB_BUILD_FAISS_PQ_TOOLS requires HNSWLIB_ENABLE_EDGE_QUANT_V0=ON")
endif()

if(HNSWLIB_ENABLE_EDGE_QUANT_V0)
    target_compile_definitions(hnswlib INTERFACE HNSWLIB_ENABLE_EDGE_QUANT_V0=1)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(hnswlib INTERFACE -fno-fast-math)
        if(COMMAND target_link_options)
            target_link_options(hnswlib INTERFACE -fno-fast-math)
        else()
            # CMake before 3.13 has no target_link_options. A leading-dash
            # item in target_link_libraries is forwarded to the link driver.
            target_link_libraries(hnswlib INTERFACE -fno-fast-math)
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(hnswlib INTERFACE /fp:precise)
    endif()
endif()

if(HNSWLIB_ENABLE_V0_SHADOW_VALIDATION)
    target_compile_definitions(hnswlib INTERFACE HNSWLIB_ENABLE_V0_SHADOW_VALIDATION=1)
endif()

if(HNSWLIB_ENABLE_V0_APPROX_SHADOW)
    target_compile_definitions(hnswlib INTERFACE HNSWLIB_ENABLE_V0_APPROX_SHADOW=1)
endif()

if(HNSWLIB_ENABLE_V0_REAL_PRUNING)
    target_compile_definitions(hnswlib INTERFACE HNSWLIB_ENABLE_V0_REAL_PRUNING=1)
endif()

target_include_directories(hnswlib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Install
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hnswlib
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS hnswlib
    EXPORT hnswlibTargets)

install(EXPORT hnswlibTargets
    FILE hnswlibConfig.cmake
    NAMESPACE hnswlib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hnswlib)

# Examples and tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(HNSWLIB_EXAMPLES "Build examples and tests." ON)
else()
    option(HNSWLIB_EXAMPLES "Build examples and tests." OFF)
endif()
if(HNSWLIB_EXAMPLES)
    set(CMAKE_CXX_STANDARD 11)
    enable_testing()

    if(HNSWLIB_ENABLE_EDGE_QUANT_V0)
      set(HNSWLIB_OPTIMIZATION_FLAG "-O3")
    else()
      set(HNSWLIB_OPTIMIZATION_FLAG "-Ofast")
    endif()

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      SET( CMAKE_CXX_FLAGS  "${HNSWLIB_OPTIMIZATION_FLAG} -std=c++11 -DHAVE_CXX0X -openmp -fpic -ftree-vectorize" )
      check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_NATIVE_FLAG)
      if(COMPILER_SUPPORT_NATIVE_FLAG)
        SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native" )
        message("set -march=native flag")
      else()
        check_cxx_compiler_flag("-mcpu=apple-m1" COMPILER_SUPPORT_M1_FLAG)
        if(COMPILER_SUPPORT_M1_FLAG)
          SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1" )
          message("set -mcpu=apple-m1 flag")
        endif()
      endif()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      SET( CMAKE_CXX_FLAGS  "${HNSWLIB_OPTIMIZATION_FLAG} -std=c++11 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
      if(NOT MINGW)
        SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lrt" )
      endif()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      SET( CMAKE_CXX_FLAGS  "/O2 -DHAVE_CXX0X /W1 /openmp /EHsc" )
    endif()

    # examples
    add_executable(example_search examples/cpp/example_search.cpp)
    target_link_libraries(example_search hnswlib)

    add_executable(example_epsilon_search examples/cpp/example_epsilon_search.cpp)
    target_link_libraries(example_epsilon_search hnswlib)

    add_executable(example_multivector_search examples/cpp/example_multivector_search.cpp)
    target_link_libraries(example_multivector_search hnswlib)

    add_executable(example_filter examples/cpp/example_filter.cpp)
    target_link_libraries(example_filter hnswlib)

    add_executable(example_replace_deleted examples/cpp/example_replace_deleted.cpp)
    target_link_libraries(example_replace_deleted hnswlib)

    add_executable(example_mt_search examples/cpp/example_mt_search.cpp)
    target_link_libraries(example_mt_search hnswlib)

    add_executable(example_mt_filter examples/cpp/example_mt_filter.cpp)
    target_link_libraries(example_mt_filter hnswlib)

    add_executable(example_mt_replace_deleted examples/cpp/example_mt_replace_deleted.cpp)
    target_link_libraries(example_mt_replace_deleted hnswlib)

    # tests
    add_executable(multivector_search_test tests/cpp/multivector_search_test.cpp)
    target_link_libraries(multivector_search_test hnswlib)

    add_executable(epsilon_search_test tests/cpp/epsilon_search_test.cpp)
    target_link_libraries(epsilon_search_test hnswlib)

    add_executable(test_updates tests/cpp/updates_test.cpp)
    target_link_libraries(test_updates hnswlib)

    add_executable(searchKnnCloserFirst_test tests/cpp/searchKnnCloserFirst_test.cpp)
    target_link_libraries(searchKnnCloserFirst_test hnswlib)

    add_executable(searchKnnWithFilter_test tests/cpp/searchKnnWithFilter_test.cpp)
    target_link_libraries(searchKnnWithFilter_test hnswlib)

    add_executable(multiThreadLoad_test tests/cpp/multiThreadLoad_test.cpp)
    target_link_libraries(multiThreadLoad_test hnswlib)

    add_executable(multiThread_replace_test tests/cpp/multiThread_replace_test.cpp)
    target_link_libraries(multiThread_replace_test hnswlib)

    add_executable(main tests/cpp/main.cpp tests/cpp/sift_1b.cpp)
    target_link_libraries(main hnswlib)

    add_executable(real_data_trace_runner examples/cpp/real_data_trace_runner.cpp)
    target_link_libraries(real_data_trace_runner hnswlib)

    if(HNSWLIB_ENABLE_BASELINE_TRACE)
      add_executable(baseline_trace_runner examples/cpp/baseline_trace_runner.cpp)
      target_link_libraries(baseline_trace_runner hnswlib)

      add_executable(baseline_trace_test tests/cpp/baseline_trace_test.cpp)
      target_link_libraries(baseline_trace_test hnswlib)
      add_test(NAME baseline_trace_test COMMAND baseline_trace_test)
    endif()

    if(HNSWLIB_ENABLE_EDGE_QUANT_V0)
      add_executable(v0_edge_sampler examples/cpp/v0_edge_sampler.cpp)
      target_link_libraries(v0_edge_sampler hnswlib)

      add_executable(v0_sidecar_inspector examples/cpp/v0_sidecar_inspector.cpp)
      target_link_libraries(v0_sidecar_inspector hnswlib)

      add_executable(v0_search_runner examples/cpp/v0_search_runner.cpp)
      target_link_libraries(v0_search_runner hnswlib)
      add_test(NAME v0_search_runner_cli_test COMMAND v0_search_runner --help)

      add_executable(v0_feature_isolation_test tests/cpp/v0_feature_isolation_test.cpp)
      target_link_libraries(v0_feature_isolation_test hnswlib)
      add_test(NAME v0_feature_isolation_test COMMAND v0_feature_isolation_test)

      add_executable(v0_graph_access_test tests/cpp/v0_graph_access_test.cpp)
      target_link_libraries(v0_graph_access_test hnswlib)
      add_test(NAME v0_graph_access_test COMMAND v0_graph_access_test)

      add_executable(v0_metadata_layout_test tests/cpp/v0_metadata_layout_test.cpp)
      target_link_libraries(v0_metadata_layout_test hnswlib)
      add_test(NAME v0_metadata_layout_test COMMAND v0_metadata_layout_test)

      add_executable(v0_sidecar_roundtrip_test tests/cpp/v0_sidecar_roundtrip_test.cpp)
      target_link_libraries(v0_sidecar_roundtrip_test hnswlib)
      add_test(NAME v0_sidecar_roundtrip_test COMMAND v0_sidecar_roundtrip_test)

      add_executable(v0_sidecar_mismatch_test tests/cpp/v0_sidecar_mismatch_test.cpp)
      target_link_libraries(v0_sidecar_mismatch_test hnswlib)
      add_test(NAME v0_sidecar_mismatch_test COMMAND v0_sidecar_mismatch_test)

      add_executable(v0_edge_sampler_test tests/cpp/v0_edge_sampler_test.cpp)
      target_link_libraries(v0_edge_sampler_test hnswlib)
      add_test(NAME v0_edge_sampler_test COMMAND v0_edge_sampler_test)

      add_executable(v0_codebook_reader_test tests/cpp/v0_codebook_reader_test.cpp)
      target_link_libraries(v0_codebook_reader_test hnswlib)
      add_test(NAME v0_codebook_reader_test COMMAND v0_codebook_reader_test)

      add_executable(v0_all_edge_encoder_test tests/cpp/v0_all_edge_encoder_test.cpp)
      target_link_libraries(v0_all_edge_encoder_test hnswlib)
      add_test(NAME v0_all_edge_encoder_test COMMAND v0_all_edge_encoder_test)

      add_executable(v0_query_metadata_test tests/cpp/v0_query_metadata_test.cpp)
      target_link_libraries(v0_query_metadata_test hnswlib)
      add_test(NAME v0_query_metadata_test COMMAND v0_query_metadata_test)

      add_executable(v0_query_lut_test tests/cpp/v0_query_lut_test.cpp)
      target_link_libraries(v0_query_lut_test hnswlib)
      add_test(NAME v0_query_lut_test COMMAND v0_query_lut_test)

      add_executable(v0_bound_evaluator_test tests/cpp/v0_bound_evaluator_test.cpp)
      target_link_libraries(v0_bound_evaluator_test hnswlib)
      add_test(NAME v0_bound_evaluator_test COMMAND v0_bound_evaluator_test)

      if(HNSWLIB_ENABLE_V0_SHADOW_VALIDATION)
        add_executable(v0_shadow_validation_test tests/cpp/v0_shadow_validation_test.cpp)
        target_link_libraries(v0_shadow_validation_test hnswlib)
        add_test(NAME v0_shadow_validation_test COMMAND v0_shadow_validation_test)
      endif()

      if(HNSWLIB_ENABLE_V0_REAL_PRUNING)
        add_executable(v0_real_pruning_test tests/cpp/v0_real_pruning_test.cpp)
        target_link_libraries(v0_real_pruning_test hnswlib)
        add_test(NAME v0_real_pruning_test COMMAND v0_real_pruning_test)
      endif()

      if(HNSWLIB_BUILD_FAISS_PQ_TOOLS)
        find_package(faiss CONFIG QUIET)
        if(TARGET faiss)
          set(HNSWLIB_V0_FAISS_TARGET faiss)
        elseif(TARGET faiss::faiss)
          set(HNSWLIB_V0_FAISS_TARGET faiss::faiss)
        elseif(FAISS_INCLUDE_DIR AND FAISS_LIBRARY)
          if(NOT EXISTS "${FAISS_INCLUDE_DIR}/faiss/impl/ProductQuantizer.h")
            message(FATAL_ERROR
              "FAISS_INCLUDE_DIR does not contain faiss/impl/ProductQuantizer.h")
          endif()
          if(NOT EXISTS "${FAISS_LIBRARY}")
            message(FATAL_ERROR "FAISS_LIBRARY does not exist")
          endif()
          add_library(hnswlib_v0_faiss UNKNOWN IMPORTED)
          set_target_properties(
            hnswlib_v0_faiss
            PROPERTIES
              IMPORTED_LOCATION "${FAISS_LIBRARY}"
              INTERFACE_INCLUDE_DIRECTORIES "${FAISS_INCLUDE_DIR}"
          )
          set(HNSWLIB_V0_FAISS_TARGET hnswlib_v0_faiss)
        else()
          message(FATAL_ERROR
            "Faiss C++ development files not found. Provide a Faiss CMake package or set FAISS_INCLUDE_DIR and FAISS_LIBRARY.")
        endif()

        add_executable(
          v0_faiss_edge_encoder
          examples/cpp/v0_faiss_edge_encoder.cpp
        )
        set_target_properties(
          v0_faiss_edge_encoder
          PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED YES
        )
        target_link_libraries(
          v0_faiss_edge_encoder
          PRIVATE hnswlib ${HNSWLIB_V0_FAISS_TARGET}
        )
      endif()

      find_package(PythonInterp 3 QUIET)
      if(PYTHONINTERP_FOUND)
        add_test(
          NAME v0_pq_local_test
          COMMAND ${PYTHON_EXECUTABLE}
                  ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/v0_pq_local_test.py
        )
        set_tests_properties(
          v0_pq_local_test
          PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
        add_test(
          NAME v0_sidecar_validator_test
          COMMAND ${PYTHON_EXECUTABLE}
                  ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/v0_sidecar_validator_test.py
        )
        set_tests_properties(
          v0_sidecar_validator_test
          PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
        add_test(
          NAME v0_performance_summary_test
          COMMAND ${PYTHON_EXECUTABLE}
                  ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/v0_performance_summary_test.py
        )
        set_tests_properties(
          v0_performance_summary_test
          PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
        add_test(
          NAME v0_margin_frontier_test
          COMMAND ${PYTHON_EXECUTABLE}
                  ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/v0_margin_frontier_test.py
        )
        set_tests_properties(
          v0_margin_frontier_test
          PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
        add_test(
          NAME v0_retry_shadow_analysis_test
          COMMAND ${PYTHON_EXECUTABLE}
                  ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/v0_retry_shadow_analysis_test.py
        )
        set_tests_properties(
          v0_retry_shadow_analysis_test
          PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
      endif()
    endif()
endif()
