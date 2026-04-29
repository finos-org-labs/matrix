cmake_minimum_required(VERSION 3.16)
project(finkit-matrix VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Include SIMD detection and compiler flags
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/DetectSIMD.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CompilerFlags.cmake)

# Sanitizer configuration
if(FC_ENABLE_SANITIZERS)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.github/sanitizers.cmake)
        include(${CMAKE_CURRENT_SOURCE_DIR}/.github/sanitizers.cmake)
    endif()
endif()

# Generate cgo_source.go with detected SIMD flags
# For runtime dispatch, we don't need SIMD flags in cgo since we use __attribute__((target(...)))
set(FC_SIMD_CFLAGS "")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cgo_source.go.in
    ${CMAKE_CURRENT_SOURCE_DIR}/cgo_source.go
    @ONLY
)

# Add platform submodule
add_subdirectory(modules/platform EXCLUDE_FROM_ALL)

# Matrix module sources
set(MATRIX_SOURCES
    matrix-c/decompose.c
    matrix-c/gemm_avx2.c
    matrix-c/gemm_avx512.c
    matrix-c/gemm_dispatch.c
    matrix-c/gemm_scalar.c
    matrix-c/gemm_sse42.c
    matrix-c/gemv.c
    matrix-c/solve.c
    matrix-c/transpose.c
    matrix-c/tridiag.c
    matrix-c/vector_ops.c
)

# Create object library
add_library(finkit_matrix OBJECT ${MATRIX_SOURCES})

target_include_directories(finkit_matrix
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/platform/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_definitions(finkit_matrix PRIVATE FC_STATIC_BUILD FC_ENABLE_INTERNAL_TESTS)

# Set compile flags for SIMD variants using detected flags
set_source_files_properties(matrix-c/gemm_sse42.c PROPERTIES COMPILE_FLAGS "${FC_SSE42_FLAGS}")
set_source_files_properties(matrix-c/gemm_avx2.c PROPERTIES COMPILE_FLAGS "${FC_AVX2_FLAGS}")
set_source_files_properties(matrix-c/gemm_avx512.c PROPERTIES COMPILE_FLAGS "${FC_AVX512_FLAGS}")

# For files with multiple SIMD paths, use the highest available
if(FC_SIMD_LEVEL STREQUAL "AVX512")
    set_source_files_properties(matrix-c/gemv.c PROPERTIES COMPILE_FLAGS "${FC_AVX512_FLAGS}")
    set_source_files_properties(matrix-c/transpose.c PROPERTIES COMPILE_FLAGS "${FC_AVX512_FLAGS}")
elseif(FC_SIMD_LEVEL STREQUAL "AVX2")
    set_source_files_properties(matrix-c/gemv.c PROPERTIES COMPILE_FLAGS "${FC_AVX2_FLAGS}")
    set_source_files_properties(matrix-c/transpose.c PROPERTIES COMPILE_FLAGS "${FC_AVX2_FLAGS}")
elseif(FC_SIMD_LEVEL STREQUAL "SSE42")
    set_source_files_properties(matrix-c/gemv.c PROPERTIES COMPILE_FLAGS "${FC_SSE42_FLAGS}")
    set_source_files_properties(matrix-c/transpose.c PROPERTIES COMPILE_FLAGS "${FC_SSE42_FLAGS}")
endif()

# Create static library
add_library(finkit_matrix_static STATIC $<TARGET_OBJECTS:finkit_matrix>)
target_include_directories(finkit_matrix_static
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link platform and math library
target_link_libraries(finkit_matrix_static PUBLIC finkit_platform_static m)

# Create alias
add_library(finkit::matrix ALIAS finkit_matrix_static)

# Install
install(TARGETS finkit_matrix_static
    EXPORT finkit-matrix-targets
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include/matrix
    FILES_MATCHING PATTERN "*.h"
)

install(EXPORT finkit-matrix-targets
    FILE finkit-matrixTargets.cmake
    NAMESPACE finkit::
    DESTINATION lib/cmake/finkit-matrix
)

# Config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/finkit-matrixConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/finkit-matrixConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/finkit-matrixConfig.cmake"
    INSTALL_DESTINATION lib/cmake/finkit-matrix
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/finkit-matrixConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/finkit-matrixConfigVersion.cmake"
    DESTINATION lib/cmake/finkit-matrix
)

# Tests
option(FC_BUILD_TESTS "Build tests" ON)
if(FC_BUILD_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
option(FC_BUILD_BENCHMARKS "Build benchmarks" ON)
if(FC_BUILD_BENCHMARKS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
    add_subdirectory(benchmarks)
endif()

# Print configuration
message(STATUS "")
message(STATUS "============================================================")
message(STATUS " finkit-matrix ${PROJECT_VERSION}")
message(STATUS "============================================================")
message(STATUS "  C Compiler: ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  System: ${CMAKE_SYSTEM_NAME}")
message(STATUS "============================================================")
message(STATUS "")
