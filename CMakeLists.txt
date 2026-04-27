cmake_minimum_required(VERSION 3.16)
project(finkit-matrix VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compile options
add_compile_options(-Wall -Wextra -Wpedantic -ffp-contract=off)

# Sanitizer configuration
if(FC_ENABLE_SANITIZERS)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.github/sanitizers.cmake)
        include(${CMAKE_CURRENT_SOURCE_DIR}/.github/sanitizers.cmake)
    endif()
endif()

# Add platform submodule
add_subdirectory(modules/platform EXCLUDE_FROM_ALL)

# Matrix module sources
set(MATRIX_SOURCES
    src/mean_var.c
)

# Create object library
add_library(finkit_matrix OBJECT ${MATRIX_SOURCES})

target_include_directories(finkit_matrix
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/platform/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_definitions(finkit_matrix PRIVATE FC_STATIC_BUILD)

# Set compile flags for SIMD variants in matrix.c
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(src/mean_var.c PROPERTIES COMPILE_FLAGS "-msse4.2 -mavx2 -mfma")
elseif(MSVC)
    set_source_files_properties(src/mean_var.c PROPERTIES COMPILE_FLAGS "/arch:AVX2")
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
