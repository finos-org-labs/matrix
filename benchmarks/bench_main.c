/**
 * @file bench_main.c
 * @brief Benchmark runner entry point for matrix module
 */

#include "bench_framework.h"
#include <platform.h>
#include <simd_detect.h>
#include <stdio.h>

/* External benchmark suites */
extern void bench_matrix_run(void);

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    fc_init();

    fc_simd_level_t simd_level = fc_get_simd_level();
    fc_bench_init();

    printf("matrix performance benchmarks v%s\n", FC_BENCH_VERSION);
    printf("SIMD level: %s\n", fc_simd_level_string(simd_level));

    bench_matrix_run();

    fc_bench_cleanup();
    return 0;
}
