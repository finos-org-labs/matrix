/**
 * @file bench_main.c
 * @brief Benchmark runner entry point for matrix module
 */

#include "bench_framework.h"
#include <simd_detect.h>
#include <stdio.h>

/* External benchmark suites */
extern void bench_matrix_run(void);

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    fc_bench_init();
    fc_detect_simd();

    printf("matrix performance benchmarks v%s\n", FC_BENCH_VERSION);
    printf("SIMD level: %s\n", fc_simd_level_string(fc_detect_simd()));

    bench_matrix_run();

    fc_bench_cleanup();
    return 0;
}
