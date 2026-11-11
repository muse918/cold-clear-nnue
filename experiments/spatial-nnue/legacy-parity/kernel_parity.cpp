// Minimal source-equivalent reproducer, not a build of the Rust repository.
// Source: muse918/cold-clear-nnue @ 31713b0bd20c0bddf9dfa11daacdbd2d67888ae4
// bot/src/evaluation/nnue.rs: LinearClampLayer::forward_simd,
// forward_reduce, and EncodeLayer::forward_simd_2.
// The chosen integer-valued floats make the expected results exact.
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

struct Results {
    float dense_original_error;
    float dense_corrected_error;
    float dot_reference;
    float dot_original;
    float dot_corrected;
};

__attribute__((target("avx2,fma")))
Results run_kernels() {
    alignas(32) float weights[8][8] = {};
    alignas(32) float inp[8] = {1,1,1,1,1,1,1,1};
    for (int j = 0; j < 8; ++j) weights[0][j] = float(j - 4);
    __m256 ret = _mm256_setzero_ps();
    for (int i = 0; i < 8; ++i)
        ret = _mm256_fmadd_ps(_mm256_loadu_ps(weights[i]), _mm256_set1_ps(inp[i]), ret);
    alignas(32) float original[8], corrected[8];
    _mm256_storeu_ps(original, ret); // Original Rust code returns here.
    ret = _mm256_max_ps(ret, _mm256_setzero_ps());
    _mm256_storeu_ps(corrected, ret);
    float eo = 0, ec = 0;
    for (int j = 0; j < 8; ++j) {
        float expected = std::max(0.0f, float(j - 4));
        eo = std::max(eo, std::abs(original[j] - expected));
        ec = std::max(ec, std::abs(corrected[j] - expected));
    }

    alignas(32) float x[32], w[32];
    float expected = 0;
    for (int i = 0; i < 32; ++i) {
        x[i] = float(i+1); w[i] = float(i+2);
        expected += x[i] * w[i];
    }
    __m256 bad = _mm256_setzero_ps(), good = _mm256_setzero_ps();
    for (int i = 0; i < 32; i += 8) {
        // Rust macro uses set_ps(weight[0],...,weight[7]), reversing lanes.
        __m256 wb = _mm256_set_ps(w[i], w[i+1], w[i+2], w[i+3],
                                 w[i+4], w[i+5], w[i+6], w[i+7]);
        __m256 wg = _mm256_loadu_ps(w+i); // equivalently setr_ps
        __m256 xv = _mm256_loadu_ps(x+i);
        bad = _mm256_fmadd_ps(wb, xv, bad);
        good = _mm256_fmadd_ps(wg, xv, good);
    }
    alignas(32) float badv[8], goodv[8];
    _mm256_storeu_ps(badv, bad); _mm256_storeu_ps(goodv, good);
    float bs = 0, gs = 0;
    for (int i = 0; i < 8; ++i) { bs += badv[i]; gs += goodv[i]; }
    return {eo, ec, expected, bs, gs};
}

int main() {
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) {
        std::cout << "{\"status\":\"unavailable\",\"reason\":\"AVX2/FMA unavailable\"}\n";
        return 2;
    }
    auto r = run_kernels();
    // Same loop count as the original 40-row prefetched embedding loop:
    // 38 loop iterations then 1 tail add => 39, rather than 40.
    int original_rows = 0;
    for (int i = 0; i < 40 - 2; ++i) ++original_rows;
    ++original_rows;
    int corrected_rows = 0;
    for (int i = 0; i < 40 - 1; ++i) ++corrected_rows;
    ++corrected_rows;
    bool pass = r.dense_original_error == 4.0f
        && r.dense_corrected_error == 0.0f
        && r.dot_reference == 11968.0f
        && r.dot_original == 11632.0f
        && r.dot_corrected == r.dot_reference
        && original_rows == 39 && corrected_rows == 40;
    std::cout << "{\n"
        << "  \"scope\": \"minimal source-equivalent C++ kernels, deterministic synthetic values\",\n"
        << "  \"base_commit\": \"31713b0bd20c0bddf9dfa11daacdbd2d67888ae4\",\n"
        << "  \"avx2_fma\": true,\n"
        << "  \"dense_original_max_abs_error\": " << r.dense_original_error << ",\n"
        << "  \"dense_corrected_max_abs_error\": " << r.dense_corrected_error << ",\n"
        << "  \"dot_reference\": " << r.dot_reference << ",\n"
        << "  \"dot_original\": " << r.dot_original << ",\n"
        << "  \"dot_corrected\": " << r.dot_corrected << ",\n"
        << "  \"row_adds_original\": " << original_rows << ",\n"
        << "  \"row_adds_corrected\": " << corrected_rows << ",\n"
        << "  \"reproduced_and_corrected\": " << (pass ? "true" : "false") << "\n}\n";
    return pass ? 0 : 1;
}
