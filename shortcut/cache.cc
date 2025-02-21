#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <limits>
#include <x86intrin.h>
#include <algorithm>

typedef float f32x8 __attribute__ ((vector_size (8 * sizeof(float))));

constexpr float INF = std::numeric_limits<float>::infinity();

constexpr f32x8 INFx8 {
    INF, INF, INF, INF, INF, INF, INF, INF
};

static inline f32x8 swap4(f32x8 x) {
    return _mm256_permute2f128_ps(x, x, 0b00000001);
}

static inline f32x8 swap2(f32x8 x) {
    return _mm256_permute_ps(x, 0b01001110);
}

static inline f32x8 swap1(f32x8 x) {
    return _mm256_permute_ps(x, 0b10110001);
}

static inline float f32x8_reduce_min(f32x8 x) {
    float r = INF;
    for (int i = 0; i < 8; ++i) {
        r = std::min(r, x[i]);
    }
    return r;
}

static inline f32x8 f32x8_min(f32x8 x, f32x8 y) {
    return x < y ? x : y;
}

void random_graph(int n, std::vector<float>& graph) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                graph[i * n + j] = 0.0;
            } else {
                graph[i * n + j] = dis(gen);
            }
        }
    }
}

void step(std::vector<float>& graph, std::vector<float>& res, int n) {
    // num of rows after padding
    int nr = (n + 8 - 1) / 8;

    // c++17 aligned
    std::vector<f32x8> origin(n * nr, INFx8);
    std::vector<f32x8> trans(n * nr, INFx8);

    #pragma omp parallel for
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < 8; ++k) {
                int m = 8 * i + k;
                origin[n * i + j][k] = (m < n) ? graph[n * m + j] : INF;
                trans[n * i + j][k] = (m < n) ? graph[n * j + m] : INF;
            }
        }
    }

    // for better cache performance
    std::vector<std::tuple<int, int, int>> rows(nr * nr);
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nr; ++j) {
            // interleave i and j
            int ij = _pdep_u32(i, 0x55555555) | _pdep_u32(j, 0xAAAAAAAA);
            rows[i * nr + j] = std::make_tuple(ij, i, j);
        }
    }
    std::sort(rows.begin(), rows.end());

    #pragma omp parallel for
    for (const auto& row: rows) {
        int i = std::get<1>(row);
        int j = std::get<2>(row);
        f32x8 vv000 = INFx8;
        f32x8 vv001 = INFx8;
        f32x8 vv010 = INFx8;
        f32x8 vv011 = INFx8;
        f32x8 vv100 = INFx8;
        f32x8 vv101 = INFx8;
        f32x8 vv110 = INFx8;
        f32x8 vv111 = INFx8;
        for (int k = 0; k < n; ++k) {
            // prefetch that 20 iterations later
            // out-of-bound access is harmless here
            constexpr int PF = 20;
            __builtin_prefetch(&origin[n * i + k + PF]);
            __builtin_prefetch(&trans[n * j + k + PF]);
            f32x8 a000 = origin[n * i + k];
            f32x8 b000 = trans[n * j + k];
            f32x8 a100 = swap4(a000);
            f32x8 a010 = swap2(a000);
            f32x8 a110 = swap2(a100);
            f32x8 b001 = swap1(b000);
            vv000 = f32x8_min(vv000, a000 + b000);
            vv001 = f32x8_min(vv001, a000 + b001);
            vv010 = f32x8_min(vv010, a010 + b000);
            vv011 = f32x8_min(vv011, a010 + b001);
            vv100 = f32x8_min(vv100, a100 + b000);
            vv101 = f32x8_min(vv101, a100 + b001);
            vv110 = f32x8_min(vv110, a110 + b000);
            vv111 = f32x8_min(vv111, a110 + b001);
        }
        f32x8 vv[8] = {vv000, vv001, vv010, vv011, vv100, vv101, vv110, vv111};
        for (int k = 1; k < 8; k += 2) {
            vv[k] = swap1(vv[k]);
        }
        for (int ri = 0; ri < 8; ++ri) {
            for (int rj = 0; rj < 8; ++rj) {
                int r = rj + i * 8;
                int c = ri + j * 8;
                if (r < n && c < n) {
                    res[n * r + c] = vv[rj ^ ri][ri];
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int n = 5;
    if (argc > 1) {
        n = std::stoi(argv[1]);
    }
    std::cout << "n: " << n << std::endl;

    std::vector<float> graph(n * n), res(n * n, 0.0);
    random_graph(n, graph);

    auto start = std::chrono::steady_clock::now();
    step(graph, res, n);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    return 0;
}