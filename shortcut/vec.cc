#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <limits>

typedef float f32x8 __attribute__ ((vector_size (8 * sizeof(float))));

constexpr float INF = std::numeric_limits<float>::infinity();

constexpr f32x8 INFx8 {
    INF, INF, INF, INF, INF, INF, INF, INF
};

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
    constexpr int nb = 8;
    int block = (n + nb - 1) / nb;

    // c++17 aligned
    std::vector<f32x8> origin(n * block);
    std::vector<f32x8> trans(n * block);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < block; ++j) {
            for (int k = 0; k < nb; ++k) {
                int m = j * nb + k;
                origin[i * block + j][k] = (m < n) ? graph[i * n + m] : INF;
                trans[i * block + j][k] = (m < n) ? graph[m * n + i] : INF;
            }
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            f32x8 row = INFx8;
            for (int k = 0; k < block; ++k) {
                f32x8 x = origin[i * block + k];
                f32x8 y = trans[j * block + k];
                f32x8 z = x + y;
                row = f32x8_min(row, z);
            }
            res[i * n + j] = f32x8_reduce_min(row);
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