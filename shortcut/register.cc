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
    constexpr int nx = 8;
    constexpr int by = 3;
    int row_b = (n + nx - 1) / nx;
    int col_b = (n + by - 1) / by;
    int rows = col_b * by;

    // c++17 aligned
    std::vector<f32x8> origin(row_b * rows, INFx8);
    std::vector<f32x8> trans(row_b * rows, INFx8);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < row_b; ++j) {
            for (int k = 0; k < nx; ++k) {
                int m = j * nx + k;
                origin[i * row_b + j][k] = (m < n) ? graph[i * n + m] : INF;
                trans[i * row_b + j][k] = (m < n) ? graph[m * n + i] : INF;
            }
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < col_b; ++i) {
        for (int j = 0; j < col_b; ++j) {
            // init a [by x by] block
            f32x8 vv[by][by];
            for (int vi = 0; vi < by; ++vi) {
                for (int vj = 0; vj < by; ++vj) {
                    vv[vi][vj] = INFx8;
                }
            }
            for (int m =0; m < row_b; ++m) {
                f32x8 x0 = origin[row_b * (j * by + 0) + m];
                f32x8 x1 = origin[row_b * (j * by + 1) + m];
                f32x8 x2 = origin[row_b * (j * by + 2) + m];
                f32x8 y0 = trans[row_b * (i * by + 0) + m];
                f32x8 y1 = trans[row_b * (i * by + 1) + m];
                f32x8 y2 = trans[row_b * (i * by + 2) + m];
                vv[0][0] = f32x8_min(vv[0][0], x0 + y0);
                vv[0][1] = f32x8_min(vv[0][1], x0 + y1);
                vv[0][2] = f32x8_min(vv[0][2], x0 + y2);
                vv[1][0] = f32x8_min(vv[1][0], x1 + y0);
                vv[1][1] = f32x8_min(vv[1][1], x1 + y1);
                vv[1][2] = f32x8_min(vv[1][2], x1 + y2);
                vv[2][0] = f32x8_min(vv[2][0], x2 + y0);
                vv[2][1] = f32x8_min(vv[2][1], x2 + y1);
                vv[2][2] = f32x8_min(vv[2][2], x2 + y2);
            }
            for (int vi = 0; vi < by; ++vi) {
                for (int vj = 0; vj < by; ++vj) {
                    int ri = i * by + vi;
                    int rj = j * by + vj;
                    if (ri < n && rj < n) {
                        res[ri * n + rj] = f32x8_reduce_min(vv[vi][vj]);
                    }
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