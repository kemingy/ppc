#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <limits>

constexpr float INF = std::numeric_limits<float>::infinity();

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
    constexpr int nb = 4;
    int block = (n + nb - 1) / nb;
    int length = block * nb;
    std::vector<float> origin(n * length, INF);
    std::vector<float> trans(n * length, INF);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            origin[i * n + j] = graph[i * n + j];
            trans[i * n + j] = graph[j * n + i];
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float v[nb];
            for (int m = 0; m < nb; ++m) {
                v[m] = INF;
            }
            for (int k = 0; k < block; ++k) {
                for (int m = 0; m < nb; ++m) {
                    float x = origin[i * length + k * nb + m];
                    float y = trans[j * length + k * nb + m];
                    v[m] = std::min(v[m], x + y);
                }
            }
            float r = INF;
            for (int m = 0; m < nb; ++m) {
                r = std::min(r, v[m]);
            }
            res[i * n + j] = r;
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