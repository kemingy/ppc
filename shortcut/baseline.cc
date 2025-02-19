#include <iostream>
#include <vector>
#include <random>
#include <chrono>

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
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                res[i * n + j] = std::min(res[i * n + j], graph[i * n + k] + graph[k * n + j]);
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