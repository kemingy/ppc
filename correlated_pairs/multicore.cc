/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

#include <cmath>
#include <vector>

constexpr int block = 4;

float pearson_correlation(int n, int num, const float* x, const float* y, double mean_x, double mean_y, double norm_x, double norm_y) {
    double mv[block] = {0.0};
    for (int i = 0; i < num; ++i) {
        for (int j = 0; j < block; ++j) {
            mv[j] += (double(x[i * block + j]) - mean_x) * (double(y[i * block + j]) - mean_y);
        }
    }
    double exp = 0.0;
    for (int i = 0; i < block; ++i) {
        exp += mv[i];
    }
    for (int i = num * block; i < n; ++i) {
        exp += (double(x[i]) - mean_x) * (double(y[i]) - mean_y);
    }
    return float(exp / (norm_x * norm_y));
}

double compute_mean(int n, int num, const float* x) {
    double mv[block] = {0.0};
    for (int i = 0; i < num; ++i) {
        for (int j = 0; j < block; ++j) {
            mv[j] += double(x[i * block + j]);
        }
    }
    double mean = 0.0;
    for (int i = 0; i < block; ++i) {
        mean += mv[i];
    }
    for (int i = num * block; i < n; ++i) {
        mean += double(x[i]);
    }
    return mean /= double(n);
}

double compute_norm(int n, int num, const float* x, double mean) {
    double mv[block] = {0.0};
    for (int i = 0; i < num; ++i) {
        for (int j = 0; j < block; ++j) {
            mv[j] += (double(x[i * block + j]) - mean) * (double(x[i * block + j]) - mean);
        }
    }
    double norm = 0.0;
    for (int i = 0; i < block; ++i) {
        norm += mv[i];
    }
    for (int i = num * block; i < n; ++i) {
        norm += (double(x[i]) - mean) * (double(x[i]) - mean);
    }
    return sqrt(norm);
}

// ny row, nx col
void correlate(int ny, int nx, const float *data, float *result) {
    int num = nx / block;

    std::vector<double> mean(ny, 0.0), norm(ny, 0.0);
    #pragma omp parallel for
    for (int i = 0; i < ny; ++i) {
        mean[i] = compute_mean(nx, num, data + i * nx);
        norm[i] = compute_norm(nx, num, data + i * nx, mean[i]);
    }

    #pragma omp parallel for
    for (int i = 0; i < ny; ++i) {
        for (int j = i; j < ny; ++j) {
            if (i == j) {
                result[j + i * ny] = 1.0;
                continue;
            }
            result[j + i * ny] = pearson_correlation(nx, num, data + i * nx, data + j * nx, mean[i], mean[j], norm[i], norm[j]);
        }
    }
}
