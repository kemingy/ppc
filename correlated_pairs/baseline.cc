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

float pearson_correlation(int n, const float* x, const float* y, double mean_x, double mean_y, double norm_x, double norm_y) {
    double exp = 0.0;
    for (int i = 0; i < n; ++i) {
        exp += (double(x[i]) - mean_x) * (double(y[i]) - mean_y);
    }
    return float(exp / (norm_x * norm_y));
}

double compute_mean(int n, const float* x) {
    double mean = 0.0;
    for (int i = 0; i < n; ++i) {
        mean += double(x[i]);
    }
    return mean /= double(n);
}

double compute_norm(int n, const float* x, double mean) {
    double norm = 0.0;
    for (int i = 0; i < n; ++i) {
        norm += (double(x[i]) - mean) * (double(x[i]) - mean);
    }
    return sqrt(norm);
}

// ny row, nx col
void correlate(int ny, int nx, const float *data, float *result) {
    std::vector<double> mean(ny, 0.0), norm(ny, 0.0);
    for (int i = 0; i < ny; ++i) {
        mean[i] = compute_mean(nx, data + i*nx);
        norm[i] = compute_norm(nx, data + i*nx, mean[i]);
    }
    for (int i = 0; i < ny; ++i) {
        for (int j = i; j < ny; ++j) {
            if (i == j) {
                result[j + i*ny] = 1.0;
                continue;
            }
            result[j + i*ny] = pearson_correlation(nx, data + i*nx, data + j*nx, mean[i], mean[j], norm[i], norm[j]);
        }
    }
}
