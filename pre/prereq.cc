struct Result {
    float avg[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- horizontal position: 0 <= x0 < x1 <= nx
- vertical position: 0 <= y0 < y1 <= ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
- output: avg[c]
*/
Result calculate(int ny, int nx, const float *data, int y0, int x0, int y1, int x1) {
    Result result{{0.0f, 0.0f, 0.0f}};
    if (x0 == x1 || y0 == y1) {
        return result;
    }
    double r = 0.0, g = 0.0, b = 0.0;
    for (int i = x0; i < x1; i++) {
        for (int j = y0; j < y1; j++) {
            r += double(data[3 * i + 3 * nx * j]);
            g += double(data[1 + 3 * i + 3 * nx * j]);
            b += double(data[2 + 3 * i + 3 * nx * j]);
        }
    }
    double total = double((x1 - x0) * (y1 - y0));
    result.avg[0] = float(r / total);
    result.avg[1] = float(g / total);
    result.avg[2] = float(b / total);
    return result;
}
