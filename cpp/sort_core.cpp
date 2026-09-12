#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <omp.h>

struct Point {
    double x, y;
    int id;
};

int main() {
    const int N = 1000;
    std::vector<Point> grid(N * N);

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            grid[i * N + j] = { dist(rng), dist(rng), i * N + j };

    auto is_monotone = [&]() -> bool {
        for (int i = 0; i < N; ++i)
            for (int j = 1; j < N; ++j)
                if (grid[i*N + j].x < grid[i*N + j - 1].x) return false;
        for (int j = 0; j < N; ++j)
            for (int i = 1; i < N; ++i)
                if (grid[i*N + j].y < grid[(i-1)*N + j].y) return false;
        return true;
    };

    const int nthreads = omp_get_max_threads();
    std::vector<std::vector<Point>> thread_cols(nthreads, std::vector<Point>(N));

    auto start = std::chrono::high_resolution_clock::now();
    int iterations = 0;
    const int MAX_ITER = 1000;

    while (iterations < MAX_ITER) {
        if (is_monotone()) break;

        // Lines
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            std::sort(grid.begin() + i * N, grid.begin() + (i + 1) * N,
                      [](const Point& a, const Point& b) { return a.x < b.x; });
        }

        // Columns
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < N; ++j) {
            int tid = omp_get_thread_num();
            auto& col = thread_cols[tid];
            for (int i = 0; i < N; ++i) col[i] = grid[i * N + j];
            std::sort(col.begin(), col.end(),
                      [](const Point& a, const Point& b) { return a.y < b.y; });
            for (int i = 0; i < N; ++i) grid[i * N + j] = col[i];
        }

        ++iterations;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double runtime_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Available threads   : " << omp_get_max_threads() << "\n";
    std::cout << "Sort iterations     : " << iterations << "\n";
    std::cout << "Total runtime (ms)        : " << runtime_ms << "\n";
    std::cout << "check convergence   : " << (is_monotone() ? "yes" : "no") << "\n";
}
