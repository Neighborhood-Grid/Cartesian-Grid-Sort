#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <string>
#include <omp.h>

struct Point {
    double x, y;
    int id;
};

// ---------- Sort functions ----------

// 1. ROW sort (x increasing)
void sort_row(std::vector<Point>& grid, int N) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        std::sort(grid.begin() + i * N, grid.begin() + (i + 1) * N,
                  [](const Point& a, const Point& b) { return a.x < b.x; });
    }
}

// 2. COLUMN sort (y increasing)
void sort_column(std::vector<Point>& grid, int N, std::vector<std::vector<Point>>& thread_buf) {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < N; ++j) {
        int tid = omp_get_thread_num();
        auto& buf = thread_buf[tid];
        buf.resize(N);
        for (int i = 0; i < N; ++i) buf[i] = grid[i * N + j];
        std::sort(buf.begin(), buf.end(),
                  [](const Point& a, const Point& b) { return a.y < b.y; });
        for (int i = 0; i < N; ++i) grid[i * N + j] = buf[i];
    }
}

// 3. UPLEFT diagonal sort (i - j = cst, x + y increasing)
void sort_upleft(std::vector<Point>& grid, int N, std::vector<std::vector<Point>>& thread_buf) {
    #pragma omp parallel for schedule(dynamic)
    for (int d = -(N - 1); d <= (N - 1); ++d) {
        int tid = omp_get_thread_num();
        auto& buf = thread_buf[tid];
        buf.clear();

        for (int i = 0; i < N; ++i) {
            int j = i - d;
            if (j >= 0 && j < N) buf.push_back(grid[i * N + j]);
        }
        if (buf.size() <= 1) continue;

        std::sort(buf.begin(), buf.end(),
                  [](const Point& a, const Point& b) {
                      return (a.x + a.y) < (b.x + b.y);
                  });

        size_t k = 0;
        for (int i = 0; i < N; ++i) {
            int j = i - d;
            if (j >= 0 && j < N) grid[i * N + j] = buf[k++];
        }
    }
}

// 4. DOWNLEFT diagonal sort (i + j = cst, x - y increasing)
void sort_downleft(std::vector<Point>& grid, int N, std::vector<std::vector<Point>>& thread_buf) {
    #pragma omp parallel for schedule(dynamic)
    for (int s = 0; s <= 2 * (N - 1); ++s) {
        int tid = omp_get_thread_num();
        auto& buf = thread_buf[tid];
        buf.clear();

        for (int i = 0; i < N; ++i) {
            int j = s - i;
            if (j >= 0 && j < N) buf.push_back(grid[i * N + j]);
        }
        if (buf.size() <= 1) continue;

        std::sort(buf.begin(), buf.end(),
                  [](const Point& a, const Point& b) {
                      return (a.x - a.y) > (b.x - b.y);
                  });

        size_t k = 0;
        for (int i = 0; i < N; ++i) {
            int j = s - i;
            if (j >= 0 && j < N) grid[i * N + j] = buf[k++];
        }
    }
}

// ---------- Disorder count, OT loss, check progress ----------

bool is_monotone(const std::vector<Point>& grid, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 1; j < N; ++j)
            if (grid[i * N + j].x < grid[i * N + j - 1].x) return false;

    for (int j = 0; j < N; ++j)
        for (int i = 1; i < N; ++i)
            if (grid[i * N + j].y < grid[(i - 1) * N + j].y) return false;

    for (int d = -(N - 1); d <= (N - 1); ++d) {
        double prev = -1e300;
        for (int i = 0; i < N; ++i) {
            int j = i - d;
            if (j < 0 || j >= N) continue;
            double val = grid[i * N + j].x + grid[i * N + j].y;
            if (val < prev) return false;
            prev = val;
        }
    }

    for (int s = 0; s <= 2 * (N - 1); ++s) {
        double prev = 1e300;
        for (int i = 0; i < N; ++i) {
            int j = s - i;
            if (j < 0 || j >= N) continue;
            double val = grid[i * N + j].x - grid[i * N + j].y;
            if (val > prev) return false;
            prev = val;
        }
    }
    return true;
}

int count_inversions(const std::vector<Point>& grid, int N) {
    int inv = 0;
    for (int i = 0; i < N; ++i)
        for (int j = 1; j < N; ++j)
            if (grid[i * N + j].x < grid[i * N + j - 1].x) inv++;

    for (int j = 0; j < N; ++j)
        for (int i = 1; i < N; ++i)
            if (grid[i * N + j].y < grid[(i - 1) * N + j].y) inv++;

    for (int d = -(N - 1); d <= (N - 1); ++d) {
        double prev = -1e300;
        bool first = true;
        for (int i = 0; i < N; ++i) {
            int j = i - d;
            if (j < 0 || j >= N) continue;
            double val = grid[i * N + j].x + grid[i * N + j].y;
            if (!first && val < prev) inv++;
            prev = val;
            first = false;
        }
    }

    for (int s = 0; s <= 2 * (N - 1); ++s) {
        double prev = 1e300;
        bool first = true;
        for (int i = 0; i < N; ++i) {
            int j = s - i;
            if (j < 0 || j >= N) continue;
            double val = grid[i * N + j].x - grid[i * N + j].y;
            if (!first && val > prev) inv++;
            prev = val;
            first = false;
        }
    }
    return inv;
}

double compute_ot_loss(const std::vector<Point>& grid, int N) {
    double loss = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double target_x = static_cast<double>(j) / (N - 1);
            double target_y = static_cast<double>(i) / (N - 1);
            const auto& p = grid[i * N + j];
            double dx = p.x - target_x;
            double dy = p.y - target_y;
            loss += dx * dx + dy * dy;
        }
    }
    return loss;
}

int main() {
    const int N = 1000;
    std::vector<Point> grid(N * N);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            grid[i * N + j] = { dist(rng), dist(rng), i * N + j };

    const int nthreads = omp_get_max_threads();
    std::vector<std::vector<Point>> thread_buf(nthreads);

    auto start = std::chrono::high_resolution_clock::now();
    int iterations = 0;
    const int MAX_ITER = 100;

    std::vector<double> loss_history;
    std::vector<int> inversion_history;

    loss_history.push_back(compute_ot_loss(grid, N));
    inversion_history.push_back(count_inversions(grid, N));

    while (iterations < MAX_ITER && !is_monotone(grid, N)) {
        sort_row(grid, N);
        sort_column(grid, N, thread_buf);
        sort_upleft(grid, N, thread_buf);
        sort_downleft(grid, N, thread_buf);

        ++iterations;
        loss_history.push_back(compute_ot_loss(grid, N));
        inversion_history.push_back(count_inversions(grid, N));
    }

    auto end = std::chrono::high_resolution_clock::now();
    double runtime_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Threads OpenMP      : " << omp_get_max_threads() << "\n";
    std::cout << "Iterations          : " << iterations << "\n";
    std::cout << "Total time (ms)    : " << runtime_ms << "\n";
    std::cout << "Check is_monotone() : " << (is_monotone(grid, N) ? "YES" : "NO") << "\n";

    std::cout << "\nHistory:\n";
    std::cout << std::setw(10) << "Iter"
              << std::setw(15) << "Inversions"
              << std::setw(20) << "OT Loss" << "\n";

    for (size_t i = 0; i < loss_history.size(); ++i) {
        std::cout << std::setw(10) << i
                  << std::setw(15) << inversion_history[i]
                  << std::setw(20) << loss_history[i]
                  << "\n";
    }

    return 0;
}
