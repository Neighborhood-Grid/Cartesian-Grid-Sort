//cl /O2 /openmp /EHsc /std:c++17 sort_core_diagonal_3D.cpp
//>> .\sort_core_diagonal_3D.exe


#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <array>
#include <omp.h>
#include <fstream>

// Define 3D points
struct Point {
    double x, y, z;
    int id;
};

// Unitary direction vectors for all the diagonals
const std::array<std::array<int,3>,13> DIRS = {{
    {{1, 0, 0}}, {{0, 1, 0}}, {{0, 0, 1}},
    {{1, 1, 0}}, {{1,-1, 0}}, {{1, 0, 1}}, {{1, 0,-1}},
    {{0, 1, 1}}, {{0, 1,-1}},
    {{1, 1, 1}}, {{1, 1,-1}}, {{1,-1, 1}}, {{1,-1,-1}}
}};

inline int idx(int i, int j, int k, int N) {
    return i * N * N + j * N + k;
}

// Count the total number of inversions across the 13 directions
long long count_inversions(const std::vector<Point>& grid, int N) {
    long long inv = 0;
    for (const auto& u : DIRS) {
        int ux = u[0], uy = u[1], uz = u[2];
        for (int i0 = 0; i0 < N; ++i0)
        for (int j0 = 0; j0 < N; ++j0)
        for (int k0 = 0; k0 < N; ++k0) {
            int pi = i0 - ux, pj = j0 - uy, pk = k0 - uz;
            if (pi >= 0 && pi < N && pj >= 0 && pj < N && pk >= 0 && pk < N)
                continue;

            double prev = -1e300;
            bool first = true;
            int i = i0, j = j0, k = k0;
            while (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
                const Point& p = grid[idx(i,j,k,N)];
                double val = p.x*ux + p.y*uy + p.z*uz;
                if (!first && val < prev) ++inv;
                prev = val;
                first = false;
                i += ux; j += uy; k += uz;
            }
        }
    }
    return inv;
}

void sort_direction(std::vector<Point>& grid, int N, int ux, int uy, int uz,
                    std::vector<std::vector<Point>>& thread_buf) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& buf = thread_buf[tid];
        buf.clear();
        buf.reserve(N);

        #pragma omp for schedule(dynamic) 
        for (int i0 = 0; i0 < N; ++i0)
        for (int j0 = 0; j0 < N; ++j0)
        for (int k0 = 0; k0 < N; ++k0) {
            int pi = i0 - ux, pj = j0 - uy, pk = k0 - uz;
            if (pi >= 0 && pi < N && pj >= 0 && pj < N && pk >= 0 && pk < N)
                continue;

            buf.clear();
            int i = i0, j = j0, k = k0;
            while (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
                buf.push_back(grid[idx(i,j,k,N)]);
                i += ux; j += uy; k += uz;
            }
            if (buf.size() <= 1) continue;

            std::sort(buf.begin(), buf.end(),
                [ux,uy,uz](const Point& a, const Point& b) {
                    return (a.x*ux + a.y*uy + a.z*uz) <
                           (b.x*ux + b.y*uy + b.z*uz);
                });

            i = i0; j = j0; k = k0;
            size_t t = 0;
            while (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
                grid[idx(i,j,k,N)] = buf[t++];
                i += ux; j += uy; k += uz;
            }
        }
    }
}

int main() {
    const int N = 100;
    const int NNN = N * N * N;

    std::vector<Point> grid(NNN);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                grid[idx(i,j,k,N)] = {dist(rng), dist(rng), dist(rng), idx(i,j,k,N)};

    const int nthreads = omp_get_max_threads();
    std::vector<std::vector<Point>> thread_buf(nthreads);

    // Initial disorder
    long long inv0 = count_inversions(grid, N);
    std::cout << "Initial inversions   : " << inv0 << "\n";
    std::cout << "Threads              : " << nthreads << "\n";
    std::cout << "Timeout              : 30 s\n\n";
    std::cout << std::setw(6) << "Iter"
              << std::setw(14) << "Inversions"
              << std::setw(12) << "Time (s)" << "\n";
    std::cout << std::string(50, '-') << "\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    int iterations = 0;
    const double TIMEOUT = 30.0;
    std::vector<long long> history;
    history.push_back(inv0);

    while (true) {
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - t0).count();
        if (elapsed >= TIMEOUT) {
            std::cout << "\n>>> TIMEOUT: 30 s reached\n";
            break;
        }

        // One complete iteration (13 directions)
        for (const auto& u : DIRS)
            sort_direction(grid, N, u[0], u[1], u[2], thread_buf);

        ++iterations;
        long long inv = count_inversions(grid, N);
        history.push_back(inv);

        double pct = 100.0 * inv / inv0;
        std::cout << std::setw(6) << iterations
                  << std::setw(14) << inv
                  << std::setw(12) << std::setprecision(2) << elapsed
                  << std::endl;

        if (inv == 0) {
            std::cout << "\n>>> CONVERGENCE reached!\n";
            break;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\n=== Summary ===\n";
    std::cout << "Number of points : " << NNN << "\n";
    std::cout << "Iterations completed : " << iterations << "\n";
    std::cout << "Total time           : " << std::fixed << std::setprecision(2) << total << " s\n";
    std::cout << "Remaining disorder   : " << std::setprecision(2)
              << 100.0 * history.back() / inv0 << " %\n";

}
