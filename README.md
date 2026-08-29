# Cartesian Grid Sort Algorithm

<p align="center">
<img src="cartesian_sort_illust.png" />
</p>

This repository illustrates the core cartesian grid sort procedure of the [`SquareNet`](https://github.com/Space-filling-net/SquareNet) ❒ gridification engine for demonstration purposes.

It showcases the live progress of the grid sorting algorithm and includes an [animated GIF](https://github.com/Space-filling-net/Cartesian-Grid-Sort/blob/main/sort.gif) alongside the simple Python script used to generate it, as well as both the full python (slow, simple) and C++ implementation (optimized).

<p align="center">
<img src="cartesian_sort_illust2.png" />
</p>

---

## Quick Start

The `Cartesian Grid Sort` allows to structure arbitrary point clouds as a multi-dimensional grid 𝄜. The algorithm is quite simple once one gets the main idea and could be reused in various contexts where a spatially coherent multi index structure can be useful. Note that end users should rather refer to SquareNet gridfication package itself (see for example this [tutorial](https://github.com/Space-filling-net/SquareNet/blob/main/00_getting_started.ipynb), or this [benchmark](https://github.com/Space-filling-net/SquareNet/blob/main/benchmark/squarenet_benchmark_kdtree.ipynb) with kd-tree) which simply requires to 
```python
pip install squarenet
```

Regarding this auxiliary repository:
- Run `main.py` to reproduce the quick visual animated demo that showcases gridification in progress.
- Feel free to look at what's inside `sort.py` (not optimized, for illustration purposes) to fully understand how the algorithm works.
- See `sort_core.cpp` for a C++ optimized version (multi-threaded).  
  It achieves **< 200 ms** on 1 million 2D points (tested on an old Ryzen 3 3250U). To reproduce the experiment:

**Windows (MSVC):**
```powershell
cl /O2 /openmp /EHsc /std:c++17 sort_core.cpp
.\sort_core.exe
```
**Linux/macOS:**
```Bash
g++ -O3 -fopenmp -std=c++17 sort_core.cpp -o sort_core
# or
clang++ -O3 -fopenmp -std=c++17 sort_core.cpp -o sort_core
./sort_core
```

- See `sort_core_diagonal.cpp` for the generalised cartesian sort algorithm (including diagonal sort).
roughly 5 times slower than the basic approach, but improves the resulting grid.

## Algorithm (2D Version)

*Note: The generalization to higher dimensions is straightforward.*

**Initialization:**
Take the  $N$ arbitrary Euclidean points to be processed:

$$ (x_k, y_k)_{1 \le k \le N} $$

In the main animation example, $N = 4225 = M^2 = 65 \times 65$ points. If N is not a perfect square, pad with dummy $\pm \infty$ points that will fall in empty slots of the grid to complete $N$ to the nearest square:

$$ N \leftarrow M^2 $$ 

Randomly assign the flat key $k$ to a 2D multi-key to form a grid (purely random initialization): 

$$ k \leftrightarrow [i, j]_k \quad i,j \in 1,2,3,..., M $$

$$ (x_k, y_k) \leftrightarrow (x_{ij}, y_{ij}) $$

**Iterative Sorting Procedure 🔄:**
1. Sort the Points according to their $x$-coordinate along the row key $i$: update $[i, j] \leftarrow [i', j]$ where $i'$ ensures the $x_{ij}$ coordinates are sorted along the $i$-axis (all columns $j$ are processed in parallel).
2. Sort the Points according to their $y$-coordinate along the column key $j$: update $[i, j] \leftarrow [i, j']$ to ensure monotonic y-coordinates.
3. Check if the $x$-sorting was broken by applying the $y$-sorting step (which is highly probable). If so, return to step 1 and repeat until both dimensions are simultaneously satisfied.

## Output

The algorithm produces a **bijective mapping** from the raw points `RP`, shape `[4225, 2]`: $(x_k, y_k)$ to a gridded tensor `GT`, shape `[65, 65, 2]`: $(x_{ij}, y_{ij})$. 

Upon termination, the resulting gridded view `GT` is guaranteed to be monotonic 📈 :
* $x$ strictly increases along $i$ ($\rightarrow$)
* $y$ strictly increases along $j$ ($\uparrow$)

This ensures that the multi-key $[i, j]$ is spatially coherent, meaning local neighborhoods are roughly preserved: the nearest spatial neighbors of a point with multi-key $[i, j]$ will likely have adjacent multi-keys $[i\pm1, j\pm1]$. Though a few outliers will unavoidably land at $[i\pm2, j\pm2]$ or further.

By construction, the transformation is a bijective assignment between the raw point key $k$ and the grid multi-key $[i, j]$. This allows for seamless data transfer between the flat point list and the grid using simple fancy indexing operations.

## Generalized Cartesian Sort - Diagonal improvement

The axis-monotonic criterion allows to sort point cloud with a simple and fast axis based procedure. But this basic version can be enhanced with diagonal steps. Diagonal (`up-right` / `down-right` ) 1D sorts works exactly as the row / column 1D steps, besides that they are applyed on diagonal levels of the grid. An optimization step of the generalized cartesian algorithm is thus:

- ➡️ row sort          
- ⬆️ column sort   
- ↗️ up-right sort  
- ↘️ down-right sort 

The full optimization step is repeated unutil convergence. The up-right sort will make $x+y$ increasing on upright levels (i-j = cst) and the down-right sort will make $x-y$ increasing on downright levels. Diagonal improvement makes the overal runtime of the algorithm roughly 10 times slower, but ensures a "stronger" monotony property of the resulting grid, which is not only monotonic on the natural axes of the grid, but also on the diagonals.

## Proof of Termination & link to Optimal Transport

A notable aspect of the `Cartesian Grid Sort` (both basic and generalized) algorithm is its proof of termination, which is relatively simple and establishes a link to `Optimal Transport` 🚙 (though the cartesian grid sort algorithm doesn't provide exact optimal transport but greedy and fast convergence to a good local minimum). 

In fact, Cartesian Grid Sort can be seen as a collective Coordinate Descent applyied on the Optimal Transport loss. Bue to the classical [Rearrangement Inequality](https://en.wikipedia.org/wiki/Rearrangement_inequality), each sorting step freezes all axes of the grid but one and solves the corresponding one-dimentional subproblem, making following quantity  (total transport energy of the grid) decreasing:

$$ \sum_{i,j} \left( (x_{ij} - i)^2 + (y_{ij} - j)^2 \right) $$

The transport energy of the grid is therefore a monovariant, garanteeing mathematicall termination of the algorithm because no cycle can occur. In practical—and even adversarial—cases, no more than 100 total iterations are typically required.

## Grid versus tree & link to KDTree

There is an interesting parallel to draw between the data structure that `Cartesian Grid Sort` produces (a spatially coherent, monotonic multi index) and standard `KDTree` 🌲 data structure. In fact, in cases where N is an exact power of 2, the `KDTree` recursive  partitioning path of each point of the cloud (left-down-left-up-right-up...) can directly be converted to a [i, j] multi-index, and it turns out that the corresponding grid $(x_{ij}, y_{ij})$ will already be sorted (x coordinate increasing along i and y coordinate along j) by construction of the tree. Notably, the construction of the tree and the cartesian sort grid have essentially the same runtime (see benchmark mentioned in Quick Start section).

The main difference between KDTree and Cartesian Grid Sort is the paradigm. KDTree is coarse to fine and threshold based (left/right, up/down)  while Cartesian Grid Sort is purely linear (i/i+1, j/j+1), with row / column axes progressively following the local structure of the point cloud without strong discontinuities. What will really make a difference is the context where the data structure is used: if e.g. one is interested for the neighbors of a single target, KDTree is probably the best choice. If one need the neighbors of all the points, e.g. to apply a Convolution Neural Network on a geometric dataset, Cartesian Grid Sort offers a simple interface between efficient tensor based frameworks and unstructured point clouds.

---

## Take home

The idea of `Cartesian grid sort` is simple: loop over 1D Cartesian projections of the point cloud (x, y, z, ...) and sort points along the corresponding grid axis (rows, columns, etc). Each 1D sort is O(N log N). Since sorting along one axis partially undoes the ordering along previous axes, you repeat the full sorting loop until all axes are sorted simultaneously — typically fewer than 50 iterations.

**What you don't get:**

- **Optimal Transport.** `Cartesian Grid Sort` trades exactness for speed. If you need the provably optimal assignment, this isn't the right tool.
- **Reverse neighborhood.** Close in space → close in grid, but *not* the other way around. Holes, clusters, and gaps in your data will be "closed" by the grid, which can place unrelated points next to each other.
- **Angular preservation.** Volume and angles can't both be conserved in the general case by a mapping (classical result). Expect some angular distortion, especially near boundaries.

**What you get:**

- **Speed.** ⏱️ Millions of points in seconds. All operations are native tensor ops.
- **Coordinate monotonicity.** x increases along rows, y along columns, etc. This enables e.g. the generalised searchsorted query tool of `SquareNet` for approximate k-NN.
- **Neighborhood preservation.** Points close in space land close in the grid. Concrete experimental results on a 1M-point 2D dataset (France map distribution 🗼):
  - Requesting a 11×11 square window  arround a query point [i,j]: [i-5:i+6, j-5:j+6] = 0.01% of candidates → recovers ~97% of the physical nearest neighbors
  - Requesting a 31×31 square window ([i-15:i+16, j-15:j+16] = 0.1% of candidates) → recovers ~99.5%

## Note: Theoretical Background

Cartesian Grid Sort was implemented independently as a research project, starting from a concrete practical problem involving massive neighborhood queries. It turns out, however, that the problem has also been studied from a theoretical perspective: the data structure presented in this project was introduced by Joselli et al. [1, 2] and formally analyzed by Skrodzki, Reitebuch, and Polthier [3]. Both are particularly interesting resources, and their reading is highly recommended for readers interested in the mathematical aspects of the method.

In their framework, the multi-index structure corresponds to what they call a **Neighborhood Grid** [3, Section 2.1], and the monotonicity property along the Cartesian axes a **stable state** [3, Definition 1].

[1] Joselli et al., "A Neighborhood Grid Data Structure for Massive 3D Crowd Simulation on GPU", *VIII Brazilian Symposium on Games and Digital Entertainment (SBGames)*, IEEE, 2009, pp. 121–131.

[2] Joselli et al., "Neighborhood Grid: A Novel Data Structure for Fluids Animation with GPU Computing", *Journal of Parallel and Distributed Computing*, vol. 75, 2015, pp. 20–28.

[3] Skrodzki, Reitebuch, Polthier, "Combinatorial and Asymptotical Results on the Neighborhood Grid", arXiv:1710.03435, 2018.
(Published in: Skrodzki, PhD thesis, Freie Universität Berlin, 2019.)

[1] Joselli et al., SBGAMES 2009.
[2] Joselli et al., *Journal of Parallel and Distributed Computing*, 2015.
[3] Skrodzki, Reitebuch, Polthier, arXiv:1710.03435, 2018.
(Published in: Skrodzki, PhD thesis, Freie Universität Berlin, 2019.)

