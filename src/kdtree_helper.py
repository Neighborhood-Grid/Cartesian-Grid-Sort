import numpy as np


def kdtree_order(X, shape, eps=1e-10):
    X = np.asarray(X)
    shape = tuple(map(int, shape))
    N, D = X.shape
    assert D in (2, 3)
    assert len(shape) == D and np.prod(shape) == N
    assert all(s >= 1 for s in shape)

    rng = np.random.default_rng(42)
    X = X + eps * rng.random(X.shape)
    grid_coord = np.empty((N, D), dtype=np.int64)

    def split(idx, depth, origin, sizes):
        if np.prod(sizes) == 1:
            grid_coord[idx[0]] = origin
            return

        axis = depth % D
        for k in range(D):
            a = (axis + k) % D
            if sizes[a] > 1:
                axis = a
                break

        left_size = sizes[axis] // 2
        right_size = sizes[axis] - left_size

        left_sizes = list(sizes)
        left_sizes[axis] = left_size
        n_left = int(np.prod(left_sizes))

        p = np.argpartition(X[idx, axis], n_left - 1)
        idx = idx[p]
        left, right = idx[:n_left], idx[n_left:]

        lo = list(origin)
        split(left, depth + 1, tuple(lo), tuple(left_sizes))

        ro = list(origin)
        ro[axis] += left_size
        right_sizes = list(sizes)
        right_sizes[axis] = right_size
        split(right, depth + 1, tuple(ro), tuple(right_sizes))

    split(np.arange(N), 0, (0,) * D, shape)
    return np.argsort(np.ravel_multi_index(grid_coord.T, shape))