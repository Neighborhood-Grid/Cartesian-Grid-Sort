import numpy as np


def kdtree_order(X, shape, eps=1e-10):
    """
    Build a KD-tree order matching an arbitrary 2D or 3D grid shape.

    Parameters
    ----------
    X : ndarray, shape (N, D)
        Point cloud, with D = 2 or 3.
    shape : tuple of int
        Target grid shape. Must have len(shape) == D and
        np.prod(shape) == N.

    Returns
    -------
    flat_order : ndarray, shape (N,)
        Permutation such that

            X[flat_order].reshape(*shape, D)

        follows the recursive KD-tree partitioning.

    Notes
    -----
    Each KD-tree node corresponds to a pure hyperrectangle of the
    logical grid. Splits are made along one grid dimension, with
    unequal halves when its size is odd.
    """
    rng = np.random.default_rng(42)
    X = np.asarray(X)  + eps * rng.random(X.shape)
    shape = tuple(shape)

    N, D = X.shape

    assert D in (2, 3), "Only 2D and 3D point clouds are supported"
    assert len(shape) == D, "shape must have the same dimension as X"
    assert np.prod(shape) == N, "np.prod(shape) must equal len(X)"
    assert all(s >= 1 for s in shape), "shape dimensions must be >= 1"

    # Logical grid coordinate of every point
    grid_coord = np.empty((N, D), dtype=np.int64)

    def split(indices, depth, origin, sizes):
        # Leaf
        if all(s == 1 for s in sizes):
            grid_coord[indices[0]] = origin
            return

        # Prefer cyclic KD axis, but skip dimensions already collapsed
        preferred_axis = depth % D

        for offset in range(D):
            axis = (preferred_axis + offset) % D
            if sizes[axis] > 1:
                break

        # Stable KD split
        order = np.argsort(X[indices, axis], kind="stable")
        indices = indices[order]

        # Split logical dimension
        left_size = sizes[axis] // 2
        right_size = sizes[axis] - left_size

        # Number of points/cells in the left hyperrectangle
        left_sizes = list(sizes)
        left_sizes[axis] = left_size
        n_left = int(np.prod(left_sizes))

        # Left child
        left_origin = list(origin)

        split(
            indices[:n_left],
            depth + 1,
            tuple(left_origin),
            tuple(left_sizes),
        )

        # Right child
        right_origin = list(origin)
        right_origin[axis] += left_size

        right_sizes = list(sizes)
        right_sizes[axis] = right_size

        split(
            indices[n_left:],
            depth + 1,
            tuple(right_origin),
            tuple(right_sizes),
        )

    # Full logical grid
    split(
        np.arange(N),
        depth=0,
        origin=(0,) * D,
        sizes=shape,
    )

    # Multi-index -> C-order linear grid index
    grid_key = np.ravel_multi_index(
        grid_coord.T,
        shape,
    )

    return np.argsort(grid_key)