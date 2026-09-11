import numpy as np
from numba import njit
from .kdtree_helper import kdtree_order


# ── Direction helpers ─────────────────────────────────────────────────────────

def _canonical_directions(directions):
    first_nz = (directions != 0).argmax(axis=1)
    first_nz_sign = directions[np.arange(len(directions)), first_nz]
    return directions[first_nz_sign > 0]


def direction_blocks(level, dim=3):
    """Return canonical lattice directions grouped by squared norm, up to `level`."""
    offsets = np.arange(-1, 2)
    lattice = np.stack(np.meshgrid(*(offsets,) * dim, indexing="ij"), axis=-1).reshape(-1, dim)
    sq_norms = np.sum(lattice ** 2, axis=-1)
    dirs = _canonical_directions(lattice[(sq_norms > 0) & (sq_norms <= level ** 2)])
    return [dirs[np.sum(dirs ** 2, axis=-1) == k] for k in range(1, level + 1)]


# ── Geometric fiber sorting ───────────────────────────────────────────────────
@njit()
def _was_sorted(buf, count):
    for i in range(count - 1):
        if buf[i] > buf[i + 1]:
            return False
    return True

@njit
def _sort_oblique_fibers_inplace(X, s1, s2):
    n0, n1, n2 = X.shape
    was_sorted = True

    for o1 in range(-n1 + 1, n1):
        for o2 in range(-n2 + 1, n2):

            # Find first valid point
            k0 = 0
            while k0 < n0:
                i1, i2 = o1 + k0 * s1, o2 + k0 * s2
                if 0 <= i1 < n1 and 0 <= i2 < n2:
                    break
                k0 += 1

            if k0 >= n0:
                continue

            # Skip if this fiber has a valid predecessor:
            # it is the same fiber already represented by another offset.
            if k0 > 0:
                pi1, pi2 = o1 + (k0 - 1) * s1, o2 + (k0 - 1) * s2
                if 0 <= pi1 < n1 and 0 <= pi2 < n2:
                    continue

            buf = np.empty(n0, dtype=np.float64)
            count = 0

            for k in range(k0, n0):
                i1, i2 = o1 + k * s1, o2 + k * s2
                if 0 <= i1 < n1 and 0 <= i2 < n2:
                    buf[count] = X[k, i1, i2]
                    count += 1

            if count <= 1:
                continue

            # Don't sort if already sorted
            if _was_sorted(buf, count):
                continue

            was_sorted = False
            buf[:count].sort()

            j = 0
            for k in range(k0, n0):
                i1, i2 = o1 + k * s1, o2 + k * s2
                if 0 <= i1 < n1 and 0 <= i2 < n2:
                    X[k, i1, i2] = buf[j]
                    j += 1

    return was_sorted


def _sort_triaxial_fibers(X, ax0, ax1, ax2, s1, s2):
    perm = [ax0, ax1, ax2] + [i for i in range(X.ndim) if i not in (ax0, ax1, ax2)]
    Y = np.ascontiguousarray(np.transpose(X, perm)).copy()
    was_sorted = _sort_oblique_fibers_inplace(Y, s1, s2)
    return was_sorted, np.transpose(Y, list(np.argsort(perm)))


def sort_along_direction(X, direction):
    """Sort values of `X` in-place along geometric fibers defined by `direction`.

    Returns True if `X` was already sorted along that direction, False otherwise.
    """
    level = int(np.abs(direction).sum())

    if level == 1:
        axis = int(np.argmax(np.abs(direction)))
        if (np.diff(X, axis=axis) >= 0).all():
            return True
        X.sort(axis=axis, )
        return False

    if level == 2:
        ax0, ax1 = np.where(direction != 0)[0]
        flip = tuple(
            slice(None, None, int(direction[i])) if i in (ax0, ax1) else slice(None)
            for i in range(X.ndim)
        )
        X[...] = X[flip]
        was_sorted = True
        for offset in range(-max(X.shape[ax0], X.shape[ax1]) + 1, max(X.shape[ax0], X.shape[ax1])):
            diag = X.diagonal(offset=offset, axis1=int(ax0), axis2=int(ax1))
            if (np.diff(diag) >= 0).all():
                continue
            was_sorted = False
            diag.flags.writeable = True
            diag.sort(axis=-1, )
        X[...] = X[flip]
        return was_sorted

    if level == 3:
        ax0, ax1, ax2 = np.where(direction != 0)[0]
        _, s1, s2 = np.sign(direction[[ax0, ax1, ax2]]).astype(int)
        was_sorted, X[...] = _sort_triaxial_fibers(X, int(ax0), int(ax1), int(ax2), int(s1), int(s2))
        return was_sorted 

    raise ValueError(f"Unsupported direction with L1-norm {level}")


# ── monotonic_lagrangian loop construction ─────────────────────────────────────────────────────

def build_monotonic_lagrangian_loop(X, directions):
    N, K = len(X), len(directions)
    projections = X @ directions.T  # (N, K)

    rank = np.empty((K, N), dtype=np.int32)
    for k in range(K):
        order = np.argsort(projections[:, k])
        rank[k, order] = np.arange(N, dtype=np.int32)

    identity = np.arange(N, dtype=np.int32)
    h = [identity] + [rank[k] for k in range(K)]
    h_next = [rank[k] for k in range(K)] + [identity]

    perms = []
    for hk, hk_next in zip(h, h_next):
        sigma = np.empty(N, dtype=np.int32)
        sigma[hk] = hk_next
        perms.append(sigma)

    init_perm, end_perm = perms[0].copy(), perms[-1].copy()
    cycle_perms = perms[:-1]
    cycle_perms[0] = init_perm[end_perm]

    return init_perm, cycle_perms, end_perm


# ── monotonic_lagrangian iterative sort ────────────────────────────────────────────────────────

def sort_loop(loop, directions, shape, n_iter):
    idx_to_cycle, directions_cycle, cycle_to_idx = loop
    grid = idx_to_cycle.reshape(shape) #(N1, N2, ..., ND)
    first_step = True

    for it in range(n_iter):
        print("-", end="", flush=True)
        fully_sorted = True
        for direction, along_direction in zip(directions, directions_cycle):
            if not first_step:
                grid = along_direction[grid]
            first_step = False
            fully_sorted = fully_sorted & sort_along_direction(grid, direction)
        if fully_sorted:
            break

    print("it:", it)
    return (cycle_to_idx[grid]).ravel(), fully_sorted


# ── Public API ────────────────────────────────────────────────────────────────

def monotonic_lagrangian_argsort(X, shape, level, n_iter = 50, eps=1e-10, init = "kdtree"):
    """Compute a multi-directional sorted ordering of point cloud `X`.

    Parameters
    ----------
    X : ndarray of shape (N, D), point cloud
    level : int, max order of lattice directions considered
    eps : float, ties breaker. Must be smaller than minimal 
    spacing between points and bigger than machine precision

    Returns
    -------
    order : ndarray of shape (N,), permutation indice such that 
    X[order].reshape(*shape, D) is a monotonic sorted grid
    converged : bool
    """
    if init == "kdtree":
        print("performing kdtree initialisation...")
        X = X[kdtree_order(X, shape = shape)]
        print("done")
    directions = np.vstack(direction_blocks(level=level, dim=X.shape[-1])).astype(np.float64)
    rng = np.random.default_rng(42)
    loop = build_monotonic_lagrangian_loop(X.astype(np.float64) + eps * rng.random(X.shape), directions)
    return sort_loop(loop, directions, shape, n_iter)


def monotonic_lagrangian_sort(X, shape, level = 1, inplace=False, n_iter = 50):
    """Sort point cloud `X` using the monotonic lagrangian algorithm.

    Parameters
    ----------
    X : ndarray of shape (N, D)
    level : int, max order of lattice directions considered
    inplace : bool, if True modifies `X` in place

    Returns
    -------
    X_sorted : ndarray of shape (N, D), or None if `inplace=True`,
    such that X_sorted.reshape(*shape, D) is a monotonic sorted grid
    converged : bool
    """
    order, converged = monotonic_lagrangian_argsort(X, shape, level = level, n_iter = n_iter)
    if not converged:
        if level >= 2:
            ordernew, converged = monotonic_lagrangian_argsort(
                X[order], shape, level = level - 1, n_iter = n_iter
            )
            order = order[ordernew]
        import warnings
        warnings.warn("Convergence failed, increase n-iter")
    if inplace:
        X[:] = X[order]
        return None
    return X[order]