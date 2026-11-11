"""Post-hoc graph label smoothing: synchronous kNN voting and sequential ICM.

All nodes outside indices_test are clamped. Only observed training labels and
classifier predictions may be supplied in target; never pass true test labels.
Dependencies: NumPy and SciPy.
"""
import numpy as np
from scipy import sparse


def contextual_graph(A, symmetrize=False):
    """Unweighted CSR support, without self-loops.

    Nonzero entries denote edges; distance magnitudes are ignored. For ICM,
    union symmetrization is used (max(B,B.T)): each unordered edge counts once.
    Explicit zero entries are absent, matching the GSA graph builder.
    """
    B = sparse.csr_matrix(A, dtype=float, copy=True)
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError('A must be square')
    if not np.isfinite(B.data).all() or np.any(B.data < 0):
        raise ValueError('A must contain finite nonnegative values')
    B.sum_duplicates()
    B.setdiag(0)
    B.eliminate_zeros()
    B.data[:] = 1.
    if symmetrize:
        B = B.maximum(B.T).tocsr()
    B.sort_indices()
    return B


def _inputs(B, target, indices_test, classes, max_iter):
    target = np.asarray(target)
    if target.ndim != 1 or len(target) != B.shape[0] or len(target) == 0:
        raise ValueError('target must be a nonempty vector of length n')
    if classes is None:
        classes, encoded = np.unique(target, return_inverse=True)
    else:
        classes = np.asarray(classes)
        if classes.ndim != 1 or not len(classes) or len(np.unique(classes)) != len(classes):
            raise ValueError('classes must be a nonempty vector of unique labels')
        mapping = {label: j for j, label in enumerate(classes.tolist())}
        try:
            encoded = np.array([mapping[label] for label in target.tolist()], dtype=int)
        except KeyError as exc:
            raise ValueError('target contains a label missing from classes') from exc
    ids = np.asarray(indices_test)
    if ids.ndim != 1 or (ids.size and not np.issubdtype(ids.dtype, np.integer)):
        raise ValueError('indices_test must contain integer node indices')
    ids = ids.astype(np.intp)
    if np.any((ids < 0) | (ids >= len(target))):
        raise ValueError('indices_test contains invalid indices')
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
        raise ValueError('max_iter must be a nonnegative integer')
    return classes, encoded.copy(), np.unique(ids)


def _counts(B, y, c):
    indicator = sparse.csr_matrix((np.ones(len(y)), (np.arange(len(y)), y)), shape=(len(y), c))
    return (B @ indicator).toarray()


def knn_smoothing(A, target, indices_test, max_iter=1, classes=None,
                  symmetrize=False, return_info=False):
    """Majority of current neighbor labels, with simultaneous updates.

    max_iter=1 is the one-shot baseline. Higher values iterate voting and can
    cycle. An immediate period-two cycle is detected and reported; longer
    cycles remain bounded by max_iter. Ties keep the current class if it is
    among the winners, otherwise choose the first class in classes order.
    Isolated nodes are unchanged. Returns labels or (labels, diagnostics).
    """
    B = contextual_graph(A, symmetrize=symmetrize)
    classes, y, ids = _inputs(B, target, indices_test, classes, max_iter)
    changes = []
    previous = None
    status = 'no_test_nodes' if not ids.size else 'max_iter'
    for _ in range(max_iter if ids.size else 0):
        counts = _counts(B[ids], y, len(classes))
        winners = counts.argmax(axis=1)
        tied = counts[np.arange(len(ids)), y[ids]] == counts.max(axis=1)
        winners[tied] = y[ids[tied]]
        nxt = y.copy()
        nxt[ids] = winners
        changed = int(np.count_nonzero(nxt != y))
        changes.append(changed)
        cycle = previous is not None and np.array_equal(nxt, previous)
        previous, y = y, nxt
        if not changed:
            status = 'fixed_point'
            break
        if cycle:
            status = 'period_two_cycle'
            break
    info = dict(n_iter=len(changes), changes=changes, status=status,
                converged=status in ('fixed_point', 'no_test_nodes'))
    return (classes[y], info) if return_info else classes[y]


def _unary(initial, c, probabilities, unary_costs, eps):
    n = len(initial)
    if probabilities is not None and unary_costs is not None:
        raise ValueError('provide probabilities OR unary_costs, not both')
    if unary_costs is not None:
        D = np.asarray(unary_costs, dtype=float)
        if D.shape != (n, c) or not np.isfinite(D).all():
            raise ValueError('unary_costs must have finite shape (n, n_classes)')
        return D.copy()
    if probabilities is not None:
        P = np.asarray(probabilities, dtype=float)
        if P.shape != (n, c) or not np.isfinite(P).all() or np.any(P < 0):
            raise ValueError('probabilities must have finite nonnegative shape (n, n_classes)')
        sums = P.sum(axis=1, keepdims=True)
        if np.any(sums <= 0) or not np.isfinite(sums).all():
            raise ValueError('each probability row must have a finite positive sum')
        if not np.isfinite(eps) or not 0 < eps < 1:
            raise ValueError('eps must lie in (0,1)')
        P = np.maximum(P / sums, eps)
        P /= P.sum(axis=1, keepdims=True)
        return -np.log(P)
    # Fixed fidelity to the initial labels, not recomputed after updates.
    D = np.ones((n, c))
    D[np.arange(n), initial] = 0.
    return D


def icm(A, target, indices_test, probabilities=None, classes=None, beta=1.,
        unary_weight=1., max_iter=50, order='sequential', random_state=None,
        unary_costs=None, eps=1e-12, return_info=False):
    """ICM for E(y)=unary_weight*sum_{i in test} D_i(y_i)
                         + beta*sum_{unordered edges {i,j}} [y_i != y_j].

    A is always converted to unweighted symmetric union support. Updating only
    outgoing neighbors of a directed graph would not minimize this energy.
    probabilities: shape (n,C), columns aligned EXACTLY with explicit classes
    (e.g. clf.classes_). If classes=None, columns must follow np.unique(target).
    Without probabilities or unary_costs, D_i(c)=[c != initial_label_i].
    unary_weight=0 gives pure Potts ICM; beta=0 gives independent unary decisions.

    Updates are sequential/in-place. Counts for affected neighbors are updated
    incrementally after each accepted move. Only strict local improvements are
    accepted; ties retain the current label. Convergence is a coordinatewise
    local optimum, not a guaranteed global optimum. Random order is reproducible
    with random_state. Energy history contains initial energy and each sweep.
    """
    if not np.isfinite(beta) or beta < 0 or not np.isfinite(unary_weight) or unary_weight < 0:
        raise ValueError('beta and unary_weight must be finite and nonnegative')
    if order not in ('sequential', 'random'):
        raise ValueError("order must be 'sequential' or 'random'")
    B = contextual_graph(A, symmetrize=True)
    classes, y, ids = _inputs(B, target, indices_test, classes, max_iter)
    D = _unary(y, len(classes), probabilities, unary_costs, eps)
    counts = _counts(B, y, len(classes))
    upper = sparse.triu(B, k=1).tocoo()
    def energy():
        return float(unary_weight * D[ids, y[ids]].sum()
                     + beta * np.count_nonzero(y[upper.row] != y[upper.col]))
    energies = [energy()]
    changes = []
    rng = np.random.default_rng(random_state)
    status = 'no_test_nodes' if not ids.size else 'max_iter'
    for _ in range(max_iter if ids.size else 0):
        changed = 0
        visit = ids if order == 'sequential' else rng.permutation(ids)
        for i in visit:
            # degree_i*beta is constant across candidates and can be dropped.
            costs = unary_weight * D[i] - beta * counts[i]
            winner = int(costs.argmin())
            old = y[i]
            if costs[winner] < costs[old]:
                neighbors = B.indices[B.indptr[i]:B.indptr[i+1]]
                counts[neighbors, old] -= 1.
                counts[neighbors, winner] += 1.
                y[i] = winner
                changed += 1
        changes.append(changed)
        energies.append(energy())
        if changed == 0:
            status = 'fixed_point'
            break
    info = dict(n_iter=len(changes), changes=changes, energy=energies, status=status,
                converged=status in ('fixed_point', 'no_test_nodes'),
                beta=float(beta), unary_weight=float(unary_weight),
                unary='custom' if unary_costs is not None else ('probabilities' if probabilities is not None else 'hard_labels'),
                graph='unweighted_symmetric_union', order=order)
    return (classes[y], info) if return_info else classes[y]
