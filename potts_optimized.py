"""Fisher/Potts GSA: sparse, vectorized implementation.
Original method: Alexandre L. M. Levada. Optimization: algebraic moments.
Run: python third_set_of_experiments_optimized.py --dataset digits
"""
from dataclasses import dataclass
import argparse
import warnings
import numpy as np
from scipy import sparse, optimize
from scipy.special import softmax
from sklearn.neighbors import kneighbors_graph


def build_KNN_Graph(dados, k):
    """Directed CSR distance graph; exclude zero distances as in original A.nonzero().

    No geodesic distances are computed. Duplicate points at distance zero do not
    contribute, deliberately preserving the uploaded implementation's semantics.
    """
    graph = kneighbors_graph(dados, n_neighbors=k, mode='distance').tocsr()
    graph.eliminate_zeros()
    return graph


def _adjacency(A):
    graph = sparse.csr_matrix(A, dtype=float, copy=True)
    if graph.shape[0] != graph.shape[1]:
        raise ValueError('A must be square')
    graph.sum_duplicates()
    graph.eliminate_zeros()
    graph.data[:] = 1.0  # Original code uses presence, not distance weights.
    return graph


def _labels(target, n):
    y = np.asarray(target)
    if y.ndim != 1 or len(y) != n or n == 0:
        raise ValueError('target must be a nonempty vector with len(target)==len(A)')
    classes, encoded = np.unique(y, return_inverse=True)
    return classes, encoded


def _counts(graph, encoded, c):
    onehot = np.zeros((len(encoded), c), dtype=float)
    onehot[np.arange(len(encoded)), encoded] = 1.0
    return np.asarray(graph @ onehot)


def neighborhood_counts(A, target):
    """Return U[i,c], ordered classes, and encoded labels."""
    graph = _adjacency(A)
    classes, encoded = _labels(target, graph.shape[0])
    return _counts(graph, encoded, len(classes)), classes, encoded


def _moments(U, beta):
    beta = float(beta)
    if not np.isfinite(beta):
        raise ValueError('beta must be finite')
    # Center before multiplying: stable for large positive/negative beta.
    center = U.max(axis=1, keepdims=True) if beta >= 0 else U.min(axis=1, keepdims=True)
    with np.errstate(over='ignore'):
        p = softmax(beta * (U - center), axis=1)
    mu = np.sum(p * U, axis=1)
    variance = np.sum(p * (U - mu[:, None])**2, axis=1)
    return mu, variance


def FisherInformation(A, beta, target):
    """Explicit target replaces original global variable; returns PHI, PSI."""
    U, _, encoded = neighborhood_counts(A, target)
    mu, variance = _moments(U, beta)
    phi = (U[np.arange(len(encoded)), encoded] - mu)**2
    return phi, variance


def free_energy(A, target):
    U, _, encoded = neighborhood_counts(A, target)
    return float(U[np.arange(len(encoded)), encoded].sum())


def pseudo_likelihood(beta, A, target):
    """Score (derivative of log pseudolikelihood), not the likelihood itself."""
    U, _, encoded = neighborhood_counts(A, target)
    mu, _ = _moments(U, beta)
    return float(U[np.arange(len(encoded)), encoded].sum() - mu.sum())


@dataclass(frozen=True)
class BetaEstimate:
    root: float
    converged: bool
    status: str
    score: float


def estimate_beta(A, target, bounds=(-5.0, 5.0)):
    """Cache histograms once. Solve monotone score by Brent in supplied bounds.

    Boundary results are explicitly reported as constrained optima (not roots).
    A flat likelihood returns zero when bounds contain zero.
    """
    U, _, encoded = neighborhood_counts(A, target)
    observed = U[np.arange(len(encoded)), encoded].sum()
    lo, hi = map(float, bounds)
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError('bounds must be finite and increasing')
    def score(beta):
        return float(observed - _moments(U, beta)[0].sum())
    if np.all(np.ptp(U, axis=1) == 0):
        root = float(np.clip(0., lo, hi))
        return BetaEstimate(root, False, 'unidentifiable_flat', score(root))
    flo, fhi = score(lo), score(hi)
    if flo < 0 or fhi > 0:
        root = lo if flo < 0 else hi
        return BetaEstimate(root, False, 'constrained_boundary', score(root))
    result = optimize.root_scalar(score, bracket=(lo, hi), method='brentq', xtol=1e-12)
    return BetaEstimate(result.root, result.converged, 'root', score(result.root))


def gsac(A, dados, target, indices_test, beta, alpha=1.0, MAX_IT=1,
         random_state=None):
    """Synchronous GSA, training nodes fixed, with original positional signature.

    For positive PSI, maximizing PSI/PHI is exactly minimizing (U_c-mu)^2.
    Equal minima choose the first class (np.unique ordering). If PSI is zero,
    keep the current label: the original ratio is undefined or uninformative.
    random_state=None uses np.random, preserving original seeding and draw order.
    Draws are still consumed for every test node, even when alpha is 0 or 1.
    dados is used only for validation: unused Gaussian fits have been removed.
    """
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must lie in [0,1]')
    if not isinstance(MAX_IT, (int, np.integer)) or MAX_IT < 0:
        raise ValueError('MAX_IT must be a nonnegative integer')
    graph = _adjacency(A)
    classes, labels = _labels(target, graph.shape[0])
    if dados is not None and np.shape(dados)[0] != graph.shape[0]:
        raise ValueError('dados and A have different numbers of samples')
    ids = np.asarray(indices_test)
    if ids.ndim != 1 or (ids.size and not np.issubdtype(ids.dtype, np.integer)):
        raise ValueError('indices_test must be a vector of integer indices')
    ids = ids.astype(np.intp)
    if np.any((ids < 0) | (ids >= len(labels))):
        raise ValueError('indices_test out of bounds')
    ids = np.unique(ids)  # Original loop visits nodes in ascending order once.
    subgraph = graph[ids]
    rng = np.random if random_state is None else np.random.default_rng(random_state)
    for _ in range(MAX_IT):
        U = _counts(subgraph, labels, len(classes))
        mu, variance = _moments(U, beta)
        winners = np.argmin((U - mu[:, None])**2, axis=1)
        change = (rng.random(len(ids)) < alpha) & (variance > 0)
        labels[ids[change]] = winners[change]
    return classes[labels]


def run_experiment(dataset='digits', k=15, seed=42, iterations=1, alpha=1.0):
    from sklearn import datasets, metrics
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    data = getattr(datasets, 'load_' + dataset)()
    X, y = data.data, data.target
    train, test = train_test_split(np.arange(len(y)), train_size=.5,
                                   random_state=seed, stratify=y)
    X = StandardScaler().fit(X[train]).transform(X)
    A = build_KNN_Graph(X, k)
    rows = []
    for name, clf in [('NB', GaussianNB()), ('DT', DecisionTreeClassifier(random_state=seed))]:
        pred = clf.fit(X[train], y[train]).predict(X[test])
        labels = y.copy()
        labels[test] = pred
        result = estimate_beta(A, labels)
        if not result.converged:
            warnings.warn(f'beta: {result.status}; beta={result.root}')
        smoothed = gsac(A, X, labels, test, result.root, alpha, iterations, seed)
        row = dict(dataset=dataset, classifier=name, beta=result.root, beta_status=result.status,
                   balanced_accuracy=metrics.balanced_accuracy_score(y[test], pred),
                   smoothed_balanced_accuracy=metrics.balanced_accuracy_score(y[test], smoothed[test]))
        rows.append(row)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['iris','wine','breast_cancer','digits'], default='digits')
    parser.add_argument('--k', type=int, default=15)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    for row in run_experiment(args.dataset, args.k, args.seed, args.iterations, args.alpha):
        print(row)
