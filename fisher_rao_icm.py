"""Hard-fidelity Potts ICM with fixed Fisher-Rao neighborhood affinities.

Use true labels ONLY at training nodes; all test entries in initial_labels must
be predictions. Fisher weights are computed once and never changed by ICM.
"""
import numpy as np
from scipy import sparse
from contextual_baselines import contextual_graph, _inputs, _counts


def normalize_mean_degree(W):
    """Global scaling, preserving symmetry. Mean weighted degree becomes one.

    Unlike row normalization, this retains an undirected pairwise energy.
    Both weighted and unweighted comparison graphs use the same normalization.
    """
    W = sparse.csr_matrix(W, dtype=float, copy=True)
    mean_degree = float(W.sum()) / W.shape[0] if W.shape[0] else 0.
    if mean_degree > 0:
        W.data /= mean_degree
    return W


def fisher_rao_weights(A, initial_labels, eta=.5, tau=1., classes=None,
                       normalize=True, return_info=False):
    """Construct W_ij=B_ij exp(-d_FR(q_i,q_j)^2/(2*tau^2)).

    B is the unweighted symmetric union support of A, excluding self-loops.
    q_ic=(initial neighbor count_ic + eta)/(degree_i + C*eta).
    d_FR=2 arccos(sum_c sqrt(q_ic*q_jc)). tau is in radians; +inf recovers B.
    Uses only edges: no dense n-by-n distance matrix. O(nC + |E|C) work/storage
    for the vectorized edge computation. normalize=True scales mean degree to 1.
    Returned diagnostics do not expose any ground truth beyond supplied labels.
    """
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError('eta must be finite and positive')
    if np.isnan(tau) or tau <= 0:
        raise ValueError('tau must be positive (infinity is allowed)')
    B = contextual_graph(A, symmetrize=True)
    classes, y, _ = _inputs(B, initial_labels, [], classes, 0)
    U = _counts(B, y, len(classes))
    degree = np.asarray(B.sum(axis=1)).ravel()
    q = (U + eta) / (degree[:, None] + len(classes)*eta)
    edges = sparse.triu(B, k=1).tocoo()
    roots = np.sqrt(q)
    # Limit edge-by-class temporary allocation for large graphs.
    distances = np.empty(len(edges.data))
    for start in range(0, len(distances), 65536):
        stop = min(start + 65536, len(distances))
        r,c = edges.row[start:stop], edges.col[start:stop]
        coefficient = np.einsum('ij,ij->i', roots[r], roots[c])
        distances[start:stop] = 2*np.arccos(np.clip(coefficient, 0., 1.))
    with np.errstate(over='ignore', under='ignore'):
        weights = np.exp(-.5*(distances/tau)**2)
    W = sparse.csr_matrix((weights,(edges.row,edges.col)),shape=B.shape)
    W = (W+W.T).tocsr()
    raw_mean = float(W.sum())/len(y)
    if normalize:
        W = normalize_mean_degree(W)
    W.eliminate_zeros()
    info = dict(eta=float(eta), tau=float(tau), n_classes=len(classes),
                n_edges=len(weights), raw_mean_degree=raw_mean,
                normalized_mean_degree=float(W.sum())/len(y),
                mean_raw_weight=float(weights.mean()) if len(weights) else 0.,
                mean_fisher_distance=float(distances.mean()) if len(distances) else 0.,
                normalization='global_mean_degree' if normalize else 'none')
    return (W, info) if return_info else W


def weighted_icm_hard(W, initial_labels, indices_test, gamma=2., fidelity=1.,
                      max_iter=50, classes=None, order='sequential',
                      random_state=None, tol=1e-12, return_info=False):
    """Minimize fidelity*sum_test [y_i != initial_i]
                   + gamma*sum_{i<j} W_ij [y_i != y_j].

    W MUST already contain affinities, not Euclidean distances. Symmetric,
    nonnegative, finite weights required. No binarization or renormalization.
    Sequential best responses; ties/changes below tol keep the current label.
    Counts update incrementally. Converged means a tol-coordinatewise minimum,
    not a global minimum. Diagnostics include energy after every full sweep.
    """
    for name,v in [('gamma',gamma),('fidelity',fidelity),('tol',tol)]:
        if not np.isfinite(v) or v < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if order not in ('sequential','random'):
        raise ValueError('order must be sequential or random')
    W = sparse.csr_matrix(W,dtype=float,copy=True)
    if W.shape[0] != W.shape[1] or not np.isfinite(W.data).all() or np.any(W.data<0):
        raise ValueError('W must be square with finite nonnegative weights')
    W.sum_duplicates(); W.setdiag(0); W.eliminate_zeros()
    difference = W-W.T
    if difference.nnz and np.max(np.abs(difference.data)) > 1e-12:
        raise ValueError('W must be symmetric')
    W = ((W+W.T)*.5).tocsr()
    classes,y,ids = _inputs(W,initial_labels,indices_test,classes,max_iter)
    initial = y.copy()
    support = _counts(W,y,len(classes))
    upper = sparse.triu(W,k=1).tocoo()
    def energy():
        return float(fidelity*np.count_nonzero(y[ids]!=initial[ids]) +
            gamma*np.sum(upper.data[y[upper.row]!=y[upper.col]]))
    energies=[energy()]; changes=[]
    status='no_test_nodes' if not len(ids) else 'max_iter'
    rng=np.random.default_rng(random_state)
    for _ in range(max_iter if len(ids) else 0):
        n_changed=0
        for i in (ids if order=='sequential' else rng.permutation(ids)):
            costs = -gamma*support[i].copy()
            costs[initial[i]] -= fidelity  # constant fidelity omitted
            winner=int(costs.argmin()); old=y[i]
            if costs[old]-costs[winner] > tol:
                start,stop=W.indptr[i:i+2]
                neighbors=W.indices[start:stop]; weights=W.data[start:stop]
                support[neighbors,old] -= weights
                support[neighbors,winner] += weights
                y[i]=winner; n_changed+=1
        changes.append(n_changed); energies.append(energy())
        if not n_changed:
            status='fixed_point'; break
    info=dict(energy=energies,changes=changes,n_iter=len(changes),status=status,
              converged=status in ('fixed_point','no_test_nodes'),gamma=float(gamma),
              fidelity=float(fidelity),tol=float(tol))
    return (classes[y],info) if return_info else classes[y]


def fisher_rao_icm(A, initial_labels, indices_test, gamma=2., fidelity=1.,
                   eta=.5, tau=1., normalize=True, max_iter=50, classes=None,
                   order='sequential', random_state=None, return_info=False):
    """Convenience wrapper: build frozen Fisher weights and run hard ICM."""
    W,weights_info=fisher_rao_weights(A,initial_labels,eta,tau,classes,normalize,True)
    labels,info=weighted_icm_hard(W,initial_labels,indices_test,gamma,fidelity,
                                max_iter,classes,order,random_state,return_info=True)
    info['weights']=weights_info
    return (labels,info) if return_info else labels
