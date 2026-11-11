"""Repeated holdout with inner validation for Fisher-Rao ICM vs ICM and kNN.
Run: python compare_fisher_rao.py --classifiers NB DT MLP RF SVM KNN --seeds 42 43
Classifier hyperparameters are fixed; inner validation selects smoothing parameters.
MLP max_iter=50 is preserved from the supplied script; inspect convergence warnings.
"""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score
from threadpoolctl import threadpool_limits
from contextual_baselines import contextual_graph, knn_smoothing
from potts_optimized import build_KNN_Graph
from fisher_rao_icm import fisher_rao_weights, normalize_mean_degree, weighted_icm_hard

CLASSIFIERS=['NB','DT','MLP','RF','SVM','KNN']

GAMMAS=[0.,.5,1.,2.,4.,8.]
KNN_ITERATIONS=[0,1,3,10,30]
METHODS=['pointwise','kNN_1','kNN_val','ICM_fixed','FR_fixed','ICM_val','FR_val']


def classifier(name,seed):
    if name == 'NB':
        return GaussianNB()
    elif name == 'DT':
        return DecisionTreeClassifier(random_state=seed)
    elif name == 'MLP':
        return MLPClassifier(random_state=seed)
    elif name == 'RF':
        return RandomForestClassifier(max_depth=5, random_state=seed)
    elif name == 'SVM':
        return SVC(gamma='auto')
    elif name == 'KNN':
        return KNeighborsClassifier()
    raise ValueError(f'Unknown classifier: {name!r}. Choose from {CLASSIFIERS}')

def graph_for(X,train,k):
    Z=StandardScaler().fit(X[train]).transform(X)
    A=contextual_graph(build_KNN_Graph(Z,min(k,len(X)-1)),True)
    return Z,A


def initial_predictions(Z,y,train,test,name,seed):
    clf=classifier(name,seed).fit(Z[train],y[train])
    initial=np.empty_like(y)
    initial[train]=y[train]
    initial[test]=clf.predict(Z[test])
    return initial


def select(A,initial,val,yval,eta,tau):
    """Only validation labels enter selection; no external test data enter."""
    W0=normalize_mean_degree(A)
    WF=fisher_rao_weights(A,initial,eta=eta,tau=tau)
    records={}
    for name,W in [('ICM',W0),('FR',WF)]:
        scores=[]
        for gamma in GAMMAS:
            out=weighted_icm_hard(W,initial,val,gamma=gamma)
            scores.append(float(balanced_accuracy_score(yval,out[val])))
        # First/smallest parameter breaks exact score ties, no test tie-break.
        records[name]=dict(selected=GAMMAS[int(np.argmax(scores))],grid=GAMMAS,scores=scores)
    scores=[]
    for it in KNN_ITERATIONS:
        out=knn_smoothing(A,initial,val,max_iter=it)
        scores.append(float(balanced_accuracy_score(yval,out[val])))
    records['kNN']=dict(selected=KNN_ITERATIONS[int(np.argmax(scores))],grid=KNN_ITERATIONS,scores=scores)
    return records


def run(names,seeds,k=15,eta=.5,tau=1.,classifiers=None):
    classifiers=list(dict.fromkeys(CLASSIFIERS if classifiers is None else classifiers))
    if not classifiers:
        raise ValueError('Select at least one classifier')
    for name in classifiers:
        classifier(name, 0)  # Validate before running any experiments.
    rows=[]; selections=[]
    for name in names:
        data=getattr(datasets,'load_'+name)(); X,y=data.data,data.target
        for seed in seeds:
            train,test=train_test_split(np.arange(len(y)),train_size=.5,stratify=y,random_state=seed)
            # Inner graph excludes ALL external test samples, including features.
            inner_train,val=train_test_split(np.arange(len(train)),train_size=.7,
                                             stratify=y[train],random_state=seed+10000)
            Zi,Ai=graph_for(X[train],inner_train,k)
            Z,A=graph_for(X,train,k)
            W0=normalize_mean_degree(A)
            for clf in classifiers:
                inner_initial=initial_predictions(Zi,y[train],inner_train,val,clf,seed)
                start=time.perf_counter()
                chosen=select(Ai,inner_initial,val,y[train][val],eta,tau)
                selection_seconds=time.perf_counter()-start
                initial=initial_predictions(Z,y,train,test,clf,seed)
                start=time.perf_counter()
                WF,weight_info=fisher_rao_weights(A,initial,eta=eta,tau=tau,return_info=True)
                weight_seconds=time.perf_counter()-start
                selections.append(dict(dataset=name,classifier=clf,seed=seed,
                    train_indices=train.tolist(),test_indices=test.tolist(),
                    inner_train_indices_within_train=inner_train.tolist(),
                    validation_indices_within_train=val.tolist(),
                    choices=chosen,selection_seconds=selection_seconds,weights=weight_info))
                for method in METHODS:
                    start=time.perf_counter()
                    if method=='pointwise':
                        out=initial.copy(); info={}; parameter=None
                    elif method.startswith('kNN'):
                        parameter=1 if method=='kNN_1' else chosen['kNN']['selected']
                        out,info=knn_smoothing(A,initial,test,max_iter=parameter,return_info=True)
                    else:
                        family='FR' if method.startswith('FR') else 'ICM'
                        parameter=2. if method.endswith('fixed') else chosen[family]['selected']
                        out,info=weighted_icm_hard(WF if family=='FR' else W0,initial,test,
                                                  gamma=parameter,return_info=True)
                        assert np.all(np.diff(info['energy'])<=1e-8),info
                    elapsed=time.perf_counter()-start
                    np.testing.assert_array_equal(out[train],y[train])
                    before=initial[test]==y[test]; after=out[test]==y[test]
                    rows.append(dict(dataset=name,classifier=clf,seed=seed,method=method,
                        parameter=parameter,balanced_accuracy=float(balanced_accuracy_score(y[test],out[test])),
                        per_class_recall=recall_score(y[test],out[test],labels=np.unique(y),average=None).tolist(),
                        corrected=int(np.sum(~before & after)),spoiled=int(np.sum(before & ~after)),
                        changed=int(np.sum(out[test]!=initial[test])),n_test=len(test),
                        smoothing_seconds=elapsed+(weight_seconds if method.startswith('FR') else 0.),
                        diagnostics=info))
            print(f'{name}: seed {seed} complete',flush=True)
    summary=[]; paired=[]
    for name in names:
        for clf in classifiers:
            for method in METHODS:
                group=[r for r in rows if (r['dataset'],r['classifier'],r['method'])==(name,clf,method)]
                values=np.array([r['balanced_accuracy'] for r in group])
                summary.append(dict(dataset=name,classifier=clf,method=method,
                    mean=float(values.mean()),std=float(values.std(ddof=1)) if len(values)>1 else 0.,
                    corrected_mean=float(np.mean([r['corrected'] for r in group])),
                    spoiled_mean=float(np.mean([r['spoiled'] for r in group])),
                    seconds_mean=float(np.mean([r['smoothing_seconds'] for r in group]))))
            for baseline in ['pointwise','kNN_1','kNN_val','ICM_val']:
                differences=[]
                for seed in seeds:
                    def score(method):
                        return next(r['balanced_accuracy'] for r in rows if
                            (r['dataset'],r['classifier'],r['seed'],r['method'])==(name,clf,seed,method))
                    differences.append(score('FR_val')-score(baseline))
                d=np.array(differences)
                paired.append(dict(dataset=name,classifier=clf,baseline=baseline,
                    mean_difference_pp=float(100*d.mean()),wins=int(np.sum(d>1e-12)),
                    ties=int(np.sum(np.abs(d)<=1e-12)),losses=int(np.sum(d< -1e-12))))
    return dict(environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                 sklearn=sklearn.__version__,threads=1),
        protocol=dict(seeds=seeds,classifiers=classifiers,
            classifier_parameters={name: classifier(name,0).get_params() for name in classifiers},
            classifier_seed_policy="random_state=split seed for DT, MLP and RF",outer_train_fraction=.5,inner_train_fraction=.7,k=k,eta=eta,tau=tau,
            gamma_grid=GAMMAS,knn_iterations_grid=KNN_ITERATIONS,fixed_gamma=2.,fidelity=1.,
            normalization='global mean weighted degree = 1, for both ICM variants',
            graph='shared symmetric union kNN support',max_icm_iter=50,
            note='Scaler/model fit on training only at each level; weights use initial predictions, never true held-out labels. Inner graph excludes external test features. Settings fixed before outer results. Repeated splits overlap; descriptive statistics only.'),
        summary=summary,paired=paired,selections=selections,rows=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',nargs='+',type=int,default=list(range(42,52)))
    parser.add_argument('--datasets',nargs='+',choices=['iris','wine','breast_cancer','digits'],
                        default=['iris','wine','breast_cancer','digits'])
    parser.add_argument('--classifiers',nargs='+',choices=CLASSIFIERS,default=CLASSIFIERS,
                        help='Pointwise classifiers to compare (default: all six). KNN is distinct from kNN label smoothing.')
    parser.add_argument('--k',type=int,default=15)
    parser.add_argument('--eta',type=float,default=.5)
    parser.add_argument('--tau',type=float,default=1.)
    parser.add_argument('--output',default='fisher_rao_results.json')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        result=run(args.datasets,args.seeds,args.k,args.eta,args.tau,classifiers=args.classifiers)
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(f'Saved {args.output}: {len(result["rows"])} evaluations')
