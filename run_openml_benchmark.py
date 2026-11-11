"""OpenML benchmark with joint gamma/tau selection for Fisher-Rao ICM.

python run_openml_benchmark.py --datasets iris digits --seeds 42
python run_openml_benchmark.py --max-samples 10000 --pca 50 --output results_openml
Resume by repeating the same command. Only successful dataset/seed/model units skip.

Default joint search: gamma in {0,.5,1,2,4,8}, tau in {.25,.5,1,2,4}.
Use --tau 1 for fixed-tau mode; otherwise --tau-grid controls the search.
Outer split: 50% training pool / 50% test, stratified, seed=s.
Inner split: 70%/30% of training pool (35%/15% of total), seed=s+10000.
After selecting parameters, preprocessing and classifier refit on the full
50% training pool. Final graph includes test features, never true test labels.
Use a NEW output directory for joint-search experiments (default results_openml_tau).
Each checkpoint records all FR candidates, scores, selected gamma and tau.
For gamma=0, tau is not identifiable and its reported value is a tie-break only.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import socket
import time
import traceback
import warnings

from pathlib import Path
import numpy as np
import pandas as pd
import scipy as sp
from scipy import sparse
import sklearn
from sklearn import datasets as skd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.multiclass import type_of_target
from threadpoolctl import threadpool_limits

from compare_fisher_rao import classifier, GAMMAS, KNN_ITERATIONS
from contextual_baselines import contextual_graph, knn_smoothing
from fisher_rao_icm import fisher_rao_weights, normalize_mean_degree, weighted_icm_hard
from potts_optimized import build_KNN_Graph

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

# DATASETS = [
#     'breast-w', 'digits', 'mfeat-zernike', 'mfeat-karhunen', 'mfeat-fourier',
#     'mfeat-factors', 'mfeat-pixel', 'semeion', 'satimage', 'USPS',
#     'UMIST_Faces_Cropped', 'Olivetti_Faces', 'pendigits', 'texture',
#     'coil-20', 'one-hundred-plants-margin', 'Smartphone-Based_Recognition_of_Human_Activities',
#     'AP_Ovary_Kidney', 'AP_Lung_Kidney', 'AP_Breast_Colon', 'OVA_Breast',
#     'Indian_pines', 'hayes-roth', 'seeds', 'har', 'isolet', 'led7', 'optdigits',
#     'LED-display-domain-7digit', 'JapaneseVowels', 'Flare', 'Satellite', 'mushroom', 'Lung',
#     'car-evaluation', 'leukemia', 'MNIST_784', 'Fashion-MNIST', 'Kuzushiji-MNIST'
# ]

DATASETS = [
    'digits', 'mfeat-karhunen', 'mfeat-fourier', 'mfeat-pixel', 'semeion', 'satimage', 'USPS', 'letter',
    'UMIST_Faces_Cropped', 'Olivetti_Faces', 'texture', 'coil-20', 'har', 'isolet', 'optdigits',
    'LED-display-domain-7digit', 'JapaneseVowels', 'Flare', 'Lung', 'MNIST_784', 'Fashion-MNIST', 'Kuzushiji-MNIST'
]

METHODS=['pointwise','kNN_1','kNN_val','ICM_val','FR_val']



def take(X, ids):
    return X.iloc[ids].copy() if isinstance(X,pd.DataFrame) else X[ids].copy()


def load_dataset(name, version=1, data_home=None, data_id=None,
                 target_column='default-target', return_metadata=False, download_timeout=60.):
    """Based on the supplied loader: local datasets first; otherwise OpenML v1.

    Deliberate changes: retain categorical columns and missing features until
    training-only preprocessing; retain sparse storage until after sampling.
    Labels are integer encoded; missing target rows are dropped and recorded.
    Multioutput/regression targets are rejected, not silently flattened/binned.
    A data_id explicitly overrides the local shortcut and name/version lookup.
    """
    local={'wine':skd.load_wine,'iris':skd.load_iris,'digits':skd.load_digits,
           'breast_cancer':skd.load_breast_cancer}
    if name in local and data_id is None:
        bundle=local[name]()
        meta=dict(source='sklearn',requested_name=name)
    else:
        request=dict(data_home=data_home,as_frame='auto',cache=True,parser='auto',
                     target_column=target_column,n_retries=2,delay=1.)
        request.update(data_id=int(data_id)) if data_id is not None else request.update(name=name,version=version)
        previous_timeout=socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(download_timeout)
            bundle=skd.fetch_openml(**request)
        finally:
            socket.setdefaulttimeout(previous_timeout)
        details=getattr(bundle,'details',{})
        meta=dict(source='OpenML',requested_name=name,requested_version=version,
                  data_id=details.get('id',data_id),resolved_name=details.get('name',name),
                  resolved_version=details.get('version'),target_column=target_column,
                  url=getattr(bundle,'url',None))
    X=bundle['data']; raw=np.asarray(bundle['target'])
    if raw.ndim==2 and raw.shape[1]==1: raw=raw[:,0]
    if raw.ndim!=1 or len(raw)!=X.shape[0]:
        raise ValueError('Target absent or multioutput. Set target_column in --dataset-map.')
    valid=~pd.isna(raw)
    if np.issubdtype(raw.dtype,np.number): valid &= np.isfinite(raw.astype(float))
    original_rows=np.flatnonzero(valid)
    X=take(X,original_rows); raw=raw[valid]
    if not len(raw): raise ValueError('No nonmissing target values')
    # Conversion avoids object arrays of Python numbers being classified unknown.
    if raw.dtype==object:
        raw=pd.Series(raw).infer_objects().to_numpy()
    kind=type_of_target(raw)
    if kind not in ('binary','multiclass'):
        raise ValueError(f'Expected a classification target, got {kind}')
    classes,y=np.unique(raw,return_inverse=True)
    if len(classes)<2: raise ValueError('At least two classes are required')
    if sparse.issparse(X):
        X=X.tocsr().astype(float)
        X.data[~np.isfinite(X.data)]=np.nan
    else:
        X=X.copy() if isinstance(X,pd.DataFrame) else pd.DataFrame(X)
        X.columns=[f'x{i}' for i in range(X.shape[1])]
        for col in X:
            if pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col]):
                X[col]=pd.to_numeric(X[col],errors='coerce').astype(float).replace([np.inf,-np.inf],np.nan)
            else:
                # Objects/strings/categories/bools; avoid mixed scalar types in encoders.
                X[col]=X[col].astype(object).map(lambda v: np.nan if pd.isna(v) else str(v))
    meta.update(n_original=len(valid),n_rows=len(y),n_features=X.shape[1],
                dropped_missing_target=int(np.sum(~valid)),class_labels=[str(c) for c in classes],
                original_row_indices=original_rows.tolist(),sparse_input=sparse.issparse(X))
    return (X,y.astype('int64'),meta) if return_metadata else (X,y.astype('int64'))


def select_subset(X,y,max_samples,seed):
    if max_samples and len(y)>max_samples:
        ids,_=train_test_split(np.arange(len(y)),train_size=max_samples,stratify=y,random_state=seed)
        if len(np.unique(y[ids]))!=len(np.unique(y)):
            raise ValueError('Sample cap removed a rare class; increase --max-samples')
    else: ids=np.arange(len(y))
    return take(X,ids),y[ids],ids


def dense_checked(X,max_mb):
    estimate=X.shape[0]*X.shape[1]*8
    if estimate>max_mb*1024**2:
        raise MemoryError(f'Dense feature matrix needs {estimate/1024**2:.1f} MiB > --max-dense-mb={max_mb}. Reduce --max-samples or change encoding.')
    return np.asarray(X.toarray() if sparse.issparse(X) else X,dtype=float)


def prepare_features(X,train,args):
    """Fit imputation, encoding, scaling and optional PCA ONLY on train."""
    if sparse.issparse(X):
        imputer=SimpleImputer(strategy='median',keep_empty_features=True)
        imputer.fit(X[train])
        Z=dense_checked(imputer.transform(X),args.max_dense_mb)
        scaler=StandardScaler().fit(Z[train]); Z=scaler.transform(Z)
    else:
        numeric=[c for c in X if pd.api.types.is_numeric_dtype(X[c])]
        categorical=[c for c in X if c not in numeric]
        parts=[]
        if numeric:
            parts.append(('numeric',make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()),numeric))
        if categorical:
            encoder=(OneHotEncoder(handle_unknown='ignore',sparse_output=True) if args.categorical=='onehot'
                     else OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))
            parts.append(('categorical',make_pipeline(SimpleImputer(strategy='constant',fill_value='__missing__',keep_empty_features=True),encoder),categorical))
        prep=ColumnTransformer(parts,sparse_threshold=1.)
        prep.fit(X.iloc[train])
        n_encoded=len(prep.get_feature_names_out())
        if len(X)*n_encoded*8 > args.max_dense_mb*1024**2:
            raise MemoryError("Encoded dense features exceed --max-dense-mb; reduce --max-samples or change encoding")
        Z=dense_checked(prep.transform(X),args.max_dense_mb)
    before=Z.shape[1]; used=0
    if args.pca and before>args.pca:
        used=min(args.pca,before,len(train)-1)
        if used<1: raise ValueError('Not enough training rows for PCA')
        model=PCA(n_components=used,svd_solver='randomized',random_state=args.preprocess_seed)
        model.fit(Z[train]); Z=model.transform(Z)
    if not np.isfinite(Z).all(): raise ValueError('Nonfinite transformed features')
    return Z,dict(encoded_features=before,final_features=Z.shape[1],pca_components=used)


def initial_predictions(Z,y,train,test,model,seed):
    clf=classifier(model,seed).fit(Z[train],y[train])
    initial=np.empty_like(y); initial[train]=y[train]; initial[test]=clf.predict(Z[test])
    return initial


def tau_candidates(args):
    """Explicit --tau requests fixed-tau mode; otherwise search --tau-grid."""
    return [args.tau] if args.tau is not None else sorted(set(args.tau_grid),reverse=True)


def choose_parameters(A,initial,val,yval,args):
    """Joint FR search using ONLY inner validation labels.

    Tie rule: smallest gamma, then largest tau (weaker edge modulation).
    Weights are built once per tau, not once per gamma. Gamma=0 predictions
    do not depend on weights and are scored once. All candidates are recorded.
    """
    choices={}
    gammas=sorted(set(args.gammas))
    W0=normalize_mean_degree(A)
    scores=[]
    for gamma in gammas:
        z=weighted_icm_hard(W0,initial,val,gamma=gamma,max_iter=args.max_iter)
        scores.append(float(balanced_accuracy_score(yval,z[val])))
    choices['ICM']=dict(selected=gammas[int(np.argmax(scores))],grid=gammas,scores=scores)

    taus=sorted(tau_candidates(args),reverse=True)
    candidates=[]
    zero_score=float(balanced_accuracy_score(yval,initial[val]))
    for tau in taus:
        W=fisher_rao_weights(A,initial,eta=args.eta,tau=tau)
        for gamma in gammas:
            if gamma==0:
                score=zero_score
            else:
                z=weighted_icm_hard(W,initial,val,gamma=gamma,max_iter=args.max_iter)
                score=float(balanced_accuracy_score(yval,z[val]))
            candidates.append(dict(gamma=float(gamma),tau=float(tau),score=score))
    best=min(candidates,key=lambda row:(-row['score'],row['gamma'],-row['tau']))
    choices['FR']=dict(selected=best['gamma'],selected_tau=best['tau'],
        selected_score=best['score'],grid=gammas,tau_grid=taus,
        scores=[max(row['score'] for row in candidates if row['gamma']==g) for g in gammas],
        candidates=candidates,selection='joint_gamma_tau' if len(taus)>1 else 'fixed_tau',
        tie_break='highest validation score; smallest gamma; largest tau',
        tau_identifiable=best['gamma']>0)
    scores=[]
    for it in args.knn_iterations:
        z=knn_smoothing(A,initial,val,max_iter=it)
        scores.append(float(balanced_accuracy_score(yval,z[val])))
    choices['kNN']=dict(selected=args.knn_iterations[int(np.argmax(scores))],grid=args.knn_iterations,scores=scores)
    return choices


def evaluate(A,Z,y,train,test,model,seed,chosen,args):
    initial=initial_predictions(Z,y,train,test,model,seed)
    W0=normalize_mean_degree(A)
    start=time.perf_counter()
    WF,wi=fisher_rao_weights(A,initial,eta=args.eta,tau=chosen['FR']['selected_tau'],return_info=True)
    weight_seconds=time.perf_counter()-start
    records=[]
    for method in METHODS:
        start=time.perf_counter(); param=None
        if method=='pointwise': out=initial.copy(); info={}
        elif method.startswith('kNN'):
            param=1 if method=='kNN_1' else chosen['kNN']['selected']
            out,info=knn_smoothing(A,initial,test,max_iter=param,return_info=True)
        else:
            family='FR' if method=='FR_val' else 'ICM'; param=chosen[family]['selected']
            out,info=weighted_icm_hard(WF if family=='FR' else W0,initial,test,gamma=param,
                                     max_iter=args.max_iter,return_info=True)
            if not np.all(np.diff(info['energy'])<=1e-7): raise RuntimeError('ICM energy increased')
        np.testing.assert_array_equal(out[train],y[train])
        before=initial[test]==y[test]; after=out[test]==y[test]
        records.append(dict(method=method,parameter=param,
            tau=chosen['FR']['selected_tau'] if method=='FR_val' else None,
            balanced_accuracy=float(balanced_accuracy_score(y[test],out[test])),
            f1_macro=float(f1_score(y[test],out[test],average='macro',zero_division=0)),
            recall_per_class=recall_score(y[test],out[test],labels=np.unique(y),average=None,zero_division=0).tolist(),
            corrected=int(np.sum(~before & after)),spoiled=int(np.sum(before & ~after)),
            changed=int(np.sum(initial[test]!=out[test])),diagnostics=info,
            smoothing_seconds=time.perf_counter()-start+(weight_seconds if method=='FR_val' else 0.)))
    return records,wi


def atomic_json(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)


def key(name):
    return re.sub(r'[^A-Za-z0-9_-]+','_',name)[:80]+'_'+hashlib.sha256(name.encode()).hexdigest()[:8]


def configuration(args,mapping):
    omit={'datasets','seeds','classifiers','output','data_home','list_datasets','summarize_only'}
    config={k:v for k,v in vars(args).items() if k not in omit and k!='dataset_map'}
    config['dataset_mapping']=mapping
    config['versions']=dict(numpy=np.__version__,scipy=sp.__version__,pandas=pd.__version__,sklearn=sklearn.__version__)
    config['source_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ['run_openml_benchmark.py','compare_fisher_rao.py','fisher_rao_icm.py','contextual_baselines.py','potts_optimized.py']}
    return config


def initialize_output(root,config):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    path=root/'configuration.json'
    if path.exists():
        if json.loads(path.read_text())!=config:
            raise ValueError('Output directory has a different configuration/code/version. Use a NEW --output directory, or --summarize-only to report existing results.')
    else: atomic_json(path,config)
    (root/'runs').mkdir(exist_ok=True)


def summarize(root):
    """Aggregate ALL completed checkpoints, including previously saved seeds.

    The long-format summary remains compatible with the previous script.
    Wide tables are generated separately for each classifier and metric.
    """
    root=Path(root)
    groups={}
    seen=set()
    for path in sorted((root/'runs').glob('*.json')):
        run=json.loads(path.read_text())
        identity=(run['dataset'],run['classifier'],run['seed'])
        if identity in seen:
            raise ValueError(f'Duplicate dataset/classifier/seed checkpoint: {identity}')
        seen.add(identity)
        for row in run['results']:
            group=(run['dataset'],run['classifier'],row['method'])
            groups.setdefault(group,[]).append((run['seed'],row))
    result=[]
    for (dataset,model,method),items in sorted(groups.items()):
        seeds=[seed for seed,_ in items]; rows=[row for _,row in items]
        values=np.array([r['balanced_accuracy'] for r in rows])
        f1=np.array([r['f1_macro'] for r in rows])
        result.append(dict(dataset=dataset,classifier=model,method=method,n_runs=len(rows),
            seeds=sorted(seeds),
            balanced_accuracy_mean=float(values.mean()),balanced_accuracy_std=float(values.std(ddof=1)) if len(values)>1 else None,
            f1_macro_mean=float(f1.mean()),f1_macro_std=float(f1.std(ddof=1)) if len(f1)>1 else None,
            corrected_mean=float(np.mean([r['corrected'] for r in rows])),spoiled_mean=float(np.mean([r['spoiled'] for r in rows]))))
    atomic_json(root/'summary.json',result)
    lines=['# OpenML benchmark','', 'Acurácia balanceada em %. Desvio-padrão amostral; n=1 não fornece desvio. Somente execuções concluídas.','',
           '| Dataset | Modelo | Método | n | Média (%) | DP (p.p.) |','|---|---|---|---:|---:|---:|']
    for r in result:
        sd='—' if r['balanced_accuracy_std'] is None else f"{100*r['balanced_accuracy_std']:.2f}"
        lines.append(f"| {r['dataset']} | {r['classifier']} | {r['method']} | {r['n_runs']} | {100*r['balanced_accuracy_mean']:.2f} | {sd} |")
    (root/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_comparisons(root,result)
    return result


def metric_cell(record,metric):
    if record is None:
        return '—'
    mean=100*record[metric+'_mean']; std=record[metric+'_std']
    deviation='n/d' if std is None else f'{100*std:.2f}'
    return f"{mean:.2f} ± {deviation} (n={record['n_runs']})"


def dataset_comparison(result,dataset):
    records=[r for r in result if r['dataset']==dataset]
    lines=[f'## Dataset: {dataset}','',
           'Média ± DP amostral (%) entre sementes concluídas; n = número de sementes.','']
    if not records:
        return '\n'.join(lines+['Nenhuma execução concluída para este dataset.',''])
    for metric,label in [('balanced_accuracy','Acurácia balanceada'),('f1_macro','F1 macro')]:
        lines += [f'### {label}','', '| Classificador | '+' | '.join(METHODS)+' |',
                  '|---|'+'---:|'*len(METHODS)]
        for model in sorted({r['classifier'] for r in records}):
            lookup={r['method']:r for r in records if r['classifier']==model}
            lines.append('| '+model+' | '+' | '.join(metric_cell(lookup.get(m),metric) for m in METHODS)+' |')
        lines.append('')
    for model in sorted({r['classifier'] for r in records}):
        seeds=sorted({seed for r in records if r['classifier']==model for seed in r['seeds']})
        lines.append(f'Sementes disponíveis ({model}): {seeds}.')
    return '\n'.join(lines)+'\n'


def general_comparison(result):
    """Macro-average gives every eligible dataset equal weight, not every seed.

    Include a dataset in the final macro row only when every method has results
    for the SAME seed set. Individual dataset rows still show partial results.
    """
    lines=['# Comparação geral entre datasets','',
           'Cada célula: média ± DP amostral (%) entre sementes; n = sementes concluídas.',
           'A linha final é a média das médias por dataset, com pesos iguais por dataset.',
           'Essa linha usa somente datasets com todos os métodos e o mesmo conjunto de sementes; não mistura classificadores.',
           'Sementes com falha não contam como zero. Holdouts repetidos não são amostras independentes.','']
    macro=[]
    for model in sorted({r['classifier'] for r in result}):
        records=[r for r in result if r['classifier']==model]
        names=sorted({r['dataset'] for r in records})
        lookup={(r['dataset'],r['method']):r for r in records}
        eligible=[]
        for name in names:
            entries=[lookup.get((name,m)) for m in METHODS]
            if all(e is not None for e in entries) and len({tuple(e['seeds']) for e in entries})==1:
                eligible.append(name)
        for metric,label in [('balanced_accuracy','Acurácia balanceada'),('f1_macro','F1 macro')]:
            lines += [f'## {model} — {label}','', '| Dataset | '+' | '.join(METHODS)+' |',
                      '|---|'+'---:|'*len(METHODS)]
            for name in names:
                safe_name=name.replace('|',r'\|').replace('\n',' ')
                lines.append('| '+safe_name+' | '+' | '.join(metric_cell(lookup.get((name,m)),metric) for m in METHODS)+' |')
            values=[]
            for method in METHODS:
                value=float(np.mean([lookup[(name,method)][metric+'_mean'] for name in eligible])) if eligible else None
                values.append('—' if value is None else f'**{100*value:.2f}**')
                macro.append(dict(classifier=model,metric=metric,method=method,
                                  n_datasets=len(eligible),datasets=eligible,mean=value))
            lines.append(f'| **Média entre datasets (D={len(eligible)})** | '+' | '.join(values)+' |')
            lines.append('')
            excluded=sorted(set(names)-set(eligible))
            if excluded: lines += ['Fora da média final por resultados incompletos: '+', '.join(excluded)+'.','']
    if not result: lines += ['Nenhuma execução concluída encontrada.','']
    return '\n'.join(lines)+'\n',macro


def write_comparisons(root,result):
    directory=Path(root)/'comparisons'; directory.mkdir(parents=True,exist_ok=True)
    for dataset in sorted({r['dataset'] for r in result}):
        (directory/(key(dataset)+'.md')).write_text(dataset_comparison(result,dataset),encoding='utf-8')
    text,macro=general_comparison(result)
    (Path(root)/'comparison_overall.md').write_text(text,encoding='utf-8')
    atomic_json(Path(root)/'comparison_overall.json',dict(
        units='fractions [0,1]; Markdown tables display percent',
        dataset_results=result,macro_averages=macro))


def report_dataset(root,dataset):
    result=summarize(root)
    print('\n'+dataset_comparison(result,dataset),flush=True)


def log_error(root,dataset,seed,model,stage,exc):
    record=dict(dataset=dataset,seed=seed,classifier=model,stage=stage,error=str(exc),type=type(exc).__name__,traceback=traceback.format_exc())
    with (root/'errors.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(record)+'\n')
    print(f'ERROR [{stage}] {dataset}/{seed}/{model}: {exc}',flush=True)


def run(args):
    mapping=json.loads(Path(args.dataset_map).read_text()) if args.dataset_map else {}
    if not isinstance(mapping,dict): raise ValueError('--dataset-map must contain a JSON object')
    root=Path(args.output); initialize_output(root,configuration(args,mapping))
    succeeded=skipped=failed=0
    for name in args.datasets:
        pending=[(seed,model) for seed in args.seeds for model in args.classifiers
                 if not (root/'runs'/f'{key(name)}_{seed}_{model}.json').exists()]
        skipped+=len(args.seeds)*len(args.classifiers)-len(pending)
        if not pending:
            print(f'SKIP {name}: all requested runs already saved',flush=True)
            report_dataset(root,name)
            continue
        try:
            print(f'LOAD {name}',flush=True)
            override=mapping.get(name,{})
            allowed={'data_id','version','target_column'}
            if not isinstance(override,dict) or set(override)-allowed: raise ValueError('Invalid dataset mapping keys')
            X,y,meta=load_dataset(name,version=override.get('version',args.version),data_home=args.data_home,
                data_id=override.get('data_id'),target_column=override.get('target_column','default-target'),return_metadata=True,download_timeout=args.download_timeout)
            X,y,subset=select_subset(X,y,args.max_samples,args.sample_seed)
            meta['selected_original_row_indices']=[meta['original_row_indices'][i] for i in subset]
            del meta['original_row_indices']
            meta['n_used']=len(y); meta['class_counts']=np.bincount(y).tolist()
            meta_path=root/'datasets'/f'{key(name)}.json'
            if meta_path.exists() and json.loads(meta_path.read_text())!=meta:
                raise ValueError('Dataset metadata or sample selection changed since checkpoint')
            atomic_json(meta_path,meta)
            print(f'  n={len(y)}, p={X.shape[1]}, classes={len(np.unique(y))}',flush=True)
        except Exception as exc:
            failed+=len(pending); log_error(root,name,None,None,'load',exc)
            report_dataset(root,name)
            continue
        for seed in args.seeds:
            models=[m for s,m in pending if s==seed]
            if not models: continue
            try:
                train,test=train_test_split(np.arange(len(y)),train_size=args.train_fraction,stratify=y,random_state=seed)
                tr,val=train_test_split(np.arange(len(train)),train_size=args.inner_train_fraction,stratify=y[train],random_state=seed+10000)
                if len(np.unique(y[train][tr]))!=len(np.unique(y)) or len(np.unique(y[train][val]))!=len(np.unique(y)) or len(np.unique(y[test]))!=len(np.unique(y)):
                    raise ValueError('Rare classes absent in a split; increase sample size or revise split fractions')
                Zi,inner_info=prepare_features(take(X,train),tr,args)
                Z,outer_info=prepare_features(X,train,args)
                Ai=contextual_graph(build_KNN_Graph(Zi,min(args.k,len(Zi)-1)),True)
                A=contextual_graph(build_KNN_Graph(Z,min(args.k,len(Z)-1)),True)
            except Exception as exc:
                failed+=len(models); log_error(root,name,seed,None,'preprocessing_or_graph',exc); continue
            for model in models:
                try:
                    start=time.perf_counter()
                    initial=initial_predictions(Zi,y[train],tr,val,model,seed)
                    chosen=choose_parameters(Ai,initial,val,y[train][val],args)
                    print(f"SELECT {name}/{seed}/{model}: FR gamma={chosen['FR']['selected']}, tau={chosen['FR']['selected_tau']}, validation BA={100*chosen['FR']['selected_score']:.2f}%",flush=True)
                    selection_seconds=time.perf_counter()-start
                    results,weights=evaluate(A,Z,y,train,test,model,seed,chosen,args)
                    record=dict(dataset=name,classifier=model,seed=seed,dataset_metadata_file=str(meta_path.relative_to(root)),
                        n_samples=len(y),n_features_raw=X.shape[1],train_indices=train.tolist(),test_indices=test.tolist(),
                        inner_train_indices=tr.tolist(),inner_validation_indices=val.tolist(),
                        preprocessing=dict(inner=inner_info,outer=outer_info),choices=chosen,weights=weights,
                        selection_seconds=selection_seconds,results=results)
                    atomic_json(root/'runs'/f'{key(name)}_{seed}_{model}.json',record)
                    succeeded+=1
                    print(f'OK {name}/{seed}/{model}: '+', '.join(f"{r['method']}={100*r['balanced_accuracy']:.2f}" for r in results),flush=True)
                except Exception as exc:
                    failed+=1; log_error(root,name,seed,model,'model_or_smoothing',exc)
            summarize(root)
        report_dataset(root,name)
    final_summary=summarize(root)
    print('\n'+general_comparison(final_summary)[0],flush=True)
    status=dict(completed_now=succeeded,skipped_existing=skipped,failed_now=failed,python=platform.python_version())
    atomic_json(root/'last_run_status.json',status)
    print(json.dumps(status),flush=True)
    return 0 if failed==0 else 1


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',default=DATASETS)
    p.add_argument('--seeds',nargs='+',type=int,default=range(30))
    p.add_argument('--classifiers',nargs='+',choices=['NB','DT', 'MLP', 'RF', 'SVM', 'KNN'],default=['NB','DT'])
    p.add_argument('--version',type=int,default=1)
    p.add_argument('--dataset-map',help='JSON: dataset name -> data_id/version/target_column overrides')
    p.add_argument('--data-home',default=None)
    p.add_argument('--download-timeout',type=float,default=60.,help='Socket timeout in seconds; fetch retries may extend total time')
    p.add_argument('--output',default='results_openml_tau')
    p.add_argument('--max-samples',type=int,default=0,help='0 uses all rows; otherwise stratified cap before preprocessing')
    p.add_argument('--sample-seed',type=int,default=42)
    p.add_argument('--pca',type=int,default=0,help='0 disables; fit randomized PCA on training only')
    p.add_argument('--preprocess-seed',type=int,default=42)
    p.add_argument('--categorical',choices=['onehot','ordinal'],default='onehot')
    p.add_argument('--max-dense-mb',type=float,default=1024.)
    p.add_argument('--train-fraction',type=float,default=.5)
    p.add_argument('--inner-train-fraction',type=float,default=.7)
    p.add_argument('--k',type=int,default=20)
    p.add_argument('--eta',type=float,default=.5)
    p.add_argument('--tau',type=float,default=None,help='Fix tau at this value instead of searching tau-grid')
    p.add_argument('--tau-grid',nargs='+',type=float,default=[.25,.5,1.,2.,4.],help='Inner-validation candidates; ignored when --tau is supplied')
    p.add_argument('--gammas',nargs='+',type=float,default=GAMMAS)
    p.add_argument('--knn-iterations',nargs='+',type=int,default=KNN_ITERATIONS)
    p.add_argument('--max-iter',type=int,default=50)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--list-datasets',action='store_true')
    p.add_argument('--summarize-only',action='store_true',help='Rebuild tables from ALL saved runs in --output; no download, fitting or config migration')
    return p


def main():
    p=parser(); args=p.parse_args()
    if args.list_datasets:
        print('\n'.join(DATASETS)); return 0
    if args.summarize_only:
        root=Path(args.output)
        if not (root/'runs').is_dir():
            p.error('No runs/ directory found in --output')
        result=summarize(root)
        for dataset in sorted({r['dataset'] for r in result}):
            print(dataset_comparison(result,dataset),flush=True)
        print(general_comparison(result)[0],flush=True)
        return 0
    if args.k<1 or args.max_samples<0 or args.pca<0 or args.max_iter<1 or args.threads<1:
        p.error('k, max-iter, threads must be positive; max-samples and pca nonnegative')
    if not 0<args.train_fraction<1 or not 0<args.inner_train_fraction<1:
        p.error('split fractions must lie in (0,1)')
    if not np.isfinite([args.eta,args.max_dense_mb,args.download_timeout]).all() or min(args.eta,args.max_dense_mb,args.download_timeout)<=0:
        p.error('eta, tau, max-dense-mb and download-timeout must be finite and positive')
    if args.tau is not None and (not np.isfinite(args.tau) or args.tau<=0):
        p.error('--tau must be finite and positive')
    if not np.isfinite(args.tau_grid).all() or min(args.tau_grid)<=0:
        p.error('--tau-grid values must be finite and positive')
    args.tau_grid=sorted(set(args.tau_grid))
    if not np.isfinite(args.gammas).all() or min(args.gammas)<0 or min(args.knn_iterations)<0:
        p.error('gamma and iteration grids must be finite and nonnegative')
    args.gammas=sorted(set(args.gammas)); args.knn_iterations=sorted(set(args.knn_iterations))
    args.seeds=list(dict.fromkeys(args.seeds)); args.datasets=list(dict.fromkeys(args.datasets)); args.classifiers=list(dict.fromkeys(args.classifiers))
    with threadpool_limits(limits=args.threads): return run(args)

if __name__=='__main__': raise SystemExit(main())
