"""
实验 6 (v3, 完整对比版): 所有训练方法的跨数据集 transfer - 两种 setting 同时跑

两种 setting 同时测试：
- Setting 1 (balanced test): source balance → 50:50 split → train, target balance → 50:50 split → test
- Setting 2 (natural test): source 50:50 split → train_half → balance, target 50:50 split → test_half (natural)

关键对比：
- Isotonic 在两种 setting 下的 cross-domain 表现
- popularity 在两种 setting 下的鲁棒性

复现命令：
    cd /path/to/repository
    python3 code/analysis_correlation/transfer_all_methods_v3.py

输出：
    code/analysis_correlation/transfer_all_methods_v3_results.json
"""
import json
import numpy as np
import sys
sys.path.insert(0, 'code/analysis_correlation')

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64, 32), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

def compute_ece(probs, accs, n_bins=10):
    probs = np.asarray(probs, dtype=float); accs = np.asarray(accs, dtype=float)
    n = len(probs)
    if n == 0: return float('nan')
    bins = np.linspace(0, 1, n_bins + 1); ece_val = 0.0
    for i in range(n_bins):
        mask = ((probs >= bins[i]) & (probs < bins[i+1])) if i < n_bins-1 else ((probs >= bins[i]) & (probs <= bins[i+1]))
        if mask.sum() == 0: continue
        ece_val += (mask.sum() / n) * abs(accs[mask].mean() - probs[mask].mean())
    return ece_val

def compute_brier(probs, accs):
    return float(np.mean((np.asarray(probs) - np.asarray(accs)) ** 2))

def compute_logloss(probs, accs, eps=1e-7):
    probs = np.clip(np.asarray(probs), eps, 1 - eps); accs = np.asarray(accs)
    return float(-np.mean(accs * np.log(probs) + (1 - accs) * np.log(1 - probs)))

def find_optimal_threshold(signal, y):
    best_thre, best_acc = None, -1
    for thre in np.unique(signal):
        acc = ((signal >= thre).astype(int) == y).mean()
        if acc > best_acc: best_acc, best_thre = acc, thre
    if best_thre is None: best_thre = np.median(signal)
    return best_thre

def run_mlp_train(X_train, y_train, hidden_dims=(128, 64, 32)):
    if len(X_train) > 100:
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=SEED, stratify=y_train)
    else:
        X_tr, X_val, y_tr, y_val = X_train, X_train, y_train, y_train
    Xtr_t = torch.tensor(X_tr, dtype=torch.float32); ytr_t = torch.tensor(y_tr, dtype=torch.float32)
    Xval_t = torch.tensor(X_val, dtype=torch.float32); yval_t = torch.tensor(y_val, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=256, shuffle=True)
    model = MLP(X_train.shape[1], hidden_dims=hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_val = float('inf'); best_state = None; no_improve = 0
    for epoch in range(100):
        model.train()
        for Xb, yb in loader:
            optimizer.zero_grad(); loss = criterion(model(Xb), yb); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): vl = criterion(model(Xval_t), yval_t).item()
        if vl < best_val - 1e-5:
            best_val = vl; best_state = {k: v.clone() for k, v in model.state_dict().items()}; no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 10: break
    if best_state is not None: model.load_state_dict(best_state)
    return model

def mlp_predict(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).numpy()
    return 1 / (1 + np.exp(-np.clip(logits, -500, 500)))

def balance_subset(samples, seed=SEED):
    rng = np.random.default_rng(seed)
    pos = [s for s in samples if s['acc'] == 1]
    neg = [s for s in samples if s['acc'] == 0]
    n_min = min(len(pos), len(neg))
    pos_idx = rng.choice(len(pos), n_min, replace=False)
    neg_idx = rng.choice(len(neg), n_min, replace=False)
    balanced = [pos[i] for i in pos_idx] + [neg[i] for i in neg_idx]
    rng.shuffle(balanced)
    return balanced

def get_setting1_samples(dataset, model, full_dict, co_occu, single_occr):
    """Setting 1: balance 全量 → 50:50 split → train + test (both balanced)"""
    samples = collect_samples(dataset, model, full_dict, co_occu, single_occr, {}, {}, {}, {}, {})
    if len(samples) < 50: return None, None
    y = np.array([s['acc'] for s in samples])
    pos_ratio = y.mean()
    if pos_ratio < 0.25 or pos_ratio > 0.75:
        samples = balance_subset(samples)
    if len(samples) < 50: return None, None
    y = np.array([s['acc'] for s in samples])
    idx = np.arange(len(samples))
    idx_train, idx_test = train_test_split(idx, test_size=0.5, random_state=SEED, stratify=y)
    return [samples[i] for i in idx_train], [samples[i] for i in idx_test]

def get_setting2_samples(dataset, model, full_dict, co_occu, single_occr):
    """Setting 2: 50:50 split → train_half (balance) + test_half (natural)"""
    samples = collect_samples(dataset, model, full_dict, co_occu, single_occr, {}, {}, {}, {}, {})
    if len(samples) < 50: return None, None
    y = np.array([s['acc'] for s in samples])
    idx = np.arange(len(samples))
    idx_train, idx_test = train_test_split(idx, test_size=0.5, random_state=SEED, stratify=y)
    samples_train_full = [samples[i] for i in idx_train]
    samples_test_natural = [samples[i] for i in idx_test]
    y_train_full = np.array([s['acc'] for s in samples_train_full])
    pos_ratio_train = y_train_full.mean()
    if pos_ratio_train < 0.25 or pos_ratio_train > 0.75:
        samples_train_balanced = balance_subset(samples_train_full)
    else:
        samples_train_balanced = samples_train_full
    if len(samples_train_balanced) < 50: return None, None
    return samples_train_balanced, samples_test_natural

from calibration_experiment import collect_samples, load_resources

print('加载资源...')
full_dict, co_occu, single_occr = load_resources()

DATASETS = ['movies', 'songs', 'basketball']
MODELS = ['llama8b', 'qwen2', 'chatgpt', 'Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']

# 预加载两种 setting 的数据
print('预加载所有 (dataset, model) 数据（两种 setting）...')
data_cache = {'setting1': {}, 'setting2': {}}
for ds in DATASETS:
    for m in MODELS:
        s1_train, s1_test = get_setting1_samples(ds, m, full_dict, co_occu, single_occr)
        s2_train, s2_test = get_setting2_samples(ds, m, full_dict, co_occu, single_occr)
        if s1_train is not None:
            data_cache['setting1'][(ds, m)] = {'train': s1_train, 'test': s1_test}
        if s2_train is not None:
            data_cache['setting2'][(ds, m)] = {'train': s2_train, 'test': s2_test}

for setting in ['setting1', 'setting2']:
    print(f'\n  {setting}:')
    for ds in DATASETS:
        for m in MODELS:
            if (ds, m) in data_cache[setting]:
                d = data_cache[setting][(ds, m)]
                acc_tr = np.mean([s['acc'] for s in d['train']])
                acc_te = np.mean([s['acc'] for s in d['test']])
                print(f'    {ds}×{m}: train={len(d["train"])} (acc={acc_tr:.3f}), test={len(d["test"])} (acc={acc_te:.3f})')

def train_method_on_source(method, source_ds, source_m, setting):
    samples_train = data_cache[setting][(source_ds, source_m)]['train']
    y_tr = np.array([s['acc'] for s in samples_train])
    conf_tr = np.array([s['conf'] for s in samples_train])
    
    if method == 'temp':
        eps = 1e-7
        p_train = np.clip(conf_tr, eps, 1 - eps)
        logits_train = np.log(p_train / (1 - p_train))
        best_T, best_loss = 1.0, float('inf')
        for T in np.concatenate([np.linspace(0.1, 2.0, 50), np.linspace(2.0, 5.0, 20)]):
            scaled = 1 / (1 + np.exp(-logits_train / T))
            bce = -np.mean(y_tr * np.log(scaled + eps) + (1 - y_tr) * np.log(1 - scaled + eps))
            if bce < best_loss: best_loss, best_T = bce, T
        probs_tr = 1 / (1 + np.exp(-logits_train / best_T))
        thre = find_optimal_threshold(probs_tr, y_tr)
        return {'T': best_T, 'thre': thre}
    elif method == 'platt':
        clf = LogisticRegression(max_iter=1000, random_state=SEED)
        clf.fit(conf_tr.reshape(-1, 1), y_tr)
        probs_tr = clf.predict_proba(conf_tr.reshape(-1, 1))[:, 1]
        thre = find_optimal_threshold(probs_tr, y_tr)
        return {'clf': clf, 'thre': thre}
    elif method == 'iso':
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
        iso.fit(conf_tr, y_tr)
        probs_tr = iso.predict(conf_tr)
        thre = find_optimal_threshold(probs_tr, y_tr)
        return {'iso': iso, 'thre': thre}
    elif method == 'mlp1':
        X_tr = conf_tr.reshape(-1, 1)
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        model = run_mlp_train(X_tr_s, y_tr)
        probs_tr = mlp_predict(model, X_tr_s)
        thre = find_optimal_threshold(probs_tr, y_tr)
        return {'model': model, 'scaler': scaler, 'thre': thre}
    elif method == 'mlp2':
        gene_coo_tr = np.array([s['gene_coo'] for s in samples_train])
        X_tr = np.column_stack([conf_tr, gene_coo_tr])
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        model = run_mlp_train(X_tr_s, y_tr)
        probs_tr = mlp_predict(model, X_tr_s)
        thre = find_optimal_threshold(probs_tr, y_tr)
        return {'model': model, 'scaler': scaler, 'thre': thre}

def eval_method_on_target(trained, method, target_ds, target_m, setting):
    samples_test = data_cache[setting][(target_ds, target_m)]['test']
    y_te = np.array([s['acc'] for s in samples_test])
    conf_te = np.array([s['conf'] for s in samples_test])
    
    if method == 'temp':
        eps = 1e-7
        p_te = np.clip(conf_te, eps, 1 - eps)
        logits_te = np.log(p_te / (1 - p_te))
        probs_te = 1 / (1 + np.exp(-logits_te / trained['T']))
    elif method == 'platt':
        probs_te = trained['clf'].predict_proba(conf_te.reshape(-1, 1))[:, 1]
    elif method == 'iso':
        probs_te = trained['iso'].predict(conf_te)
    elif method == 'mlp1':
        X_te = conf_te.reshape(-1, 1)
        X_te_s = trained['scaler'].transform(X_te)
        probs_te = mlp_predict(trained['model'], X_te_s)
    elif method == 'mlp2':
        gene_coo_te = np.array([s['gene_coo'] for s in samples_test])
        X_te = np.column_stack([conf_te, gene_coo_te])
        X_te_s = trained['scaler'].transform(X_te)
        probs_te = mlp_predict(trained['model'], X_te_s)
    
    thre = trained['thre']
    align = float(((probs_te >= thre).astype(int) == y_te).mean())
    ece = compute_ece(probs_te, y_te)
    brier = compute_brier(probs_te, y_te)
    auc = roc_auc_score(y_te, probs_te) if len(set(y_te)) > 1 else float('nan')
    logloss = compute_logloss(probs_te, y_te)
    sep = probs_te[y_te==1].mean() - probs_te[y_te==0].mean() if (y_te==0).sum()>0 and (y_te==1).sum()>0 else float('nan')
    return {'align': align, 'ece': ece, 'brier': brier, 'auc': auc, 'logloss': logloss, 'sep': sep, 'n_test': len(samples_test), 'mean_acc_test': float(y_te.mean())}

METHODS = [('temp', 'Temp'), ('platt', 'Platt'), ('iso', 'Isotonic'), ('mlp1', 'MLP (conf only)'), ('mlp2', 'MLP (conf+gene_coo)')]

# 跑两种 setting 的所有 transfer
results = {'setting1': {'in_domain': {}, 'cross_dataset': {}, 'cross_model': {}},
           'setting2': {'in_domain': {}, 'cross_dataset': {}, 'cross_model': {}}}

for setting in ['setting1', 'setting2']:
    print(f'\n=== 跑 {setting} ===')
    
    # In-domain
    print(f'\n--- {setting}: In-domain baseline ---')
    for ds in DATASETS:
        for m in MODELS:
            if (ds, m) not in data_cache[setting]: continue
            for method_key, _ in METHODS:
                trained = train_method_on_source(method_key, ds, m, setting)
                res = eval_method_on_target(trained, method_key, ds, m, setting)
                results[setting]['in_domain'][f'{ds}__{m}__{method_key}'] = res
    
    # Cross-dataset
    print(f'--- {setting}: Cross-dataset ---')
    for src_ds in DATASETS:
        for tgt_ds in DATASETS:
            if src_ds == tgt_ds: continue
            for m in MODELS:
                if (src_ds, m) not in data_cache[setting] or (tgt_ds, m) not in data_cache[setting]: continue
                for method_key, _ in METHODS:
                    trained = train_method_on_source(method_key, src_ds, m, setting)
                    res = eval_method_on_target(trained, method_key, tgt_ds, m, setting)
                    results[setting]['cross_dataset'][f'{src_ds}->{tgt_ds}__{m}__{method_key}'] = res
    
    # Cross-model
    print(f'--- {setting}: Cross-model ---')
    for ds in DATASETS:
        for src_m in MODELS:
            for tgt_m in MODELS:
                if src_m == tgt_m: continue
                if (ds, src_m) not in data_cache[setting] or (ds, tgt_m) not in data_cache[setting]: continue
                for method_key, _ in METHODS:
                    trained = train_method_on_source(method_key, ds, src_m, setting)
                    res = eval_method_on_target(trained, method_key, ds, tgt_m, setting)
                    results[setting]['cross_model'][f'{ds}__{src_m}->{tgt_m}__{method_key}'] = res

# 保存结果
def to_native(d):
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in d.items()}
results_native = {s: {cat: {k: to_native(v) for k, v in pairs.items()} for cat, pairs in cats.items()} for s, cats in results.items()}
with open('code/analysis_correlation/transfer_all_methods_v3_results.json', 'w') as f:
    json.dump(results_native, f, indent=2)

# 汇总
print('\n=== 实验 6 (v3) 汇总：两种 setting 对比 ===')
for setting in ['setting1', 'setting2']:
    print(f'\n{"="*100}')
    print(f'  {setting.upper()}')
    print(f'{"="*100}')
    for cat, pairs in results[setting].items():
        if not pairs: continue
        print(f'\n--- {setting} / {cat} ---')
        by_method = {mk: {'align': [], 'ece': [], 'brier': [], 'auc': [], 'logloss': [], 'sep': [], 'mean_acc_test': []} for mk, _ in METHODS}
        for k, v in pairs.items():
            method_key = k.split('__')[-1]
            for metric in by_method[method_key]:
                by_method[method_key][metric].append(v[metric])
        print(f'{"Method":<22} {"test_acc":>9} {"Align":>8} {"ECE":>8} {"Brier":>8} {"AUC":>8} {"LogLoss":>8} {"Δsep":>9}')
        for method_key, method_label in METHODS:
            d = by_method[method_key]
            if not d['ece']: continue
            print(f'{method_label:<22} {np.mean(d["mean_acc_test"]):>9.4f} {np.mean(d["align"])*100:>7.2f}% {np.mean(d["ece"]):>8.4f} {np.mean(d["brier"]):>8.4f} {np.mean(d["auc"]):>8.4f} {np.mean(d["logloss"]):>8.4f} {np.mean(d["sep"]):>+9.4f}')

print('\n结果已保存到 code/analysis_correlation/transfer_all_methods_v3_results.json')
