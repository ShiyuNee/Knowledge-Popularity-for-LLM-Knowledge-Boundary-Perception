"""
实验 6: 所有训练方法的跨数据集 transfer

实验类型：
1. In-domain baseline: 同 (D, M) 训测 (18 pairs)
2. Cross-dataset: 同 model 跨 dataset (36 pairs)
3. Cross-model: 同 dataset 跨 model (90 pairs)

训练方法（5 种）：
- Temp: grid search T 使 BCE 最小
- Platt: LogisticRegression(max_iter=1000, random_state=42)
- Isotonic: IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
- MLP (1d, conf only): 128/64/32, BatchNorm, dropout=0.3, Adam (lr=5e-4, wd=1e-4),
                        batch_size=256, 100 epochs, patience=10, val 10% early stopping
- MLP (2d, conf + gene_coo): 同上，输入 (conf, gene_coo)

训练流程（所有方法一致）：
1. 在 source (D1, M1) 上加载全量数据
2. 如果 pos_ratio < 0.25 或 > 0.75，下采样到 1:1 (balance)
3. 50:50 stratified split (seed=42) → train + test
4. 在 train 上训练方法
5. 在 train 概率上找最优二值化阈值

评估流程：
1. 在 target (D2, M2) 上加载全量数据
2. 同样的 balance + 50:50 split → 取 test_half
3. 用 source 训练的参数 transform target 数据
   - Temp: 用 source 的 T 缩放 target 的 conf
   - Platt: 用 source 的 LR predict target 的 conf
   - Isotonic: 用 source 的 iso.predict(target 的 conf)
   - MLP (1d): 用 source 的 scaler.transform + MLP predict target 的 conf
   - MLP (2d): 用 source 的 scaler.transform + MLP predict target 的 (conf, gene_coo)
4. 用 source 的 threshold 在 target test 上算指标

评估指标：
- Alignment, ECE, Brier, AUC, LogLoss, Δsep

复现命令：
    cd /path/to/repository
    python3 code/analysis_correlation/transfer_all_methods.py

输出：
    code/analysis_correlation/transfer_all_methods_results.json
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
    """训练 MLP，返回 (trained_model, scaler)"""
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

def get_train_test_samples(dataset, model, full_dict, co_occu, single_occr):
    """加载 (dataset, model) 数据，做 balance + 50:50 split，返回 (train_samples, test_samples)"""
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

from calibration_experiment import collect_samples, load_resources

print('加载资源...')
full_dict, co_occu, single_occr = load_resources()

DATASETS = ['movies', 'songs', 'basketball']
MODELS = ['llama8b', 'qwen2', 'chatgpt', 'Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']

# 准备所有 (dataset, model) 的 train + test samples
print('预加载所有 (dataset, model) 数据...')
data_cache = {}
for ds in DATASETS:
    for m in MODELS:
        train_s, test_s = get_train_test_samples(ds, m, full_dict, co_occu, single_occr)
        if train_s is not None:
            data_cache[(ds, m)] = {'train': train_s, 'test': test_s}
            print(f'  {ds}×{m}: train={len(train_s)}, test={len(test_s)}, acc_train={np.mean([s["acc"] for s in train_s]):.3f}, acc_test={np.mean([s["acc"] for s in test_s]):.3f}')

def train_method_on_source(method, source_ds, source_m):
    """在 source 上训练方法，返回 trained artifacts"""
    samples_train = data_cache[(source_ds, source_m)]['train']
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

def eval_method_on_target(trained, method, target_ds, target_m):
    """在 target test set 上评估方法"""
    samples_test = data_cache[(target_ds, target_m)]['test']
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
    return {'align': align, 'ece': ece, 'brier': brier, 'auc': auc, 'logloss': logloss, 'sep': sep, 'n_test': len(samples_test)}

METHODS = [('temp', 'Temp'), ('platt', 'Platt'), ('iso', 'Isotonic'), ('mlp1', 'MLP (conf only)'), ('mlp2', 'MLP (conf+gene_coo)')]

# === 跑所有 transfer pair ===
# 1. In-domain baseline
# 2. Cross-dataset: 同 model 跨 dataset
# 3. Cross-model: 同 dataset 跨 model

results = {'in_domain': {}, 'cross_dataset': {}, 'cross_model': {}}

print('\n=== 跑 in-domain baseline ===')
for ds in DATASETS:
    for m in MODELS:
        if (ds, m) not in data_cache: continue
        for method_key, method_label in METHODS:
            trained = train_method_on_source(method_key, ds, m)
            res = eval_method_on_target(trained, method_key, ds, m)
            results['in_domain'][f'{ds}__{m}__{method_key}'] = res
            print(f'  {ds}×{m} [{method_label}]: Align={res["align"]*100:.2f}%, ECE={res["ece"]:.4f}, Brier={res["brier"]:.4f}, AUC={res["auc"]:.4f}, LogLoss={res["logloss"]:.4f}')

print('\n=== 跑 cross-dataset (同 model, 跨 dataset) ===')
for src_ds in DATASETS:
    for tgt_ds in DATASETS:
        if src_ds == tgt_ds: continue
        for m in MODELS:
            if (src_ds, m) not in data_cache or (tgt_ds, m) not in data_cache: continue
            for method_key, method_label in METHODS:
                trained = train_method_on_source(method_key, src_ds, m)
                res = eval_method_on_target(trained, method_key, tgt_ds, m)
                results['cross_dataset'][f'{src_ds}->{tgt_ds}__{m}__{method_key}'] = res
            print(f'  {src_ds}->{tgt_ds}×{m}: done')

print('\n=== 跑 cross-model (同 dataset, 跨 model) ===')
for ds in DATASETS:
    for src_m in MODELS:
        for tgt_m in MODELS:
            if src_m == tgt_m: continue
            if (ds, src_m) not in data_cache or (ds, tgt_m) not in data_cache: continue
            for method_key, method_label in METHODS:
                trained = train_method_on_source(method_key, ds, src_m)
                res = eval_method_on_target(trained, method_key, ds, tgt_m)
                results['cross_model'][f'{ds}__{src_m}->{tgt_m}__{method_key}'] = res
            print(f'  {ds}×{src_m}->{tgt_m}: done')

# 保存结果
with open('/tmp/exp_a_results.json', 'w') as f:
    # convert numpy types
    def to_native(d):
        return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in d.items()}
    results_native = {cat: {k: to_native(v) for k, v in pairs.items()} for cat, pairs in results.items()}
    json.dump(results_native, f, indent=2)

# 汇总
print('\n=== 实验 A 汇总 ===')
for cat, pairs in results.items():
    if not pairs: continue
    print(f'\n--- {cat} ---')
    by_method = {mk: {'align': [], 'ece': [], 'brier': [], 'auc': [], 'logloss': [], 'sep': []} for mk, _ in METHODS}
    for k, v in pairs.items():
        method_key = k.split('__')[-1]
        for metric in by_method[method_key]:
            by_method[method_key][metric].append(v[metric])
    print(f'{"Method":<22} {"Align":>8} {"ECE":>8} {"Brier":>8} {"AUC":>8} {"LogLoss":>8} {"Δsep":>9}')
    for method_key, method_label in METHODS:
        d = by_method[method_key]
        if not d['ece']: continue
        print(f'{method_label:<22} {np.mean(d["align"])*100:>7.2f}% {np.mean(d["ece"]):>8.4f} {np.mean(d["brier"]):>8.4f} {np.mean(d["auc"]):>8.4f} {np.mean(d["logloss"]):>8.4f} {np.mean(d["sep"]):>+9.4f}')

print('\n结果已保存到 /tmp/exp_a_results.json')
