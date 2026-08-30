"""
补充实验：所有训练方法的 Cross-relation transfer (Movies+Songs → Basketball)

背景：transfer_all_methods_v3.py 已跑完 in-domain + cross-dataset + cross-model 的 5 个方法，
但 cross-relation 部分缺失。本脚本补充 cross-relation 的 5 个方法 × 2 setting × 6 模型。

数据划分（与 v3 完全一致）：
- Setting 1 (balanced test): source balance → 50:50 split → train, target balance → 50:50 split → test
- Setting 2 (natural test): source 50:50 split → train_half → balance, target 50:50 split → test_half (natural)

Cross-relation 特殊处理：
- Source = Movies + Songs 合并后的 train samples (按 setting 处理)
- Target = Basketball 的 test samples (按 setting 处理)

方法：Temp / Platt / Isotonic / MLP (1d, conf only) / MLP (2d, conf+gene_coo)

复现命令：
    cd /path/to/repository
    python3 code/analysis_correlation/transfer_cross_relation_all_methods.py

输出：
    code/analysis_correlation/transfer_cross_relation_all_methods_results.json
"""
import json
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, 'code/analysis_correlation')

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 复用 transfer_all_methods_v3.py 的工具函数（仅函数，不导入全局变量以避免触发实验）
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "transfer_all_methods_v3",
    "code/analysis_correlation/transfer_all_methods_v3.py"
)
# 直接定义工具函数和常量，避免 import v3 时触发全局实验代码
SEED = 42
DATASETS = ['movies', 'songs', 'basketball']
MODELS = ['llama8b', 'qwen2', 'chatgpt', 'Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']
# 只跑 3 个 1 维 baseline 方法（MLP-2d 在实验 4 中已跑，MLP-1d 在实验 6 v3 中已跑）
METHODS = [('temp', 'Temp'), ('platt', 'Platt'), ('iso', 'Isotonic')]


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
        mask = ((probs >= bins[i]) & (probs < bins[i + 1])) if i < n_bins - 1 else ((probs >= bins[i]) & (probs <= bins[i + 1]))
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


def mlp_predict(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).numpy()
    return 1 / (1 + np.exp(-np.clip(logits, -500, 500)))


def run_mlp_train_safe(X_train, y_train, hidden_dims=(128, 64, 32)):
    """与 transfer_all_methods_v3.run_mlp_train 等价，但增加了 BatchNorm 安全处理：
    - 动态 batch_size = min(256, len(X_tr))
    - 跳过 batch_size < 2 的 batch（避免 BatchNorm 错误）
    """
    if len(X_train) > 100:
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=SEED, stratify=y_train)
    else:
        X_tr, X_val, y_tr, y_val = X_train, X_train, y_train, y_train

    # 动态调整 batch_size
    effective_batch_size = min(256, len(X_tr))
    if effective_batch_size < 2:
        effective_batch_size = 2

    Xtr_t = torch.tensor(X_tr, dtype=torch.float32); ytr_t = torch.tensor(y_tr, dtype=torch.float32)
    Xval_t = torch.tensor(X_val, dtype=torch.float32); yval_t = torch.tensor(y_val, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=effective_batch_size, shuffle=True)
    model = MLP(X_train.shape[1], hidden_dims=hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_val = float('inf'); best_state = None; no_improve = 0
    for epoch in range(100):
        model.train()
        for Xb, yb in loader:
            if Xb.shape[0] < 2:
                continue  # BatchNorm 需要 batch_size >= 2
            optimizer.zero_grad(); loss = criterion(model(Xb), yb); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            vl_input = Xval_t
            if Xval_t.shape[0] < 2:
                vl_input = Xval_t.repeat(2, 1)
                vl_target = yval_t.repeat(2)
            else:
                vl_target = yval_t
            vl = criterion(model(vl_input), vl_target).item()
        if vl < best_val - 1e-5:
            best_val = vl; best_state = {k: v.clone() for k, v in model.state_dict().items()}; no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 10: break
    if best_state is not None: model.load_state_dict(best_state)
    return model


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
    from calibration_experiment import collect_samples
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
    from calibration_experiment import collect_samples
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


# 加载资源
from calibration_experiment import load_resources
print('加载资源...')
full_dict, co_occu, single_occr = load_resources()

# 预加载两种 setting 的数据（只加载 cross-relation 需要的）
print('预加载 (movies, songs, basketball) × 6 模型 × 2 setting 数据...')
data_cache = {'setting1': {}, 'setting2': {}}
for ds in DATASETS:
    for m in MODELS:
        s1_train, s1_test = get_setting1_samples(ds, m, full_dict, co_occu, single_occr)
        s2_train, s2_test = get_setting2_samples(ds, m, full_dict, co_occu, single_occr)
        if s1_train is not None:
            data_cache['setting1'][(ds, m)] = {'train': s1_train, 'test': s1_test}
        if s2_train is not None:
            data_cache['setting2'][(ds, m)] = {'train': s2_train, 'test': s2_test}


def train_method_on_combined_source(method, source_samples):
    y_tr = np.array([s['acc'] for s in source_samples])
    conf_tr = np.array([s['conf'] for s in source_samples])
    if method == 'temp':
        eps = 1e-7
        p_train = np.clip(conf_tr, eps, 1 - eps)
        logits_train = np.log(p_train / (1 - p_train))
        best_T, best_loss = 1.0, float('inf')
        for T in np.concatenate([np.linspace(0.1, 2.0, 50), np.linspace(2.0, 5.0, 20)]):
            scaled = 1 / (1 + np.exp(-logits_train / T))
            bce = -np.mean(y_tr * np.log(scaled + eps) + (1 - y_tr) * np.log(1 - scaled + eps))
            if bce < best_loss:
                best_loss, best_T = bce, T
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
        model = run_mlp_train_safe(X_tr_s, y_tr)
        probs_tr = mlp_predict(model, X_tr_s)
        thre = find_optimal_threshold(probs_tr, y_tr)
        return {'model': model, 'scaler': scaler, 'thre': thre}
    elif method == 'mlp2':
        gene_coo_tr = np.array([s['gene_coo'] for s in source_samples])
        X_tr = np.column_stack([conf_tr, gene_coo_tr])
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        model = run_mlp_train_safe(X_tr_s, y_tr)
        probs_tr = mlp_predict(model, X_tr_s)
        thre = find_optimal_threshold(probs_tr, y_tr)
        return {'model': model, 'scaler': scaler, 'thre': thre}


def eval_method_on_target_v2(trained, method, target_samples):
    """在 target samples 上评估指定方法。与 transfer_all_methods_v3.py 的 eval_method_on_target 等价，
    但接受 target_samples 作为参数（而不是从 data_cache 读取）。
    """
    y_te = np.array([s['acc'] for s in target_samples])
    conf_te = np.array([s['conf'] for s in target_samples])

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
        gene_coo_te = np.array([s['gene_coo'] for s in target_samples])
        X_te = np.column_stack([conf_te, gene_coo_te])
        X_te_s = trained['scaler'].transform(X_te)
        probs_te = mlp_predict(trained['model'], X_te_s)

    thre = trained['thre']
    align = float(((probs_te >= thre).astype(int) == y_te).mean())
    ece = compute_ece(probs_te, y_te)
    brier = compute_brier(probs_te, y_te)
    auc = roc_auc_score(y_te, probs_te) if len(set(y_te)) > 1 else float('nan')
    logloss = compute_logloss(probs_te, y_te)
    sep = probs_te[y_te == 1].mean() - probs_te[y_te == 0].mean() if (y_te == 0).sum() > 0 and (y_te == 1).sum() > 0 else float('nan')
    return {
        'align': align, 'ece': ece, 'brier': brier, 'auc': auc, 'logloss': logloss, 'sep': sep,
        'n_test': len(target_samples), 'mean_acc_test': float(y_te.mean())
    }


def get_combined_source_samples(setting, model):
    if (DATASETS[0], model) not in data_cache[setting] or (DATASETS[1], model) not in data_cache[setting]:
        return None
    movies_train = data_cache[setting][(DATASETS[0], model)]['train']
    songs_train = data_cache[setting][(DATASETS[1], model)]['train']
    combined = movies_train + songs_train
    return combined


def to_native(v):
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


# 跑两种 setting 的 cross-relation
results = {'setting1': {'cross_relation': {}}, 'setting2': {'cross_relation': {}}}

print('\n' + '=' * 100)
print('  Cross-relation transfer (Movies+Songs → Basketball) - 5 methods × 2 settings × 6 models')
print('=' * 100)

for setting in ['setting1', 'setting2']:
    print(f'\n=== {setting} ===')
    print(f'{"Method":<22} {"Model":<15} {"Align":>8} {"ECE":>8} {"Brier":>8} {"AUC":>8} {"LogLoss":>8} {"Δsep":>9} {"N_test":>8} {"test_acc":>9}')

    for m in MODELS:
        # 检查 Basketball 是否在 data_cache 中
        if (DATASETS[2], m) not in data_cache[setting]:
            continue

        # 合并 Movies + Songs 的 source samples
        combined_source = get_combined_source_samples(setting, m)
        if combined_source is None or len(combined_source) < 50:
            print(f'  [SKIP] {m}: combined source 数据不足')
            continue

        # Target = Basketball test samples
        target_samples = data_cache[setting][(DATASETS[2], m)]['test']

        for method_key, method_label in METHODS:
            try:
                trained = train_method_on_combined_source(method_key, combined_source)
                res = eval_method_on_target_v2(trained, method_key, target_samples)
                key = f'Movies+Songs__Basketball__{m}__{method_key}'
                results[setting]['cross_relation'][key] = {k: to_native(v) for k, v in res.items()}
                print(f'{method_label:<22} {m:<15} {res["align"] * 100:>7.2f}% {res["ece"]:>8.4f} {res["brier"]:>8.4f} {res["auc"]:>8.4f} {res["logloss"]:>8.4f} {res["sep"]:>+9.4f} {res["n_test"]:>8} {res["mean_acc_test"]:>9.4f}')
            except Exception as e:
                print(f'  [ERROR] {method_label} × {m}: {e}')

# 保存结果
output_path = Path(__file__).with_name(
    'transfer_cross_relation_all_methods_results.json'
)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\n详细结果已保存到 {output_path}')

# 汇总
print('\n' + '=' * 100)
print('  Cross-relation 汇总（6 模型平均）')
print('=' * 100)
print(f'{"Setting":<10} {"Method":<22} {"Align":>8} {"ECE":>8} {"Brier":>8} {"AUC":>8} {"LogLoss":>8} {"Δsep":>9}')
for setting in ['setting1', 'setting2']:
    by_method = {mk: {'align': [], 'ece': [], 'brier': [], 'auc': [], 'logloss': [], 'sep': [], 'mean_acc_test': []} for mk, _ in METHODS}
    for k, v in results[setting]['cross_relation'].items():
        method_key = k.split('__')[-1]
        for metric in by_method[method_key]:
            by_method[method_key][metric].append(v[metric])
    for method_key, method_label in METHODS:
        d = by_method[method_key]
        if not d['ece']:
            continue
        print(f'{setting:<10} {method_label:<22} {np.mean(d["align"]) * 100:>7.2f}% {np.mean(d["ece"]):>8.4f} {np.mean(d["brier"]):>8.4f} {np.mean(d["auc"]):>8.4f} {np.mean(d["logloss"]):>8.4f} {np.mean(d["sep"]):>+9.4f}')
