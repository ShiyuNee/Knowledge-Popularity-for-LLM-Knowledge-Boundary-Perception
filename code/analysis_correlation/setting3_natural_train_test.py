"""
实验 7 (Setting 3): natural train + natural test

数据划分：
1. 全量数据 (dataset, model) → 50:50 stratified split (seed=42) → train_half + test_half
2. train_half 不做 balance (保持自然分布, mean(acc) ≈ 0.30)
3. test_half 不做 balance (保持自然分布, mean(acc) ≈ 0.30)
4. 在 train_half_natural 上训练所有方法
5. 在 test_half_natural 上评估所有方法

与 Setting 2 的区别：
- Setting 2: train_half 做 balance + test_half natural
- Setting 3: train_half 不做 balance + test_half natural (即 Setting 2 去掉 balance 步骤)

与 Setting 1 / Setting 2 共享的关键设置（确保实验间一致性）：
- 全局 SEED = 42
- 50:50 stratified split (random_state=42)
- 6 模型 × 3 数据集 = 18 (dataset, model) pairs
- 6 方法: Prob (raw) / Temp / Platt / Isotonic / MLP (1d) / MLP (2d)
- MLP: 128/64/32, BatchNorm, dropout=0.3, Adam (lr=5e-4, wd=1e-4), batch_size=256, 100 epochs, patience=10
- 不排除任何低准确率模型
- macro average (算术平均)

训练方法：
- Prob (raw): 无训练，直接用 train 上的最优阈值
- Temp: grid search T 使 BCE 最小
- Platt: LogisticRegression(max_iter=1000, random_state=42)
- Isotonic: IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
- MLP (1d): 128/64/32, BatchNorm, dropout=0.3, Adam (lr=5e-4, wd=1e-4),
            batch_size=256, 100 epochs, patience=10, val 10% early stopping
- MLP (2d): 同上，输入 (conf, gene_coo)

评估指标：
- Alignment: accuracy((pred >= threshold).astype(int) == y)
- ECE: 10-bin Expected Calibration Error
- Brier: mean((pred - y)^2)
- AUC: ROC-AUC (排序能力)
- LogLoss: 严格负对数似然
- Δsep: mean(pred|y=1) - mean(pred|y=0) (区分度)

复现命令：
    cd /path/to/repository
    python3 code/analysis_correlation/setting3_natural_train_test.py

输出：
    code/analysis_correlation/setting3_results.json
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


def run_mlp_with_train_probs(X_train, y_train, X_test, hidden_dims=(128, 64, 32)):
    """训练 MLP 并返回 (train_probs, test_probs)"""
    # 动态调整 batch_size，避免 BatchNorm 在 batch_size=1 时报错
    effective_batch_size = min(256, len(X_train))
    if effective_batch_size < 2:
        effective_batch_size = 2  # 至少 2 个样本才能做 BatchNorm

    if len(X_train) > 100:
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=SEED, stratify=y_train)
    else:
        X_tr, X_val, y_tr, y_val = X_train, X_train, y_train, y_train
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
        with torch.no_grad(): vl = criterion(model(Xval_t), yval_t).item()
        if vl < best_val - 1e-5:
            best_val = vl; best_state = {k: v.clone() for k, v in model.state_dict().items()}; no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 10: break
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_train = model(torch.tensor(X_train, dtype=torch.float32)).numpy()
        logits_test = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    probs_train = 1 / (1 + np.exp(-np.clip(logits_train, -500, 500)))
    probs_test = 1 / (1 + np.exp(-np.clip(logits_test, -500, 500)))
    return probs_train, probs_test


def run_temperature(conf_train, y_train, conf_test):
    eps = 1e-7
    p_train = np.clip(conf_train, eps, 1 - eps)
    logits_train = np.log(p_train / (1 - p_train))
    best_T, best_loss = 1.0, float('inf')
    for T in np.concatenate([np.linspace(0.1, 2.0, 50), np.linspace(2.0, 5.0, 20)]):
        scaled = 1 / (1 + np.exp(-logits_train / T))
        bce = -np.mean(y_train * np.log(scaled + eps) + (1 - y_train) * np.log(1 - scaled + eps))
        if bce < best_loss: best_loss, best_T = bce, T
    scaled_train = 1 / (1 + np.exp(-logits_train / best_T))
    logits_test = np.log(np.clip(conf_test, eps, 1 - eps) / (1 - np.clip(conf_test, eps, 1 - eps)))
    scaled_test = 1 / (1 + np.exp(-logits_test / best_T))
    return scaled_train, scaled_test, best_T


from calibration_experiment import collect_samples, load_resources

print('加载资源...')
full_dict, co_occu, single_occr = load_resources()

DATASETS = ['movies', 'songs', 'basketball']
MODELS = ['llama8b', 'qwen2', 'chatgpt', 'Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']

agg = {k: {'ece': [], 'brier': [], 'auc': [], 'logloss': [], 'sep': [], 'align': [], 'mean_acc': [], 'mean_pred': [], 'std_pred': []}
       for k in ['raw', 'temp', 'platt', 'iso', 'mlp1', 'mlp2']}

# 同时保存每个 (dataset, model, method) 的详细数据，用于一致性验证
per_pair_results = {}

print('\n=== 实验 7：Setting 3 (natural train + natural test) ===')
header = f'{"Dataset":<10} {"Model":<15} {"Method":<22} {"test_acc":>9} {"mean(pred)":>11} {"std(pred)":>10} {"Δsep":>9} {"Align":>8} {"ECE":>8} {"Brier":>8} {"AUC":>8} {"LogLoss":>8}'
print(header)

for ds in DATASETS:
    for m in MODELS:
        samples = collect_samples(ds, m, full_dict, co_occu, single_occr, {}, {}, {}, {}, {})
        if len(samples) < 50: continue
        y = np.array([s['acc'] for s in samples])
        idx = np.arange(len(samples))
        # 与 Setting 2 完全相同的 split
        idx_train, idx_test = train_test_split(idx, test_size=0.5, random_state=SEED, stratify=y)
        samples_train_natural = [samples[i] for i in idx_train]
        samples_test_natural = [samples[i] for i in idx_test]

        # 关键区别于 Setting 2: 不做 balance，train 和 test 都保持自然分布
        samples_train_balanced = samples_train_natural  # 不做 balance
        y_tr = np.array([s['acc'] for s in samples_train_balanced])
        y_te = np.array([s['acc'] for s in samples_test_natural])
        conf_tr = np.array([s['conf'] for s in samples_train_balanced])
        conf_te = np.array([s['conf'] for s in samples_test_natural])
        mean_acc_te = y_te.mean()
        mean_acc_tr = y_tr.mean()

        methods_results = {}

        # 1) Raw prob
        thre = find_optimal_threshold(conf_tr, y_tr)
        probs_te = conf_te
        align = float(((probs_te >= thre).astype(int) == y_te).mean())
        ece = compute_ece(probs_te, y_te); brier = compute_brier(probs_te, y_te)
        auc = roc_auc_score(y_te, probs_te) if len(set(y_te)) > 1 else float('nan')
        logloss = compute_logloss(probs_te, y_te)
        sep = probs_te[y_te == 1].mean() - probs_te[y_te == 0].mean() if (y_te == 0).sum() > 0 and (y_te == 1).sum() > 0 else float('nan')
        methods_results['raw'] = (mean_acc_te, probs_te.mean(), probs_te.std(), sep, align, ece, brier, auc, logloss)

        # 2) Temp
        probs_tr_t, probs_te_t, T = run_temperature(conf_tr, y_tr, conf_te)
        thre = find_optimal_threshold(probs_tr_t, y_tr)
        align = float(((probs_te_t >= thre).astype(int) == y_te).mean())
        ece = compute_ece(probs_te_t, y_te); brier = compute_brier(probs_te_t, y_te)
        auc = roc_auc_score(y_te, probs_te_t) if len(set(y_te)) > 1 else float('nan')
        logloss = compute_logloss(probs_te_t, y_te)
        sep = probs_te_t[y_te == 1].mean() - probs_te_t[y_te == 0].mean() if (y_te == 0).sum() > 0 and (y_te == 1).sum() > 0 else float('nan')
        methods_results['temp'] = (mean_acc_te, probs_te_t.mean(), probs_te_t.std(), sep, align, ece, brier, auc, logloss)

        # 3) Platt
        clf = LogisticRegression(max_iter=1000, random_state=SEED)
        clf.fit(conf_tr.reshape(-1, 1), y_tr)
        probs_tr_p = clf.predict_proba(conf_tr.reshape(-1, 1))[:, 1]
        probs_te_p = clf.predict_proba(conf_te.reshape(-1, 1))[:, 1]
        thre = find_optimal_threshold(probs_tr_p, y_tr)
        align = float(((probs_te_p >= thre).astype(int) == y_te).mean())
        ece = compute_ece(probs_te_p, y_te); brier = compute_brier(probs_te_p, y_te)
        auc = roc_auc_score(y_te, probs_te_p) if len(set(y_te)) > 1 else float('nan')
        logloss = compute_logloss(probs_te_p, y_te)
        sep = probs_te_p[y_te == 1].mean() - probs_te_p[y_te == 0].mean() if (y_te == 0).sum() > 0 and (y_te == 1).sum() > 0 else float('nan')
        methods_results['platt'] = (mean_acc_te, probs_te_p.mean(), probs_te_p.std(), sep, align, ece, brier, auc, logloss)

        # 4) Isotonic
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
        iso.fit(conf_tr, y_tr)
        probs_tr_i = iso.predict(conf_tr)
        probs_te_i = iso.predict(conf_te)
        thre = find_optimal_threshold(probs_tr_i, y_tr)
        align = float(((probs_te_i >= thre).astype(int) == y_te).mean())
        ece = compute_ece(probs_te_i, y_te); brier = compute_brier(probs_te_i, y_te)
        auc = roc_auc_score(y_te, probs_te_i) if len(set(y_te)) > 1 else float('nan')
        logloss = compute_logloss(probs_te_i, y_te)
        sep = probs_te_i[y_te == 1].mean() - probs_te_i[y_te == 0].mean() if (y_te == 0).sum() > 0 and (y_te == 1).sum() > 0 else float('nan')
        methods_results['iso'] = (mean_acc_te, probs_te_i.mean(), probs_te_i.std(), sep, align, ece, brier, auc, logloss)

        # 5) MLP 1d
        X_tr_1d = conf_tr.reshape(-1, 1); X_te_1d = conf_te.reshape(-1, 1)
        scaler1 = StandardScaler().fit(X_tr_1d)
        probs_tr_m1, probs_te_m1 = run_mlp_with_train_probs(scaler1.transform(X_tr_1d), y_tr, scaler1.transform(X_te_1d))
        thre = find_optimal_threshold(probs_tr_m1, y_tr)
        align = float(((probs_te_m1 >= thre).astype(int) == y_te).mean())
        ece = compute_ece(probs_te_m1, y_te); brier = compute_brier(probs_te_m1, y_te)
        auc = roc_auc_score(y_te, probs_te_m1) if len(set(y_te)) > 1 else float('nan')
        logloss = compute_logloss(probs_te_m1, y_te)
        sep = probs_te_m1[y_te == 1].mean() - probs_te_m1[y_te == 0].mean() if (y_te == 0).sum() > 0 and (y_te == 1).sum() > 0 else float('nan')
        methods_results['mlp1'] = (mean_acc_te, probs_te_m1.mean(), probs_te_m1.std(), sep, align, ece, brier, auc, logloss)

        # 6) MLP 2d
        gene_coo_tr = np.array([s['gene_coo'] for s in samples_train_balanced])
        gene_coo_te = np.array([s['gene_coo'] for s in samples_test_natural])
        X_tr_2d = np.column_stack([conf_tr, gene_coo_tr])
        X_te_2d = np.column_stack([conf_te, gene_coo_te])
        scaler2 = StandardScaler().fit(X_tr_2d)
        probs_tr_m2, probs_te_m2 = run_mlp_with_train_probs(scaler2.transform(X_tr_2d), y_tr, scaler2.transform(X_te_2d))
        thre = find_optimal_threshold(probs_tr_m2, y_tr)
        align = float(((probs_te_m2 >= thre).astype(int) == y_te).mean())
        ece = compute_ece(probs_te_m2, y_te); brier = compute_brier(probs_te_m2, y_te)
        auc = roc_auc_score(y_te, probs_te_m2) if len(set(y_te)) > 1 else float('nan')
        logloss = compute_logloss(probs_te_m2, y_te)
        sep = probs_te_m2[y_te == 1].mean() - probs_te_m2[y_te == 0].mean() if (y_te == 0).sum() > 0 and (y_te == 1).sum() > 0 else float('nan')
        methods_results['mlp2'] = (mean_acc_te, probs_te_m2.mean(), probs_te_m2.std(), sep, align, ece, brier, auc, logloss)

        for k, label in [('raw', 'Prob (raw)'), ('temp', 'Temp'), ('platt', 'Platt'), ('iso', 'Isotonic'), ('mlp1', 'MLP (conf only)'), ('mlp2', 'MLP (conf+gene_coo)')]:
            r = methods_results[k]
            print(f'{ds:<10} {m:<15} {label:<22} {r[0]:>9.4f} {r[1]:>11.4f} {r[2]:>10.4f} {r[3]:>+9.4f} {r[4] * 100:>7.2f}% {r[5]:>8.4f} {r[6]:>8.4f} {r[7]:>8.4f} {r[8]:>8.4f}')
            agg[k]['ece'].append(r[5]); agg[k]['brier'].append(r[6])
            agg[k]['auc'].append(r[7]); agg[k]['logloss'].append(r[8])
            agg[k]['sep'].append(r[3]); agg[k]['align'].append(r[4])
            agg[k]['mean_acc'].append(r[0]); agg[k]['mean_pred'].append(r[1]); agg[k]['std_pred'].append(r[2])

            # 保存每个 pair 的详细数据
            pair_key = f'{ds}__{m}__{k}'
            per_pair_results[pair_key] = {
                'align': float(r[4]),
                'ece': float(r[5]),
                'brier': float(r[6]),
                'auc': float(r[7]) if not np.isnan(r[7]) else None,
                'logloss': float(r[8]),
                'sep': float(r[3]) if not np.isnan(r[3]) else None,
                'n_train': int(len(samples_train_balanced)),
                'n_test': int(len(samples_test_natural)),
                'mean_acc_train': float(mean_acc_tr),
                'mean_acc_test': float(mean_acc_te),
                'mean_pred': float(r[1]),
                'std_pred': float(r[2]),
            }
        print()

print('\n=== 实验 7 汇总（18 pairs 平均, Setting 3: natural train + natural test）===')
print(f'{"Method":<22} {"test_acc":>9} {"mean(pred)":>11} {"std(pred)":>10} {"Δsep":>9} {"Align":>8} {"ECE":>8} {"Brier":>8} {"AUC":>8} {"LogLoss":>8}')
for name, key in [('Prob (raw)', 'raw'), ('Temp', 'temp'), ('Platt', 'platt'), ('Isotonic', 'iso'), ('MLP (conf only)', 'mlp1'), ('MLP (conf+gene_coo)', 'mlp2')]:
    d = agg[key]
    print(f'{name:<22} {np.mean(d["mean_acc"]):>9.4f} {np.mean(d["mean_pred"]):>11.4f} {np.mean(d["std_pred"]):>10.4f} {np.mean(d["sep"]):>+9.4f} {np.mean(d["align"]) * 100:>7.2f}% {np.mean(d["ece"]):>8.4f} {np.mean(d["brier"]):>8.4f} {np.mean(d["auc"]):>8.4f} {np.mean(d["logloss"]):>8.4f}')

results = {
    'setting': 'natural_train_natural_test',
    'description': 'Setting 3: 50:50 split → train_half (natural, no balance) + test_half (natural)',
    'aggregate': {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in agg.items()},
    'per_pair': per_pair_results,
}
output_path = Path(__file__).with_name('setting3_results.json')
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\n详细结果已保存到 {output_path}')
