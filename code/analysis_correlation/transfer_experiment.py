"""
Transfer 实验：用 (train_dataset, train_model) 训练的 MLP 在 (test_dataset, test_model) 上测试。

回应审稿人意见：
> "any transfer across datasets, models, or relations"

实验类型：
1. Cross-dataset transfer: 同 model 跨 dataset (e.g., Movies → Songs, Movies → Basketball)
2. Cross-model transfer: 同 dataset 跨 model (e.g., Llama-8B → Qwen2.5-32B)
3. Cross-relation transfer: 跨关系类型 (Movies+Songs [人-作品] → Basketball [人-地点])

特征：固定使用 conf + RPop_Ge（即 gene_coo）= 2 维，与主表 Prob+RPop_Ge (MLP) 行一致。
指标：Alignment, Conf_w, ECE 在目标 (test_dataset, test_model) 的 test set 上算。

注：所有 (train, test) 对使用相同的 train/test split (seed=42, stratify) 以保证可复现。
平衡策略与主表一致：若 (dataset, model) 准确率 < 0.25 或 > 0.75，对训练数据做 balance。
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from calibration_experiment import (
    DATASETS, MODELS, RES_TO_SC, RES_TO_VC, RES_TO_POP,
    PATTERN, SEED, TEST_RATIO, BALANCE_THRESHOLD, SINGLE_OCC_THRESHOLD, LLM_POP_RUN,
    read_jsonl, remove_punctuation_edges, load_resources, get_pop,
    load_sc_confidence, load_vc_confidence, load_llm_pop,
    collect_samples, build_feature_matrix, balance_samples,
    compute_ece, compute_conf_wrong, find_optimal_threshold, compute_alignment,
    MLP,
)

# ─── 路径配置 ─────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# This historical transfer implementation predates the consolidated v3 script.
# Seed it explicitly so its retained historical tables can be regenerated.
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


# ─── 训练一个 MLP（在 source pair 上）────────────────────────────────────────
def train_mlp_on_source(source_dataset, source_model, feat_keys,
                        hidden_dims=(128, 64, 32), dropout=0.3,
                        lr=5e-4, epochs=100, batch_size=256, patience=10,
                        full_dict=None, co_occu=None, single_occr=None,
                        sc_conf_dict=None, vc_conf_dict=None,
                        llm_gene_pop_list=None, llm_question_pop_list=None, llm_coo_pop_list=None,
                        balance=True):
    """在 (source_dataset, source_model) 上训练 MLP，返回 (model, scaler, threshold, train_acc)."""
    samples = collect_samples(source_dataset, source_model,
                              full_dict, co_occu, single_occr,
                              sc_conf_dict, vc_conf_dict,
                              llm_gene_pop_list, llm_question_pop_list, llm_coo_pop_list)
    if len(samples) < 50:
        return None

    # 类别平衡（与主表一致）
    pos_ratio = sum(s['acc'] for s in samples) / len(samples)
    if balance and (pos_ratio < BALANCE_THRESHOLD or pos_ratio > 1 - BALANCE_THRESHOLD):
        samples = balance_samples(samples)

    # NaN 过滤
    samples = [s for s in samples
               if all(s.get(k) is not None and
                      not (isinstance(s.get(k), float) and np.isnan(s.get(k)))
                      for k in feat_keys)]
    if len(samples) < 50:
        return None

    X_raw, y = build_feature_matrix(samples, feat_keys)

    if y.mean() < 0.02 or y.mean() > 0.98:
        return None

    # train/test split
    indices = np.arange(len(samples))
    idx_train, idx_test = train_test_split(
        indices, test_size=TEST_RATIO, random_state=SEED, stratify=y
    )
    scaler = StandardScaler().fit(X_raw[idx_train])
    X_train = scaler.transform(X_raw[idx_train])
    X_test = scaler.transform(X_raw[idx_test])
    y_train, y_test = y[idx_train], y[idx_test]

    # 训练 MLP（与 calibration_experiment.run_mlp 一致）
    if len(X_train) > 100:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=SEED, stratify=y_train
        )
    else:
        X_tr, X_val, y_tr, y_val = X_train, X_train, y_train, y_train

    def to_tensor(X, y):
        return (torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32))

    Xtr_t, ytr_t = to_tensor(X_tr, y_tr)
    Xval_t, yval_t = to_tensor(X_val, y_val)

    # 动态调整 batch_size，避免 BatchNorm 在 batch_size=1 时报错（修复 drop_last bug 后的副作用）
    effective_batch_size = min(batch_size, len(X_tr))
    if effective_batch_size < 2:
        effective_batch_size = 2  # 至少 2 个样本才能做 BatchNorm
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=effective_batch_size, shuffle=True)

    model = MLP(X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for Xb, yb in loader:
            if Xb.shape[0] < 2:
                continue  # BatchNorm 需要 batch_size >= 2
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xval_t), yval_t).item()
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # 在 train 上找最优阈值（与 calibration_experiment.run_mlp 一致）
    Xtrain_t = torch.tensor(X_train, dtype=torch.float32)
    with torch.no_grad():
        logits_train = model(Xtrain_t).numpy()
    probs_train = 1 / (1 + np.exp(-np.clip(logits_train, -500, 500)))
    threshold = find_optimal_threshold(probs_train, y_train)

    return {
        'model': model,
        'scaler': scaler,
        'threshold': threshold,
        'n_train': len(X_train),
    }


# ─── 在 (test_dataset, test_model) 的 test set 上评估 MLP ────────────────────
def eval_mlp_on_target(trained, test_dataset, test_model, feat_keys,
                       full_dict=None, co_occu=None, single_occr=None,
                       sc_conf_dict=None, vc_conf_dict=None,
                       llm_gene_pop_list=None, llm_question_pop_list=None, llm_coo_pop_list=None,
                       balance=True):
    """加载训练好的 (model, scaler, threshold)，在 (test_dataset, test_model) 的 test set 上算指标。"""
    samples = collect_samples(test_dataset, test_model,
                              full_dict, co_occu, single_occr,
                              sc_conf_dict, vc_conf_dict,
                              llm_gene_pop_list, llm_question_pop_list, llm_coo_pop_list)
    if len(samples) < 50:
        return None

    # 类别平衡（与主表一致）
    pos_ratio = sum(s['acc'] for s in samples) / len(samples)
    if balance and (pos_ratio < BALANCE_THRESHOLD or pos_ratio > 1 - BALANCE_THRESHOLD):
        samples = balance_samples(samples)

    # NaN 过滤
    samples = [s for s in samples
               if all(s.get(k) is not None and
                      not (isinstance(s.get(k), float) and np.isnan(s.get(k)))
                      for k in feat_keys)]
    if len(samples) < 50:
        return None

    X_raw, y = build_feature_matrix(samples, feat_keys)
    if y.mean() < 0.02 or y.mean() > 0.98:
        return None

    # 用与训练时相同的 split seed 取 test set
    indices = np.arange(len(samples))
    _, idx_test = train_test_split(
        indices, test_size=TEST_RATIO, random_state=SEED, stratify=y
    )
    # 必须用 source 的 scaler 来 transform target 的 raw 数据（避免数据泄漏）
    X_raw_test = X_raw[idx_test]
    y_test = y[idx_test]
    X_test_rescaled = trained['scaler'].transform(X_raw_test)
    n_test = len(idx_test)

    # 用 trained model 预测
    Xte_t = torch.tensor(X_test_rescaled, dtype=torch.float32)
    with torch.no_grad():
        logits_test = trained['model'](Xte_t).numpy()
    probs_test = 1 / (1 + np.exp(-np.clip(logits_test, -500, 500)))

    # 计算 Alignment / Conf_w / ECE
    pred = (probs_test >= trained['threshold']).astype(int)
    alignment = float((pred == y_test).mean())
    conf_w = compute_conf_wrong(probs_test, y_test)
    ece_val = compute_ece(probs_test, y_test)

    return {
        'alignment': round(alignment * 100, 2),
        'conf_w': round(float(conf_w), 4) if not np.isnan(conf_w) else None,
        'ece': round(float(ece_val), 4) if not np.isnan(ece_val) else None,
        'n_test': n_test,
    }


# ─── 收集 source 和 target 数据的工具 ───────────────────────────────────────
def get_pair_data(dataset, model, full_dict, co_occu, single_occr):
    """加载某个 (dataset, model) 的所有辅助数据。返回的 dict key 与 collect_samples 参数名一致。"""
    sc_conf_dict = load_sc_confidence(dataset, model)
    vc_conf_dict = load_vc_confidence(dataset, model)
    llm_gene_pop = load_llm_pop(dataset, model, 'gene_pop')
    llm_qpop = load_llm_pop(dataset, model, 'question_pop')
    llm_coo = load_llm_pop(dataset, model, 'coo_pop')
    return {
        'full_dict': full_dict, 'co_occu': co_occu, 'single_occr': single_occr,
        'sc_conf_dict': sc_conf_dict, 'vc_conf_dict': vc_conf_dict,
        'llm_gene_pop_list': llm_gene_pop,
        'llm_question_pop_list': llm_qpop,
        'llm_coo_pop_list': llm_coo,
    }


def run_transfer_for_feat(feat_keys, label='conf+gene_coo'):
    """对给定特征组合，跑所有 transfer 实验。"""
    print(f"\n{'='*80}")
    print(f"  Transfer experiments: feature = {label} ({feat_keys})")
    print(f"{'='*80}")

    full_dict, co_occu, single_occr = load_resources()

    # 预加载所有 (dataset, model) 的辅助数据
    pair_data = {}
    for ds in DATASETS:
        for m in MODELS:
            res_path = os.path.join(BASE, 'res', ds, f'{ds}_{m}_temperature1.jsonl')
            if not os.path.exists(res_path):
                continue
            pair_data[(ds, m)] = get_pair_data(ds, m, full_dict, co_occu, single_occr)
            print(f"  Loaded aux data for ({ds}, {m})")

    results = {
        'cross_dataset': {},   # 同 model, 跨 dataset
        'cross_model': {},     # 同 dataset, 跨 model
        'cross_relation': {},  # 跨关系类型 (Movies+Songs → Basketball)
        'in_domain': {},      # 对角线，作为 baseline 对照
    }

    # ─── 1. In-domain baseline（对角线）──
    print(f"\n  [In-domain baseline]")
    for ds in DATASETS:
        for m in MODELS:
            if (ds, m) not in pair_data:
                continue
            d = pair_data[(ds, m)]
            trained = train_mlp_on_source(ds, m, feat_keys, **d)
            if trained is None:
                continue
            eval_res = eval_mlp_on_target(trained, ds, m, feat_keys, **d)
            if eval_res is not None:
                results['in_domain'][f'{ds}__{m}'] = eval_res
                print(f"    {ds} × {m}: Align={eval_res['alignment']}%  Cw={eval_res['conf_w']}  ECE={eval_res['ece']}")

    # ─── 2. Cross-dataset transfer ──
    print(f"\n  [Cross-dataset transfer]")
    for m in MODELS:
        for src_ds in DATASETS:
            for tgt_ds in DATASETS:
                if src_ds == tgt_ds:
                    continue
                if (src_ds, m) not in pair_data or (tgt_ds, m) not in pair_data:
                    continue
                src_d = pair_data[(src_ds, m)]
                tgt_d = pair_data[(tgt_ds, m)]
                trained = train_mlp_on_source(src_ds, m, feat_keys, **src_d)
                if trained is None:
                    continue
                eval_res = eval_mlp_on_target(trained, tgt_ds, m, feat_keys, **tgt_d)
                if eval_res is not None:
                    key = f'{src_ds}__{tgt_ds}__{m}'
                    results['cross_dataset'][key] = eval_res
                    print(f"    {src_ds} → {tgt_ds} × {m}: Align={eval_res['alignment']}%  Cw={eval_res['conf_w']}  ECE={eval_res['ece']}")

    # ─── 3. Cross-model transfer ──
    print(f"\n  [Cross-model transfer]")
    for ds in DATASETS:
        for src_m in MODELS:
            for tgt_m in MODELS:
                if src_m == tgt_m:
                    continue
                if (ds, src_m) not in pair_data or (ds, tgt_m) not in pair_data:
                    continue
                src_d = pair_data[(ds, src_m)]
                tgt_d = pair_data[(ds, tgt_m)]
                trained = train_mlp_on_source(ds, src_m, feat_keys, **src_d)
                if trained is None:
                    continue
                eval_res = eval_mlp_on_target(trained, ds, tgt_m, feat_keys, **tgt_d)
                if eval_res is not None:
                    key = f'{ds}__{src_m}__{tgt_m}'
                    results['cross_model'][key] = eval_res
                    print(f"    {ds}: {src_m} → {tgt_m}: Align={eval_res['alignment']}%  Cw={eval_res['conf_w']}  ECE={eval_res['ece']}")

    # ─── 4. Cross-relation transfer (Movies+Songs → Basketball) ──
    print(f"\n  [Cross-relation transfer: Movies+Songs → Basketball]")
    for m in MODELS:
        # 合并 Movies 和 Songs 的 source samples
        if (DATASETS[0], m) not in pair_data or (DATASETS[1], m) not in pair_data:
            continue
        if (DATASETS[2], m) not in pair_data:
            continue

        # 加载 Movies 和 Songs 的 source samples
        src_movies_d = pair_data[(DATASETS[0], m)]
        src_songs_d = pair_data[(DATASETS[1], m)]
        tgt_basket_d = pair_data[(DATASETS[2], m)]

        movies_samples = collect_samples(DATASETS[0], m, **src_movies_d)
        songs_samples = collect_samples(DATASETS[1], m, **src_songs_d)
        combined_samples = movies_samples + songs_samples

        # 类别平衡
        pos_ratio = sum(s['acc'] for s in combined_samples) / len(combined_samples)
        if pos_ratio < BALANCE_THRESHOLD or pos_ratio > 1 - BALANCE_THRESHOLD:
            combined_samples = balance_samples(combined_samples)

        # NaN 过滤
        combined_samples = [s for s in combined_samples
                            if all(s.get(k) is not None and
                                   not (isinstance(s.get(k), float) and np.isnan(s.get(k)))
                                   for k in feat_keys)]
        if len(combined_samples) < 50:
            continue

        # 训练
        X_raw, y = build_feature_matrix(combined_samples, feat_keys)
        if y.mean() < 0.02 or y.mean() > 0.98:
            continue

        indices = np.arange(len(combined_samples))
        idx_train, _ = train_test_split(indices, test_size=TEST_RATIO, random_state=SEED, stratify=y)
        scaler = StandardScaler().fit(X_raw[idx_train])
        X_train = scaler.transform(X_raw[idx_train])
        y_train = y[idx_train]

        # 复用 calibration_experiment.run_mlp 的训练逻辑
        if len(X_train) > 100:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.1, random_state=SEED, stratify=y_train
            )
        else:
            X_tr, X_val, y_tr, y_val = X_train, X_train, y_train, y_train

        Xtr_t = torch.tensor(X_tr, dtype=torch.float32)
        ytr_t = torch.tensor(y_tr, dtype=torch.float32)
        Xval_t = torch.tensor(X_val, dtype=torch.float32)
        yval_t = torch.tensor(y_val, dtype=torch.float32)

        # 动态调整 batch_size，避免 BatchNorm 在 batch_size=1 时报错（修复 drop_last bug 后的副作用）
        effective_batch_size = min(256, len(X_tr))
        if effective_batch_size < 2:
            effective_batch_size = 2  # 至少 2 个样本才能做 BatchNorm
        loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=effective_batch_size, shuffle=True)
        model = MLP(X_train.shape[1], hidden_dims=(128, 64, 32), dropout=0.3)
        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        best_state = None
        no_improve = 0
        for epoch in range(100):
            model.train()
            for Xb, yb in loader:
                if Xb.shape[0] < 2:
                    continue  # BatchNorm 需要 batch_size >= 2
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(Xval_t), yval_t).item()
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 10:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        # 在 train 上找阈值
        Xtrain_t = torch.tensor(X_train, dtype=torch.float32)
        with torch.no_grad():
            logits_train = model(Xtrain_t).numpy()
        probs_train = 1 / (1 + np.exp(-np.clip(logits_train, -500, 500)))
        threshold = find_optimal_threshold(probs_train, y_train)

        # 在 Basketball 上评估
        trained = {'model': model, 'scaler': scaler, 'threshold': threshold, 'n_train': len(X_train)}
        eval_res = eval_mlp_on_target(trained, DATASETS[2], m, feat_keys, **tgt_basket_d)
        if eval_res is not None:
            key = f'Movies+Songs__Basketball__{m}'
            results['cross_relation'][key] = eval_res
            print(f"    Movies+Songs → Basketball × {m}: Align={eval_res['alignment']}%  Cw={eval_res['conf_w']}  ECE={eval_res['ece']}")

    return results


def main():
    # 主特征组合：conf + RPop_Ge（与主表 Prob+RPop_Ge 一致）
    feat_keys = ['conf', 'gene_coo']
    results = run_transfer_for_feat(feat_keys, label='conf+RPop_Ge')

    # 保存结果
    out_path = os.path.join(BASE, 'code', 'analysis_correlation', 'transfer_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")

    # 汇总打印
    print_summary(results)


def print_summary(results):
    """打印 transfer 结果汇总表。"""
    print(f"\n{'='*100}")
    print("  Transfer Summary: Alignment / Conf_w / ECE")
    print(f"{'='*100}")

    # In-domain baseline
    print(f"\n  [In-domain (diagonal) baseline]")
    print(f"  {'Pair':<30} {'Align':>8} {'Cw':>8} {'ECE':>8} {'N':>6}")
    for k, v in sorted(results.get('in_domain', {}).items()):
        print(f"  {k:<30} {v['alignment']:>7.2f}% {v['conf_w']:>8.4f} {v['ece']:>8.4f} {v['n_test']:>6}")

    # Cross-dataset
    print(f"\n  [Cross-dataset transfer]")
    print(f"  {'Transfer':<40} {'Align':>8} {'Cw':>8} {'ECE':>8} {'N':>6}")
    for k, v in sorted(results.get('cross_dataset', {}).items()):
        print(f"  {k:<40} {v['alignment']:>7.2f}% {v['conf_w']:>8.4f} {v['ece']:>8.4f} {v['n_test']:>6}")

    # Cross-model
    print(f"\n  [Cross-model transfer]")
    print(f"  {'Transfer':<45} {'Align':>8} {'Cw':>8} {'ECE':>8} {'N':>6}")
    for k, v in sorted(results.get('cross_model', {}).items()):
        print(f"  {k:<45} {v['alignment']:>7.2f}% {v['conf_w']:>8.4f} {v['ece']:>8.4f} {v['n_test']:>6}")

    # Cross-relation
    print(f"\n  [Cross-relation transfer]")
    print(f"  {'Transfer':<50} {'Align':>8} {'Cw':>8} {'ECE':>8} {'N':>6}")
    for k, v in sorted(results.get('cross_relation', {}).items()):
        print(f"  {k:<50} {v['alignment']:>7.2f}% {v['conf_w']:>8.4f} {v['ece']:>8.4f} {v['n_test']:>6}")


if __name__ == '__main__':
    main()
