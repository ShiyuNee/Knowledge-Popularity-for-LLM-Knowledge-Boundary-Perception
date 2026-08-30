"""
分类实验：利用 confidence 和 popularity 信号预测模型是否答对（acc=0/1）

实验设计（测试时无 ground truth answer，只能用以下特征）：
  - conf        : 模型 token 概率均值
  - question_pop: 问题实体的 Wikidata sitelinks（查询时可得）
  - gene_pop    : 模型生成实体的 Wikidata sitelinks（生成后可查）
  - gene_coo    : question_entity 与生成实体的 Wikipedia 共现次数（生成后可查）

五组实验：
  A. conf 单独                          → 线性回归（阈值搜索）
  B. conf + question_pop                → MLP
  C. conf + gene_pop                    → MLP
  D. conf + gene_coo                    → MLP
  E. conf + question_pop + gene_pop + gene_coo → MLP（全特征）

过滤规则（与分析脚本一致）：
  - Res=None 跳过
  - single_occurrence > 6000 跳过（movies/songs 三实体，basketball 仅 question_entity）
  - 统计文件中找不到实体的样本跳过
  - popularity=No 的样本：question_pop/gene_pop 设为 0（不跳过，模拟实际应用场景）

评估指标：
  - Alignment（预测准确率）
  - AUROC

数据划分：50% train / 50% test，固定 seed=42，每个 dataset × model 独立
"""

import json
import math
import re
import os
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ─── 路径配置 ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES_DIR = os.path.join(BASE, 'res')

POP_PATH   = os.path.join(RES_DIR, 'gt_gene_entity_popularity_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.jsonl')
COO_PATH   = os.path.join(RES_DIR, 'cooccurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')
SINGLE_PATH= os.path.join(RES_DIR, 'single_occurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')

DATASETS = ['movies', 'songs', 'basketball']
MODELS   = ['llama8b', 'qwen2', 'chatgpt', 'Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']

PATTERN = {
    'movies':     'Who is the director of the movie ',
    'songs':      'Who is the performer of the song ',
    'basketball': 'Where is the birthplace of the basketball player '
}

SINGLE_OCC_THRESHOLD = 6000
SEED = 42
TEST_RATIO = 0.5

# 样本平衡：对多数类做 undersample，使正负样本 1:1
# 仅在 pos_ratio < BALANCE_THRESHOLD 或 > 1-BALANCE_THRESHOLD 时触发
BALANCE_THRESHOLD = 0.25  # acc < 25% 或 > 75% 时做平衡


def balance_samples(samples, seed=SEED):
    """
    对样本做 undersample 平衡（多数类随机下采样到与少数类相同数量）。
    返回平衡后的样本列表。
    """
    rng = np.random.default_rng(seed)
    pos = [s for s in samples if s['acc'] == 1]
    neg = [s for s in samples if s['acc'] == 0]
    n_min = min(len(pos), len(neg))
    pos_idx = rng.choice(len(pos), n_min, replace=False)
    neg_idx = rng.choice(len(neg), n_min, replace=False)
    balanced = [pos[i] for i in pos_idx] + [neg[i] for i in neg_idx]
    rng.shuffle(balanced)
    return balanced


# ─── 工具函数 ─────────────────────────────────────────────────────────────────
def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]


def remove_punctuation_edges(s, name='movies'):
    s = s.replace('\n', '').split('(')[0].strip()
    if name == 'basketball':
        s = s.split(',')[0].strip()
    elif len(s) <= 20:
        s = s.split(',')[0].strip()
    return re.sub(r'^[^\w]+|[^\w]+$', '', s).strip()


def load_resources():
    pop_data = read_jsonl(POP_PATH)
    full_dict = {}
    for d in pop_data:
        full_dict.update(d)
    co_occu   = json.loads(open(COO_PATH,    encoding='utf-8').read())
    single_occr = json.loads(open(SINGLE_PATH, encoding='utf-8').read())
    return full_dict, co_occu, single_occr


def get_pop(full_dict, entity):
    """获取实体流行度，找不到或 No 返回 0"""
    info = full_dict.get(entity, {})
    pop  = info.get('popularity', 0) if isinstance(info, dict) else info
    if pop == 'No' or pop is None:
        return 0
    try:
        return int(pop)
    except:
        return 0


def collect_samples(dataset, model, full_dict, co_occu, single_occr):
    """
    收集一个 dataset × model 的所有有效样本。
    返回 list of dict，每个 dict 包含：
      conf, question_pop, gene_pop, gene_coo, acc
    注意：
      - popularity=No 不跳过，设为 0
      - gene_entity 在统计文件中找不到时，gene_pop=0, gene_coo=0
    """
    res_path = os.path.join(RES_DIR, dataset, f'{dataset}_{model}_temperature1.jsonl')
    if not os.path.exists(res_path):
        return []

    model_res = read_jsonl(res_path)
    samples = []

    for item in model_res:
        if not item.get('Res') or item['Res'] is None:
            continue

        question_entity = item['question'].replace(PATTERN[dataset], '').lower()
        gene_entity     = remove_punctuation_edges(item['Res'], dataset)

        # question_entity 必须在统计文件中
        if question_entity not in co_occu:
            continue
        if question_entity not in single_occr:
            continue

        # single_occurrence 过滤（异常高频实体）
        if dataset in ['movies', 'songs']:
            q_single = single_occr.get(question_entity, 0)
            g_single = single_occr.get(gene_entity.lower(), 0)
            if q_single > SINGLE_OCC_THRESHOLD or g_single > SINGLE_OCC_THRESHOLD:
                continue
        else:  # basketball：只过滤 question_entity
            if single_occr.get(question_entity, 0) > SINGLE_OCC_THRESHOLD:
                continue

        # confidence
        if 'gpt' in model.lower():
            probs = [math.exp(t) for t in item['Log_p']['token_logprobs']]
        else:
            probs = item['Log_p']['token_probs']
        conf = sum(probs) / len(probs)

        # question_pop（item 中直接有，No 设为 0）
        qpop_raw = item.get('popularity', 0)
        question_pop = 0 if (qpop_raw == 'No' or qpop_raw is None) else int(qpop_raw)

        # gene_pop（查字典，找不到设为 0）
        gene_pop = get_pop(full_dict, gene_entity)

        # gene_coo（question_entity 与生成实体的共现，找不到设为 0）
        gene_coo = co_occu[question_entity].get(gene_entity.lower(), 0)

        samples.append({
            'conf':         conf,
            'question_pop': question_pop,
            'gene_pop':     gene_pop,
            'gene_coo':     gene_coo,
            'acc':          item['has_answer'],
        })

    return samples


# ─── 特征归一化 ───────────────────────────────────────────────────────────────
def build_feature_matrix(samples, feature_keys):
    """
    构建未标准化的特征矩阵。StandardScaler 在划分后仅用训练集拟合。
    """
    X = np.array([[s[k] for k in feature_keys] for s in samples], dtype=float)
    y = np.array([s['acc'] for s in samples], dtype=int)
    return X, y


# ─── 线性回归（单特征：阈值搜索）────────────────────────────────────────────
def run_linear_threshold(X_train, y_train, X_test, y_test):
    """
    单特征线性分类：在训练集上搜索最优阈值，在测试集上评估。
    X 是 (n,1) 的归一化特征。
    """
    feat_train = X_train[:, 0]
    feat_test  = X_test[:, 0]

    # 搜索最优阈值（在训练集上）
    best_thre, best_align = None, -1
    for thre in np.unique(feat_train):
        pred = (feat_train >= thre).astype(int)
        align = accuracy_score(y_train, pred)
        if align > best_align:
            best_align = align
            best_thre  = thre

    # 测试集评估
    pred_test = (feat_test >= best_thre).astype(int)
    alignment = accuracy_score(y_test, pred_test)

    # AUROC（用原始特征值，不用二值化）
    try:
        auroc = roc_auc_score(y_test, feat_test)
    except:
        auroc = 0.5

    return alignment, auroc


# ─── Temperature Scaling ──────────────────────────────────────────────────────
def run_temperature_scaling(X_train, y_train, X_test, y_test):
    """
    Temperature scaling: 在 confidence 上搜索最优温度 T，使得校准后的 BCE 最小。
    作为标准的单特征后处理校准 baseline。
    """
    feat_train = X_train[:, 0]
    feat_test  = X_test[:, 0]

    # 避免 logit 计算中的数值问题，裁剪到 (0, 1)
    eps = 1e-7
    p_train = np.clip(feat_train, eps, 1 - eps)
    p_test  = np.clip(feat_test, eps, 1 - eps)

    logits_train = np.log(p_train / (1 - p_train))

    best_T, best_loss = 1.0, float('inf')
    for T in np.concatenate([np.linspace(0.1, 2.0, 50), np.linspace(2.0, 5.0, 20)]):
        scaled_logits = logits_train / T
        scaled_probs = 1 / (1 + np.exp(-scaled_logits))
        # BCE
        bce = -np.mean(y_train * np.log(scaled_probs + eps) +
                       (1 - y_train) * np.log(1 - scaled_probs + eps))
        if bce < best_loss:
            best_loss = bce
            best_T = T

    # 测试集评估
    logits_test = np.log(p_test / (1 - p_test))
    scaled_logits_test = logits_test / best_T
    probs = 1 / (1 + np.exp(-scaled_logits_test))
    preds = (probs >= 0.5).astype(int)
    alignment = accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, probs)
    except:
        auroc = 0.5
    return alignment, auroc


# ─── Platt Scaling ────────────────────────────────────────────────────────────
def run_platt_scaling(X_train, y_train, X_test, y_test):
    """
    Platt scaling: 在 confidence 上训练 logistic regression，测试集评估 AUROC。
    作为 conf_only 的 stronger baseline（学习非线性决策边界）。
    """
    feat_train = X_train[:, 0].reshape(-1, 1)
    feat_test  = X_test[:, 0].reshape(-1, 1)

    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(feat_train, y_train)

    probs = clf.predict_proba(feat_test)[:, 1]
    preds = clf.predict(feat_test)
    alignment = accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, probs)
    except:
        auroc = 0.5
    return alignment, auroc


# ─── Logistic Regression（多特征）─────────────────────────────────────────────
def run_logistic_regression(X_train, y_train, X_test, y_test):
    """
    多特征 logistic regression（MLP 的线性对照）。
    """
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)
    alignment = accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, probs)
    except:
        auroc = 0.5
    return alignment, auroc


# ─── MLP 模型 ─────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32), dropout=0.3):
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


def run_mlp(X_train, y_train, X_test, y_test,
            hidden_dims=(64, 32), dropout=0.3,
            lr=1e-3, epochs=100, batch_size=256, patience=15):
    """
    训练 MLP 二分类器，返回 (alignment, auroc)。
    使用 early stopping（基于验证集 loss）。
    """
    # 从训练集中划出 10% 做验证
    if len(X_train) > 100:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=SEED, stratify=y_train
        )
    else:
        X_tr, X_val, y_tr, y_val = X_train, X_train, y_train, y_train

    # 转 tensor
    def to_tensor(X, y):
        return (torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32))

    Xtr_t, ytr_t = to_tensor(X_tr, y_tr)
    Xval_t, yval_t = to_tensor(X_val, y_val)
    Xte_t, yte_t  = to_tensor(X_test, y_test)

    loader = DataLoader(TensorDataset(Xtr_t, ytr_t),
                        batch_size=batch_size, shuffle=True)

    model = MLP(X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_state    = None
    no_improve    = 0

    for epoch in range(epochs):
        model.train()
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xval_t), yval_t).item()
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 测试集评估
    model.eval()
    with torch.no_grad():
        logits = model(Xte_t).numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    alignment = accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, probs)
    except:
        auroc = 0.5

    return alignment, auroc


# ─── 单组实验 ─────────────────────────────────────────────────────────────────
EXPERIMENTS = [
    # (名称,          特征列表,                                    分类器类型)
    # 分类器类型: 'threshold' | 'temp' | 'platt' | 'logreg' | 'mlp'
    ('conf_only',     ['conf'],                                    'threshold'),
    ('conf_temp',     ['conf'],                                    'temp'),
    ('conf_platt',    ['conf'],                                    'platt'),
    ('conf+qpop',     ['conf', 'question_pop'],                    'logreg'),
    ('conf+gene_pop', ['conf', 'gene_pop'],                        'logreg'),
    ('conf+gene_coo', ['conf', 'gene_coo'],                        'logreg'),
    ('conf+all_pop',  ['conf', 'question_pop', 'gene_pop', 'gene_coo'], 'logreg'),
    ('conf+qpop_mlp', ['conf', 'question_pop'],                    'mlp'),
    ('conf+gene_pop_mlp', ['conf', 'gene_pop'],                    'mlp'),
    ('conf+gene_coo_mlp', ['conf', 'gene_coo'],                    'mlp'),
    ('conf+all_pop_mlp',  ['conf', 'question_pop', 'gene_pop', 'gene_coo'], 'mlp'),
]

# MLP 超参（可调）
MLP_PARAMS = dict(hidden_dims=(128, 64, 32), dropout=0.3,
                  lr=5e-4, epochs=100, batch_size=256, patience=10)


def run_one(samples, label=''):
    """对一组样本跑所有实验，返回结果 dict"""
    if len(samples) < 50:
        return None

    pos_ratio = sum(s['acc'] for s in samples) / len(samples)
    need_balance = (pos_ratio < BALANCE_THRESHOLD or pos_ratio > 1 - BALANCE_THRESHOLD)

    results = {}
    for exp_name, feat_keys, classifier in EXPERIMENTS:
        # 样本平衡（在特征构建前做，保证 train/test 都来自平衡后的集合）
        if need_balance:
            used_samples = balance_samples(samples)
        else:
            used_samples = samples

        X_raw, y = build_feature_matrix(used_samples, feat_keys)

        # 检查类别平衡（极端不平衡时 AUROC 无意义）
        pos_ratio_used = y.mean()
        if pos_ratio_used < 0.02 or pos_ratio_used > 0.98:
            results[exp_name] = {'alignment': None, 'auroc': None, 'note': 'class_imbalance'}
            continue

        idx_train, idx_test = train_test_split(
            np.arange(len(y)), test_size=TEST_RATIO, random_state=SEED, stratify=y
        )
        scaler = StandardScaler().fit(X_raw[idx_train])
        X_train = scaler.transform(X_raw[idx_train])
        X_test = scaler.transform(X_raw[idx_test])
        y_train, y_test = y[idx_train], y[idx_test]

        if classifier == 'threshold':
            alignment, auroc = run_linear_threshold(X_train, y_train, X_test, y_test)
        elif classifier == 'temp':
            alignment, auroc = run_temperature_scaling(X_train, y_train, X_test, y_test)
        elif classifier == 'platt':
            alignment, auroc = run_platt_scaling(X_train, y_train, X_test, y_test)
        elif classifier == 'logreg':
            alignment, auroc = run_logistic_regression(X_train, y_train, X_test, y_test)
        elif classifier == 'mlp':
            alignment, auroc = run_mlp(X_train, y_train, X_test, y_test, **MLP_PARAMS)

        results[exp_name] = {
            'alignment': round(alignment * 100, 2),
            'auroc':     round(auroc, 4),
            'balanced':  need_balance,
            'n_used':    len(used_samples),
        }

    return results


# ─── 主函数 ───────────────────────────────────────────────────────────────────
def main():
    print("加载资源文件...")
    full_dict, co_occu, single_occr = load_resources()

    all_results = {}  # {dataset: {model: {exp_name: {alignment, auroc}}}}

    for dataset in DATASETS:
        all_results[dataset] = {}
        for model in MODELS:
            res_path = os.path.join(RES_DIR, dataset, f'{dataset}_{model}_temperature1.jsonl')
            if not os.path.exists(res_path):
                continue

            print(f"\n{'='*60}")
            print(f"  {dataset} × {model}")
            print(f"{'='*60}")

            samples = collect_samples(dataset, model, full_dict, co_occu, single_occr)
            if len(samples) < 50:
                print(f"  样本不足 ({len(samples)})，跳过")
                continue

            acc_rate = sum(s['acc'] for s in samples) / len(samples)
            need_bal = (acc_rate < BALANCE_THRESHOLD or acc_rate > 1 - BALANCE_THRESHOLD)
            print(f"  n={len(samples)}, acc={acc_rate:.3f}{'  [将做样本平衡]' if need_bal else ''}")

            results = run_one(samples, label=f'{dataset}×{model}')
            if results is None:
                continue

            all_results[dataset][model] = results

            # 打印结果
            print(f"\n  {'实验':<20} {'Alignment':>10} {'AUROC':>8}")
            print(f"  {'-'*40}")
            for exp_name, res in results.items():
                if res.get('note'):
                    print(f"  {exp_name:<20} {'skip('+res['note']+')':>20}")
                else:
                    bal_tag = f"  (balanced n={res['n_used']})" if res.get('balanced') else ""
                    print(f"  {exp_name:<20} {res['alignment']:>9.2f}%  {res['auroc']:>7.4f}{bal_tag}")

    # ── 汇总打印 ──────────────────────────────────────────────────────────────
    print_summary(all_results)
    return all_results


def print_summary(all_results):
    """按实验类型打印汇总表"""
    exp_names = [e[0] for e in EXPERIMENTS]

    print("\n\n" + "="*80)
    print("  汇总结果（Alignment %）")
    print("="*80)

    header = f"{'Dataset':<12} {'Model':<16} " + " ".join(f"{e:<16}" for e in exp_names)
    print(header)
    print("-" * len(header))

    for dataset in DATASETS:
        for model in MODELS:
            if model not in all_results.get(dataset, {}):
                continue
            res = all_results[dataset][model]
            row = f"{dataset:<12} {model:<16} "
            for exp_name in exp_names:
                r = res.get(exp_name, {})
                if r.get('note'):
                    row += f"{'skip':<16} "
                elif r.get('alignment') is not None:
                    row += f"{r['alignment']:<16.2f} "
                else:
                    row += f"{'N/A':<16} "
            print(row)

    print("\n" + "="*80)
    print("  汇总结果（AUROC）")
    print("="*80)
    print(header)
    print("-" * len(header))

    for dataset in DATASETS:
        for model in MODELS:
            if model not in all_results.get(dataset, {}):
                continue
            res = all_results[dataset][model]
            row = f"{dataset:<12} {model:<16} "
            for exp_name in exp_names:
                r = res.get(exp_name, {})
                if r.get('note'):
                    row += f"{'skip':<16} "
                elif r.get('auroc') is not None:
                    row += f"{r['auroc']:<16.4f} "
                else:
                    row += f"{'N/A':<16} "
            print(row)

    # ── 按实验计算各数据集平均提升 ──────────────────────────────────────────
    print("\n\n" + "="*80)
    print("  各实验相对 conf_only 的平均 Alignment 提升（百分点）")
    print("="*80)
    for dataset in DATASETS:
        gains = {e: [] for e in exp_names[1:]}
        for model in MODELS:
            if model not in all_results.get(dataset, {}):
                continue
            res = all_results[dataset][model]
            base = res.get('conf_only', {}).get('alignment')
            if base is None:
                continue
            for exp_name in exp_names[1:]:
                r = res.get(exp_name, {})
                if r.get('alignment') is not None:
                    gains[exp_name].append(r['alignment'] - base)
        print(f"\n  {dataset}:")
        for exp_name, g in gains.items():
            if g:
                print(f"    {exp_name:<20}: avg gain = {np.mean(g):+.2f} pp  (range [{min(g):+.2f}, {max(g):+.2f}])")


if __name__ == '__main__':
    main()
