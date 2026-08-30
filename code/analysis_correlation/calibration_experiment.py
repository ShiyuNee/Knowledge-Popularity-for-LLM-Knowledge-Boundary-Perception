"""
校准实验 v2：Alignment (%) 作为唯一评估指标

评估指标：Alignment
  - 在 train (dev) 集上对信号找最优二值化阈值
  - 在 test 集上用该阈值二值化，计算与 correctness 的匹配率
  - Alignment = accuracy_score(y_test, pred_test)

特征说明：
  - conf:            模型 token 概率均值（原始 confidence）
  - question_pop:    问题实体 Wikidata sitelinks（外部）
  - gene_pop:        生成实体 Wikidata sitelinks（外部）
  - gene_coo:        问题实体与生成实体 Wikipedia 共现次数（外部）
  - sc_conf:         Self-Consistency confidence（n_consistent/n_sampled）
  - vc_conf:         Verbalized confidence（0/1 二值：certain=1, uncertain=0）
  - llm_question_pop: 模型自评的问题实体熟悉度（1-10）
  - llm_gene_pop:    模型自评的生成实体熟悉度（1-10）
  - llm_coo_pop:     模型自评的问题-生成实体关系熟悉度（1-10）

过滤规则（校准场景，不过滤 popularity 查不到的数据）：
  - Res=None 跳过
  - single_occurrence > 6000 跳过（模糊高频实体名，质量过滤）
  - popularity/coo 找不到 → 设为 0（不跳过，符合实际部署场景）

分类器：Threshold / Temperature Scaling / Platt Scaling / MLP（无 LR）
"""

import json
import math
import re
import os
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.isotonic import IsotonicRegression
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
SC_DIR  = os.path.join(BASE, 'baselines', 'self_consistency', 'consis_judge_res')
VC_DIR  = os.path.join(BASE, 'baselines', 'verbalized_confidence')
LLM_POP_DIR = os.path.join(BASE, 'llm_pop_generation')

POP_PATH    = os.path.join(RES_DIR, 'gt_gene_entity_popularity_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.jsonl')
COO_PATH    = os.path.join(RES_DIR, 'cooccurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')
SINGLE_PATH = os.path.join(RES_DIR, 'single_occurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')

DATASETS = ['movies', 'songs', 'basketball']
MODELS   = ['llama8b', 'qwen2', 'chatgpt', 'Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']

PATTERN = {
    'movies':     'Who is the director of the movie ',
    'songs':      'Who is the performer of the song ',
    'basketball': 'Where is the birthplace of the basketball player '
}

# 模型名映射：res 文件名 → SC 文件名
RES_TO_SC = {
    'llama8b':    'llama3-8b',
    'qwen2':      'qwen2',
    'chatgpt':    'chatgpt',
    'Qwen2.5-7B':  'qwen2.5-7b',
    'Qwen2.5-14B': 'qwen2.5-14b',
    'Qwen2.5-32B': 'qwen2.5-32b',
}

# 模型名映射：res 文件名 → LLM pop 文件名
RES_TO_POP = {
    'llama8b':    'llama8b',
    'qwen2':      'qwen2',
    'chatgpt':    'chatgpt',
    'Qwen2.5-7B':  'qwen25-7b',
    'Qwen2.5-14B': 'qwen25-14b',
    'Qwen2.5-32B': 'qwen25-32b',
}

# 模型名映射：res 文件名 → VC 文件名
RES_TO_VC = {
    'llama8b':    'llama3-8b',
    'qwen2':      'qwen2',
    'chatgpt':    'chatgpt',
    'Qwen2.5-7B':  'qwen2.5-7b',
    'Qwen2.5-14B': 'qwen2.5-14b',
    'Qwen2.5-32B': 'qwen2.5-32b',
}

SINGLE_OCC_THRESHOLD = 6000
SEED = 42
TEST_RATIO = 0.5
BALANCE_THRESHOLD = 0.25
LLM_POP_RUN = '0'  # 使用 _0 后缀（temperature=0，确定性输出）

# Make MLP initialization, dropout, and DataLoader shuffling reproducible.
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


# ─── 样本平衡 ─────────────────────────────────────────────────────────────────
def balance_samples(samples, seed=SEED):
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
    co_occu    = json.loads(open(COO_PATH,    encoding='utf-8').read())
    single_occr = json.loads(open(SINGLE_PATH, encoding='utf-8').read())
    return full_dict, co_occu, single_occr


def get_pop(full_dict, entity):
    info = full_dict.get(entity, {})
    pop  = info.get('popularity', 0) if isinstance(info, dict) else info
    if pop == 'No' or pop is None:
        return 0
    try:
        return int(pop)
    except:
        return 0


def load_sc_confidence(dataset, model):
    sc_key = RES_TO_SC.get(model)
    if not sc_key:
        return {}
    sc_path = os.path.join(SC_DIR, f'{dataset}_{sc_key}_sc.jsonl')
    if not os.path.exists(sc_path):
        return {}
    sc_data = read_jsonl(sc_path)
    return {item['question']: item['sc_confidence'] for item in sc_data}


def load_llm_pop(dataset, model, pop_type, run_suffix=LLM_POP_RUN):
    pop_key = RES_TO_POP.get(model)
    if not pop_key:
        return []
    pop_path = os.path.join(LLM_POP_DIR, dataset,
                            f'{dataset}_{pop_key}_{pop_type}_{run_suffix}.jsonl')
    if not os.path.exists(pop_path):
        return []
    data = read_jsonl(pop_path)
    result = []
    for item in data:
        try:
            result.append(int(item['Res']))
        except (ValueError, TypeError):
            result.append(0)
    return result


def load_vc_confidence(dataset, model):
    vc_key = RES_TO_VC.get(model)
    if not vc_key:
        return {}
    vc_path = os.path.join(VC_DIR, f'{dataset}_{vc_key}_vc.jsonl')
    if not os.path.exists(vc_path):
        return {}
    vc_data = read_jsonl(vc_path)
    return {item['question']: item['vc_confidence'] for item in vc_data}


# ─── 数据收集 ─────────────────────────────────────────────────────────────────
def collect_samples(dataset, model, full_dict, co_occu, single_occr,
                    sc_conf_dict, vc_conf_dict, llm_gene_pop_list, llm_question_pop_list, llm_coo_pop_list):
    res_path = os.path.join(RES_DIR, dataset, f'{dataset}_{model}_temperature1.jsonl')
    if not os.path.exists(res_path):
        return []

    model_res = read_jsonl(res_path)
    samples = []
    has_sc = bool(sc_conf_dict)
    has_vc = bool(vc_conf_dict)
    has_llm_gene = len(llm_gene_pop_list) > 0
    has_llm_qpop = len(llm_question_pop_list) > 0
    has_llm_coo  = len(llm_coo_pop_list) > 0

    for i, item in enumerate(model_res):
        if not item.get('Res') or item['Res'] is None:
            continue

        question_entity = item['question'].replace(PATTERN[dataset], '').lower()
        gene_entity     = remove_punctuation_edges(item['Res'], dataset)

        # single_occurrence 过滤
        if dataset in ['movies', 'songs']:
            q_single = single_occr.get(question_entity, 0)
            g_single = single_occr.get(gene_entity.lower(), 0)
            if q_single > SINGLE_OCC_THRESHOLD or g_single > SINGLE_OCC_THRESHOLD:
                continue
        else:
            if single_occr.get(question_entity, 0) > SINGLE_OCC_THRESHOLD:
                continue

        # confidence
        if 'gpt' in model.lower():
            probs = [math.exp(t) for t in item['Log_p']['token_logprobs']]
        else:
            probs = item['Log_p']['token_probs']
        conf = sum(probs) / len(probs)

        # 外部特征（找不到 → 0，不跳过）
        qpop_raw = item.get('popularity', 0)
        question_pop = 0 if (qpop_raw == 'No' or qpop_raw is None) else int(qpop_raw)
        gene_pop = get_pop(full_dict, gene_entity)
        gene_coo = co_occu.get(question_entity, {}).get(gene_entity.lower(), 0)

        # Self-Consistency confidence
        sc_conf = sc_conf_dict.get(item['question'], np.nan) if has_sc else np.nan

        # Verbalized Confidence
        vc_conf = vc_conf_dict.get(item['question'], np.nan) if has_vc else np.nan

        # LLM-generated popularity
        llm_gpop = llm_gene_pop_list[i] if has_llm_gene and i < len(llm_gene_pop_list) else np.nan
        llm_qpop = llm_question_pop_list[i] if has_llm_qpop and i < len(llm_question_pop_list) else np.nan
        llm_coo  = llm_coo_pop_list[i] if has_llm_coo and i < len(llm_coo_pop_list) else np.nan

        samples.append({
            'conf':         conf,
            'question_pop': question_pop,
            'gene_pop':     gene_pop,
            'gene_coo':     gene_coo,
            'sc_conf':      sc_conf,
            'vc_conf':      vc_conf,
            'llm_gene_pop': llm_gpop,
            'llm_question_pop': llm_qpop,
            'llm_coo_pop':  llm_coo,
            'acc':          item['has_answer'],
        })

    return samples


# ─── 特征归一化 ───────────────────────────────────────────────────────────────
def build_feature_matrix(samples, feature_keys):
    X = np.array([[s.get(k, 0) for k in feature_keys] for s in samples], dtype=float)
    X = np.nan_to_num(X, nan=0.0)
    y = np.array([s['acc'] for s in samples], dtype=int)
    return X, y


# ─── Alignment 核心：在 train 上找最优阈值 ─────────────────────────────────────
def find_optimal_threshold(signal_train, y_train):
    """在 train 集上搜索最优二值化阈值，返回使 alignment 最大的阈值。"""
    best_thre, best_acc = None, -1
    for thre in np.unique(signal_train):
        pred = (signal_train >= thre).astype(int)
        acc = accuracy_score(y_train, pred)
        if acc > best_acc:
            best_acc = acc
            best_thre = thre
    # 如果所有值都一样（如 VC 二值信号），取中位数
    if best_thre is None:
        best_thre = np.median(signal_train)
    return best_thre


def compute_alignment(signal_test, y_test, threshold):
    """用给定阈值在 test 集上计算 Alignment。"""
    pred = (signal_test >= threshold).astype(int)
    return accuracy_score(y_test, pred)


def compute_ece(probs, accs, n_bins=10):
    """计算 Expected Calibration Error (ECE)。
    probs: 校准后概率 [0,1]
    accs:  正确性 0/1
    """
    probs = np.asarray(probs, dtype=float)
    accs = np.asarray(accs, dtype=float)
    n = len(probs)
    if n == 0:
        return float('nan')
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
        else:
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece_val += (mask.sum() / n) * abs(accs[mask].mean() - probs[mask].mean())
    return ece_val


def compute_conf_wrong(probs, accs):
    """计算 Conf_w：错误样本上校准后概率的均值。"""
    probs = np.asarray(probs, dtype=float)
    accs = np.asarray(accs, dtype=float)
    wrong_mask = (accs == 0)
    if wrong_mask.sum() == 0:
        return float('nan')
    return probs[wrong_mask].mean()


# ─── 分类器 ────────────────────────────────────────────────────────────────────
def run_threshold(X_train, y_train, X_test, y_test):
    """单信号最优阈值：在 train 上找最优阈值，在 test 上算 alignment。
    返回 (alignment, probs_test)：probs_test = signal_test（原始信号作为概率）。
    """
    signal_train = X_train[:, 0]
    signal_test  = X_test[:, 0]
    thre = find_optimal_threshold(signal_train, y_train)
    alignment = compute_alignment(signal_test, y_test, thre)
    return alignment, signal_test


def run_temperature_scaling(X_train, y_train, X_test, y_test):
    """Temperature scaling：在 train 上拟合 T，在 train 上找最优阈值，在 test 上算 alignment。
    返回 (alignment, scaled_probs_test)。
    """
    signal_train = X_train[:, 0]
    signal_test  = X_test[:, 0]
    eps = 1e-7
    p_train = np.clip(signal_train, eps, 1 - eps)
    p_test  = np.clip(signal_test, eps, 1 - eps)

    # 拟合最优 T
    logits_train = np.log(p_train / (1 - p_train))
    best_T, best_loss = 1.0, float('inf')
    for T in np.concatenate([np.linspace(0.1, 2.0, 50), np.linspace(2.0, 5.0, 20)]):
        scaled_logits = logits_train / T
        scaled_probs = 1 / (1 + np.exp(-scaled_logits))
        bce = -np.mean(y_train * np.log(scaled_probs + eps) +
                       (1 - y_train) * np.log(1 - scaled_probs + eps))
        if bce < best_loss:
            best_loss = bce
            best_T = T

    # 在 train 上得到校准后概率，找最优阈值
    scaled_probs_train = 1 / (1 + np.exp(-logits_train / best_T))
    thre = find_optimal_threshold(scaled_probs_train, y_train)

    # 在 test 上算 alignment
    logits_test = np.log(p_test / (1 - p_test))
    scaled_probs_test = 1 / (1 + np.exp(-logits_test / best_T))
    alignment = compute_alignment(scaled_probs_test, y_test, thre)
    return alignment, scaled_probs_test


def run_platt_scaling(X_train, y_train, X_test, y_test):
    """Platt scaling：在 train 上拟合 LR，在 train 概率上找最优阈值，在 test 上算 alignment。
    返回 (alignment, probs_test)。
    """
    signal_train = X_train[:, 0].reshape(-1, 1)
    signal_test  = X_test[:, 0].reshape(-1, 1)
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(signal_train, y_train)

    # 在 train 上得到校准后概率，找最优阈值
    probs_train = clf.predict_proba(signal_train)[:, 1]
    thre = find_optimal_threshold(probs_train, y_train)

    # 在 test 上算 alignment
    probs_test = clf.predict_proba(signal_test)[:, 1]
    alignment = compute_alignment(probs_test, y_test, thre)
    return alignment, probs_test


def run_isotonic_scaling(X_train, y_train, X_test, y_test):
    """Isotonic regression：在 train 上拟合单调映射，在 train 概率上找最优阈值，在 test 上算 alignment。
    返回 (alignment, probs_test)。
    """
    signal_train = X_train[:, 0]
    signal_test  = X_test[:, 0]
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(signal_train, y_train)

    # 在 train 上得到校准后概率，找最优阈值
    probs_train = iso.predict(signal_train)
    thre = find_optimal_threshold(probs_train, y_train)

    # 在 test 上算 alignment
    probs_test = iso.predict(signal_test)
    alignment = compute_alignment(probs_test, y_test, thre)
    return alignment, probs_test


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
            hidden_dims=(128, 64, 32), dropout=0.3,
            lr=5e-4, epochs=100, batch_size=256, patience=10,
            X_full=None):
    """MLP：内部 train/val split 用于 early stopping，在完整 train 上找最优阈值，在 test 上算 alignment。
    如果传入 X_full，则同时返回全样本上的预测概率（用于计算 Conf_w/ECE）。
    返回 (alignment, probs_test, probs_full)。
    """
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

    # 在完整 X_train 上预测，找最优阈值
    Xtrain_t = torch.tensor(X_train, dtype=torch.float32)
    with torch.no_grad():
        logits_train = model(Xtrain_t).numpy().astype(np.float64)
    probs_train = 1 / (1 + np.exp(-np.clip(logits_train, -500, 500)))
    thre = find_optimal_threshold(probs_train, y_train)

    # 在 test 上预测，算 alignment
    Xte_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        logits_test = model(Xte_t).numpy().astype(np.float64)
    probs_test = 1 / (1 + np.exp(-np.clip(logits_test, -500, 500)))
    alignment = compute_alignment(probs_test, y_test, thre)

    # 在全样本上预测（用于计算 Conf_w/ECE）
    if X_full is not None:
        Xfull_t = torch.tensor(X_full, dtype=torch.float32)
        with torch.no_grad():
            logits_full = model(Xfull_t).numpy().astype(np.float64)
        probs_full = 1 / (1 + np.exp(-np.clip(logits_full, -500, 500)))
    else:
        probs_full = None

    return alignment, probs_test, probs_full


# ─── 实验配置 ─────────────────────────────────────────────────────────────────
# 表格行（9 行）+ 补充实验（文字讨论用）
# 无 LR，只用 MLP

EXPERIMENTS = [
    # ── Table Part 1: Baseline confidence signals ──
    ('conf_only',            ['conf'],                                    'threshold'),
    ('sc_only',              ['sc_conf'],                                 'threshold'),
    ('vc_only',              ['vc_conf'],                                 'threshold'),

    # ── Table Part 2: Rescaling baselines (conf only) ──
    ('conf_temp',            ['conf'],                                    'temp'),
    ('conf_platt',           ['conf'],                                    'platt'),
    ('conf_isotonic',        ['conf'],                                    'isotonic'),
    # Same-capacity MLP baseline (conf only, MLP) — addresses reviewer concern about MLP capacity confound
    ('conf_mlp',             ['conf'],                                    'mlp'),

    # ── Table Part 3: External popularity + MLP ──
    ('conf+gene_coo_mlp',    ['conf', 'gene_coo'],                        'mlp'),
    ('conf+all_ext_mlp',     ['conf', 'question_pop', 'gene_pop', 'gene_coo'], 'mlp'),

    # ── Table Part 4: LLM-generated popularity + MLP ──
    ('conf+llm_coo_pop_mlp', ['conf', 'llm_coo_pop'],                     'mlp'),
    ('conf+llm_all_mlp',     ['conf', 'llm_question_pop', 'llm_gene_pop', 'llm_coo_pop'], 'mlp'),

    # ── 补充实验（文字讨论用）──
    # SC/VC + ext pop + MLP
    ('sc+gene_coo_mlp',      ['sc_conf', 'gene_coo'],                     'mlp'),
    ('sc+all_ext_mlp',       ['sc_conf', 'question_pop', 'gene_pop', 'gene_coo'], 'mlp'),
    ('vc+gene_coo_mlp',      ['vc_conf', 'gene_coo'],                     'mlp'),
    ('vc+all_ext_mlp',       ['vc_conf', 'question_pop', 'gene_pop', 'gene_coo'], 'mlp'),
    # conf+sc combination
    ('conf+sc',              ['conf', 'sc_conf'],                          'threshold'),
    ('conf+sc+gene_coo_mlp', ['conf', 'sc_conf', 'gene_coo'],             'mlp'),
    ('conf+sc+all_ext_mlp',  ['conf', 'sc_conf', 'question_pop', 'gene_pop', 'gene_coo'], 'mlp'),
    # Hybrid: LLM + external RPop_Ge
    ('conf+llm_all+gene_coo_mlp', ['conf', 'llm_question_pop', 'llm_gene_pop', 'llm_coo_pop', 'gene_coo'], 'mlp'),
    # Per-feature standalone threshold
    ('question_pop_only',    ['question_pop'],                            'threshold'),
    ('gene_pop_only',        ['gene_pop'],                                'threshold'),
    ('gene_coo_only',        ['gene_coo'],                                'threshold'),
    # Per-feature + Conf (MLP)
    ('conf+question_pop_mlp',['conf', 'question_pop'],                    'mlp'),
    ('conf+gene_pop_mlp',    ['conf', 'gene_pop'],                        'mlp'),
    ('conf+llm_gene_pop_mlp',['conf', 'llm_gene_pop'],                    'mlp'),
    # SC/VC rescaling baselines
    ('sc_temp',              ['sc_conf'],                                 'temp'),
    ('sc_platt',             ['sc_conf'],                                 'platt'),
    ('vc_temp',              ['vc_conf'],                                 'temp'),
    ('vc_platt',             ['vc_conf'],                                 'platt'),
]


def _run_all_experiments(samples, skip_experiments=None):
    """在给定 samples 上跑所有实验，返回每个实验的指标 dict。
    samples 应该已经过 NaN 过滤和（可选的）balance 处理。
    """
    if len(samples) < 50:
        return None

    pos_ratio = sum(s['acc'] for s in samples) / len(samples)
    results = {}
    skip = set(skip_experiments or [])

    for exp_name, feat_keys, classifier in EXPERIMENTS:
        if exp_name in skip:
            continue

        # 过滤含 NaN 特征的样本（而非跳过整个实验）
        filtered_samples = [s for s in samples
                            if all(s.get(k) is not None and
                                   not (isinstance(s.get(k), float) and np.isnan(s.get(k)))
                                   for k in feat_keys)]
        if len(filtered_samples) < 50:
            results[exp_name] = {'alignment': None, 'note': 'too_few_after_nan_filter'}
            continue

        used_samples = filtered_samples
        X_raw, y = build_feature_matrix(used_samples, feat_keys)

        pos_ratio_used = y.mean()
        if pos_ratio_used < 0.02 or pos_ratio_used > 0.98:
            results[exp_name] = {'alignment': None, 'note': 'class_imbalance'}
            continue

        # 保留 raw signal（未标准化），用于 threshold + (conf/sc_conf/vc_conf) 计算 Conf_w/ECE
        raw_signal_full = np.array([s.get(feat_keys[0], 0) for s in used_samples], dtype=float) if len(feat_keys) == 1 else None

        # 用同样的 indices 划分 train/test，保证 raw_signal 与 X 对齐
        indices = np.arange(len(used_samples))
        idx_train, idx_test = train_test_split(
            indices, test_size=TEST_RATIO, random_state=SEED, stratify=y
        )
        # MLP 的 scaler 只在 training split 上拟合，避免 test-distribution leakage。
        # threshold/temp/platt/isotonic 使用原始信号。
        if classifier == 'mlp':
            scaler = StandardScaler().fit(X_raw[idx_train])
            X_train = scaler.transform(X_raw[idx_train])
            X_test = scaler.transform(X_raw[idx_test])
        else:  # threshold / temp / platt / isotonic 用 raw
            X_train, X_test = X_raw[idx_train], X_raw[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]
        raw_signal_test = raw_signal_full[idx_test] if raw_signal_full is not None else None

        if classifier == 'threshold':
            alignment, probs_test = run_threshold(X_train, y_train, X_test, y_test)
        elif classifier == 'temp':
            alignment, probs_test = run_temperature_scaling(X_train, y_train, X_test, y_test)
        elif classifier == 'platt':
            alignment, probs_test = run_platt_scaling(X_train, y_train, X_test, y_test)
        elif classifier == 'isotonic':
            alignment, probs_test = run_isotonic_scaling(X_train, y_train, X_test, y_test)
        elif classifier == 'mlp':
            alignment, probs_test, _ = run_mlp(
                X_train, y_train, X_test, y_test,
                X_full=None  # 不需要全样本预测
            )

        # 计算 Conf_w 和 ECE：用 test set（与 Alignment 同一个数据，确保公平）
        # - threshold + (conf/sc_conf/vc_conf) → 用 raw_signal_test
        # - temp/platt/isotonic/mlp → 用校准后 probs_test
        # - threshold + popularity → 不算（设为 None）
        is_threshold_conf = (
            classifier == 'threshold' and len(feat_keys) == 1 and feat_keys[0] in ('conf', 'sc_conf', 'vc_conf')
        )
        if is_threshold_conf:
            probs_for_ece = np.clip(raw_signal_test, 0, 1)
        elif classifier in ('temp', 'platt', 'isotonic', 'mlp'):
            probs_for_ece = np.clip(probs_test, 0, 1)
        else:
            probs_for_ece = None

        if probs_for_ece is not None:
            conf_w = compute_conf_wrong(probs_for_ece, y_test)
            ece_val = compute_ece(probs_for_ece, y_test)
        else:
            conf_w = None
            ece_val = None

        results[exp_name] = {
            'alignment': round(alignment * 100, 2),
            'conf_w':    round(float(conf_w), 4) if conf_w is not None and not np.isnan(conf_w) else None,
            'ece':       round(float(ece_val), 4) if ece_val is not None and not np.isnan(ece_val) else None,
            'n_used':    len(used_samples),
        }

    return results


def run_one(samples, label='', skip_experiments=None):
    """对一组样本跑所有实验，返回结果 dict。

    当 (dataset, model) 准确率偏离 [0.25, 0.75] 时，同时报告两套指标：
    - 主表（balanced）：class-balanced test set 上的 Alignment/Conf_w/ECE
    - 附录（natural）：natural-distribution test set 上的 Alignment/Conf_w/ECE
    """
    if len(samples) < 50:
        return None

    pos_ratio = sum(s['acc'] for s in samples) / len(samples)
    need_balance = (pos_ratio < BALANCE_THRESHOLD or pos_ratio > 1 - BALANCE_THRESHOLD)

    # 主实验：在 balanced samples 上跑（与论文 Table 6 一致）
    if need_balance:
        balanced_samples = balance_samples(samples)
        balanced_results = _run_all_experiments(balanced_samples, skip_experiments)
    else:
        balanced_results = _run_all_experiments(samples, skip_experiments)

    if balanced_results is None:
        return None

    # 同时计算 natural-distribution 指标（不 balance）
    # 仅在 need_balance=True 时计算（need_balance=False 时两者相同）
    natural_results = None
    if need_balance:
        natural_results = _run_all_experiments(samples, skip_experiments)

    # 合并结果：把 balanced 指标作为主指标，natural 指标作为 natural_* 字段
    results = {}
    for exp_name, res in balanced_results.items():
        if res.get('alignment') is None:
            results[exp_name] = res
            continue

        merged = dict(res)
        merged['balanced'] = need_balance

        if natural_results is not None and exp_name in natural_results:
            nat = natural_results[exp_name]
            if nat.get('alignment') is not None:
                merged['natural_alignment'] = nat['alignment']
                merged['natural_conf_w'] = nat['conf_w']
                merged['natural_ece'] = nat['ece']
                merged['natural_n_used'] = nat['n_used']
            else:
                merged['natural_alignment'] = None
                merged['natural_note'] = nat.get('note', 'skip')
        # need_balance=False 时，natural == balanced，不再重复存储

        results[exp_name] = merged

    return results


# ─── 汇总打印 ─────────────────────────────────────────────────────────────────
def print_summary(all_results):
    """打印汇总表"""
    exp_names = [e[0] for e in EXPERIMENTS]

    # 主要表格行（12 行）
    table_rows = [
        'conf_only', 'sc_only', 'vc_only',
        'conf_temp', 'conf_platt', 'conf_isotonic', 'conf_mlp',
        'conf+gene_coo_mlp', 'conf+all_ext_mlp',
        'conf+llm_coo_pop_mlp', 'conf+llm_all_mlp',
    ]

    print(f"\n{'='*100}")
    print("  Main Table: Alignment (%) — averaged over open-source models")
    print(f"{'='*100}")

    header = f"{'Method':<35} {'Movies':>8} {'Songs':>8} {'Bask.':>8}"
    print(header)
    print("-" * len(header))

    # 行标签映射
    LABELS = {
        'conf_only':              'Conf only (Threshold)',
        'sc_only':                'SC only (Threshold)',
        'vc_only':                'VC only (Threshold)',
        'conf_temp':              'Conf + Temp Scaling',
        'conf_platt':             'Conf + Platt Scaling',
        'conf_isotonic':          'Conf + Isotonic',
        'conf_mlp':               'Conf (MLP)',
        'conf+gene_coo_mlp':      'Conf + RPop_Ge (MLP)',
        'conf+all_ext_mlp':       'Conf + All ext (MLP)',
        'conf+llm_coo_pop_mlp':   'Conf + LLM RPop_Ge (MLP)',
        'conf+llm_all_mlp':       'Conf + All LLM (MLP)',
    }

    for dataset in DATASETS:
        for exp_name in table_rows:
            vals = []
            for model in MODELS:
                if model == 'chatgpt':
                    continue  # 排除 chatgpt，只算 open-source 平均
                r = all_results.get(dataset, {}).get(model, {}).get(exp_name, {})
                if r.get('alignment') is not None:
                    vals.append(r['alignment'])
            avg = np.mean(vals) if vals else float('nan')
            label = LABELS.get(exp_name, exp_name)
            print(f"  {label:<33} {avg:>7.2f}%  ", end='')
            # 仅第一个 dataset 打印行标签
        print()

    # 按 dataset 分列打印
    for exp_name in table_rows:
        label = LABELS.get(exp_name, exp_name)
        row = f"  {label:<33}"
        for dataset in DATASETS:
            vals = []
            for model in MODELS:
                if model == 'chatgpt':
                    continue
                r = all_results.get(dataset, {}).get(model, {}).get(exp_name, {})
                if r.get('alignment') is not None:
                    vals.append(r['alignment'])
            avg = np.mean(vals) if vals else float('nan')
            row += f" {avg:>7.2f}%"
        print(row)

    # 补充实验
    print(f"\n{'='*100}")
    print("  Supplementary: Alignment (%) — averaged over open-source models")
    print(f"{'='*100}")

    for exp_name in exp_names:
        if exp_name in table_rows:
            continue
        row = f"  {exp_name:<35}"
        for dataset in DATASETS:
            vals = []
            for model in MODELS:
                if model == 'chatgpt':
                    continue
                r = all_results.get(dataset, {}).get(model, {}).get(exp_name, {})
                if r.get('alignment') is not None:
                    vals.append(r['alignment'])
            avg = np.mean(vals) if vals else float('nan')
            if np.isnan(avg):
                row += f" {'N/A':>7}"
            else:
                row += f" {avg:>6.2f}%"
        print(row)

    # 每 model 细节
    print(f"\n{'='*100}")
    print("  Per-model details: Alignment (%)")
    print(f"{'='*100}")
    for dataset in DATASETS:
        print(f"\n  {dataset}:")
        for model in MODELS:
            print(f"    {model}:")
            for exp_name in table_rows:
                r = all_results.get(dataset, {}).get(model, {}).get(exp_name, {})
                label = LABELS.get(exp_name, exp_name)
                if r.get('alignment') is not None:
                    bal_tag = " (bal)" if r.get('balanced') else ""
                    print(f"      {label:<33} {r['alignment']:>6.2f}%{bal_tag}")
                else:
                    print(f"      {label:<33} {'skip('+str(r.get('note','?'))+')'}")


# ─── 主函数 ───────────────────────────────────────────────────────────────────
def main():
    print("加载外部资源文件...")
    full_dict, co_occu, single_occr = load_resources()

    all_results = {}

    for dataset in DATASETS:
        all_results[dataset] = {}
        for model in MODELS:
            res_path = os.path.join(RES_DIR, dataset, f'{dataset}_{model}_temperature1.jsonl')
            if not os.path.exists(res_path):
                continue

            print(f"\n{'='*60}")
            print(f"  {dataset} × {model}")
            print(f"{'='*60}")

            # 加载 SC confidence
            sc_conf_dict = load_sc_confidence(dataset, model)
            has_sc = bool(sc_conf_dict)

            # 加载 Verbalized Confidence
            vc_conf_dict = load_vc_confidence(dataset, model)
            has_vc = bool(vc_conf_dict)

            # 加载 LLM-generated popularity
            llm_gene_pop = load_llm_pop(dataset, model, 'gene_pop')
            llm_qpop = load_llm_pop(dataset, model, 'question_pop')
            llm_coo  = load_llm_pop(dataset, model, 'coo_pop')
            has_llm = bool(llm_gene_pop)

            print(f"  SC confidence: {'✓' if has_sc else '✗'}")
            print(f"  VC confidence: {'✓' if has_vc else '✗'}")
            print(f"  LLM pop:       {'✓' if has_llm else '✗'}")

            # 收集样本
            samples = collect_samples(dataset, model, full_dict, co_occu, single_occr,
                                      sc_conf_dict, vc_conf_dict, llm_gene_pop, llm_qpop, llm_coo)
            if len(samples) < 50:
                print(f"  样本不足 ({len(samples)})，跳过")
                continue

            acc_rate = sum(s['acc'] for s in samples) / len(samples)
            need_bal = (acc_rate < BALANCE_THRESHOLD or acc_rate > 1 - BALANCE_THRESHOLD)
            print(f"  n={len(samples)}, acc={acc_rate:.3f}{'  [将做样本平衡]' if need_bal else ''}")

            # 根据数据可用性跳过不需要的实验
            skip_exps = []
            if not has_sc:
                skip_exps.extend([e[0] for e in EXPERIMENTS if 'sc_conf' in e[1]])
            if not has_vc:
                skip_exps.extend([e[0] for e in EXPERIMENTS if 'vc_conf' in e[1]])
            if not has_llm:
                skip_exps.extend([e[0] for e in EXPERIMENTS if 'llm_' in e[1]])

            results = run_one(samples, label=f'{dataset}×{model}', skip_experiments=skip_exps)
            if results is None:
                continue

            all_results[dataset][model] = results

            # 打印结果
            print(f"\n  {'实验':<40} {'Alignment':>10}")
            print(f"  {'-'*55}")
            for exp_name, res in results.items():
                if res.get('alignment') is not None:
                    bal_tag = f"  (balanced n={res['n_used']})" if res.get('balanced') else ""
                    print(f"  {exp_name:<40} {res['alignment']:>8.2f}%{bal_tag}")
                else:
                    print(f"  {exp_name:<40} {'skip('+res.get('note','?')+')':>20}")

    print_summary(all_results)

    # 保存结果为 JSON
    out_path = os.path.join(BASE, 'code', 'analysis_correlation', 'calibration_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")

    return all_results


if __name__ == '__main__':
    main()
