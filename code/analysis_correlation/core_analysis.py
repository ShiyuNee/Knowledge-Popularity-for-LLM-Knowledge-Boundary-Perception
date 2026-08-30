"""
核心分析脚本（core_analysis.py）

【论文数据来源说明】
本文论文 demo 中的所有统计数据均来自 verify_per_model.py（逐模型分析），
而非本文件。本文件用于探索性分析和补充验证，输出格式较松散。
verify_per_model.py 生成规范表格，是论文 Table 1-9 的主要来源。

【函数与论文对应关系】
- rq1_overconfidence_decomposition(): 论文 §4 (RQ1) + §5 (asymmetry)
    [1.1] Spearman 相关 → Table 1 (bivariate)
    [1.2-1.3] 做错样本偏相关 → §5.1, Table 1 (multivariate partial)
    [1.4] 分组控制 → §5.1
    [1.5] 做对 vs 做错 → §5.1 asymmetry (2–4× 差异)
- rq2_relation_specificity(): 论文 §4 (RQ2)
    [2.1] 分布统计 → §4.1
    [2.2] Spearman 相关 → Table 2
    [2.3] 偏相关（控制 gt_coo）→ §4.2, Finding #1
    [2.4] 四象限分析 → §4.3
    [2.5] gt_coo=0 样本分析 → §4.4
- rq2_cross_dataset_comparison(): 论文 §4.4 (basketball 弱的原因)
- rq2_quality_within_coo_bins(): 论文 §4.3 (固定 coo 区间内 quality_v2 对比)
- rq3_basketball_filtering(): 论文 §4.4 (>6000 过滤讨论)

【方法论原则】
所有分析均为 per dataset × model，从不跨模型合并。
合并会引入异质性误差分布（不同模型的 confidence 分布和错误模式不同），
导致虚假相关。

RQ1: 过度自信来源分解
  - 假设：模型看到熟悉的实体名（高 pop），不论对错，都会更自信
  - 具体实验：
    1. 直接 Spearman: answer_pop/gene_pop vs conf（做错样本中）
    2. 分组控制：固定 coo 区间 + 固定 question_pop 区间，看 answer_pop 对 conf 的影响
    3. 在做对/做错样本中分别分析 gene_pop 对 conf 的影响

RQ2: 共现数量 ≠ 关系强度 — 互信息/特异性是否独立影响 acc
  - 关系特异性度量：
    quality_v2 = gt_coo / q_single
    PMI = log2(gt_coo * N / (single[x] * single[y]))
    NPMI = PMI / -log2(P(x,y))
  - 实验：
    1. 三个数据集的 quality/PMI 分布对比
    2. 控制 gt_coo 后，quality/PMI 对 acc 的偏相关
    3. basketball 特殊性分析

RQ3: basketball >6000 过滤对比
"""

import json
import math
import re
import os
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu, pearsonr, rankdata

# ─── 路径配置 ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES_DIR = os.path.join(BASE, 'res')

POP_PATH = os.path.join(RES_DIR, 'gt_gene_entity_popularity_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.jsonl')
COO_PATH = os.path.join(RES_DIR, 'cooccurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')
SINGLE_PATH = os.path.join(RES_DIR, 'single_occurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')

DATASETS = ['movies', 'songs', 'basketball']
MODELS = ['llama8b', 'qwen2', 'chatgpt', 'Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']

PATTERN = {
    'movies': 'Who is the director of the movie ',
    'songs': 'Who is the performer of the song ',
    'basketball': 'Where is the birthplace of the basketball player '
}

SINGLE_OCC_THRESHOLD = 6000
# quality_v2 分析专用：q_single 的合理区间
# 下界 50：q_single<50 的实体 Wikipedia 文档极少，gt_coo/q_single 失去区分度
# 上界 6000：与 SINGLE_OCC_THRESHOLD 一致，排除歧义性极高的常见词
Q_SINGLE_LO = 50
Q_SINGLE_HI = 6000
# Wikipedia 估算总文档数（用于 PMI 计算）
N_WIKI = 6_000_000


# ─── 工具函数 ─────────────────────────────────────────────────────────────────
def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]


def remove_punctuation_edges(s, name='movies'):
    s = s.replace('\n', '')
    s = s.split('(')[0].strip()
    if name in ['basketball']:
        s = s.split(',')[0].strip()
    else:
        if len(s) <= 20:
            s = s.split(',')[0].strip()
    s = re.sub(r'^[^\w]+|[^\w]+$', '', s)
    return s.strip()


def load_popularity():
    pop_data = read_jsonl(POP_PATH)
    full_dict = {}
    for d in pop_data:
        full_dict.update(d)
    return full_dict


def partial_spearman(x, y, z):
    """控制 z 后，x 与 y 的偏相关（rank 线性残差法）"""
    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)
    rz = rankdata(z).astype(float)
    rz_c = rz - rz.mean()
    if np.dot(rz_c, rz_c) == 0:
        return 0, 1
    beta_x = np.dot(rz_c, rx) / np.dot(rz_c, rz_c)
    res_x = rx - beta_x * rz_c
    beta_y = np.dot(rz_c, ry) / np.dot(rz_c, rz_c)
    res_y = ry - beta_y * rz_c
    return pearsonr(res_x, res_y)


def _ece(acc, conf, n_bins=10):
    acc_arr = np.asarray(acc, dtype=float)
    conf_arr = np.asarray(conf, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(acc_arr)
    for i in range(n_bins):
        mask = (conf_arr >= bins[i]) & (conf_arr < bins[i + 1]) if i < n_bins - 1 else (conf_arr >= bins[i]) & (conf_arr <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(acc_arr[mask].mean() - conf_arr[mask].mean())
    return ece


def collect_samples(dataset, model, full_dict, co_occu, single_occr,
                    apply_filter=True):
    """
    为单个 dataset × model 组合收集样本。

    提取的字段：
    - question_pop, gt_pop, gene_pop: Wikidata popularity (sitelinks)
    - coo, gene_coo: Wikipedia co-occurrence counts
    - q_single, gt_single, gene_single: Wikipedia single-occurrence counts
    - total_coo: sum of all co-occurrences for question_entity
    - quality_v2 = gt_coo / q_single: relation specificity
    - pmi: Pointwise Mutual Information
    - npmi: Normalized PMI
    - conf: average token probability (confidence)
    - acc: has_answer (0/1)

    过滤条件：
    - movies/songs: 排除 single_occurrence > 6000 的 question/gene/ref 实体
    - basketball: 仅排除 single_occurrence > 6000 的 question_entity
    - apply_filter=False: 仅用于 RQ3 过滤对比分析
    """

    res_path = os.path.join(RES_DIR, dataset, f'{dataset}_{model}_temperature1.jsonl')
    if not os.path.exists(res_path):
        return []

    model_res = read_jsonl(res_path)
    samples = []
    skip_stat_miss = 0

    for item in model_res:
        if not item.get('Res') or item['Res'] is None:
            continue
        if item.get('popularity') == 'No': # skip samples without question popularity
            continue
        # 提取question, ref, gene entity
        question_entity = item['question'].replace(PATTERN[dataset], '').lower()
        ref = remove_punctuation_edges(item['reference'][0], dataset)
        gene_entity = remove_punctuation_edges(item['Res'], dataset)

        # 检查统计文件
        if question_entity not in co_occu:
            skip_stat_miss += 1
            continue
        if question_entity.lower() not in single_occr:
            skip_stat_miss += 1
            continue
        if gene_entity.lower() not in single_occr:
            skip_stat_miss += 1
            continue
        if ref.lower() not in single_occr:
            skip_stat_miss += 1
            continue

        # 过滤 single_occurrence 异常大的样本
        if apply_filter:
            if dataset in ['movies', 'songs']:
                if (single_occr[question_entity.lower()] > SINGLE_OCC_THRESHOLD or
                        single_occr[gene_entity.lower()] > SINGLE_OCC_THRESHOLD or
                        single_occr[ref.lower()] > SINGLE_OCC_THRESHOLD):
                    continue
            else:
                if single_occr[question_entity.lower()] > SINGLE_OCC_THRESHOLD:
                    continue

        # popularity
        question_pop = item['popularity']
        ref_pop_info = full_dict.get(ref, {})
        ref_pop = ref_pop_info.get('popularity', 0) if isinstance(ref_pop_info, dict) else ref_pop_info
        if ref_pop == 'No' or ref_pop is None:
            ref_pop = 0
        gene_pop_info = full_dict.get(gene_entity, {})
        gene_pop = gene_pop_info.get('popularity', 0) if isinstance(gene_pop_info, dict) else gene_pop_info
        if gene_pop == 'No' or gene_pop is None:
            gene_pop = 0

        # confidence
        if 'gpt' in model.lower():
            probs = [math.exp(t) for t in item['Log_p']['token_logprobs']]
        else:
            probs = item['Log_p']['token_probs']
        conf = sum(probs) / len(probs)

        # co-occurrence
        gt_coo = co_occu[question_entity].get(ref.lower(), 0)
        gene_coo = co_occu[question_entity].get(gene_entity.lower(), 0)

        # single occurrence
        q_single = single_occr[question_entity.lower()]
        gt_single = single_occr[ref.lower()]
        gene_single = single_occr[gene_entity.lower()]

        # 关系质量
        # quality_v2 = gt_coo / q_single:
        # question_entity 的 Wikipedia 文档中，有多少比例同时提到了正确答案
        # 分母 q_single 完全来自 Wikipedia 单独出现次数，不依赖模型预测，无数据泄露
        total_coo = sum(co_occu[question_entity].values())
        quality_v2 = gt_coo / q_single if q_single > 0 else 0

        # PMI (Pointwise Mutual Information)
        # PMI(x,y) = log2(P(x,y) / (P(x)*P(y))) = log2(gt_coo * N / (single[x] * single[y]))
        pmi = 0
        if gt_coo > 0 and q_single > 0 and gt_single > 0:
            pmi = math.log2((gt_coo * N_WIKI) / (q_single * gt_single))

        # NPMI (Normalized PMI): NPMI = PMI / -log2(P(x,y)) ∈ [-1, 1]
        # P(x,y) = gt_coo / N
        npmi = 0
        if gt_coo > 0:
            p_xy = gt_coo / N_WIKI
            npmi = pmi / (-math.log2(p_xy)) if p_xy > 0 and p_xy < 1 else 0
            # clamp to [-1, 1]
            npmi = max(-1, min(1, npmi))

        samples.append({
            'question_pop': question_pop,
            'gt_pop': int(ref_pop) if ref_pop else 0,
            'gene_pop': int(gene_pop) if gene_pop else 0,
            'coo': gt_coo,
            'gene_coo': gene_coo,
            'q_single': q_single,
            'gt_single': gt_single,
            'gene_single': gene_single,
            'total_coo': total_coo,
            'quality_v2': quality_v2,
            'pmi': pmi,
            'npmi': npmi,
            'conf': conf,
            'acc': item['has_answer'],
        })

    if skip_stat_miss > 0:
        print(f"  [{dataset} × {model}] 跳过 {skip_stat_miss} 条统计文件中缺失实体的样本")

    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# RQ1: 过度自信来源分解
# ═══════════════════════════════════════════════════════════════════════════════
def rq1_overconfidence_decomposition(samples, label=''):
    """
    RQ1: 过度自信来源分解（对应论文 §4 + §5）

    论文引用位置：
    - [1.1] 各流行度因子与 conf/acc 的 Spearman 相关 → Table 1 (bivariate)
    - [1.2] 做错样本中 gene_pop/gene_coo vs conf → §5.1
    - [1.3] 偏相关（控制 coo）→ Table 1 (multivariate partial), §5.1
    - [1.4] 分组控制（固定 coo+qpop 区间）→ §5.1
    - [1.5] 做对 vs 做错 gene_pop→conf 对比 → §5.1 asymmetry (Movies 2–4×)

    注意：论文数据来自 verify_per_model.py 表1-表3，本函数输出格式较松散，
    主要用于探索性观察。
    """
    if len(samples) < 30:
        print(f"  [skip] 样本不足: {len(samples)}")
        return

    qpop = np.array([s['question_pop'] for s in samples], dtype=float)
    gt_pop = np.array([s['gt_pop'] for s in samples], dtype=float)
    gene_pop = np.array([s['gene_pop'] for s in samples], dtype=float)
    coo = np.array([s['coo'] for s in samples], dtype=float)
    gene_coo = np.array([s['gene_coo'] for s in samples], dtype=float)
    conf = np.array([s['conf'] for s in samples], dtype=float)
    acc = np.array([s['acc'] for s in samples], dtype=float)

    print(f"\n{'='*70}")
    print(f"  RQ1: 过度自信来源分解 — {label}  (n={len(samples)})")
    print(f"{'='*70}")

    # ── 1.1 直接 Spearman 相关：各个 pop 因子与 conf/acc 的关系 ──────────
    print("\n[1.1] 各流行度因子与 conf/acc 的 Spearman 相关")
    for name, arr in [('question_pop', qpop), ('gt_pop (answer)', gt_pop),
                      ('gene_pop (generated)', gene_pop),
                      ('coo (gt_coo)', coo), ('gene_coo', gene_coo)]:
        if np.std(arr) == 0:
            print(f"  {name:25s}: 方差为0，跳过")
            continue
        r_acc, p_acc = spearmanr(arr, acc)
        r_conf, p_conf = spearmanr(arr, conf)
        print(f"  {name:25s} vs acc : r={r_acc:+.3f}  p={p_acc:.3e}")
        print(f"  {name:25s} vs conf: r={r_conf:+.3f}  p={p_conf:.3e}")

    # ── 1.2 做错样本中：answer_pop 和 gene_pop 对 conf 的影响 ──────────
    print("\n[1.2] 做错样本中（acc=0）的流行度-过度自信分析")
    wrong_mask = acc == 0
    n_wrong = wrong_mask.sum()
    if n_wrong < 20:
        print("  做错样本不足，跳过")
        return

    w_qpop = qpop[wrong_mask]
    w_gt_pop = gt_pop[wrong_mask]
    w_gene_pop = gene_pop[wrong_mask]
    w_coo = coo[wrong_mask]
    w_gene_coo = gene_coo[wrong_mask]
    w_conf = conf[wrong_mask]

    print(f"  做错样本 n={n_wrong}  avg_conf={w_conf.mean():.3f}")

    for name, arr in [('question_pop', w_qpop), ('gt_pop (answer)', w_gt_pop),
                      ('gene_pop (generated)', w_gene_pop),
                      ('coo (gt_coo)', w_coo), ('gene_coo', w_gene_coo)]:
        if np.std(arr) == 0:
            continue
        r, p = spearmanr(arr, w_conf)
        print(f"  Spearman({name:25s}, conf|acc=0): r={r:+.3f}  p={p:.3e}")

    # ── 1.3 偏相关：控制 coo 后，gt_pop/gene_pop 对 conf 的独立影响 ──
    print("\n[1.3] 偏相关：控制 coo 后，answer_pop/gene_pop 对 conf 的独立影响")
    # 在做错样本中
    r_gt_conf_coo, p = partial_spearman(w_gt_pop, w_conf, w_coo)
    print(f"  做错样本: gt_pop vs conf (控制 coo): r={r_gt_conf_coo:+.3f}  p={p:.3e}")
    r_gene_conf_coo, p = partial_spearman(w_gene_pop, w_conf, w_coo)
    print(f"  做错样本: gene_pop vs conf (控制 coo): r={r_gene_conf_coo:+.3f}  p={p:.3e}")
    r_qpop_conf_coo, p = partial_spearman(w_qpop, w_conf, w_coo)
    print(f"  做错样本: question_pop vs conf (控制 coo): r={r_qpop_conf_coo:+.3f}  p={p:.3e}")

    # 在全部样本中
    r_gt_conf_all, p = partial_spearman(gt_pop, conf, coo)
    print(f"  全样本:   gt_pop vs conf (控制 coo): r={r_gt_conf_all:+.3f}  p={p:.3e}")
    r_gene_conf_all, p = partial_spearman(gene_pop, conf, coo)
    print(f"  全样本:   gene_pop vs conf (控制 coo): r={r_gene_conf_all:+.3f}  p={p:.3e}")

    # ── 1.4 分组控制：固定 coo 区间，看 gene_pop 对 conf 的影响（做错样本）──
    print("\n[1.4] 分组控制：固定 coo+qpop 区间，gene_pop 对 conf 的影响（做错样本）")
    # 将 coo 分为低/高两组，question_pop 分为低/高两组，共4个格子
    coo_med = np.median(w_coo)
    qpop_med = np.median(w_qpop)
    for coo_label, coo_mask in [('低coo', w_coo <= coo_med), ('高coo', w_coo > coo_med)]:
        for qpop_label, qpop_mask in [('低qpop', w_qpop <= qpop_med), ('高qpop', w_qpop > qpop_med)]:
            cell_mask = coo_mask & qpop_mask
            n_cell = cell_mask.sum()
            if n_cell < 20:
                continue
            gp = w_gene_pop[cell_mask]
            wc = w_conf[cell_mask]
            if np.std(gp) == 0:
                continue
            r, p = spearmanr(gp, wc)
            gene_med = np.median(gp)
            high_gene = wc[gp > gene_med]
            low_gene = wc[gp <= gene_med]
            diff = high_gene.mean() - low_gene.mean() if len(high_gene) > 0 and len(low_gene) > 0 else 0
            print(f"  {coo_label}+{qpop_label} (n={n_cell}): "
                  f"Spearman(gene_pop, conf)={r:+.3f} p={p:.3e}  "
                  f"高gene_pop conf={high_gene.mean():.3f} 低={low_gene.mean():.3f} diff={diff:+.3f}")

    # ── 1.5 全样本：做对 vs 做错，gene_pop 对 conf 的影响 ──────────────
    print("\n[1.5] 做对 vs 做错样本中 gene_pop 对 conf 的影响对比")
    right_mask = acc == 1
    if right_mask.sum() >= 20:
        r_gene_conf_right, p = spearmanr(gene_pop[right_mask], conf[right_mask])
        r_gene_conf_wrong, p = spearmanr(gene_pop[wrong_mask], conf[wrong_mask])
        print(f"  做对: Spearman(gene_pop, conf) = {r_gene_conf_right:+.3f}")
        print(f"  做错: Spearman(gene_pop, conf) = {r_gene_conf_wrong:+.3f}")
        # 做错时 gene_pop > gt_pop 的比例
        gene_larger = (w_gene_pop > w_gt_pop).mean()
        print(f"  做错时 gene_pop > gt_pop 的比例: {gene_larger:.3f}")
        # 做错时 gene_coo vs gt_coo
        gene_coo_larger = (w_gene_coo > w_coo).mean()
        print(f"  做错时 gene_coo > gt_coo 的比例: {gene_coo_larger:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# RQ2: 共现数量 ≠ 关系强度
# ═══════════════════════════════════════════════════════════════════════════════
def rq2_relation_specificity(samples, label=''):
    """
    RQ2: 关系特异性分析（对应论文 §4）

    论文引用位置：
    - [2.1] 关系特异性指标分布 → §4.1
    - [2.2] 各指标与 acc/conf 的 Spearman 相关 → Table 2
    - [2.3] 偏相关（控制 gt_coo）→ §4.2, Finding #1 (quality_v2 15/15 显著)
    - [2.4] 四象限分析（coo × quality_v2）→ §4.3
    - [2.5] gt_coo=0 样本分析 → §4.4

    过滤条件：quality_v2 分析仅在 q_single ∈ [50, 6000) 的样本上进行，
    排除 Wikipedia 文档极少（分母噪音大）和歧义性极高（分母虚高）的实体。

    注意：论文数据来自 verify_per_model.py 表4、表7、表8，本函数用于探索。
    """
    # 对 quality_v2 分析，过滤 q_single 区间
    samples = [s for s in samples if Q_SINGLE_LO <= s['q_single'] < Q_SINGLE_HI]
    if len(samples) < 30:
        print(f"  [skip] 过滤 q_single∈[{Q_SINGLE_LO},{Q_SINGLE_HI}) 后样本不足: {len(samples)}")
        return

    qpop = np.array([s['question_pop'] for s in samples], dtype=float)
    coo = np.array([s['coo'] for s in samples], dtype=float)
    quality = np.array([s['quality_v2'] for s in samples], dtype=float)
    pmi = np.array([s['pmi'] for s in samples], dtype=float)
    npmi = np.array([s['npmi'] for s in samples], dtype=float)
    total_coo = np.array([s['total_coo'] for s in samples], dtype=float)
    conf = np.array([s['conf'] for s in samples], dtype=float)
    acc = np.array([s['acc'] for s in samples], dtype=float)

    print(f"\n{'='*70}")
    print(f"  RQ2: 关系特异性分析 — {label}  (n={len(samples)})")
    print(f"{'='*70}")

    # ── 2.1 分布统计 ──────────────────────────────────────────────────────
    print("\n[2.1] 关系特异性指标分布")
    nonzero_coo = coo > 0
    for name, arr in [('gt_coo', coo), ('quality_v2', quality), ('PMI', pmi),
                      ('NPMI', npmi), ('total_coo', total_coo)]:
        m = nonzero_coo if name != 'gt_coo' else np.ones(len(arr), dtype=bool)
        masked = arr[m]
        if len(masked) == 0:
            continue
        print(f"  {name:15s}: mean={masked.mean():.3f}  median={np.median(masked):.3f}  "
              f"std={masked.std():.3f}  min={masked.min():.3f}  max={masked.max():.3f}")

    # gt_coo=0 的比例
    print(f"  gt_coo=0 的样本比例: {(coo == 0).mean():.3f}")

    # ── 2.2 直接 Spearman 相关 ──────────────────────────────────────────────
    print("\n[2.2] 各特异性指标与 acc/conf 的 Spearman 相关")
    for name, arr in [('gt_coo', coo), ('quality_v2', quality), ('PMI', pmi),
                      ('NPMI', npmi), ('total_coo', total_coo)]:
        if np.std(arr) == 0:
            continue
        r_acc, p_acc = spearmanr(arr, acc)
        r_conf, p_conf = spearmanr(arr, conf)
        print(f"  {name:15s} vs acc : r={r_acc:+.3f}  p={p_acc:.3e}")
        print(f"  {name:15s} vs conf: r={r_conf:+.3f}  p={p_conf:.3e}")

    # ── 2.3 偏相关：控制 gt_coo 后，quality/PMI 对 acc 的独立影响 ──────
    print("\n[2.3] 偏相关：控制 gt_coo 后，特异性指标对 acc 的独立影响")
    # 只在 gt_coo > 0 的样本上做
    mask = nonzero_coo
    if mask.sum() >= 30:
        for name, arr in [('quality_v2', quality), ('PMI', pmi), ('NPMI', npmi)]:
            r_pc, p_pc = partial_spearman(arr[mask], acc[mask], coo[mask])
            print(f"  {name:15s} vs acc (控制 gt_coo, n={mask.sum()}): r={r_pc:+.3f}  p={p_pc:.3e}")
        # 额外：控制 quality_v2 后，gt_coo 对 acc 的偏相关
        r_pc, p_pc = partial_spearman(coo[mask], acc[mask], quality[mask])
        print(f"  gt_coo          vs acc (控制 quality_v2): r={r_pc:+.3f}  p={p_pc:.3e}")
    else:
        print(f"  gt_coo>0 的样本不足 ({mask.sum()})")

    # ── 2.4 分组对比：高coo+低quality vs 低coo+高quality ──────────────
    print("\n[2.4] 四象限分析：coo × quality_v2 分组 acc")
    coo_med = np.median(coo)
    q_med = np.median(quality)
    quadrants = {
        'HH(高coo+高quality)': (coo > coo_med) & (quality > q_med),
        'HL(高coo+低quality)': (coo > coo_med) & (quality <= q_med),
        'LH(低coo+高quality)': (coo <= coo_med) & (quality > q_med),
        'LL(低coo+低quality)': (coo <= coo_med) & (quality <= q_med),
    }
    print(f"  {'象限':<25} {'n':>6} {'avg_acc':>9} {'avg_conf':>9} {'avg_coo':>10} {'avg_quality':>12}")
    print(f"  {'-'*25} {'-'*6} {'-'*9} {'-'*9} {'-'*10} {'-'*12}")
    for name, qmask in quadrants.items():
        n = qmask.sum()
        if n == 0:
            continue
        print(f"  {name:<25} {n:>6} {acc[qmask].mean():>9.3f} {conf[qmask].mean():>9.3f} "
              f"{coo[qmask].mean():>10.1f} {quality[qmask].mean():>12.4f}")

    # 关键对比
    hh = quadrants['HH(高coo+高quality)']
    hl = quadrants['HL(高coo+低quality)']
    lh = quadrants['LH(低coo+高quality)']
    if hh.sum() >= 5 and hl.sum() >= 5:
        stat, p = mannwhitneyu(acc[hh], acc[hl], alternative='two-sided')
        diff = acc[hh].mean() - acc[hl].mean()
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"  → 高coo+高quality vs 高coo+低quality: diff={diff:+.3f}  p={p:.3e} {sig}")
    if lh.sum() >= 5 and hl.sum() >= 5:
        stat, p = mannwhitneyu(acc[lh], acc[hl], alternative='two-sided')
        diff = acc[lh].mean() - acc[hl].mean()
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"  → 低coo+高quality vs 高coo+低quality: diff={diff:+.3f}  p={p:.3e} {sig}")

    # ── 2.5 做错样本中 coo=0 的比例 ──────────────────────────────────────
    print("\n[2.5] gt_coo=0 样本分析")
    zero_coo = coo == 0
    nonzero_coo_mask = coo > 0
    if zero_coo.sum() > 0:
        print(f"  gt_coo=0 样本: n={zero_coo.sum()} ({zero_coo.mean():.1%})  "
              f"avg_acc={acc[zero_coo].mean():.3f}  avg_conf={conf[zero_coo].mean():.3f}")
        print(f"  gt_coo>0 样本: n={nonzero_coo_mask.sum()} ({nonzero_coo_mask.mean():.1%})  "
              f"avg_acc={acc[nonzero_coo_mask].mean():.3f}  avg_conf={conf[nonzero_coo_mask].mean():.3f}")
        # gt_coo=0 意味着 Wikipedia 中 Q 与 A 从未共现
        # 但 gt_coo=0 的样本中仍然有一定准确率，这说明模型可能有其他来源的知识


# ═══════════════════════════════════════════════════════════════════════════════
# RQ3: basketball >6000 过滤对比
# ═══════════════════════════════════════════════════════════════════════════════
def rq3_basketball_filtering(full_dict, co_occu, single_occr):
    """
    RQ3: basketball >6000 过滤对比分析（对应论文 §4.4）

    对比有/无 >6000 过滤条件下的核心指标差异，
    并分析被过滤掉的 question_entity 的特征。

    论文引用位置：§4.4 过滤条件讨论（<0.4% 样本被移除）
    """
    print(f"\n{'='*70}")
    print(f"  RQ3: basketball >6000 过滤对比")
    print(f"{'='*70}")

    dataset = 'basketball'
    for model in ['llama8b', 'chatgpt']:
        samples_filtered = collect_samples(dataset, model, full_dict, co_occu, single_occr,
                                           apply_filter=True)
        samples_unfiltered = collect_samples(dataset, model, full_dict, co_occu, single_occr,
                                             apply_filter=False)
        if not samples_filtered or not samples_unfiltered:
            continue

        print(f"\n--- basketball × {model} ---")
        print(f"  有过滤: n={len(samples_filtered)}  无过滤: n={len(samples_unfiltered)}")
        print(f"  过滤掉: {len(samples_unfiltered) - len(samples_filtered)} 样本 "
              f"({(1 - len(samples_filtered)/len(samples_unfiltered))*100:.1f}%)")

        # 对比核心指标
        for name, samples in [('有过滤', samples_filtered), ('无过滤', samples_unfiltered)]:
            acc = np.array([s['acc'] for s in samples], dtype=float)
            conf = np.array([s['conf'] for s in samples], dtype=float)
            coo = np.array([s['coo'] for s in samples], dtype=float)
            qpop = np.array([s['question_pop'] for s in samples], dtype=float)
            quality = np.array([s['quality_v2'] for s in samples], dtype=float)

            r_coo_acc, _ = spearmanr(coo, acc)
            r_qpop_conf, _ = spearmanr(qpop, conf)
            r_coo_conf, _ = spearmanr(coo, conf)
            r_q_acc, p_q = 0, 1
            if np.std(quality) > 0:
                r_q_acc, p_q = spearmanr(quality, acc)

            print(f"  [{name}] avg_acc={acc.mean():.3f}  avg_conf={conf.mean():.3f}  "
                  f"Spearman(coo,acc)={r_coo_acc:+.3f}  "
                  f"Spearman(qpop,conf)={r_qpop_conf:+.3f}  "
                  f"Spearman(coo,conf)={r_coo_conf:+.3f}  "
                  f"Spearman(quality,acc)={r_q_acc:+.3f}")

        # 分析被过滤掉的样本
        filtered_out_set = set()
        # 简单识别被过滤的样本
        res_path = os.path.join(RES_DIR, dataset, f'{dataset}_{model}_temperature1.jsonl')
        model_res = read_jsonl(res_path)
        filtered_entities = []
        for item in model_res:
            if not item.get('Res') or item['Res'] is None or item.get('popularity') == 'No':
                continue
            qe = item['question'].replace(PATTERN[dataset], '').lower()
            if qe not in co_occu or qe.lower() not in single_occr:
                continue
            if single_occr[qe.lower()] > SINGLE_OCC_THRESHOLD:
                filtered_entities.append({
                    'entity': qe,
                    'single_occ': single_occr[qe.lower()],
                    'acc': item['has_answer'],
                    'conf': sum(item['Log_p']['token_probs']) / len(item['Log_p']['token_probs']) if 'gpt' not in model.lower() else sum([math.exp(t) for t in item['Log_p']['token_logprobs']]) / len(item['Log_p']['token_logprobs'])
                })

        if filtered_entities:
            fe = filtered_entities
            print(f"\n  被过滤的 question_entity ({len(fe)} 个唯一实体):")
            # 按出现次数排序
            entity_counts = {}
            for e in fe:
                name = e['entity']
                if name not in entity_counts:
                    entity_counts[name] = {'count': 0, 'acc_sum': 0, 'conf_sum': 0, 'single_occ': e['single_occ']}
                entity_counts[name]['count'] += 1
                entity_counts[name]['acc_sum'] += e['acc']
                entity_counts[name]['conf_sum'] += e['conf']

            sorted_entities = sorted(entity_counts.items(), key=lambda x: -x[1]['count'])[:15]
            print(f"  {'实体名':<20} {'single_occ':>10} {'样本数':>6} {'avg_acc':>8} {'avg_conf':>8}")
            print(f"  {'-'*20} {'-'*10} {'-'*6} {'-'*8} {'-'*8}")
            for name, info in sorted_entities:
                avg_acc = info['acc_sum'] / info['count']
                avg_conf = info['conf_sum'] / info['count']
                print(f"  {name:<20} {info['single_occ']:>10} {info['count']:>6} {avg_acc:>8.3f} {avg_conf:>8.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# RQ2 补充：basketball 为什么弱 — 跨数据集对比
# ═══════════════════════════════════════════════════════════════════════════════
def rq2_cross_dataset_comparison(all_samples_by_dataset):
    """
    RQ2 补充：跨数据集对比（对应论文 §4.4）

    分析 basketball 为什么 coo→acc 相关性弱，核心对比：
    - 同样 gt_coo 范围内的 acc 对比（basketball vs movies）
    - 同样 quality_v2 范围内的 acc 和 coo→acc 相关对比

    论文引用位置：§4.4 "Why is basketball different?"

    注意：本函数仅做分布统计对比，不做跨模型合并的相关分析。
    """
    print(f"\n{'='*70}")
    print(f"  RQ2补充：跨数据集对比 — basketball 为什么相关性弱？")
    print(f"{'='*70}")

    print(f"\n  {'指标':<35}", end='')
    for ds in DATASETS:
        print(f"  {ds:>12}", end='')
    print()

    for metric_name, metric_fn in [
        ('gt_coo=0 比例', lambda s: (np.array([x['coo'] for x in s]) == 0).mean()),
        ('median gt_coo', lambda s: np.median([x['coo'] for x in s])),
        ('median quality_v2', lambda s: np.median([x['quality_v2'] for x in s])),
        ('median PMI (gt_coo>0)', lambda s: np.median([x['pmi'] for x in s if x['coo'] > 0])),
        ('median NPMI (gt_coo>0)', lambda s: np.median([x['npmi'] for x in s if x['coo'] > 0])),
        ('avg_acc', lambda s: np.mean([x['acc'] for x in s])),
    ]:
        print(f"  {metric_name:<35}", end='')
        for ds in DATASETS:
            samples = all_samples_by_dataset.get(ds, [])
            if samples:
                val = metric_fn(samples)
                print(f"  {val:>12.3f}", end='')
            else:
                print(f"  {'N/A':>12}", end='')
        print()

    # Spearman 相关对比
    print(f"\n  {'Spearman 相关':<35}", end='')
    for ds in DATASETS:
        print(f"  {ds:>12}", end='')
    print()

    for metric_name, key, target in [
        ('Spearman(coo, acc)', 'coo', 'acc'),
        ('Spearman(quality_v2, acc)', 'quality_v2', 'acc'),
        ('Spearman(PMI, acc)', 'pmi', 'acc'),
        ('Spearman(question_pop, conf)', 'question_pop', 'conf'),
    ]:
        print(f"  {metric_name:<35}", end='')
        for ds in DATASETS:
            samples = all_samples_by_dataset.get(ds, [])
            if len(samples) < 30:
                print(f"  {'N/A':>12}", end='')
                continue
            arr = np.array([s[key] for s in samples], dtype=float)
            tgt = np.array([s[target] for s in samples], dtype=float)
            if np.std(arr) == 0:
                print(f"  {'N/A':>12}", end='')
                continue
            r, _ = spearmanr(arr, tgt)
            print(f"  {r:>+12.3f}", end='')
        print()

    # basketball 弱的根本原因分析
    print(f"\n  [Basketball 弱的原因分析]")
    bball = all_samples_by_dataset.get('basketball', [])
    movies = all_samples_by_dataset.get('movies', [])
    if bball and movies:
        # 同样 gt_coo 范围内的 acc 对比
        for coo_range, label in [((0, 0), 'gt_coo=0'), ((1, 5), 'gt_coo 1~5'), ((6, 20), 'gt_coo 6~20'), ((21, 9999), 'gt_coo >20')]:
            lo, hi = coo_range
            b_mask = np.array([lo <= s['coo'] <= hi for s in bball])
            m_mask = np.array([lo <= s['coo'] <= hi for s in movies])
            b_acc = np.mean([s['acc'] for s, m in zip(bball, b_mask) if m]) if b_mask.sum() > 0 else float('nan')
            m_acc = np.mean([s['acc'] for s, m in zip(movies, m_mask) if m]) if m_mask.sum() > 0 else float('nan')
            b_n = b_mask.sum()
            m_n = m_mask.sum()
            print(f"  {label:<20}: basketball acc={b_acc:.3f} (n={b_n})  movies acc={m_acc:.3f} (n={m_n})")

        # 同样 quality_v2 范围
        print()
        for q_range, label in [((0, 0.1), 'quality 0~0.1'), ((0.1, 0.3), 'quality 0.1~0.3'),
                                ((0.3, 0.6), 'quality 0.3~0.6'), ((0.6, 1.01), 'quality 0.6~1.0')]:
            lo, hi = q_range
            b_mask = np.array([lo <= s['quality_v2'] <= hi for s in bball])
            m_mask = np.array([lo <= s['quality_v2'] <= hi for s in movies])
            b_acc = np.mean([s['acc'] for s, m in zip(bball, b_mask) if m]) if b_mask.sum() > 0 else float('nan')
            m_acc = np.mean([s['acc'] for s, m in zip(movies, m_mask) if m]) if m_mask.sum() > 0 else float('nan')
            b_n = b_mask.sum()
            m_n = m_mask.sum()
            # 在这个 quality 区间内，coo 与 acc 的 Spearman
            b_coo_in = [s['coo'] for s, m in zip(bball, b_mask) if m]
            b_acc_in = [s['acc'] for s, m in zip(bball, b_mask) if m]
            m_coo_in = [s['coo'] for s, m in zip(movies, m_mask) if m]
            m_acc_in = [s['acc'] for s, m in zip(movies, m_mask) if m]
            b_r, m_r = float('nan'), float('nan')
            if len(b_coo_in) > 30 and np.std(b_coo_in) > 0:
                b_r, _ = spearmanr(b_coo_in, b_acc_in)
            if len(m_coo_in) > 30 and np.std(m_coo_in) > 0:
                m_r, _ = spearmanr(m_coo_in, m_acc_in)
            print(f"  {label:<20}: bball acc={b_acc:.3f} (n={b_n}, r_coo_acc={b_r:+.3f})  "
                  f"movies acc={m_acc:.3f} (n={m_n}, r_coo_acc={m_r:+.3f})")


# ═══════════════════════════════════════════════════════════════════════════════
# RQ2 补充：固定 coo 区间，对比做对 vs 做错样本的 quality_v2 分布
# ═══════════════════════════════════════════════════════════════════════════════
def rq2_quality_within_coo_bins(samples, label=''):
    """
    RQ2 补充：固定 coo 区间内，做对 vs 做错的 quality_v2 对比（对应论文 §4.3）

    核心思路：在 coo 相近的条件下，做对的样本是否比做错的样本有更高的 quality_v2？
    方法：
      1. 将样本按 gt_coo 分成若干区间（bin）
      2. 在每个 bin 内，分别统计做对（acc=1）和做错（acc=0）样本的 quality_v2 均值/中位数
      3. 用 Mann-Whitney U 检验两组 quality_v2 是否有显著差异
      4. 汇总：在 coo 相近的情况下，做对样本的 quality_v2 是否系统性地高于做错样本

    论文引用位置：§4.3 "Quality within coo bins"

    过滤条件：quality_v2 分析仅在 q_single ∈ [Q_SINGLE_LO, Q_SINGLE_HI) 的样本上进行。
    """
    # 过滤 q_single 区间，保证 quality_v2 度量有效
    samples = [s for s in samples if Q_SINGLE_LO <= s['q_single'] < Q_SINGLE_HI]
    if len(samples) < 50:
        print(f"  [skip] 过滤 q_single∈[{Q_SINGLE_LO},{Q_SINGLE_HI}) 后样本不足: {len(samples)}")
        return

    coo = np.array([s['coo'] for s in samples], dtype=float)
    quality = np.array([s['quality_v2'] for s in samples], dtype=float)
    acc = np.array([s['acc'] for s in samples], dtype=float)

    print(f"\n{'='*70}")
    print(f"  RQ2补充：固定coo区间内，做对 vs 做错的 quality_v2 对比 — {label}")
    print(f"{'='*70}")

    # ── 方法1：按 gt_coo 分区间，逐区间对比 ──────────────────────────────
    # 区间设计：覆盖 coo 的主要分布范围
    coo_bins = [
        (1, 2,   'coo=1~2'),
        (3, 5,   'coo=3~5'),
        (6, 10,  'coo=6~10'),
        (11, 20, 'coo=11~20'),
        (21, 50, 'coo=21~50'),
        (51, 9999, 'coo>50'),
    ]

    print(f"\n[方法1] 逐 coo 区间：做对 vs 做错 的 quality_v2 对比")
    print(f"  {'coo区间':<12} {'n_right':>8} {'n_wrong':>8} "
          f"{'q_right(med)':>14} {'q_wrong(med)':>14} "
          f"{'diff(med)':>11} {'p(MWU)':>10} {'sig':>4}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*14} {'-'*14} {'-'*11} {'-'*10} {'-'*4}")

    consistent_direction = 0  # 做对 quality > 做错 quality 的区间数
    total_valid_bins = 0

    for lo, hi, bin_label in coo_bins:
        bin_mask = (coo >= lo) & (coo <= hi)
        right_mask = bin_mask & (acc == 1)
        wrong_mask = bin_mask & (acc == 0)

        n_r = right_mask.sum()
        n_w = wrong_mask.sum()

        if n_r < 10 or n_w < 10:
            print(f"  {bin_label:<12} {n_r:>8} {n_w:>8}  [样本不足，跳过]")
            continue

        q_r = quality[right_mask]
        q_w = quality[wrong_mask]

        med_r = np.median(q_r)
        med_w = np.median(q_w)
        diff = med_r - med_w

        stat, p = mannwhitneyu(q_r, q_w, alternative='two-sided')
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))

        print(f"  {bin_label:<12} {n_r:>8} {n_w:>8} "
              f"{med_r:>14.4f} {med_w:>14.4f} "
              f"{diff:>+11.4f} {p:>10.3e} {sig:>4}")

        total_valid_bins += 1
        if diff > 0:
            consistent_direction += 1

    if total_valid_bins > 0:
        print(f"\n  → 在 {total_valid_bins} 个有效区间中，{consistent_direction} 个区间做对样本 quality_v2 > 做错样本")

    # ── 方法2：按 gt_coo 分位数分组（更均匀的分组）──────────────────────
    print(f"\n[方法2] 按 gt_coo 四分位分组（gt_coo>0 样本），逐组对比 quality_v2")
    nonzero_mask = coo > 0
    if nonzero_mask.sum() < 50:
        print("  gt_coo>0 样本不足，跳过")
    else:
        coo_nz = coo[nonzero_mask]
        quality_nz = quality[nonzero_mask]
        acc_nz = acc[nonzero_mask]

        quartile_edges = np.percentile(coo_nz, [0, 25, 50, 75, 100])
        print(f"  gt_coo 四分位边界: {[f'{v:.0f}' for v in quartile_edges]}")
        print(f"  {'分组':<15} {'n_right':>8} {'n_wrong':>8} "
              f"{'q_right(med)':>14} {'q_wrong(med)':>14} "
              f"{'diff':>8} {'p':>10} {'sig':>4}")
        print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*14} {'-'*14} {'-'*8} {'-'*10} {'-'*4}")

        for i in range(4):
            lo_q, hi_q = quartile_edges[i], quartile_edges[i + 1]
            if i == 0:
                qbin_mask = (coo_nz >= lo_q) & (coo_nz <= hi_q)
            else:
                qbin_mask = (coo_nz > lo_q) & (coo_nz <= hi_q)

            r_m = qbin_mask & (acc_nz == 1)
            w_m = qbin_mask & (acc_nz == 0)
            n_r, n_w = r_m.sum(), w_m.sum()

            if n_r < 10 or n_w < 10:
                print(f"  Q{i+1}[{lo_q:.0f},{hi_q:.0f}]    {n_r:>8} {n_w:>8}  [样本不足]")
                continue

            q_r = quality_nz[r_m]
            q_w = quality_nz[w_m]
            med_r, med_w = np.median(q_r), np.median(q_w)
            diff = med_r - med_w
            stat, p = mannwhitneyu(q_r, q_w, alternative='two-sided')
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))

            print(f"  Q{i+1}[{lo_q:.0f},{hi_q:.0f}]{'':<6} {n_r:>8} {n_w:>8} "
                  f"{med_r:>14.4f} {med_w:>14.4f} "
                  f"{diff:>+8.4f} {p:>10.3e} {sig:>4}")

    # ── 方法3：整体汇总——做对 vs 做错的 quality_v2 均值/中位数 ──────────
    print(f"\n[方法4] 整体汇总（不分 coo 区间）")
    right_all = acc == 1
    wrong_all = acc == 0
    if right_all.sum() >= 10 and wrong_all.sum() >= 10:
        q_r_all = quality[right_all]
        q_w_all = quality[wrong_all]
        stat, p = mannwhitneyu(q_r_all, q_w_all, alternative='two-sided')
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"  做对(n={right_all.sum()}): quality_v2 mean={q_r_all.mean():.4f}  median={np.median(q_r_all):.4f}")
        print(f"  做错(n={wrong_all.sum()}): quality_v2 mean={q_w_all.mean():.4f}  median={np.median(q_w_all):.4f}")
        print(f"  MWU p={p:.3e} {sig}  median diff={np.median(q_r_all)-np.median(q_w_all):+.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("加载数据...")
    full_dict = load_popularity() # wikidata统计得到的entity popularity
    co_occu = json.loads(open(COO_PATH).read()) # 共现
    single_occr = json.loads(open(SINGLE_PATH).read()) # wikipedia统计的entity popularity

    # ═══ 构建 all_samples_by_dataset（用于跨数据集对比，不合并分析）═══
    # 注意：论文所有分析均为 per dataset × model，从不跨模型合并。
    # 合并会引入异质性误差分布（不同模型的 confidence 分布和错误模式不同），
    # 导致虚假相关。此处仅构建数据结构用于 rq2_cross_dataset_comparison。
    all_samples_by_dataset = {}
    for dataset in DATASETS:
        all_samples = []
        for model in MODELS:
            # 对 basketball，只用 llama8b 和 chatgpt（Qwen2.5 系列 acc 太低）
            if dataset == 'basketball' and model.startswith('Qwen2.5'):
                continue
            s = collect_samples(dataset, model, full_dict, co_occu, single_occr)
            all_samples.extend(s)
        all_samples_by_dataset[dataset] = all_samples

    # RQ2 补充：跨数据集对比（仅比较分布统计，不做跨模型合并的相关分析）
    rq2_cross_dataset_comparison(all_samples_by_dataset)

    # ═══ RQ3: basketball 过滤对比 ═══
    rq3_basketball_filtering(full_dict, co_occu, single_occr)

    # ═══ 按数据集 × 模型 单独分析（RQ1 + RQ2）═══
    print("\n\n" + "="*70)
    print("按 dataset × model 分析 (RQ1 + RQ2)")
    print("="*70)
    for dataset in DATASETS:
        for model in MODELS:
            if dataset == 'basketball' and model.startswith('Qwen2.5'):
                continue
            samples = collect_samples(dataset, model, full_dict, co_occu, single_occr)
            if len(samples) < 30:
                continue
            rq1_overconfidence_decomposition(samples, label=f'{dataset} × {model}')
            rq2_relation_specificity(samples, label=f'{dataset} × {model}')
            rq2_quality_within_coo_bins(samples, label=f'{dataset} × {model}')


if __name__ == '__main__':
    main()