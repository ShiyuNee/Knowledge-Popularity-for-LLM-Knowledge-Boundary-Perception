"""
verify_per_model.py — 论文核心数据来源脚本

【论文主要数据来源】
论文 §4–§7 和 Appendix B–C 的统计数据均可由本脚本复现。
所有分析按 dataset × model 独立进行，不跨模型合并，以避免异质性误差分布导致的虚假相关。

【偏相关分析框架】
论文采用对称三变量互控框架，根据分析目标选择不同的变量集：
  - RQ1 (acc):  {gt_coo, question_pop, gt_pop}  → 知识属性，每个控制其余两个
  - RQ2 (conf): {gene_coo, question_pop, gene_pop} → 生成答案属性，每个控制其余两个

设计理由：
  1. acc 反映模型是否学会了知识，应关注问题-正确答案对的知识属性（gt_*），
     不应引入生成答案属性（gene_*），因为只有多学正确答案对应的关系才能让模型学会
  2. conf 反映模型对自身生成答案的信心，应关注模型实际生成内容的属性（gene_*），
     因为模型不知道 gt 是什么，只知道自己生成了什么
  3. 不将 gene_pop 与 gt_pop 放入同一模型，因为做对时 gene_pop=gt_pop（ρ=0.96–0.99），
     近乎共线性导致偏相关不稳定

【按 RQ 的表格对应关系】

RQ1: What makes a model learn a fact? (§4)
  表1: 基础分布统计（median_coo, median_qv2, coo=0%）→ §4.1
  表2: RQ1 三变量偏相关 — {gt_coo, qpop, gt_pop} → acc → §4.1, Appendix B
  表3: quality_v2→acc 偏相关（控制gt_coo，q_single∈[50,6000)）→ §4.2, Finding #1
  表4: 四象限分析（coo × quality_v2 → avg_acc）→ §4.3

RQ2: What drives confidence, and how does it differ from learning? (§5)
  表5: RQ2 三变量偏相关 — {gene_coo, qpop, gene_pop} → conf（全样本）→ §5.1
  表6: 做对 vs 做错 asymmetry — gene_pop→conf 偏相关（3-var 互控）→ §5.1, Appendix C
  表7: 错误样本 — 三因子互控偏相关 → §5.2, Appendix C
  表8: 正确 vs 错误全因子分解 — 各因子→conf 在正确/错误子集上的对比 → §5.2, Appendix C
  表9: 做错时 gene_pop>gt_pop / gene_coo>gt_coo 比例 → §5.3
  表10: gene_pop 变异度（CV, P10/P90, 四分位conf趋势）→ §5.4
  表11: 做对时 Spearman(gene_pop, conf) → §5.4 (basketball 负相关验证)

§7: Using External Signals for Calibration (§7)
  表12: 固定 conf 区间，做对 vs 做错的 gene_pop/gene_coo 差异 → §7.1 特征分析

【关键过滤条件】
  - SINGLE_OCC_THRESHOLD=6000: 排除 Aho-Corasick 歧义高频实体（如 "fire", "angel"）
  - Q_SINGLE_LO=50, Q_SINGLE_HI=6000: quality_v2 分析专用，q_single 太小时分母噪音大
  - filter_pop_no=True: 过滤 Wikidata 查不到的 gt_pop/gene_pop 样本（默认关闭以保持兼容）
  - movies/songs: 对 question/gene/ref 三类实体做 SINGLE_OCC 过滤
  - basketball: 仅对 question_entity 做 SINGLE_OCC 过滤（城市名天然高频）
  - Basketball 排除 Qwen2.5 系列（acc<8%，相关分析不可靠）
"""

import json
import math
import re
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr, rankdata

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
Q_SINGLE_LO = 50   # quality_v2 分析下界：q_single 太小时分母噪音大
Q_SINGLE_HI = 6000 # quality_v2 分析上界：与 SINGLE_OCC_THRESHOLD 一致


def read_jsonl(path):
    """读取 JSONL 文件，返回 dict 列表"""
    return [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]


def remove_punctuation_edges(s, name='movies'):
    """清洗实体名称：去除括号内容、多余逗号、首尾特殊字符"""
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
    """加载 Wikidata sitelinks popularity 数据（gene/gt entity）"""
    pop_data = read_jsonl(POP_PATH)
    full_dict = {}
    for d in pop_data:
        full_dict.update(d)
    return full_dict


def partial_spearman(x, y, z):
    """偏 Spearman 相关：控制单个变量 z 后 x 与 y 的偏相关（rank 空间 OLS 残差法）

    这是 multivariate_partial 的单控制变量特化版本，用于 quality_v2 分析（控制 gt_coo）。
    对三变量互控分析，请使用 three_var_partial() 而非本函数。

    原理：
      1. 将 x, y, z 均转为 rank
      2. 用 OLS 回归去除 z 对 x 和 y 的线性影响
      3. 对残差计算 Pearson 相关
    """
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


def multivariate_partial(x, y, controls):
    """多元偏相关：控制所有 controls 后 x 与 y 的偏相关（rank 空间 OLS 残差法）

    对三变量互控框架，controls 为长度 2 的列表（被控制的另外两个变量）。
    对 quality_v2 单控制分析，controls 为长度 1 的列表。

    原理：
      1. 将 x, y, controls 均转为 rank
      2. 用 OLS 回归去除 controls 对 x 和 y 的线性影响，得到残差 res_x, res_y
      3. 计算 Pearson(res_x, res_y) 作为偏 Spearman 相关系数
    """
    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()

    if len(controls) == 0:
        return pearsonr(rx_c, ry_c)

    Z = np.column_stack([rankdata(c).astype(float) for c in controls])
    Z = Z - Z.mean(axis=0)
    try:
        beta_x = np.linalg.lstsq(Z, rx_c, rcond=None)[0]
        beta_y = np.linalg.lstsq(Z, ry_c, rcond=None)[0]
        res_x = rx_c - Z @ beta_x
        res_y = ry_c - Z @ beta_y
        return pearsonr(res_x, res_y)
    except:
        return 0, 1


def three_var_partial(samples, factor_keys, target_key):
    """三变量互控偏相关：对 factor_keys 中的三个变量，每个控制其余两个，
    计算与 target_key 的偏 Spearman 相关系数。

    这是论文核心分析框架的实现：
      - RQ1 (acc):  factor_keys=['coo', 'question_pop', 'gt_pop'], target_key='acc'
      - RQ2 (conf): factor_keys=['gene_coo', 'question_pop', 'gene_pop'], target_key='conf'

    对每个因子 f，控制 [其余两个因子]，计算 f → target 的偏相关。
    这确保了每个因子的效应是独立于另外两个因子的。

    Args:
        samples: list of dicts，由 collect_samples() 返回
        factor_keys: 三个因子在 sample dict 中的键名，如 ['coo', 'question_pop', 'gt_pop']
        target_key: 目标变量键名，'acc' 或 'conf'

    Returns:
        dict: {factor_key: partial_r}，每个因子对应的偏相关系数
    """
    results = {}
    arrs = {k: np.array([s[k] for s in samples], dtype=float) for k in factor_keys}
    tgt = np.array([s[target_key] for s in samples], dtype=float)

    for k in factor_keys:
        others = [arrs[j] for j in factor_keys if j != k]
        r, _ = multivariate_partial(arrs[k], tgt, others)
        results[k] = r
    return results


def collect_samples(dataset, model, full_dict, co_occu, single_occr,
                    filter_pop_no=False):
    """
    为单个 dataset × model 组合收集样本。

    提取的字段：
    - question_pop, gt_pop, gene_pop: Wikidata popularity (sitelinks)
    - coo, gene_coo: Wikipedia co-occurrence counts
    - q_single, gt_single, gene_single: Wikipedia single-occurrence counts
    - total_coo: sum of all co-occurrences for question_entity
    - quality_v2 = gt_coo / q_single: relation specificity
    - pmi: Pointwise Mutual Information
    - conf: average token probability (confidence)
    - acc: has_answer (0/1)

    过滤条件：
    - movies/songs: 排除 single_occurrence > 6000 的 question/gene/ref 实体
    - basketball: 仅排除 single_occurrence > 6000 的 question_entity
    - filter_pop_no=True: 同时过滤 gt_pop 或 gene_pop 在 Wikidata 中查不到的样本
      （popularity='No' 或实体不在 full_dict 中），而非置为 0
    """
    res_path = os.path.join(RES_DIR, dataset, f'{dataset}_{model}_temperature1.jsonl')
    if not os.path.exists(res_path):
        return []
    model_res = read_jsonl(res_path)
    samples = []
    for item in model_res:
        if not item.get('Res') or item['Res'] is None: # 没生成答案的不要
            continue
        if item.get('popularity') == 'No': # 没有question popularity的不要
            continue
        question_entity = item['question'].replace(PATTERN[dataset], '').lower()
        ref = remove_punctuation_edges(item['reference'][0], dataset)
        gene_entity = remove_punctuation_edges(item['Res'], dataset)

        # 对Qwen3.5-32B的movies，后来重新生成了一遍，有几个entity不一样，但是由于影响很小，没有再重新统计一下实体流行度。因此把这几个找不到的过滤一下
        if question_entity not in co_occu:
            continue
        if question_entity.lower() not in single_occr:
            continue
        if gene_entity.lower() not in single_occr:
            continue
        if ref.lower() not in single_occr:
            continue
        if dataset in ['movies', 'songs']: # basketabal数据集的实体统计不用过滤，因为是city name，都很常见
            if (single_occr[question_entity.lower()] > SINGLE_OCC_THRESHOLD or
                    single_occr[gene_entity.lower()] > SINGLE_OCC_THRESHOLD or
                    single_occr[ref.lower()] > SINGLE_OCC_THRESHOLD): # 过滤掉噪音实体
                continue
        else: # 对basketball, 只对question entity做限制，因为answer都是city name, 都很常见不做限制
            if single_occr[question_entity.lower()] > SINGLE_OCC_THRESHOLD:
                continue

        question_pop = item['popularity']
        # ── gt_pop / gene_pop 查询 ──
        ref_pop_info = full_dict.get(ref, {})
        ref_pop = ref_pop_info.get('popularity', 0) if isinstance(ref_pop_info, dict) else ref_pop_info
        gt_pop_missing = (ref_pop == 'No' or ref_pop is None or ref not in full_dict)
        if gt_pop_missing:
            ref_pop = 0

        gene_pop_info = full_dict.get(gene_entity, {})
        gene_pop = gene_pop_info.get('popularity', 0) if isinstance(gene_pop_info, dict) else gene_pop_info
        gene_pop_missing = (gene_pop == 'No' or gene_pop is None or gene_entity not in full_dict)
        if gene_pop_missing:
            gene_pop = 0

        # filter_pop_no=True: 跳过 gt_pop 或 gene_pop 在 Wikidata 查不到的样本
        if filter_pop_no and (gt_pop_missing or gene_pop_missing):
            continue
        # 计算信心
        if 'gpt' in model.lower():
            probs = [math.exp(t) for t in item['Log_p']['token_logprobs']]
        else:
            probs = item['Log_p']['token_probs']
        conf = sum(probs) / len(probs)
        # 获得共现
        gt_coo = co_occu[question_entity].get(ref.lower(), 0)
        gene_coo = co_occu[question_entity].get(gene_entity.lower(), 0)
        # 从wikipedia得到的single occurrence
        q_single = single_occr[question_entity.lower()]
        gt_single = single_occr[ref.lower()]
        total_coo = sum(co_occu[question_entity].values())
        # quality_v2 = gt_coo / q_single: question_entity 的 Wikipedia 文档中有多少比例同时提到了正确答案
        quality_v2 = gt_coo / q_single if q_single > 0 else 0

        pmi = 0
        if gt_coo > 0 and q_single > 0 and gt_single > 0:
            pmi = math.log2((gt_coo * 6_000_000) / (q_single * gt_single))

        samples.append({
            'question_pop': question_pop,
            'gt_pop': int(ref_pop) if ref_pop else 0,
            'gene_pop': int(gene_pop) if gene_pop else 0,
            'coo': gt_coo,
            'gene_coo': gene_coo,
            'q_single': q_single,
            'gt_single': gt_single,
            'total_coo': total_coo,
            'quality_v2': quality_v2,
            'pmi': pmi,
            'conf': conf,
            'acc': item['has_answer'],
        })
    return samples


def _skip_bball_qwen25(dataset, model):
    """Basketball 排除 Qwen2.5 系列（acc<8%，相关分析不可靠）"""
    return dataset == 'basketball' and model.startswith('Qwen2.5')


# ═══════════════════════════════════════════════════════════════════════════════
# RQ1: What makes a model learn a fact? (§4)
# ═══════════════════════════════════════════════════════════════════════════════

def table1_basic_stats(all_data):
    """表1: 基础分布统计（median_coo, median_qv2, coo=0%）

    对应论文 §4.1 (dataset statistics), Table 2 (bivariate correlations)
    过滤：q_single ∈ [50, 6000)，排除分母噪音
    用途：提供每个 dataset×model 的基础分布数据，验证合并是否影响统计
    """
    print("\n" + "="*100)
    print("表1 [RQ1 §4.1]: 基础分布 — 逐模型")
    print("="*100)
    print(f"{'dataset':12s} {'model':15s} {'n':>5s} | {'median_coo':>11s} {'median_qv2':>12s} {'coo=0%':>7s} | {'Spearman(coo,acc)':>18s} {'Spearman(qv2,acc)':>18s}")
    print("-" * 110)

    for dataset in DATASETS:
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = all_data.get((dataset, model), [])
            if len(samples) < 100:
                continue

            q_samples = [s for s in samples if Q_SINGLE_LO <= s['q_single'] < Q_SINGLE_HI]
            if len(q_samples) < 50:
                continue

            coo = np.array([s['coo'] for s in q_samples], dtype=float)
            quality = np.array([s['quality_v2'] for s in q_samples], dtype=float)
            acc = np.array([s['acc'] for s in q_samples], dtype=float)

            r_coo, _ = spearmanr(coo, acc) if np.std(coo) > 0 else (float('nan'), 1)
            r_qv2, _ = spearmanr(quality, acc) if np.std(quality) > 0 else (float('nan'), 1)

            print(f"{dataset:12s} {model:15s} {len(samples):5d} | "
                  f"{np.median(coo):11.1f} {np.median(quality):12.3f} {(coo==0).mean():7.1%} | "
                  f"{r_coo:+18.3f} {r_qv2:+18.3f}")


def table2_acc_three_var_partial(all_data):
    """表2: RQ1 三变量偏相关 — {gt_coo, qpop, gt_pop} → acc（全样本）

    对应论文 §4.1 (Movies 表格) + Appendix B (Songs/Basketball 表格)
    框架：对称三变量互控，每个因子控制其余两个
    变量集：gt_coo（知识共现）, question_pop（问题流行度）, gt_pop（正确答案流行度）
    核心发现：
      - Movies/Songs: gt_coo→acc 最强（+0.21 ~ +0.39），gt_pop→acc 近零
      - Basketball: gt_pop→acc 主导（+0.18 ~ +0.32），gt_coo→acc 较弱
    """
    # RQ1 偏相关因子：知识属性 {gt_coo, question_pop, gt_pop}
    ACC_FACTORS = ['coo', 'question_pop', 'gt_pop']

    print("\n" + "="*100)
    print("表2 [RQ1 §4.1]: 三变量互控偏相关 — {gt_coo, qpop, gt_pop} → acc")
    print("  每个因子控制其余两个 | 变量选择理由：acc 关注知识属性，不引入 gene_*")
    print("="*100)
    print(f"{'dataset':12s} {'model':15s} {'n':>5s} | {'gt_coo→acc':>10s} {'qpop→acc':>10s} {'gt_pop→acc':>10s}")
    print("-" * 70)

    for dataset in DATASETS:
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = all_data.get((dataset, model), [])
            if len(samples) < 100:
                continue

            res = three_var_partial(samples, ACC_FACTORS, 'acc')

            print(f"{dataset:12s} {model:15s} {len(samples):5d} | "
                  f"{res['coo']:+10.3f} {res['question_pop']:+10.3f} {res['gt_pop']:+10.3f}")


def table2_conf_three_var_partial(all_data):
    """表5: RQ2 三变量偏相关 — {gene_coo, qpop, gene_pop} → conf（全样本）

    对应论文 §5.1 (Movies 全样本表格) + Appendix C (Songs/Basketball 表格)
    框架：对称三变量互控，每个因子控制其余两个
    变量集：gene_coo（生成答案共现）, question_pop（问题流行度）, gene_pop（生成答案流行度）
    核心发现：
      - gene_coo→conf 是主要置信度驱动力（Movies/Songs +0.17 ~ +0.61）
      - gene_pop→conf 在 Movies 上中等（+0.04 ~ +0.24），在 Songs 上较弱
      - Basketball 上所有信号弱或负
    """
    # RQ2 偏相关因子：生成答案属性 {gene_coo, question_pop, gene_pop}
    CONF_FACTORS = ['gene_coo', 'question_pop', 'gene_pop']

    print("\n" + "="*100)
    print("表5 [RQ2 §5.1]: 三变量互控偏相关 — {gene_coo, qpop, gene_pop} → conf（全样本）")
    print("  每个因子控制其余两个 | 变量选择理由：conf 关注模型生成内容的属性")
    print("="*100)
    print(f"{'dataset':12s} {'model':15s} {'n':>5s} | {'gene_coo→conf':>13s} {'qpop→conf':>10s} {'gene_pop→conf':>13s}")
    print("-" * 80)

    for dataset in DATASETS:
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = all_data.get((dataset, model), [])
            if len(samples) < 100:
                continue

            res = three_var_partial(samples, CONF_FACTORS, 'conf')

            print(f"{dataset:12s} {model:15s} {len(samples):5d} | "
                  f"{res['gene_coo']:+13.3f} {res['question_pop']:+10.3f} {res['gene_pop']:+13.3f}")


def table3_quality_v2_partial(all_data):
    """表3: quality_v2→acc 偏相关（控制gt_coo）

    对应论文 §4.2, Finding #1 (single-control analysis)
    过滤：q_single ∈ [50, 6000)，排除分母噪音
    核心发现：控制 gt_coo 后，quality_v2→acc 在 15/15 组合中显著为正
    """
    print("\n" + "="*100)
    print("表3 [RQ1 §4.2]: quality_v2→acc 偏相关（控制gt_coo），逐模型逐数据集")
    print("="*100)
    print(f"{'dataset':12s} {'model':15s} {'n':>5s} | {'quality_v2→acc(控制coo)':>25s} | {'Spearman(coo,acc)':>18s}")
    print("-" * 80)

    for dataset in DATASETS:
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = all_data.get((dataset, model), [])
            if len(samples) < 100:
                continue

            q_samples = [s for s in samples if Q_SINGLE_LO <= s['q_single'] < Q_SINGLE_HI]
            if len(q_samples) < 50:
                continue

            quality = np.array([s['quality_v2'] for s in q_samples], dtype=float)
            coo = np.array([s['coo'] for s in q_samples], dtype=float)
            acc = np.array([s['acc'] for s in q_samples], dtype=float)

            r_q, _ = partial_spearman(quality, acc, coo) if np.std(quality) > 0 else (float('nan'), 1)
            r_coo_acc, _ = spearmanr(coo, acc) if np.std(coo) > 0 else (float('nan'), 1)

            print(f"{dataset:12s} {model:15s} {len(q_samples):5d} | {r_q:+25.3f} | {r_coo_acc:+18.3f}")


def table4_quadrant_analysis(all_data):
    """表4: 四象限分析 (coo × quality_v2) → avg_acc

    对应论文 §4.3 (quadrant analysis)
    核心发现：HH(高coo+高quality) avg_acc 最高，LL 最低；HH vs HL 差异显著
    """
    print("\n" + "="*100)
    print("表4 [RQ1 §4.3]: 四象限分析 (coo × quality_v2) → avg_acc，逐模型")
    print("="*100)
    for dataset in ['movies', 'basketball']:
        print(f"\n--- {dataset} ---")
        print(f"{'model':15s} | {'HH(acc/n)':>12s} {'HL(acc/n)':>12s} {'LH(acc/n)':>12s} {'LL(acc/n)':>12s} | {'HH-HL':>7s} p")
        print("-" * 85)
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = all_data.get((dataset, model), [])
            if len(samples) < 100:
                continue

            coo = np.array([s['coo'] for s in samples], dtype=float)
            quality = np.array([s['quality_v2'] for s in samples], dtype=float)
            acc = np.array([s['acc'] for s in samples], dtype=float)
            coo_med = np.median(coo)
            q_med = np.median(quality)

            quadrants = {
                'HH': (coo > coo_med) & (quality > q_med),
                'HL': (coo > coo_med) & (quality <= q_med),
                'LH': (coo <= coo_med) & (quality > q_med),
                'LL': (coo <= coo_med) & (quality <= q_med),
            }
            line = f"{model:15s} |"
            for qname in ['HH', 'HL', 'LH', 'LL']:
                m = quadrants[qname]
                n = m.sum()
                a = acc[m].mean() if n > 0 else float('nan')
                line += f" {a:.3f}/{n:4d}"
                line += f" |"
            hh = quadrants['HH']
            hl = quadrants['HL']
            if hh.sum() >= 5 and hl.sum() >= 5:
                from scipy.stats import mannwhitneyu
                stat, p = mannwhitneyu(acc[hh], acc[hl], alternative='two-sided')
                diff = acc[hh].mean() - acc[hl].mean()
                sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
                line += f" {diff:+.3f} {sig}"
            else:
                line += " N/A"
            print(line)


# ═══════════════════════════════════════════════════════════════════════════════
# RQ2: What drives confidence, and how does it differ from learning? (§5)
# ═══════════════════════════════════════════════════════════════════════════════

def table6_asymmetry_gene_pop(all_data):
    """表6: 做对 vs 做错 asymmetry — gene_pop→conf 偏相关（三变量互控）

    对应论文 §5.1 (Movies 表格) + Appendix C (Songs/Basketball 表格)
    框架：gene_pop→conf 控制 gene_coo + qpop（三变量互控中的 gene_pop 项）
    核心发现：
      - Movies: 开源模型 gene_pop→conf 从正确到错误放大 1.7–4.9×（popularity bias）
      - Songs: 放大比 0.7–1.3×，不对称性弱
      - Basketball: gene_pop→conf 为负或近零，popularity heuristic 崩溃
      - ChatGPT 在 Movies/Songs 错误样本上 gene_pop→conf 反号
    """
    CONF_FACTORS = ['gene_coo', 'question_pop', 'gene_pop']

    print("\n" + "="*100)
    print("表6 [RQ2 §5.1]: gene_pop→conf 做对 vs 做错 asymmetry（3-var 互控）")
    print("  gene_pop→conf 控制 gene_coo + qpop | 比例 = incorrect / correct")
    print("  比例为 '—' 表示两例均为负或 correct≤0 且不能合理解释放大")
    print("="*100)
    print(f"{'dataset':12s} {'model':15s} | {'correct':>8s} {'incorrect':>9s} | {'ratio':>6s}")
    print("-" * 60)

    for dataset in DATASETS:
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = all_data.get((dataset, model), [])
            correct = [s for s in samples if s['acc'] == 1]
            incorrect = [s for s in samples if s['acc'] == 0]

            r_corr = float('nan')
            r_incorr = float('nan')

            if len(correct) >= 30:
                res_c = three_var_partial(correct, CONF_FACTORS, 'conf')
                r_corr = res_c['gene_pop']
            if len(incorrect) >= 30:
                res_i = three_var_partial(incorrect, CONF_FACTORS, 'conf')
                r_incorr = res_i['gene_pop']

            # 计算放大比：仅当 correct>0 且 incorrect>0 时有意义
            if np.isnan(r_corr) or np.isnan(r_incorr) or r_corr <= 0 or r_incorr <= 0:
                ratio_str = '—'
            else:
                ratio_str = f'{r_incorr / r_corr:.1f}×'

            def fmt(v):
                return f'{v:+.3f}' if not np.isnan(v) else '—'

            print(f"{dataset:12s} {model:15s} | {fmt(r_corr):>8s} {fmt(r_incorr):>9s} | {ratio_str:>6s}")


def table7_incorrect_three_var(all_data):
    """表7: 错误样本 — 三因子互控偏相关 → conf

    对应论文 §5.2 (Movies 表格) + Appendix C (Songs/Basketball 表格)
    框架：{gene_pop, gene_coo, qpop} 每个控制其余两个 → conf
    核心发现：
      - Movies 错误样本：gene_pop→conf (+0.19~+0.31) 与 gene_coo→conf (+0.18~+0.27) 并存
      - Songs 错误样本：gene_coo→conf 主导（+0.14~+0.66），gene_pop→conf 较弱
      - Basketball 错误样本：qpop→conf 唯一稳定正信号（+0.16）
    """
    CONF_FACTORS = ['gene_coo', 'question_pop', 'gene_pop']

    print("\n" + "="*100)
    print("表7 [RQ2 §5.2]: 错误样本 — 三因子互控偏相关 → conf")
    print("  每个因子控制其余两个 | 仅做错样本")
    print("="*100)
    print(f"{'dataset':12s} {'model':15s} {'n_wrong':>7s} | {'gene_pop→conf':>13s} {'gene_coo→conf':>13s} {'qpop→conf':>10s}")
    print("-" * 80)

    for dataset in DATASETS:
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = all_data.get((dataset, model), [])
            incorrect = [s for s in samples if s['acc'] == 0]

            if len(incorrect) < 30:
                print(f"{dataset:12s} {model:15s} {len(incorrect):7d} | —")
                continue

            res = three_var_partial(incorrect, CONF_FACTORS, 'conf')

            print(f"{dataset:12s} {model:15s} {len(incorrect):7d} | "
                  f"{res['gene_pop']:+13.3f} {res['gene_coo']:+13.3f} {res['question_pop']:+10.3f}")


def table8_correct_vs_incorrect(all_data):
    """表8: 正确 vs 错误全因子分解 — 各因子→conf 在正确/错误子集上的对比

    对应论文 §5.2 (三个数据集的 correct/incorrect 全因子表) + Appendix C
    框架：{gene_pop, gene_coo, qpop} 每个控制其余两个 → conf
    核心发现：
      - Movies: gene_pop→conf 从正确到错误放大 2.0–4.9×（开源模型），gene_coo→conf 稳定
      - Songs: gene_coo→conf 放大（correct +0.01~+0.18 → incorrect +0.14~+0.66），gene_pop→conf 弱不对称
      - Basketball: gene_pop→conf 为负或近零，popularity heuristic 崩溃
      - qpop→conf 在所有数据集上衰减而非放大（正确时更强）
    """
    CONF_FACTORS = ['gene_coo', 'question_pop', 'gene_pop']

    print("\n" + "="*130)
    print("表8 [RQ2 §5.2]: 正确 vs 错误全因子分解 — 各因子→conf 在正确/错误子集上的对比")
    print("  每个因子控制其余两个 | 揭示哪些信号从正确到错误 '放大'（familiarity heuristic）")
    print("="*130)

    for dataset in DATASETS:
        print(f"\n--- {dataset.upper()} ---")
        print(f"{'model':15s} | {'gene_pop(c)':>11s} {'gene_coo(c)':>11s} {'qpop(c)':>8s} | {'gene_pop(i)':>11s} {'gene_coo(i)':>11s} {'qpop(i)':>8s}")
        print("-" * 90)

        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = all_data.get((dataset, model), [])
            correct = [s for s in samples if s['acc'] == 1]
            incorrect = [s for s in samples if s['acc'] == 0]

            r_c = {'gene_pop': float('nan'), 'gene_coo': float('nan'), 'question_pop': float('nan')}
            r_i = {'gene_pop': float('nan'), 'gene_coo': float('nan'), 'question_pop': float('nan')}

            if len(correct) >= 30:
                r_c = three_var_partial(correct, CONF_FACTORS, 'conf')
            if len(incorrect) >= 30:
                r_i = three_var_partial(incorrect, CONF_FACTORS, 'conf')

            def fmt(v):
                return f'{v:+.3f}' if not np.isnan(v) else '—'

            print(f"{model:15s} | "
                  f"{fmt(r_c['gene_pop']):>11s} {fmt(r_c['gene_coo']):>11s} {fmt(r_c['question_pop']):>8s} | "
                  f"{fmt(r_i['gene_pop']):>11s} {fmt(r_i['gene_coo']):>11s} {fmt(r_i['question_pop']):>8s}")


def table9_gene_pop_gt_pop_ratio(all_data):
    """表9: 做错时 gene_pop>gt_pop / gene_coo>gt_coo 比例

    对应论文 §5.3 (ChatGPT outlier discussion)
    方法：纯描述性统计，不涉及偏相关框架
    核心发现：ChatGPT 做错时 gene_coo>gt_coo 比例 (29.7–36.2%) 远超开源模型 (4–11%)
    """
    print("\n" + "="*130)
    print("表9 [RQ2 §5.3]: 做错时趋向更'知名'/'关联强'的错误答案，逐模型逐数据集")
    print("  gene_pop>gt_pop 比例：做错时生成实体流行度 > 正确答案流行度的比例")
    print("  gene_coo>gt_coo 比例：做错时生成实体与问题的共现 > 正确答案与问题的共现的比例")
    print("="*130)
    print(f"{'dataset':12s} {'model':15s} {'n_wrong':>7s} | "
          f"{'gene_pop>gt_pop':>16s} {'Sp(gpop,conf)':>14s} | "
          f"{'gene_coo>gt_coo':>16s} {'Sp(gcoo,conf)':>14s}")
    print("-" * 100)

    for dataset in DATASETS:
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = all_data.get((dataset, model), [])
            wrong = [s for s in samples if s['acc'] == 0]
            if len(wrong) < 30:
                continue

            w_gene_pop = np.array([s['gene_pop'] for s in wrong], dtype=float)
            w_gt_pop   = np.array([s['gt_pop']   for s in wrong], dtype=float)
            w_gene_coo = np.array([s['gene_coo'] for s in wrong], dtype=float)
            w_gt_coo   = np.array([s['coo']      for s in wrong], dtype=float)
            w_conf     = np.array([s['conf']      for s in wrong], dtype=float)

            ratio_pop = (w_gene_pop > w_gt_pop).mean()
            ratio_coo = (w_gene_coo > w_gt_coo).mean()
            r_pop, _ = spearmanr(w_gene_pop, w_conf) if np.std(w_gene_pop) > 0 else (float('nan'), 1)
            r_coo, _ = spearmanr(w_gene_coo, w_conf) if np.std(w_gene_coo) > 0 else (float('nan'), 1)

            print(f"{dataset:12s} {model:15s} {len(wrong):7d} | "
                  f"{ratio_pop:16.3f} {r_pop:+14.3f} | "
                  f"{ratio_coo:16.3f} {r_coo:+14.3f}")


def _variation_stats(arr, conf):
    """计算变异度统计量 (mean, std, CV, P10, P90, P90/P10, 四分位conf趋势)"""
    mean_v = arr.mean()
    std_v = arr.std()
    cv = std_v / max(mean_v, 1)
    p10, p90 = np.percentile(arr, [10, 90])
    ratio = p90 / max(p10, 1)

    pcts = np.percentile(arr, [0, 25, 50, 75, 100])
    q_confs = []
    for i in range(4):
        lo, hi = pcts[i], pcts[i+1]
        m = (arr >= lo) & (arr <= hi) if i == 0 else (arr > lo) & (arr <= hi)
        if m.sum() > 5:
            q_confs.append(f"{conf[m].mean():.3f}")
        else:
            q_confs.append("N/A")
    trend = "→".join(q_confs)
    return mean_v, std_v, cv, p10, p90, ratio, trend


def table10_gene_pop_variation(all_data):
    """表10: gene_pop / gene_coo / question_pop 变异度（CV, P10/P90, 四分位conf趋势）

    对应论文 §5.4 (why basketball lacks popularity asymmetry)
    方法：纯描述性统计，不涉及偏相关框架
    核心发现：basketball 的 gene_pop CV 远低于 movies/songs → 变异不足无法检测相关
    """
    print("\n" + "="*140)
    print("表10 [RQ2 §5.4]: gene_pop / gene_coo / question_pop 变异度 — 逐模型不合并（做错样本）")
    print("="*140)
    signals = [
        ('gene_pop', 'gene_pop'),
        ('gene_coo', 'gene_coo'),
        ('question_pop', 'question_pop'),
    ]
    for sig_name, sig_key in signals:
        print(f"\n--- {sig_name} (wrong answers) ---")
        print(f"{'dataset':12s} {'model':15s} | {'mean':>10s} {'std':>10s} {'CV':>6s} {'P10':>6s} {'P90':>6s} {'P90/P10':>8s} | {'四分位conf趋势':>25s}")
        print("-" * 120)
        for dataset in DATASETS:
            for model in MODELS:
                if _skip_bball_qwen25(dataset, model):
                    continue
                samples = all_data.get((dataset, model), [])
                if len(samples) < 100:
                    continue
                wrong = [s for s in samples if s['acc'] == 0]
                if len(wrong) < 30:
                    continue
                arr = np.array([s[sig_key] for s in wrong], dtype=float)
                conf = np.array([s['conf'] for s in wrong], dtype=float)
                mean_v, std_v, cv, p10, p90, ratio, trend = _variation_stats(arr, conf)
                print(f"{dataset:12s} {model:15s} | {mean_v:10.1f} {std_v:10.1f} {cv:6.2f} {p10:6.0f} {p90:6.0f} {ratio:8.1f} | {trend:>25s}")


def table11_correct_spearman(all_data):
    """表11: 做对时 Spearman(gene_pop, conf) vs 做错时

    对应论文 §5.4 (basketball negative correlation verification)
    方法：双变量 Spearman 相关（非偏相关），用于验证 basketball 的异常模式
    核心发现：basketball 做对时 gene_pop→conf 为负相关（与 movies/songs 相反）
      → 因为 basketball 的 gene_entity 是城市名，pop 变异极低
    """
    print("\n" + "="*100)
    print("表11 [RQ2 §5.4]: 做对时 Spearman(gene_pop, conf)，逐模型逐数据集")
    print("="*100)
    print(f"{'dataset':12s} {'model':15s} | {'n_right':>7s} {'Spearman(gene_pop,conf|做对)':>30s} | {'Spearman(gene_pop,conf|做错)':>30s}")
    print("-" * 100)

    for dataset in DATASETS:
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = all_data.get((dataset, model), [])
            right = [s for s in samples if s['acc'] == 1]
            wrong = [s for s in samples if s['acc'] == 0]

            def spear(arr1, arr2):
                if len(arr1) < 20 or np.std(arr1) == 0:
                    return float('nan')
                return spearmanr(arr1, arr2)[0]

            gp_right = np.array([s['gene_pop'] for s in right], dtype=float) if len(right) >= 20 else np.array([])
            conf_right = np.array([s['conf'] for s in right], dtype=float) if len(right) >= 20 else np.array([])
            gp_wrong = np.array([s['gene_pop'] for s in wrong], dtype=float) if len(wrong) >= 20 else np.array([])
            conf_wrong = np.array([s['conf'] for s in wrong], dtype=float) if len(wrong) >= 20 else np.array([])

            r_right = spear(gp_right, conf_right)
            r_wrong = spear(gp_wrong, conf_wrong)

            print(f"{dataset:12s} {model:15s} | {len(right):7d} {r_right:+30.3f} | {len(wrong):30d} {r_wrong:+30.3f}" if len(wrong) >= 20 else
                  f"{dataset:12s} {model:15s} | {len(right):7d} {r_right:+30.3f} | {'<20':>30s}")


# ═══════════════════════════════════════════════════════════════════════════════
# §7: Using External Signals for Calibration
# ═══════════════════════════════════════════════════════════════════════════════

def table12_fixed_conf_bin(all_data):
    """表12: 固定 conf 区间，做对 vs 做错的 gene_pop/gene_coo 差异

    对应论文 §7.1 (signal-aware calibration feature analysis) + Appendix D
    方法：按 conf 四分位分层，对比正确/错误样本的 gene_coo 和 gene_pop 均值差
    核心逻辑：
      - conf 相近时，做对样本 gene_coo 更高、gene_pop 更低
      - gene_coo 高说明答案可信（高 conf 有效）
      - gene_pop 高说明可能是流行度偏差（高 conf 虚高）
      - 将 gene_pop/gene_coo 加入预测，可在 conf 相同时区分做对和做错
    """
    print("\n" + "="*130)
    print("表12 [§7.1]: 固定 conf 区间，做对 vs 做错样本的 gene_pop 和 gene_coo 对比")
    print("  核心逻辑：conf 相近时，做对样本 gene_coo 是否更高？gene_pop 是否更低？")
    print("  若是，则说明在 conf 相同的情况下，gene_coo/gene_pop 能区分做对和做错，实现纠偏")
    print("="*130)

    from scipy.stats import mannwhitneyu

    for dataset in ['movies', 'songs', 'basketball']:
        print(f"\n--- {dataset} ---")
        print(f"{'model':15s} | {'conf区间':>12s} | "
              f"{'n_right':>7s} {'gcoo_right':>11s} {'gpop_right':>11s} | "
              f"{'n_wrong':>7s} {'gcoo_wrong':>11s} {'gpop_wrong':>11s} | "
              f"{'Δgcoo(r-w)':>12s} {'Δgpop(r-w)':>12s}")
        print("-" * 130)

        for model in MODELS:
            samples = all_data.get((dataset, model), [])
            if len(samples) < 100:
                continue

            all_conf = np.array([s['conf'] for s in samples], dtype=float)
            quartiles = np.percentile(all_conf, [0, 25, 50, 75, 100])
            bin_labels = ['Q1(低conf)', 'Q2', 'Q3', 'Q4(高conf)']

            first_row = True
            for i in range(4):
                lo, hi = quartiles[i], quartiles[i+1]
                if i == 3:
                    bin_samples = [s for s in samples if lo <= s['conf'] <= hi]
                else:
                    bin_samples = [s for s in samples if lo <= s['conf'] < hi]

                right = [s for s in bin_samples if s['acc'] == 1]
                wrong = [s for s in bin_samples if s['acc'] == 0]
                if len(right) < 10 or len(wrong) < 10:
                    continue

                r_gcoo = np.array([s['gene_coo'] for s in right], dtype=float)
                r_gpop = np.array([s['gene_pop'] for s in right], dtype=float)
                w_gcoo = np.array([s['gene_coo'] for s in wrong], dtype=float)
                w_gpop = np.array([s['gene_pop'] for s in wrong], dtype=float)

                delta_gcoo = r_gcoo.mean() - w_gcoo.mean()
                delta_gpop = r_gpop.mean() - w_gpop.mean()

                model_str = model if first_row else " " * 15
                first_row = False
                print(f"{model_str:15s} | {bin_labels[i]:>12s} | "
                      f"{len(right):7d} {r_gcoo.mean():11.2f} {r_gpop.mean():11.1f} | "
                      f"{len(wrong):7d} {w_gcoo.mean():11.2f} {w_gpop.mean():11.1f} | "
                      f"{delta_gcoo:+12.2f} {delta_gpop:+12.1f}")
            print()


# ═══════════════════════════════════════════════════════════════════════════════
# Table 0: Baseline Statistics (Acc, Conf, W-Conf, ECE) — 论文 Table 1
# ═══════════════════════════════════════════════════════════════════════════════

def _ece(acc_array, conf_array, n_bins=10):
    """计算 Expected Calibration Error (ECE)"""
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    n = len(acc_array)
    for i in range(n_bins):
        mask = (conf_array >= bins[i]) & (conf_array < bins[i + 1])
        bin_acc = acc_array[mask]
        bin_conf = conf_array[mask]
        if len(bin_acc) == 0:
            continue
        ece_val += (len(bin_acc) / n) * np.abs(bin_acc.mean() - bin_conf.mean())
    return ece_val


def table0_baseline_stats(all_data):
    """表0: Baseline 准确率、平均置信度、错误答案置信度、ECE

    对应论文 Table 1 (§3 Baseline: Overconfidence is Universal)
    注意：此函数输出所有 dataset × model（包括 Basketball Qwen2.5 系列），
    与论文正文中 correlation 分析排除 Qwen2.5 Basketball 不同，
    baseline 统计表需要展示所有模型的原始数据。
    """
    print("\n" + "=" * 100)
    print("表0 [§3 Baseline]: Baseline Statistics — Acc, Conf, W-Conf, ECE")
    print("  所有 dataset × model 组合，包括 Basketball Qwen2.5 系列")
    print("=" * 100)
    print(f"{'dataset':12s} {'model':15s} {'n':>6s} {'n_right':>7s} {'n_wrong':>7s} | "
          f"{'Acc':>6s} {'Conf':>6s} {'W-Conf':>7s} {'ECE':>6s}")
    print("-" * 100)

    for dataset in DATASETS:
        for model in MODELS:
            # baseline 表不过滤 Basketball Qwen2.5 — 展示所有模型原始数据
            samples = all_data.get((dataset, model), [])
            if len(samples) < 10:
                print(f"{dataset:12s} {model:15s}  (no data or insufficient samples)")
                continue

            accs = np.array([s['acc'] for s in samples], dtype=float)
            confs = np.array([s['conf'] for s in samples], dtype=float)
            n = len(accs)
            n_right = int(accs.sum())
            n_wrong = n - n_right

            avg_acc = accs.mean()
            avg_conf = confs.mean()
            wrong_confs = confs[accs == 0]
            wrong_conf = wrong_confs.mean() if len(wrong_confs) > 0 else float('nan')
            ece_val = _ece(accs, confs)

            print(f"{dataset:12s} {model:15s} {n:6d} {n_right:7d} {n_wrong:7d} | "
                  f"{avg_acc:.3f} {avg_conf:.3f} {wrong_conf:.3f} {ece_val:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# main: 按 RQ1 → RQ2 → §7 顺序输出所有表格
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("加载数据...")
    full_dict = load_popularity()
    co_occu = json.loads(open(COO_PATH).read())
    single_occr = json.loads(open(SINGLE_PATH).read())

    # 收集所有 dataset × model 数据（filter_pop_no=True 过滤无 popularity 样本）
    # correlation 分析排除 Basketball Qwen2.5（acc<8%，相关不稳定），
    # 但 baseline 统计表需要这些数据
    all_data = {}
    all_data_with_bball_qwen25 = {}  # 包含 Basketball Qwen2.5，用于 baseline 表
    for dataset in DATASETS:
        for model in MODELS:
            samples = collect_samples(dataset, model, full_dict, co_occu, single_occr,
                                        filter_pop_no=True)
            all_data_with_bball_qwen25[(dataset, model)] = samples
            if not _skip_bball_qwen25(dataset, model):
                all_data[(dataset, model)] = samples
            n = len(samples)
            n_wrong = sum(1 for s in samples if s['acc'] == 0)
            n_right = n - n_wrong
            skip_tag = " (skip corr)" if _skip_bball_qwen25(dataset, model) else ""
            print(f"  {dataset:12s} × {model:15s}: n={n:5d}  做对={n_right:5d}  做错={n_wrong:5d}{skip_tag}")

    # ── Baseline: Table 0 — 所有模型（含 Basketball Qwen2.5）的 Acc/Conf/W-Conf/ECE ──
    table0_baseline_stats(all_data_with_bball_qwen25)    # §3 Baseline 统计表

    # ── RQ1: What makes a model learn a fact? (§4) ──
    # 框架：{gt_coo, qpop, gt_pop} → acc，每个控制其余两个
    table1_basic_stats(all_data)               # §4.1 基础分布
    table2_acc_three_var_partial(all_data)     # §4.1 三变量偏相关 + Appendix B
    table3_quality_v2_partial(all_data)        # §4.2 quality_v2 分析
    table4_quadrant_analysis(all_data)         # §4.3 四象限分析

    # ── RQ2: What drives confidence? (§5) ──
    # 框架：{gene_coo, qpop, gene_pop} → conf，每个控制其余两个
    table2_conf_three_var_partial(all_data)    # §5.1 全样本三变量偏相关 + Appendix C
    table6_asymmetry_gene_pop(all_data)        # §5.1 gene_pop→conf 正确/错误不对称 + Appendix C
    table7_incorrect_three_var(all_data)       # §5.2 错误样本三因子互控 + Appendix C
    table8_correct_vs_incorrect(all_data)      # §5.2 正确/错误全因子分解 + Appendix C
    table9_gene_pop_gt_pop_ratio(all_data)     # §5.3 错误答案偏向
    table10_gene_pop_variation(all_data)       # §5.4 变异度
    table11_correct_spearman(all_data)         # §5.4 basketball 负相关验证

    # ── §7: Using External Signals for Calibration ──
    table12_fixed_conf_bin(all_data)           # §7.1 conf 分层特征分析 + Appendix D

    print("\n\n完成！所有分析均按 dataset × model 独立进行，无跨模型合并。")


if __name__ == '__main__':
    main()
