"""
模型规模 Scaling 分析（Scaling Analysis）

研究问题：
  随着模型规模增大（Qwen2.5-7B → 14B → 32B），
  哪种流行度对模型表现的影响在减弱？哪种影响持续存在？

具体分析：
  1. 三种流行度（question_pop, gt_pop, coo）与 acc/conf/alignment 的 Spearman 相关系数
     随模型规模的变化趋势
  2. 平均 acc、conf、alignment 随规模的变化
  3. 过度自信率（conf > acc 的比例）随规模的变化
  4. 低流行度样本 vs 高流行度样本的 acc gap 随规模的变化
     （如果大模型能弥补低流行度的劣势，gap 应该缩小）

预期结论：
  - coo 与 acc 的相关性随规模增大而减弱 → 大模型能更好地记住低共现知识
  - question_pop 与 conf 的相关性随规模变化不大 → 过度自信是系统性问题，scaling 无法解决
  - 如果以上成立 → 对 LLM 社区有直接实践意义：scaling 提升 acc 但不能修复 calibration
  - 如果不成立 → 说明 scaling 对两者影响一致，需要其他方法解决过度自信
"""

import json
import math
import re
import os
import numpy as np
from scipy.stats import spearmanr

# ─── 路径配置 ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES_DIR = os.path.join(BASE, 'res')

POP_PATH = os.path.join(RES_DIR, 'gt_gene_entity_popularity_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.jsonl')
COO_PATH = os.path.join(RES_DIR, 'cooccurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')
SINGLE_PATH = os.path.join(RES_DIR, 'single_occurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')

DATASETS = ['movies', 'songs', 'basketball']

# Qwen2.5 系列，按规模排序
QWEN_MODELS = ['Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']
QWEN_PARAMS = [7, 14, 32]  # 单位：B

# 所有模型（用于横向对比）
ALL_MODELS = ['llama8b', 'qwen2', 'chatgpt', 'Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']

PATTERN = {
    'movies': 'Who is the director of the movie ',
    'songs': 'Who is the performer of the song ',
    'basketball': 'Where is the birthplace of the basketball player '
}
SINGLE_OCC_THRESHOLD = 6000


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


def collect_samples(dataset, model, full_dict, co_occu, single_occr):
    res_path = os.path.join(RES_DIR, dataset, f'{dataset}_{model}_temperature1.jsonl')
    if not os.path.exists(res_path):
        return []

    model_res = read_jsonl(res_path)
    samples = []
    skip_stat_miss = 0

    for item in model_res:
        if not item.get('Res') or item['Res'] is None:
            continue
        if item.get('popularity') == 'No':
            continue

        question_entity = item['question'].replace(PATTERN[dataset], '').lower()
        ref = remove_punctuation_edges(item['reference'][0], dataset)
        gene_entity = remove_punctuation_edges(item['Res'], dataset)

        # 检查统计文件中是否有对应实体，找不到则跳过
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
        # movies/songs: 三个实体都可能有问题（歧义匹配），全部过滤
        # basketball: question_entity 是人名，可能出现歧义；gene_entity/ref 是城市名，天然高频，不过滤
        if dataset in ['movies', 'songs']:
            if (single_occr[question_entity.lower()] > SINGLE_OCC_THRESHOLD or
                    single_occr[gene_entity.lower()] > SINGLE_OCC_THRESHOLD or
                    single_occr[ref.lower()] > SINGLE_OCC_THRESHOLD):
                continue
        else:  # basketball
            if single_occr[question_entity.lower()] > SINGLE_OCC_THRESHOLD:
                continue

        ref_pop = full_dict.get(ref, {})
        ref_pop = ref_pop.get('popularity', 0) if isinstance(ref_pop, dict) else ref_pop
        if ref_pop == 'No':
            ref_pop = 0

        if 'gpt' in model.lower():
            probs = [math.exp(t) for t in item['Log_p']['token_logprobs']]
        else:
            probs = item['Log_p']['token_probs']
        conf = sum(probs) / len(probs)

        coo = co_occu[question_entity].get(ref.lower(), 0)
        align = 1 - abs(item['has_answer'] - conf)

        samples.append({
            'question_pop': item['popularity'],
            'gt_pop': ref_pop,
            'coo': coo,
            'conf': conf,
            'acc': item['has_answer'],
            'align': align,
            'overconf': conf - item['has_answer'],  # 正值=过度自信，负值=保守
        })

    if skip_stat_miss > 0:
        print(f"  [{dataset} × {model}] 跳过 {skip_stat_miss} 条统计文件中缺失实体的样本")

    return samples


def _ece(acc, conf, n_bins=10):
    """计算 Expected Calibration Error"""
    acc = np.asarray(acc, dtype=float)
    conf = np.asarray(conf, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(acc)
    for i in range(n_bins):
        mask = (conf >= bins[i]) & (conf < bins[i + 1]) if i < n_bins - 1 else (conf >= bins[i]) & (conf <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(acc[mask].mean() - conf[mask].mean())
    return ece


def compute_stats(samples):
    """计算一组样本的核心统计量"""
    if not samples:
        return None
    qpop = np.array([s['question_pop'] for s in samples], dtype=float)
    gt_pop = np.array([s['gt_pop'] for s in samples], dtype=float)
    coo = np.array([s['coo'] for s in samples], dtype=float)
    conf = np.array([s['conf'] for s in samples], dtype=float)
    acc = np.array([s['acc'] for s in samples], dtype=float)
    align = np.array([s['align'] for s in samples], dtype=float)
    overconf = np.array([s['overconf'] for s in samples], dtype=float)

    # ECE (Expected Calibration Error)
    stats = {
        'n': len(samples),
        'avg_acc': acc.mean(),
        'avg_conf': conf.mean(),
        'avg_align': align.mean(),
        'overconf_rate': (overconf > 0).mean(),  # conf > acc 的比例
        'avg_overconf_magnitude': overconf[overconf > 0].mean() if (overconf > 0).any() else 0,
        'ece': _ece(acc, conf),
    }

    # Spearman 相关
    for pop_name, pop_arr in [('question_pop', qpop), ('gt_pop', gt_pop), ('coo', coo)]:
        r_acc, _ = spearmanr(pop_arr, acc)
        r_conf, _ = spearmanr(pop_arr, conf)
        r_align, _ = spearmanr(pop_arr, align)
        stats[f'r_acc_{pop_name}'] = r_acc
        stats[f'r_conf_{pop_name}'] = r_conf
        stats[f'r_align_{pop_name}'] = r_align

    # 低/高流行度 acc gap（按 coo 分）
    coo_median = np.median(coo)
    high_coo_acc = acc[coo > coo_median].mean() if (coo > coo_median).any() else 0
    low_coo_acc = acc[coo <= coo_median].mean() if (coo <= coo_median).any() else 0
    stats['acc_gap_by_coo'] = high_coo_acc - low_coo_acc

    # 低/高 question_pop 的 conf gap
    qpop_median = np.median(qpop)
    high_qpop_conf = conf[qpop > qpop_median].mean() if (qpop > qpop_median).any() else 0
    low_qpop_conf = conf[qpop <= qpop_median].mean() if (qpop <= qpop_median).any() else 0
    stats['conf_gap_by_qpop'] = high_qpop_conf - low_qpop_conf

    return stats


# ─── 主程序 ───────────────────────────────────────────────────────────────────
def main():
    print("加载数据...")
    full_dict = load_popularity()
    co_occu = json.loads(open(COO_PATH).read())
    single_occr = json.loads(open(SINGLE_PATH).read())

    # ── 1. Qwen2.5 规模效应分析 ────────────────────────────────────────────
    print("\n" + "="*70)
    print("【分析1】Qwen2.5 规模效应：7B → 14B → 32B")
    print("="*70)

    for dataset in DATASETS:
        print(f"\n--- Dataset: {dataset} ---")
        model_stats = {}
        for model in QWEN_MODELS:
            samples = collect_samples(dataset, model, full_dict, co_occu, single_occr)
            if not samples:
                print(f"  {model}: 数据文件不存在，跳过")
                continue
            stats = compute_stats(samples)
            model_stats[model] = stats

        if len(model_stats) < 2:
            print("  数据不足，跳过")
            continue

        # 打印核心指标随规模的变化
        print(f"\n  {'指标':<35}", end='')
        for m in QWEN_MODELS:
            if m in model_stats:
                print(f"  {m:>14}", end='')
        print()
        print(f"  {'-'*35}", end='')
        for m in QWEN_MODELS:
            if m in model_stats:
                print(f"  {'-'*14}", end='')
        print()

        metrics_to_show = [
            ('avg_acc', '平均 acc'),
            ('avg_conf', '平均 conf'),
            ('avg_align', '平均 alignment'),
            ('overconf_rate', '过度自信率 (conf>acc)'),
            ('ece', 'ECE'),
            ('r_acc_coo', 'Spearman(coo, acc)'),
            ('r_conf_question_pop', 'Spearman(question_pop, conf)'),
            ('r_acc_question_pop', 'Spearman(question_pop, acc)'),
            ('r_conf_coo', 'Spearman(coo, conf)'),
            ('acc_gap_by_coo', 'acc gap (高coo - 低coo)'),
            ('conf_gap_by_qpop', 'conf gap (高qpop - 低qpop)'),
        ]

        for key, label in metrics_to_show:
            print(f"  {label:<35}", end='')
            for m in QWEN_MODELS:
                if m in model_stats:
                    val = model_stats[m].get(key, float('nan'))
                    print(f"  {val:>14.3f}", end='')
            print()

        # 趋势分析
        print(f"\n  [趋势分析]")
        available = [m for m in QWEN_MODELS if m in model_stats]
        if len(available) >= 2:
            first, last = available[0], available[-1]
            for key, label in [
                ('r_acc_coo', 'Spearman(coo, acc)'),
                ('r_conf_question_pop', 'Spearman(question_pop, conf)'),
                ('overconf_rate', '过度自信率'),
                ('ece', 'ECE'),
                ('acc_gap_by_coo', 'acc gap by coo'),
            ]:
                v_first = model_stats[first].get(key, 0)
                v_last = model_stats[last].get(key, 0)
                delta = v_last - v_first
                direction = '↑ 增大' if delta > 0.01 else ('↓ 减小' if delta < -0.01 else '→ 基本不变')
                print(f"    {label}: {v_first:.3f} → {v_last:.3f}  {direction} (Δ={delta:+.3f})")

    # ── 2. 所有模型横向对比（逐数据集，不跨数据集合并）────────────────────
    print("\n\n" + "="*70)
    print("【分析2】所有模型横向对比（逐数据集，不合并）")
    print("="*70)

    metrics_to_show = [
        ('avg_acc', '平均 acc'),
        ('avg_conf', '平均 conf'),
        ('overconf_rate', '过度自信率'),
        ('ece', 'ECE'),
        ('r_acc_coo', 'Spearman(coo, acc)'),
        ('r_conf_question_pop', 'Spearman(question_pop, conf)'),
        ('r_acc_question_pop', 'Spearman(question_pop, acc)'),
        ('r_conf_coo', 'Spearman(coo, conf)'),
        ('acc_gap_by_coo', 'acc gap (高coo - 低coo)'),
        ('conf_gap_by_qpop', 'conf gap (高qpop - 低qpop)'),
    ]

    for dataset in DATASETS:
        print(f"\n--- Dataset: {dataset} ---")
        dataset_model_stats = {}
        for model in ALL_MODELS:
            if dataset == 'basketball' and model.startswith('Qwen2.5'):
                continue  # basketball 上 Qwen2.5 acc 太低，不纳入横向对比
            samples = collect_samples(dataset, model, full_dict, co_occu, single_occr)
            if not samples:
                continue
            dataset_model_stats[model] = compute_stats(samples)

        available_models = [m for m in ALL_MODELS if m in dataset_model_stats]
        print(f"  {'指标':<35}", end='')
        for m in available_models:
            print(f"  {m:>14}", end='')
        print()

        for key, label in metrics_to_show:
            print(f"  {label:<35}", end='')
            for m in available_models:
                val = dataset_model_stats[m].get(key, float('nan'))
                print(f"  {val:>14.3f}", end='')
            print()

    # ── 3. 关键发现总结 ────────────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("【关键发现总结】")
    print("="*70)
    print("""
预期结论解读：
  A. 如果 Spearman(coo, acc) 随规模增大而增大：
     → 大模型对共现信号更敏感，低共现知识更难学会
  B. 如果 Spearman(question_pop, conf) 随规模变化不大：
     → 过度自信与 question_pop 的绑定是系统性问题，scaling 无法解决
  C. 如果 A 和 B 同时成立：
     → 核心 insight: "scaling 提升 acc 但不能修复 calibration"
  D. 如果 overconf_rate 随规模增大而增大：
     → 大模型更自信但不一定更准确，calibration 问题随 scaling 恶化
""")


if __name__ == '__main__':
    main()
