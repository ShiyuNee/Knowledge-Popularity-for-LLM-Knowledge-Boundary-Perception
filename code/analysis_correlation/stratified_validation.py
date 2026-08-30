"""
分层验证：为 §5.3 补充 Songs 和 Basketball 的分层验证结果
运行 rq2_quality_within_coo_bins 按 dataset × model
"""

import json
import math
import re
import os
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu, pearsonr, rankdata

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES_DIR = os.path.join(BASE, 'res')

POP_PATH = os.path.join(RES_DIR, 'gt_gene_entity_popularity_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.jsonl')
COO_PATH = os.path.join(RES_DIR, 'cooccurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')
SINGLE_PATH = os.path.join(RES_DIR, 'single_occurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')

DATASETS = ['songs', 'basketball']
MODELS = ['llama8b', 'qwen2']

PATTERN = {
    'movies': 'Who is the director of the movie ',
    'songs': 'Who is the performer of the song ',
    'basketball': 'Where is the birthplace of the basketball player '
}

SINGLE_OCC_THRESHOLD = 6000
Q_SINGLE_LO = 50
Q_SINGLE_HI = 6000


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

    for item in model_res:
        if not item.get('Res') or item['Res'] is None:
            continue
        if item.get('popularity') == 'No':
            continue

        question_entity = item['question'].replace(PATTERN[dataset], '').lower()
        ref = remove_punctuation_edges(item['reference'][0], dataset)
        gene_entity = remove_punctuation_edges(item['Res'], dataset)

        if question_entity not in co_occu:
            continue
        if question_entity.lower() not in single_occr:
            continue
        if gene_entity.lower() not in single_occr:
            continue
        if ref.lower() not in single_occr:
            continue

        if dataset in ['movies', 'songs']:
            if (single_occr[question_entity.lower()] > SINGLE_OCC_THRESHOLD or
                    single_occr[gene_entity.lower()] > SINGLE_OCC_THRESHOLD or
                    single_occr[ref.lower()] > SINGLE_OCC_THRESHOLD):
                continue
        else:
            if single_occr[question_entity.lower()] > SINGLE_OCC_THRESHOLD:
                continue

        question_pop = item['popularity']
        ref_pop_info = full_dict.get(ref, {})
        ref_pop = ref_pop_info.get('popularity', 0) if isinstance(ref_pop_info, dict) else ref_pop_info
        if ref_pop == 'No' or ref_pop is None:
            ref_pop = 0
        gene_pop_info = full_dict.get(gene_entity, {})
        gene_pop = gene_pop_info.get('popularity', 0) if isinstance(gene_pop_info, dict) else gene_pop_info
        if gene_pop == 'No' or gene_pop is None:
            gene_pop = 0

        if 'gpt' in model.lower():
            probs = [math.exp(t) for t in item['Log_p']['token_logprobs']]
        else:
            probs = item['Log_p']['token_probs']
        conf = sum(probs) / len(probs)

        gt_coo = co_occu[question_entity].get(ref.lower(), 0)
        gene_coo = co_occu[question_entity].get(gene_entity.lower(), 0)
        q_single = single_occr[question_entity.lower()]
        quality_v2 = gt_coo / q_single if q_single > 0 else 0

        samples.append({
            'question_pop': question_pop,
            'gt_pop': int(ref_pop) if ref_pop else 0,
            'gene_pop': int(gene_pop) if gene_pop else 0,
            'coo': gt_coo,
            'gene_coo': gene_coo,
            'q_single': q_single,
            'quality_v2': quality_v2,
            'conf': conf,
            'acc': item['has_answer'],
        })

    return samples


def stratified_by_coo(samples, label=''):
    """固定 gt_coo 区间，对比 low vs high quality_v2 的准确率"""
    samples = [s for s in samples if Q_SINGLE_LO <= s['q_single'] < Q_SINGLE_HI]
    if len(samples) < 50:
        print(f"  [skip] {label}: 过滤后样本不足 ({len(samples)})")
        return

    coo = np.array([s['coo'] for s in samples], dtype=float)
    quality = np.array([s['quality_v2'] for s in samples], dtype=float)
    acc = np.array([s['acc'] for s in samples], dtype=float)

    print(f"\n{'='*70}")
    print(f"  Stratified validation — {label} (n={len(samples)})")
    print(f"{'='*70}")

    # 按 gt_coo 分区间
    coo_bins = [
        (1, 2, 'coo=1~2'),
        (3, 5, 'coo=3~5'),
        (6, 10, 'coo=6~10'),
        (11, 20, 'coo=11~20'),
        (21, 50, 'coo=21~50'),
        (51, 9999, 'coo>50'),
    ]

    print(f"  {'coo区间':<12} {'n_lowQ':>8} {'acc_lowQ':>10} {'n_highQ':>8} {'acc_highQ':>10} {'diff':>8} {'p':>10} {'sig':>4}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*4}")

    overall_consistent = 0
    overall_total = 0

    for lo, hi, bin_label in coo_bins:
        bin_mask = (coo >= lo) & (coo <= hi)
        if bin_mask.sum() < 20:
            print(f"  {bin_label:<12} [样本不足: {bin_mask.sum()}]")
            continue

        q_med = np.median(quality[bin_mask])
        low_q_mask = bin_mask & (quality <= q_med)
        high_q_mask = bin_mask & (quality > q_med)

        n_l, n_h = low_q_mask.sum(), high_q_mask.sum()
        if n_l < 5 or n_h < 5:
            print(f"  {bin_label:<12} {n_l:>8} {'N/A':>10} {n_h:>8} {'N/A':>10} [组内样本不足]")
            continue

        acc_l = acc[low_q_mask]
        acc_h = acc[high_q_mask]

        stat, p = mannwhitneyu(acc_h, acc_l, alternative='two-sided')
        diff = acc_h.mean() - acc_l.mean()
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))

        print(f"  {bin_label:<12} {n_l:>8} {acc_l.mean():>10.3f} {n_h:>8} {acc_h.mean():>10.3f} {diff:>+8.3f} {p:>10.3e} {sig:>4}")

        overall_total += 1
        if diff > 0:
            overall_consistent += 1

    if overall_total > 0:
        print(f"\n  → {overall_consistent}/{overall_total} 个区间 high_quality > low_quality")


def main():
    print("加载数据...")
    full_dict = load_popularity()
    co_occu = json.loads(open(COO_PATH).read())
    single_occr = json.loads(open(SINGLE_PATH).read())

    for dataset in DATASETS:
        for model in MODELS:
            samples = collect_samples(dataset, model, full_dict, co_occu, single_occr)
            if len(samples) < 50:
                continue
            stratified_by_coo(samples, label=f'{dataset} × {model}')


if __name__ == '__main__':
    main()
