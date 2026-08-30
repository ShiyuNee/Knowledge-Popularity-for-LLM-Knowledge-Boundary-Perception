"""
analyze_dataset_statistics.py — 论文 Table 6/7 数据来源脚本

【论文主要数据来源】
本文 paper_demo.md §5.4 中的 Table 6（Intrinsic signal properties）和 Table 7（Signal→confidence associations）
的数据均来自本脚本。本脚本与 verify_per_model.py 的区别在于：
  - verify_per_model.py 提供按 dataset×model 的偏相关分析（RQ1/RQ2/RQ3 核心统计）
  - 本脚本提供数据集固有属性的统计分析（不涉及模型行为的因果解释），以及补充的比值变量分析

【分析对应关系】
分析1: 数据集固有统计量（全样本合并去重）→ §5.4 Table 6 前三行
  - gene_pop CV / P90/P10 / median → Popularity Gradient
  - gt_coo/question_pop median → Relation Density
  - gene_coo/gene_pop median (wrong, nonzero) → Wrong-Answer Co-occurrence
  - ρ(gene_pop, gene_coo) wrong → Signal Coupling

分析2: 做对 vs 做错各变量分布 → §5.4 解释性数据（ratio variables on wrong answers）

分析3: 做错样本各变量与confidence的Spearman相关 → §5.4 Table 7 (Spearman列)

分析4: 做错样本偏相关分析 → §5.4 Table 7 (偏相关列, 衰减百分比)
  - 衰减率 = (ρ_direct - ρ_partial) / |ρ_direct|

分析5: 比值分布对比 → §5.4 gene_coo/question_pop, gene_coo/gene_pop 描述

分析6: 正确答案的关系强度指标 → §5.4 gt_coo/question_pop, gt_coo/gene_pop 描述
"""

import json
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr

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
    import re
    s = re.sub(r'^[^\w]+|[^\w]+$', '', s)
    return s.strip()


def load_popularity():
    pop_data = read_jsonl(POP_PATH)
    full_dict = {}
    for d in pop_data:
        full_dict.update(d)
    return full_dict


def collect_samples(dataset, model, full_dict, co_occu, single_occr, filter_pop_no=True):
    res_path = os.path.join(RES_DIR, dataset, f'{dataset}_{model}_temperature1.jsonl')
    if not os.path.exists(res_path):
        return []
    model_res = read_jsonl(res_path)
    samples = []
    import math
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
        gt_pop_missing = (ref_pop == 'No' or ref_pop is None or ref not in full_dict)
        if gt_pop_missing:
            ref_pop = 0

        gene_pop_info = full_dict.get(gene_entity, {})
        gene_pop = gene_pop_info.get('popularity', 0) if isinstance(gene_pop_info, dict) else gene_pop_info
        gene_pop_missing = (gene_pop == 'No' or gene_pop is None or gene_entity not in full_dict)
        if gene_pop_missing:
            gene_pop = 0

        if filter_pop_no and (gt_pop_missing or gene_pop_missing):
            continue

        if 'gpt' in model.lower():
            probs = [math.exp(t) for t in item['Log_p']['token_logprobs']]
        else:
            probs = item['Log_p']['token_probs']
        conf = sum(probs) / len(probs)

        gt_coo = co_occu[question_entity].get(ref.lower(), 0)
        gene_coo = co_occu[question_entity].get(gene_entity.lower(), 0)
        q_single = single_occr[question_entity.lower()]
        gt_single = single_occr[ref.lower()]
        total_coo = sum(co_occu[question_entity].values())

        quality_v2 = gt_coo / q_single if q_single > 0 else 0

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
            'conf': conf,
            'acc': item['has_answer'],
        })
    return samples


def _skip_bball_qwen25(dataset, model):
    return dataset == 'basketball' and model.startswith('Qwen2.5')


def pct_nonzero(arr):
    """非零比例"""
    return (arr > 0).mean()


def main():
    print("加载数据...")
    full_dict = load_popularity()
    co_occu = json.loads(open(COO_PATH).read())
    single_occr = json.loads(open(SINGLE_PATH).read())

    # 收集所有数据
    all_data = {}
    for dataset in DATASETS:
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = collect_samples(dataset, model, full_dict, co_occu, single_occr, filter_pop_no=True)
            all_data[(dataset, model)] = samples

    # =========================================================================
    # 分析1: 数据集固有属性（合并所有模型，因为这是数据集本身的特征）
    # =========================================================================
    print("\n" + "=" * 120)
    print("分析1: 数据集固有统计量（所有模型样本合并）")
    print("=" * 120)

    for dataset in DATASETS:
        # 合并该数据集的所有模型样本
        all_samples = []
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            all_samples.extend(all_data.get((dataset, model), []))

        # 去重：同一question-entity+gene-entity组合只保留一次
        seen = set()
        unique_samples = []
        for s in all_samples:
            key = (s['question_pop'], s['gt_pop'], s['gene_pop'], s['coo'], s['gene_coo'],
                   s['q_single'], s['gt_single'])
            if key not in seen:
                seen.add(key)
                unique_samples.append(s)

        n_total = len(unique_samples)
        # 但对于做对/做错分析，不同模型做对/做错不同，所以需要保留所有
        # 对于纯粹的数据集属性（不依赖模型结果），用去重后的

        # 使用去重样本来计算数据集固有属性
        qpop = np.array([s['question_pop'] for s in unique_samples], dtype=float)
        gpop = np.array([s['gene_pop'] for s in unique_samples], dtype=float)
        gtpop = np.array([s['gt_pop'] for s in unique_samples], dtype=float)
        gcoo = np.array([s['gene_coo'] for s in unique_samples], dtype=float)
        gtcoo = np.array([s['coo'] for s in unique_samples], dtype=float)
        qsingle = np.array([s['q_single'] for s in unique_samples], dtype=float)

        # 计算比值（避免除零）
        # gene_coo / question_pop
        gcoo_qpop = np.where(qpop > 0, gcoo / qpop, 0)
        # gene_coo / gene_pop
        gcoo_gpop = np.where(gpop > 0, gcoo / gpop, 0)
        # gt_coo / question_pop
        gtcoo_qpop = np.where(qpop > 0, gtcoo / qpop, 0)
        # gt_coo / gene_pop
        gtcoo_gpop = np.where(gpop > 0, gtcoo / gpop, 0)

        print(f"\n{'='*80}")
        print(f"数据集: {dataset} (去重后 {n_total} 条)")
        print(f"{'='*80}")

        # ── 基本统计 ──
        print(f"\n--- 基本统计 ---")
        print(f"  {'变量':25s} {'mean':>10s} {'median':>10s} {'std':>10s} {'P10':>10s} {'P25':>10s} {'P75':>10s} {'P90':>10s} {'nonzero%':>10s}")
        for name, arr in [
            ('question_pop', qpop),
            ('gt_pop', gtpop),
            ('gene_pop', gpop),
            ('gt_coo', gtcoo),
            ('gene_coo', gcoo),
            ('q_single', qsingle),
            ('gene_coo/question_pop', gcoo_qpop),
            ('gene_coo/gene_pop', gcoo_gpop),
            ('gt_coo/question_pop', gtcoo_qpop),
            ('gt_coo/gene_pop', gtcoo_gpop),
        ]:
            pcts = np.percentile(arr, [10, 25, 50, 75, 90])
            print(f"  {name:25s} {arr.mean():10.2f} {np.median(arr):10.2f} {arr.std():10.2f} "
                  f"{pcts[0]:10.2f} {pcts[1]:10.2f} {pcts[3]:10.2f} {pcts[4]:10.2f} "
                  f"{pct_nonzero(arr):10.3f}")

        # ── 变异系数 ──
        print(f"\n--- 变异系数 (CV = std/mean) ---")
        for name, arr in [
            ('question_pop', qpop),
            ('gt_pop', gtpop),
            ('gene_pop', gpop),
            ('gt_coo', gtcoo),
            ('gene_coo', gcoo),
            ('gene_coo/question_pop', gcoo_qpop),
            ('gene_coo/gene_pop', gcoo_gpop),
        ]:
            mean = arr.mean()
            cv = arr.std() / mean if mean > 0 else float('inf')
            p10, p90 = np.percentile(arr, [10, 90])
            ratio = p90 / max(p10, 0.001)
            print(f"  {name:25s} CV={cv:.3f}  P90/P10={ratio:.1f}")

        # ── 相关性矩阵 ──
        print(f"\n--- Spearman相关矩阵（全样本） ---")
        vars_dict = {
            'question_pop': qpop,
            'gt_pop': gtpop,
            'gene_pop': gpop,
            'gt_coo': gtcoo,
            'gene_coo': gcoo,
        }
        var_names = list(vars_dict.keys())
        # 添加比值
        vars_dict['gcoo/qpop'] = gcoo_qpop
        vars_dict['gcoo/gpop'] = gcoo_gpop
        var_names_ext = list(vars_dict.keys())

        # 打印header
        header = f"  {'':25s} " + " ".join(f"{nm:>12s}" for nm in var_names_ext)
        print(header)
        for nm1 in var_names_ext:
            row = f"  {nm1:25s}"
            for nm2 in var_names_ext:
                if np.std(vars_dict[nm1]) > 0 and np.std(vars_dict[nm2]) > 0:
                    r, _ = spearmanr(vars_dict[nm1], vars_dict[nm2])
                    row += f" {r:+12.3f}"
                else:
                    row += f" {'N/A':>12s}"
            print(row)

    # =========================================================================
    # 分析2: 做对 vs 做错 分别统计（按模型，因为做对/做错依赖模型）
    # =========================================================================
    print("\n" + "=" * 140)
    print("分析2: 做对 vs 做错 — 各统计量在正确/错误答案下的分布")
    print("=" * 140)

    for dataset in DATASETS:
        print(f"\n{'='*140}")
        print(f"数据集: {dataset}")
        print(f"{'='*140}")
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = all_data.get((dataset, model), [])
            right = [s for s in samples if s['acc'] == 1]
            wrong = [s for s in samples if s['acc'] == 0]
            if len(right) < 30 or len(wrong) < 30:
                continue

            print(f"\n  --- {dataset} × {model} (right={len(right)}, wrong={len(wrong)}) ---")
            print(f"  {'变量':30s} | {'right_mean':>11s} {'right_med':>11s} {'right_P90':>11s} {'right_nz%':>11s} | "
                  f"{'wrong_mean':>11s} {'wrong_med':>11s} {'wrong_P90':>11s} {'wrong_nz%':>11s} | {'Δmean':>10s}")
            print(f"  {'-'*135}")

            for subset_name, subset_data in [('right', right), ('wrong', wrong)]:
                pass  # 下面统一处理

            # 统一的变量列表
            var_info = []
            for s_list, prefix in [(right, 'right'), (wrong, 'wrong')]:
                qpop = np.array([s['question_pop'] for s in s_list], dtype=float)
                gpop = np.array([s['gene_pop'] for s in s_list], dtype=float)
                gtpop = np.array([s['gt_pop'] for s in s_list], dtype=float)
                gcoo = np.array([s['gene_coo'] for s in s_list], dtype=float)
                gtcoo = np.array([s['coo'] for s in s_list], dtype=float)
                gcoo_qpop = np.where(qpop > 0, gcoo / qpop, 0)
                gcoo_gpop = np.where(gpop > 0, gcoo / gpop, 0)

                var_info.append({
                    'qpop': qpop, 'gpop': gpop, 'gtpop': gtpop,
                    'gcoo': gcoo, 'gtcoo': gtcoo,
                    'gcoo_qpop': gcoo_qpop, 'gcoo_gpop': gcoo_gpop,
                })

            for var_name in ['qpop', 'gpop', 'gtpop', 'gcoo', 'gtcoo', 'gcoo_qpop', 'gcoo_gpop']:
                r_arr = var_info[0][var_name]
                w_arr = var_info[1][var_name]
                r_mean, r_med, r_p90, r_nz = r_arr.mean(), np.median(r_arr), np.percentile(r_arr, 90), pct_nonzero(r_arr)
                w_mean, w_med, w_p90, w_nz = w_arr.mean(), np.median(w_arr), np.percentile(w_arr, 90), pct_nonzero(w_arr)
                delta = r_mean - w_mean
                print(f"  {var_name:30s} | {r_mean:11.2f} {r_med:11.2f} {r_p90:11.2f} {r_nz:11.3f} | "
                      f"{w_mean:11.2f} {w_med:11.2f} {w_p90:11.2f} {w_nz:11.3f} | {delta:+10.2f}")

    # =========================================================================
    # 分析3: 各变量与confidence的Spearman相关（做错样本）
    # =========================================================================
    print("\n" + "=" * 140)
    print("分析3: 做错样本 — 各变量与confidence的Spearman相关")
    print("=" * 140)
    print(f"{'dataset':12s} {'model':15s} | "
          f"{'gene_pop':>10s} {'question_pop':>13s} {'gene_coo':>10s} {'gt_coo':>8s} | "
          f"{'gcoo/qpop':>10s} {'gcoo/gpop':>10s} | "
          f"{'ρ(gpop,gcoo)':>14s} {'ρ(gpop,qpop)':>14s} {'ρ(gcoo,qpop)':>14s}")
    print("-" * 140)

    for dataset in DATASETS:
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = all_data.get((dataset, model), [])
            wrong = [s for s in samples if s['acc'] == 0]
            if len(wrong) < 30:
                continue

            w_conf = np.array([s['conf'] for s in wrong], dtype=float)
            w_gpop = np.array([s['gene_pop'] for s in wrong], dtype=float)
            w_qpop = np.array([s['question_pop'] for s in wrong], dtype=float)
            w_gcoo = np.array([s['gene_coo'] for s in wrong], dtype=float)
            w_gtcoo = np.array([s['coo'] for s in wrong], dtype=float)
            w_gcoo_qpop = np.where(w_qpop > 0, w_gcoo / w_qpop, 0)
            w_gcoo_gpop = np.where(w_gpop > 0, w_gcoo / w_gpop, 0)

            def sp(a, b):
                if np.std(a) > 0 and np.std(b) > 0:
                    return spearmanr(a, b)[0]
                return float('nan')

            r_gpop = sp(w_gpop, w_conf)
            r_qpop = sp(w_qpop, w_conf)
            r_gcoo = sp(w_gcoo, w_conf)
            r_gtcoo = sp(w_gtcoo, w_conf)
            r_gcoo_qpop = sp(w_gcoo_qpop, w_conf)
            r_gcoo_gpop = sp(w_gcoo_gpop, w_conf)
            # 变量间相关
            r_gpop_gcoo = sp(w_gpop, w_gcoo)
            r_gpop_qpop = sp(w_gpop, w_qpop)
            r_gcoo_qpop_val = sp(w_gcoo, w_qpop)

            print(f"{dataset:12s} {model:15s} | "
                  f"{r_gpop:+10.3f} {r_qpop:+13.3f} {r_gcoo:+10.3f} {r_gtcoo:+8.3f} | "
                  f"{r_gcoo_qpop:+10.3f} {r_gcoo_gpop:+10.3f} | "
                  f"{r_gpop_gcoo:+14.3f} {r_gpop_qpop:+14.3f} {r_gcoo_qpop_val:+14.3f}")

    # =========================================================================
    # 分析4: 做错样本 — 偏相关分析（控制gene_coo后gene_pop→conf，控制gene_pop后gene_coo→conf）
    # =========================================================================
    print("\n" + "=" * 140)
    print("分析4: 做错样本 — 偏相关 + 各比值变量→conf偏相关（控制其他变量）")
    print("=" * 140)

    from scipy.stats import rankdata

    def partial_spearman(x, y, z):
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

    print(f"{'dataset':12s} {'model':15s} | "
          f"{'gpop→conf':>10s} {'(控gcoo)':>10s} | "
          f"{'gcoo→conf':>10s} {'(控gpop)':>10s} | "
          f"{'gcoo/qpop→conf':>15s} {'(控gpop)':>10s} | "
          f"{'gcoo/gpop→conf':>15s} {'(控gpop)':>10s}")
    print("-" * 130)

    for dataset in DATASETS:
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = all_data.get((dataset, model), [])
            wrong = [s for s in samples if s['acc'] == 0]
            if len(wrong) < 30:
                continue

            w_conf = np.array([s['conf'] for s in wrong], dtype=float)
            w_gpop = np.array([s['gene_pop'] for s in wrong], dtype=float)
            w_gcoo = np.array([s['gene_coo'] for s in wrong], dtype=float)
            w_qpop = np.array([s['question_pop'] for s in wrong], dtype=float)
            w_gcoo_qpop = np.where(w_qpop > 0, w_gcoo / w_qpop, 0)
            w_gcoo_gpop = np.where(w_gpop > 0, w_gcoo / w_gpop, 0)

            # 直接Spearman
            def sp(a, b):
                if np.std(a) > 0 and np.std(b) > 0:
                    return spearmanr(a, b)[0]
                return float('nan')

            # 偏相关
            r1, _ = partial_spearman(w_gpop, w_conf, w_gcoo)
            r2, _ = partial_spearman(w_gcoo, w_conf, w_gpop)
            r3, _ = partial_spearman(w_gcoo_qpop, w_conf, w_gpop)
            r4, _ = partial_spearman(w_gcoo_gpop, w_conf, w_gpop)

            sp_gpop = sp(w_gpop, w_conf)
            sp_gcoo = sp(w_gcoo, w_conf)
            sp_gcoo_qpop = sp(w_gcoo_qpop, w_conf)
            sp_gcoo_gpop = sp(w_gcoo_gpop, w_conf)

            print(f"{dataset:12s} {model:15s} | "
                  f"{sp_gpop:+10.3f} {r1:+10.3f} | "
                  f"{sp_gcoo:+10.3f} {r2:+10.3f} | "
                  f"{sp_gcoo_qpop:+15.3f} {r3:+10.3f} | "
                  f"{sp_gcoo_gpop:+15.3f} {r4:+10.3f}")

    # =========================================================================
    # 分析5: gene_coo/gene_pop 和 gene_coo/question_pop 的分布比较
    # =========================================================================
    print("\n" + "=" * 140)
    print("分析5: 三个数据集的比值分布对比（做错样本，跨模型聚合）")
    print("=" * 140)

    for dataset in DATASETS:
        wrong_all = []
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = all_data.get((dataset, model), [])
            wrong_all.extend([s for s in samples if s['acc'] == 0])

        if not wrong_all:
            continue

        w_gpop = np.array([s['gene_pop'] for s in wrong_all], dtype=float)
        w_qpop = np.array([s['question_pop'] for s in wrong_all], dtype=float)
        w_gcoo = np.array([s['gene_coo'] for s in wrong_all], dtype=float)
        w_gtcoo = np.array([s['coo'] for s in wrong_all], dtype=float)

        gcoo_qpop = np.where(w_qpop > 0, w_gcoo / w_qpop, 0)
        gcoo_gpop = np.where(w_gpop > 0, w_gcoo / w_gpop, 0)

        print(f"\n--- {dataset} (wrong, n={len(wrong_all)}) ---")
        print(f"  gene_coo/question_pop 分布:")
        nonzero = gcoo_qpop[gcoo_qpop > 0]
        if len(nonzero) > 0:
            pcts = np.percentile(nonzero, [10, 25, 50, 75, 90])
            print(f"    nonzero: {len(nonzero)}/{len(gcoo_qpop)} ({pct_nonzero(gcoo_qpop):.3f})")
            print(f"    nonzero mean={nonzero.mean():.6f}  median={np.median(nonzero):.6f}  P10={pcts[0]:.6f}  P90={pcts[4]:.6f}")
        else:
            print(f"    全部为0")

        print(f"  gene_coo/gene_pop 分布:")
        nonzero2 = gcoo_gpop[gcoo_gpop > 0]
        if len(nonzero2) > 0:
            pcts2 = np.percentile(nonzero2, [10, 25, 50, 75, 90])
            print(f"    nonzero: {len(nonzero2)}/{len(gcoo_gpop)} ({pct_nonzero(gcoo_gpop):.3f})")
            print(f"    nonzero mean={nonzero2.mean():.6f}  median={np.median(nonzero2):.6f}  P10={pcts2[0]:.6f}  P90={pcts2[4]:.6f}")
        else:
            print(f"    全部为0")

    # =========================================================================
    # 分析6: 正确答案的对比 — gt_coo/question_pop 和 gt_coo/gt_pop
    # =========================================================================
    print("\n" + "=" * 140)
    print("分析6: 正确答案的关系强度指标（做对样本，跨模型聚合）")
    print("=" * 140)

    for dataset in DATASETS:
        right_all = []
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = all_data.get((dataset, model), [])
            right_all.extend([s for s in samples if s['acc'] == 1])

        if not right_all:
            continue

        r_qpop = np.array([s['question_pop'] for s in right_all], dtype=float)
        r_gtpop = np.array([s['gt_pop'] for s in right_all], dtype=float)
        r_gtcoo = np.array([s['coo'] for s in right_all], dtype=float)
        r_qsingle = np.array([s['q_single'] for s in right_all], dtype=float)
        r_quality_v2 = np.where(r_qsingle > 0, r_gtcoo / r_qsingle, 0)

        gtcoo_qpop = np.where(r_qpop > 0, r_gtcoo / r_qpop, 0)
        gtcoo_gtpop = np.where(r_gtpop > 0, r_gtcoo / r_gtpop, 0)

        print(f"\n--- {dataset} (right, n={len(right_all)}) ---")
        for name, arr in [
            ('gt_coo', r_gtcoo),
            ('gt_coo/question_pop', gtcoo_qpop),
            ('gt_coo/gt_pop', gtcoo_gtpop),
            ('quality_v2 (gt_coo/q_single)', r_quality_v2),
        ]:
            pcts = np.percentile(arr, [25, 50, 75, 90])
            print(f"  {name:30s} mean={arr.mean():.4f} median={np.median(arr):.4f} "
                  f"P25={pcts[0]:.4f} P75={pcts[2]:.4f} P90={pcts[3]:.4f} "
                  f"nonzero={pct_nonzero(arr):.3f}")

    print("\n\n完成！")


if __name__ == '__main__':
    main()