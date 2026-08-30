"""
quality_v2 → confidence 分析

核心问题：quality_v2 (relation specificity = gt_coo / q_single) 是否独立预测 confidence？
- 在 RQ1 中，quality_v2 对 accuracy 有独立预测力（控制 gt_coo 后仍显著）
- 如果 quality_v2 对 confidence 没有独立预测力（尤其在错误样本上），
  这将精确刻画 learning-confidence asymmetry 的边界：
  learning 需要 specificity，confidence 不需要

分析设计：
1. quality_v2 → conf 的 Spearman 相关（全样本/正确/错误）
2. quality_v2 → conf 的偏相关（控制 gt_coo），与 table3 的 quality_v2 → acc 对比
3. quality_v2 → conf 的偏相关（控制 gt_coo + question_pop），三变量框架
4. 在错误样本上重复上述分析（核心：asymmetry 证据）
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


def multivariate_partial(x, y, controls):
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


def collect_samples(dataset, model, full_dict, co_occu, single_occr,
                    filter_pop_no=False):
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


def main():
    print("加载数据...")
    full_dict = load_popularity()
    co_occu = json.loads(open(COO_PATH).read())
    single_occr = json.loads(open(SINGLE_PATH).read())

    all_data = {}
    for dataset in DATASETS:
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = collect_samples(dataset, model, full_dict, co_occu, single_occr,
                                       filter_pop_no=True)
            all_data[(dataset, model)] = samples
            n = len(samples)
            n_wrong = sum(1 for s in samples if s['acc'] == 0)
            n_right = n - n_wrong
            print(f"  {dataset:12s} × {model:15s}: n={n:5d}  right={n_right:5d}  wrong={n_wrong:5d}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 分析1: quality_v2 → conf 双变量 Spearman（全样本/正确/错误）
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("分析1: quality_v2 → conf Spearman 相关（全样本/正确/错误）")
    print("=" * 120)
    print(f"{'dataset':12s} {'model':15s} | "
          f"{'all_r':>8s} {'all_p':>10s} | "
          f"{'corr_r':>8s} {'corr_p':>10s} | "
          f"{'incorr_r':>9s} {'incorr_p':>11s} | "
          f"{'n_all':>6s} {'n_corr':>7s} {'n_inc':>6s}")
    print("-" * 120)

    for dataset in DATASETS:
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = all_data.get((dataset, model), [])
            # quality_v2 分析需要 q_single 过滤
            q_samples = [s for s in samples if Q_SINGLE_LO <= s['q_single'] < Q_SINGLE_HI]
            if len(q_samples) < 50:
                continue

            quality = np.array([s['quality_v2'] for s in q_samples], dtype=float)
            conf = np.array([s['conf'] for s in q_samples], dtype=float)
            acc = np.array([s['acc'] for s in q_samples], dtype=float)

            # 全样本
            if np.std(quality) > 0:
                r_all, p_all = spearmanr(quality, conf)
            else:
                r_all, p_all = float('nan'), float('nan')

            # 正确样本
            corr_mask = acc == 1
            q_corr = quality[corr_mask]
            c_corr = conf[corr_mask]
            if len(q_corr) >= 30 and np.std(q_corr) > 0:
                r_corr, p_corr = spearmanr(q_corr, c_corr)
            else:
                r_corr, p_corr = float('nan'), float('nan')

            # 错误样本
            incorr_mask = acc == 0
            q_inc = quality[incorr_mask]
            c_inc = conf[incorr_mask]
            if len(q_inc) >= 30 and np.std(q_inc) > 0:
                r_inc, p_inc = spearmanr(q_inc, c_inc)
            else:
                r_inc, p_inc = float('nan'), float('nan')

            def fmt_r(v):
                return f'{v:+.3f}' if not np.isnan(v) else '—'
            def fmt_p(v):
                if np.isnan(v):
                    return '—'
                return f'{v:.1e}' if v < 0.001 else f'{v:.3f}'

            print(f"{dataset:12s} {model:15s} | "
                  f"{fmt_r(r_all):>8s} {fmt_p(p_all):>10s} | "
                  f"{fmt_r(r_corr):>8s} {fmt_p(p_corr):>10s} | "
                  f"{fmt_r(r_inc):>9s} {fmt_p(p_inc):>11s} | "
                  f"{len(q_samples):6d} {int(corr_mask.sum()):7d} {int(incorr_mask.sum()):6d}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 分析2: quality_v2 → conf 偏相关（控制 gt_coo）— 与 table3 做对
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("分析2: quality_v2 → conf 偏相关（控制 gt_coo），全样本/正确/错误")
    print("  对照：table3 中 quality_v2 → acc 偏相关（控制 gt_coo）在 15/15 组合中显著为正")
    print("  如果 quality_v2 → conf 在错误样本中不显著，说明 specificity 只影响 learning 不影响 confidence")
    print("=" * 120)
    print(f"{'dataset':12s} {'model':15s} | "
          f"{'qv2→acc(ctrl coo)':>18s} | "
          f"{'qv2→conf(all)':>14s} {'qv2→conf(corr)':>15s} {'qv2→conf(incorr)':>17s} | "
          f"{'n_all':>6s} {'n_c':>6s} {'n_i':>6s}")
    print("-" * 130)

    for dataset in DATASETS:
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = all_data.get((dataset, model), [])
            q_samples = [s for s in samples if Q_SINGLE_LO <= s['q_single'] < Q_SINGLE_HI]
            if len(q_samples) < 50:
                continue

            quality = np.array([s['quality_v2'] for s in q_samples], dtype=float)
            coo = np.array([s['coo'] for s in q_samples], dtype=float)
            conf = np.array([s['conf'] for s in q_samples], dtype=float)
            acc = np.array([s['acc'] for s in q_samples], dtype=float)

            # quality_v2 → acc (控制 coo) — 复现 table3
            if np.std(quality) > 0:
                r_acc, p_acc = partial_spearman(quality, acc, coo)
            else:
                r_acc, p_acc = float('nan'), float('nan')

            # quality_v2 → conf (控制 coo) — 全样本
            if np.std(quality) > 0:
                r_conf_all, p_conf_all = partial_spearman(quality, conf, coo)
            else:
                r_conf_all, p_conf_all = float('nan'), float('nan')

            # 正确样本
            corr_mask = acc == 1
            q_c, c_c, coo_c = quality[corr_mask], conf[corr_mask], coo[corr_mask]
            if len(q_c) >= 30 and np.std(q_c) > 0:
                r_conf_corr, p_conf_corr = partial_spearman(q_c, c_c, coo_c)
            else:
                r_conf_corr, p_conf_corr = float('nan'), float('nan')

            # 错误样本
            incorr_mask = acc == 0
            q_i, c_i, coo_i = quality[incorr_mask], conf[incorr_mask], coo[incorr_mask]
            if len(q_i) >= 30 and np.std(q_i) > 0:
                r_conf_inc, p_conf_inc = partial_spearman(q_i, c_i, coo_i)
            else:
                r_conf_inc, p_conf_inc = float('nan'), float('nan')

            def fmt(v):
                return f'{v:+.3f}' if not np.isnan(v) else '—'
            def fmt_p(v):
                if np.isnan(v):
                    return ''
                sig = '***' if v < 0.001 else ('**' if v < 0.01 else ('*' if v < 0.05 else ''))
                return sig

            print(f"{dataset:12s} {model:15s} | "
                  f"{fmt(r_acc):>18s} | "
                  f"{fmt(r_conf_all):>14s} "
                  f"{fmt(r_conf_corr):>15s} "
                  f"{fmt(r_conf_inc):>17s} | "
                  f"{len(q_samples):6d} {int(corr_mask.sum()):6d} {int(incorr_mask.sum()):6d}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 分析3: quality_v2 → conf 偏相关（控制 gt_coo + question_pop）
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("分析3: quality_v2 → conf 偏相关（控制 gt_coo + question_pop）")
    print("  三变量框架：与 RQ1 中的 {gt_coo, qpop, gt_pop} → acc 对称")
    print("=" * 120)
    print(f"{'dataset':12s} {'model':15s} | "
          f"{'qv2→conf(all)':>14s} {'qv2→conf(corr)':>15s} {'qv2→conf(incorr)':>17s}")
    print("-" * 80)

    for dataset in DATASETS:
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = all_data.get((dataset, model), [])
            q_samples = [s for s in samples if Q_SINGLE_LO <= s['q_single'] < Q_SINGLE_HI]
            if len(q_samples) < 50:
                continue

            quality = np.array([s['quality_v2'] for s in q_samples], dtype=float)
            coo = np.array([s['coo'] for s in q_samples], dtype=float)
            qpop = np.array([s['question_pop'] for s in q_samples], dtype=float)
            conf = np.array([s['conf'] for s in q_samples], dtype=float)
            acc = np.array([s['acc'] for s in q_samples], dtype=float)

            # 全样本
            if np.std(quality) > 0:
                r_all, _ = multivariate_partial(quality, conf, [coo, qpop])
            else:
                r_all = float('nan')

            # 正确
            corr_mask = acc == 1
            if corr_mask.sum() >= 30 and np.std(quality[corr_mask]) > 0:
                r_corr, _ = multivariate_partial(quality[corr_mask], conf[corr_mask],
                                                  [coo[corr_mask], qpop[corr_mask]])
            else:
                r_corr = float('nan')

            # 错误
            incorr_mask = acc == 0
            if incorr_mask.sum() >= 30 and np.std(quality[incorr_mask]) > 0:
                r_inc, _ = multivariate_partial(quality[incorr_mask], conf[incorr_mask],
                                                 [coo[incorr_mask], qpop[incorr_mask]])
            else:
                r_inc = float('nan')

            def fmt(v):
                return f'{v:+.3f}' if not np.isnan(v) else '—'

            print(f"{dataset:12s} {model:15s} | "
                  f"{fmt(r_all):>14s} {fmt(r_corr):>15s} {fmt(r_inc):>17s}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 分析4: 对照表 — quality_v2 对 acc vs conf 的不对称性
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 130)
    print("分析4: quality_v2 对 acc vs conf 的不对称性（偏相关，控制 gt_coo）")
    print("  核心对比：")
    print("  - quality_v2→acc (控制coo): 在 RQ1 中 15/15 正 → specificity 影响学习")
    print("  - quality_v2→conf(incorrect)(控制coo): 如果近零 → specificity 不影响自信 → asymmetry!")
    print("=" * 130)
    print(f"{'dataset':12s} {'model':15s} | "
          f"{'qv2→acc':>8s} | "
          f"{'qv2→conf(all)':>14s} {'qv2→conf(corr)':>15s} {'qv2→conf(incorr)':>17s} | "
          f"{'acc≠conf?':>9s}")
    print("-" * 130)

    sig_count = {'acc': 0, 'conf_all': 0, 'conf_corr': 0, 'conf_inc': 0}
    
    for dataset in DATASETS:
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = all_data.get((dataset, model), [])
            q_samples = [s for s in samples if Q_SINGLE_LO <= s['q_single'] < Q_SINGLE_HI]
            if len(q_samples) < 50:
                continue

            quality = np.array([s['quality_v2'] for s in q_samples], dtype=float)
            coo = np.array([s['coo'] for s in q_samples], dtype=float)
            conf = np.array([s['conf'] for s in q_samples], dtype=float)
            acc = np.array([s['acc'] for s in q_samples], dtype=float)

            if np.std(quality) > 0:
                r_acc, p_acc = partial_spearman(quality, acc, coo)
                r_conf_all, p_conf_all = partial_spearman(quality, conf, coo)
            else:
                r_acc, p_acc = float('nan'), float('nan')
                r_conf_all, p_conf_all = float('nan'), float('nan')

            corr_mask = acc == 1
            incorr_mask = acc == 0
            q_c, c_c, coo_c = quality[corr_mask], conf[corr_mask], coo[corr_mask]
            q_i, c_i, coo_i = quality[incorr_mask], conf[incorr_mask], coo[incorr_mask]

            if len(q_c) >= 30 and np.std(q_c) > 0:
                r_conf_corr, p_conf_corr = partial_spearman(q_c, c_c, coo_c)
            else:
                r_conf_corr, p_conf_corr = float('nan'), float('nan')

            if len(q_i) >= 30 and np.std(q_i) > 0:
                r_conf_inc, p_conf_inc = partial_spearman(q_i, c_i, coo_i)
            else:
                r_conf_inc, p_conf_inc = float('nan'), float('nan')

            # 判断不对称性
            def is_pos_sig(r, p):
                return not np.isnan(r) and not np.isnan(p) and r > 0.05 and p < 0.05

            acc_sig = is_pos_sig(r_acc, p_acc)
            conf_inc_sig = is_pos_sig(r_conf_inc, p_conf_inc)
            if acc_sig:
                sig_count['acc'] += 1
            if is_pos_sig(r_conf_all, p_conf_all):
                sig_count['conf_all'] += 1
            if is_pos_sig(r_conf_corr, p_conf_corr):
                sig_count['conf_corr'] += 1
            if conf_inc_sig:
                sig_count['conf_inc'] += 1

            asym = "YES" if (acc_sig and not conf_inc_sig) else ("partial" if (acc_sig and conf_inc_sig and abs(r_acc) > abs(r_conf_inc) * 1.5) else "—")

            def fmt(v):
                return f'{v:+.3f}' if not np.isnan(v) else '—'
            def sig_star(p):
                if np.isnan(p):
                    return ''
                return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))

            print(f"{dataset:12s} {model:15s} | "
                  f"{fmt(r_acc)}{sig_star(p_acc):>4s} | "
                  f"{fmt(r_conf_all)}{sig_star(p_conf_all):>10s} "
                  f"{fmt(r_conf_corr)}{sig_star(p_conf_corr):>10s} "
                  f"{fmt(r_conf_inc)}{sig_star(p_conf_inc):>12s} | "
                  f"{asym:>9s}")

    total = sum(1 for d in DATASETS for m in MODELS if not _skip_bball_qwen25(d, m))
    print(f"\n汇总: qv2→acc 显著正: {sig_count['acc']}/{total}, "
          f"qv2→conf(all) 显著正: {sig_count['conf_all']}/{total}, "
          f"qv2→conf(corr) 显著正: {sig_count['conf_corr']}/{total}, "
          f"qv2→conf(incorr) 显著正: {sig_count['conf_inc']}/{total}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 分析5: 对比 — gt_coo→acc vs gt_coo→conf(incorrect)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 130)
    print("分析5: 对比 gt_coo→acc vs gt_coo→conf(incorrect)（偏相关，均控制 question_pop + gt_pop）")
    print("  这是 RQ2 已有结果（gene_coo→conf 在错误样本上放大）的镜像分析")
    print("  gt_coo 是正确答案的共现，模型做错时不知道 gt，所以 gt_coo→conf 应该弱于 gt_coo→acc")
    print("=" * 130)
    print(f"{'dataset':12s} {'model':15s} | "
          f"{'gt_coo→acc':>11s} {'gt_coo→conf(c)':>15s} {'gt_coo→conf(i)':>15s}")
    print("-" * 80)

    for dataset in DATASETS:
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = all_data.get((dataset, model), [])
            if len(samples) < 100:
                continue

            coo = np.array([s['coo'] for s in samples], dtype=float)
            qpop = np.array([s['question_pop'] for s in samples], dtype=float)
            gtpop = np.array([s['gt_pop'] for s in samples], dtype=float)
            conf = np.array([s['conf'] for s in samples], dtype=float)
            acc = np.array([s['acc'] for s in samples], dtype=float)

            # gt_coo → acc (控制 qpop + gt_pop)
            if np.std(coo) > 0:
                r_acc, _ = multivariate_partial(coo, acc, [qpop, gtpop])
            else:
                r_acc = float('nan')

            corr_mask = acc == 1
            incorr_mask = acc == 0

            # gt_coo → conf correct
            if corr_mask.sum() >= 30 and np.std(coo[corr_mask]) > 0:
                r_conf_c, _ = multivariate_partial(coo[corr_mask], conf[corr_mask],
                                                    [qpop[corr_mask], gtpop[corr_mask]])
            else:
                r_conf_c = float('nan')

            # gt_coo → conf incorrect
            if incorr_mask.sum() >= 30 and np.std(coo[incorr_mask]) > 0:
                r_conf_i, _ = multivariate_partial(coo[incorr_mask], conf[incorr_mask],
                                                    [qpop[incorr_mask], gtpop[incorr_mask]])
            else:
                r_conf_i = float('nan')

            def fmt(v):
                return f'{v:+.3f}' if not np.isnan(v) else '—'

            print(f"{dataset:12s} {model:15s} | "
                  f"{fmt(r_acc):>11s} {fmt(r_conf_c):>15s} {fmt(r_conf_i):>15s}")


if __name__ == '__main__':
    main()