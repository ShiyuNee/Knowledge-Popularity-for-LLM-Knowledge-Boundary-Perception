"""
针对性验证分析：
1. 为什么 basketball 的 gene_pop 不影响 conf？（用数值验证，而非猜测）
2. 什么影响模型学会(acc)？什么影响模型信心(conf)？
   → 假设：共现(coo)决定学会，实体流行度(pop)决定信心
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

DATASETS = ['movies', 'songs', 'basketball']
MODELS = ['llama8b', 'qwen2', 'chatgpt', 'Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']

PATTERN = {
    'movies': 'Who is the director of the movie ',
    'songs': 'Who is the performer of the song ',
    'basketball': 'Where is the birthplace of the basketball player '
}
SINGLE_OCC_THRESHOLD = 6000
N_WIKI = 6_000_000


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
        if question_entity not in co_occu:
            skip_stat_miss += 1; continue
        if question_entity.lower() not in single_occr:
            skip_stat_miss += 1; continue
        if gene_entity.lower() not in single_occr:
            skip_stat_miss += 1; continue
        if ref.lower() not in single_occr:
            skip_stat_miss += 1; continue
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
        if ref_pop == 'No' or ref_pop is None: ref_pop = 0
        gene_pop_info = full_dict.get(gene_entity, {})
        gene_pop = gene_pop_info.get('popularity', 0) if isinstance(gene_pop_info, dict) else gene_pop_info
        if gene_pop == 'No' or gene_pop is None: gene_pop = 0

        if 'gpt' in model.lower():
            probs = [math.exp(t) for t in item['Log_p']['token_logprobs']]
        else:
            probs = item['Log_p']['token_probs']
        conf = sum(probs) / len(probs)

        gt_coo = co_occu[question_entity].get(ref.lower(), 0)
        gene_coo = co_occu[question_entity].get(gene_entity.lower(), 0)
        q_single = single_occr[question_entity.lower()]
        gt_single = single_occr[ref.lower()]
        gene_single = single_occr[gene_entity.lower()]
        total_coo = sum(co_occu[question_entity].values())
        # quality_v2 = gt_coo / q_single: question_entity 的 Wikipedia 文档中有多少比例同时提到了正确答案
        # 分母 q_single 完全来自 Wikipedia，不依赖模型预测，无数据泄露
        quality_v2 = gt_coo / q_single if q_single > 0 else 0

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
            'conf': conf,
            'acc': item['has_answer'],
        })
    if skip_stat_miss > 0:
        print(f"  [{dataset} × {model}] 跳过 {skip_stat_miss} 条")
    return samples


def main():
    print("加载数据...")
    full_dict = load_popularity()
    co_occu = json.loads(open(COO_PATH).read())
    single_occr = json.loads(open(SINGLE_PATH).read())

    # ═══════════════════════════════════════════════════════════════════════
    # 问题1: basketball 为什么 gene_pop 不影响 conf?
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("问题1: basketball 为什么 gene_pop 对 conf 无影响？")
    print("="*70)

    # 假设验证：basketball 的 gene_entity 是城市名，
    # 城市间的 Wikidata popularity 差异小（大多数城市 pop 类似），
    # 而 movies 的 gene_entity 是人名，popularity 差异巨大（名导演 vs 小导演）

    for dataset in DATASETS:
        samples = []
        for model in ['llama8b', 'qwen2', 'chatgpt']:
            s = collect_samples(dataset, model, full_dict, co_occu, single_occr)
            samples.extend(s)
        if not samples:
            continue

        gene_pop = np.array([s['gene_pop'] for s in samples], dtype=float)
        gt_pop = np.array([s['gt_pop'] for s in samples], dtype=float)
        gene_single = np.array([s['gene_single'] for s in samples], dtype=float)
        gt_single = np.array([s['gt_single'] for s in samples], dtype=float)

        print(f"\n--- {dataset} ---")
        print(f"  gene_pop:  mean={gene_pop.mean():.1f}  std={gene_pop.std():.1f}  "
              f"CV={gene_pop.std()/max(gene_pop.mean(),1):.2f}  "
              f"min={gene_pop.min():.0f}  p25={np.percentile(gene_pop,25):.0f}  "
              f"median={np.median(gene_pop):.0f}  p75={np.percentile(gene_pop,75):.0f}  "
              f"max={gene_pop.max():.0f}")
        print(f"  gt_pop:    mean={gt_pop.mean():.1f}  std={gt_pop.std():.1f}  "
              f"CV={gt_pop.std()/max(gt_pop.mean(),1):.2f}  "
              f"min={gt_pop.min():.0f}  p25={np.percentile(gt_pop,25):.0f}  "
              f"median={np.median(gt_pop):.0f}  p75={np.percentile(gt_pop,75):.0f}  "
              f"max={gt_pop.max():.0f}")

        # 更关键：gene_pop 的 "区分度"——是否集中在一个狭窄范围
        # 如果 90% 的 gene_pop 在 [a, b] 内，那变异太小无法检测到相关
        p10, p90 = np.percentile(gene_pop, [10, 90])
        iqr = np.percentile(gene_pop, 75) - np.percentile(gene_pop, 25)
        print(f"  gene_pop IQR={iqr:.0f}  P10={p10:.0f}  P90={p90:.0f}  P90/P10={p90/max(p10,1):.1f}x")

        # 按 gene_pop 四分位看 conf 和 acc
        print(f"  gene_pop 四分位 → conf, acc:")
        pcts = np.percentile(gene_pop, [0, 25, 50, 75, 100])
        conf = np.array([s['conf'] for s in samples], dtype=float)
        acc = np.array([s['acc'] for s in samples], dtype=float)
        for i in range(4):
            lo, hi = pcts[i], pcts[i+1]
            m = (gene_pop >= lo) & (gene_pop <= hi) if i == 0 else (gene_pop > lo) & (gene_pop <= hi)
            if m.sum() < 5: continue
            print(f"    Q{i+1} [{lo:.0f},{hi:.0f}]: n={m.sum():5d}  "
                  f"avg_conf={conf[m].mean():.3f}  avg_acc={acc[m].mean():.3f}")

        # 实体类型的判断：在 basketball 中，gene_entity 和 gt_entity 都是城市
        # 城市名在 Wikipedia 中天然高频且集中——不同城市 pop 变异小
        # 而 movies 中 gene_entity 是导演，pop 变异大（0 ~ 几百）
        # 测量：gene_pop 的变异系数 (Coefficient of Variation)
        print(f"  → gene_pop 变异系数(CV): {dataset}={gene_pop.std()/max(gene_pop.mean(),1):.2f}")

    # ═══════════════════════════════════════════════════════════════════════
    # 问题1 补充: 在 basketball 中，gene_entity 是城市名，
    # 但如果我们看 gene_single (Wikipedia 文档数) 而非 gene_pop (Wikidata sitelinks)，
    # gene_single 的变异是否足够大？
    # ═══════════════════════════════════════════════════════════════════════
    print("\n\n" + "="*70)
    print("问题1 补充: gene_single (Wikipedia频率) vs gene_pop (Wikidata) 的区分度对比")
    print("="*70)

    for dataset in DATASETS:
        samples = []
        for model in ['llama8b', 'qwen2', 'chatgpt']:
            s = collect_samples(dataset, model, full_dict, co_occu, single_occr)
            samples.extend(s)
        if not samples:
            continue

        gene_pop = np.array([s['gene_pop'] for s in samples], dtype=float)
        gene_single = np.array([s['gene_single'] for s in samples], dtype=float)
        conf = np.array([s['conf'] for s in samples], dtype=float)

        # gene_pop vs conf
        r_pop, p_pop = spearmanr(gene_pop, conf)
        # gene_single vs conf
        r_single, p_single = spearmanr(gene_single, conf)

        print(f"\n--- {dataset} ---")
        print(f"  gene_pop:    CV={gene_pop.std()/max(gene_pop.mean(),1):.2f}  "
              f"Spearman(gene_pop, conf)={r_pop:+.3f}  p={p_pop:.3e}")
        print(f"  gene_single: CV={gene_single.std()/max(gene_single.mean(),1):.2f}  "
              f"Spearman(gene_single, conf)={r_single:+.3f}  p={p_single:.3e}")

    # ═══════════════════════════════════════════════════════════════════════
    # 问题2: 什么影响模型学会(acc)? 什么影响模型信心(conf)?
    # → 共现(coo)决定学会，实体流行度(pop)决定信心
    # 验证方法：
    # (A) 逐因素计算与 acc 和 conf 的偏相关（控制其他因素）
    # (B) 做对的样本中，coo vs pop 哪个影响 conf？
    # (C) 做错的样本中，coo vs pop 哪个影响 conf？
    # ═══════════════════════════════════════════════════════════════════════
    print("\n\n" + "="*70)
    print("问题2: 什么影响 acc（学会），什么影响 conf（信心）？")
    print("="*70)

    for dataset in DATASETS:
        for model in ['llama8b', 'chatgpt']:
            samples = collect_samples(dataset, model, full_dict, co_occu, single_occr)
            if len(samples) < 100:
                continue

            print(f"\n--- {dataset} × {model} (n={len(samples)}) ---")

            qpop = np.array([s['question_pop'] for s in samples], dtype=float)
            gt_pop = np.array([s['gt_pop'] for s in samples], dtype=float)
            gene_pop = np.array([s['gene_pop'] for s in samples], dtype=float)
            coo = np.array([s['coo'] for s in samples], dtype=float)
            gene_coo = np.array([s['gene_coo'] for s in samples], dtype=float)
            conf = np.array([s['conf'] for s in samples], dtype=float)
            acc = np.array([s['acc'] for s in samples], dtype=float)

            # ── (A) 全样本：各因素与 acc/conf 的简单 Spearman ──
            print("  [全样本] 各因素 vs acc / conf 的 Spearman:")
            for name, arr in [('question_pop', qpop), ('gt_pop', gt_pop),
                              ('gene_pop', gene_pop), ('coo', coo), ('gene_coo', gene_coo)]:
                if np.std(arr) == 0:
                    continue
                r_a, _ = spearmanr(arr, acc)
                r_c, _ = spearmanr(arr, conf)
                # 标记哪个更强
                acc_mark = '***' if abs(r_a) > abs(r_c) else ''
                conf_mark = '***' if abs(r_c) > abs(r_a) else ''
                print(f"    {name:15s}  vs acc: {r_a:+.3f} {acc_mark:3s}  vs conf: {r_c:+.3f} {conf_mark:3s}")

            # ── (B) 逐步回归式的偏相关：每个因素控制其余因素后对 acc/conf 的影响 ──
            print("  [全样本] 偏相关（控制其余因素后，对 acc/conf 的独立贡献）:")
            # 因素列表（去掉 gene_coo 避免共线性，因为它和 acc 强绑定）
            factors = {
                'question_pop': qpop,
                'gt_pop': gt_pop,
                'gene_pop': gene_pop,
                'coo': coo,
            }
            # 对 acc：控制其他所有因素
            target_acc = acc
            target_conf = conf
            for fname, farr in factors.items():
                other_factors = [v for k, v in factors.items() if k != fname]
                # 联合控制变量：将其他因素整合
                # 简化做法：依次控制每个其他因素，取最保守的偏相关
                # 更严谨：做多元偏相关（控制所有其他因素）
                # 实现方法：对 farr 和 target 分别对所有控制变量做 rank 回归取残差
                from scipy.stats import rankdata
                rf = rankdata(farr).astype(float)
                rt_acc = rankdata(target_acc).astype(float)
                rt_conf = rankdata(target_conf).astype(float)

                # 对所有其他因素做回归，取残差
                def resid_after_controls(y, control_arrays):
                    """y 对所有 control 做 OLS（rank空间），返回残差"""
                    y_c = y - y.mean()
                    # 构造控制矩阵
                    X = np.column_stack([rankdata(c).astype(float) for c in control_arrays])
                    X = X - X.mean(axis=0)
                    # 正规方程
                    try:
                        beta = np.linalg.lstsq(X, y_c, rcond=None)[0]
                        return y_c - X @ beta
                    except:
                        return y_c

                other_arrs = [factors[k] for k in factors if k != fname]
                res_f = resid_after_controls(rf, other_arrs)
                res_acc = resid_after_controls(rt_acc, other_arrs)
                res_conf = resid_after_controls(rt_conf, other_arrs)

                r_acc_partial, p_acc = pearsonr(res_f, res_acc)
                r_conf_partial, p_conf = pearsonr(res_f, res_conf)
                acc_mark = '***' if abs(r_acc_partial) > abs(r_conf_partial) else ''
                conf_mark = '***' if abs(r_conf_partial) > abs(r_acc_partial) else ''
                print(f"    {fname:15s}  → acc: {r_acc_partial:+.3f} (p={p_acc:.1e}) {acc_mark:3s}  "
                      f"→ conf: {r_conf_partial:+.3f} (p={p_conf:.1e}) {conf_mark:3s}")

            # ── (C) 做对/做错分开看：影响 conf 的因素是否不同？──
            print("  [做对样本] 各因素 vs conf:")
            right_mask = acc == 1
            wrong_mask = acc == 0
            for name, arr in [('question_pop', qpop), ('gt_pop', gt_pop),
                              ('gene_pop', gene_pop), ('coo', coo)]:
                if right_mask.sum() > 30 and np.std(arr[right_mask]) > 0:
                    r_r, _ = spearmanr(arr[right_mask], conf[right_mask])
                else:
                    r_r = float('nan')
                if wrong_mask.sum() > 30 and np.std(arr[wrong_mask]) > 0:
                    r_w, _ = spearmanr(arr[wrong_mask], conf[wrong_mask])
                else:
                    r_w = float('nan')
                print(f"    {name:15s}  做对: {r_r:+.3f}  做错: {r_w:+.3f}")

    # ═══════════════════════════════════════════════════════════════════════
    # 问题2 补充: 做对的样本中 conf 的驱动因素
    # 如果 coo 决定学会，那做对的样本中 coo 应该较高，
    # 而 conf 应该更多由 pop 相关因素驱动
    # ═══════════════════════════════════════════════════════════════════════
    print("\n\n" + "="*70)
    print("问题2 补充: 做对样本中，coo vs pop 对 conf 的影响对比")
    print("="*70)

    for dataset in DATASETS:
        all_right = []
        all_wrong = []
        for model in ['llama8b', 'qwen2', 'chatgpt']:
            s = collect_samples(dataset, model, full_dict, co_occu, single_occr)
            for item in s:
                if item['acc'] == 1:
                    all_right.append(item)
                else:
                    all_wrong.append(item)

        if len(all_right) < 50 or len(all_wrong) < 50:
            continue

        print(f"\n--- {dataset} ---")
        print(f"  做对: n={len(all_right)}  做错: n={len(all_wrong)}")

        for target_name, target_data in [('做对', all_right), ('做错', all_wrong)]:
            conf = np.array([s['conf'] for s in target_data], dtype=float)
            qpop = np.array([s['question_pop'] for s in target_data], dtype=float)
            gt_pop = np.array([s['gt_pop'] for s in target_data], dtype=float)
            gene_pop = np.array([s['gene_pop'] for s in target_data], dtype=float)
            coo = np.array([s['coo'] for s in target_data], dtype=float)

            print(f"  [{target_name}] conf 均值={conf.mean():.3f}")
            # 控制 coo 后偏相关
            r_qpop, _ = partial_spearman(qpop, conf, coo)
            r_gt_pop, _ = partial_spearman(gt_pop, conf, coo)
            r_gene_pop, _ = partial_spearman(gene_pop, conf, coo)
            r_coo, _ = partial_spearman(coo, conf, qpop)  # 控制 qpop

            print(f"    控制 coo 后: qpop→conf={r_qpop:+.3f}  gt_pop→conf={r_gt_pop:+.3f}  gene_pop→conf={r_gene_pop:+.3f}")
            print(f"    控制 qpop 后: coo→conf={r_coo:+.3f}")


if __name__ == '__main__':
    main()