"""
偏相关分析（Partial Correlation Analysis）

研究问题：
  "共现次数(coo)决定acc，单实体流行度(question_pop)决定conf" 是否成立？

方法：
  1. 直接 Spearman 相关：coo vs acc, question_pop vs conf
  2. 偏相关：
     - 控制 question_pop 后，coo 与 acc 的相关性是否仍显著？
     - 控制 coo 后，question_pop 与 conf 的相关性是否仍显著？
  3. 对称实验（分组控制）：
     - 将 question_pop 分4个四分位区间，在每个区间内比较高/低 coo 对 acc 和 conf 的影响
     - 将 coo 分4个四分位区间，在每个区间内比较高/低 question_pop 对 acc 和 conf 的影响
     注意：只汇报 bin1~3，bin4 内 control 变量范围过宽，控制效果不可靠
  4. 过度自信来源分析（仅 acc=0 的样本）：
     - 在做错的样本中，分别按 question_pop 和 coo 分组，观察 conf 的变化
     - 验证"模型的自信来自对问题的熟悉（question_pop），而非对正确答案的掌握（coo）"

过滤规则（与 main.py 一致）：
  - popularity == "No" 的样本跳过
  - single_occurrence > 6000 的样本跳过：
    - movies/songs：question_entity、gene_entity、ref 三个实体都过滤
    - basketball：仅过滤 question_entity（人名），gene_entity/ref 为城市名不过滤
  - 统计文件（co_occurrence / single_occurrence）中找不到对应实体的样本跳过

注意：所有分析按 dataset × model 独立进行，不跨模型合并。
"""

import json
import math
import re
import os
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

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

SINGLE_OCC_THRESHOLD = 6000  # 与 main.py 一致


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


def partial_spearman(x, y, z):
    """
    计算 x 与 y 的偏相关系数（控制 z）。
    方法：对 x 和 y 分别对 z 做 Spearman 回归（rank 线性回归），取残差后再算相关。
    实现：先对三者取 rank，再用线性回归去除 z 的影响，最后算 Pearson（等价于偏 Spearman）。
    """
    from scipy.stats import rankdata, pearsonr
    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)
    rz = rankdata(z).astype(float)

    # 对 rx 关于 rz 做线性回归，取残差
    rz_mean = rz.mean()
    rz_c = rz - rz_mean
    beta_x = np.dot(rz_c, rx) / np.dot(rz_c, rz_c)
    res_x = rx - beta_x * rz_c

    # 对 ry 关于 rz 做线性回归，取残差
    beta_y = np.dot(rz_c, ry) / np.dot(rz_c, rz_c)
    res_y = ry - beta_y * rz_c

    r, p = pearsonr(res_x, res_y)
    return r, p


def load_popularity():
    """加载 popularity 数据，返回 {entity_name: popularity_value} 字典"""
    pop_data = read_jsonl(POP_PATH)
    full_dict = {}
    for d in pop_data:
        full_dict.update(d)
    return full_dict


def collect_samples(dataset, model, full_dict, co_occu, single_occr):
    """
    收集一个 dataset+model 组合的所有有效样本。
    返回列表，每个元素是 dict：
      question_pop, gt_pop, coo, conf, acc
    """
    res_path = os.path.join(RES_DIR, dataset, f'{dataset}_{model}_temperature1.jsonl')
    if not os.path.exists(res_path):
        return []

    model_res = read_jsonl(res_path)
    samples = []
    skip_stat_miss = 0  # 统计文件中找不到实体的样本数

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
        # basketball: question_entity 是人名，可能出现歧义（如 "he jun": 77608）；
        #   但 gene_entity/ref 是城市名，天然高频（如 "chicago"），不属于异常，不过滤
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

        # confidence
        if 'gpt' in model.lower():
            probs = [math.exp(t) for t in item['Log_p']['token_logprobs']]
        else:
            probs = item['Log_p']['token_probs']
        conf = sum(probs) / len(probs)

        # co-occurrence：用 GT answer 与 question 的共现（与 main.py 中 gt_cooccurance 一致）
        coo = co_occu[question_entity].get(ref.lower(), 0)

        samples.append({
            'question_pop': item['popularity'],
            'gt_pop': ref_pop,
            'coo': coo,
            'conf': conf,
            'acc': item['has_answer'],
        })

    if skip_stat_miss > 0:
        print(f"  [{dataset} × {model}] 跳过 {skip_stat_miss} 条统计文件中缺失实体的样本")

    return samples


def run_partial_correlation(samples, label=''):
    """对一组样本做完整的偏相关分析"""
    if len(samples) < 30:
        print(f"  [skip] 样本数不足: {len(samples)}")
        return

    qpop = np.array([s['question_pop'] for s in samples], dtype=float)
    gt_pop = np.array([s['gt_pop'] for s in samples], dtype=float)
    coo = np.array([s['coo'] for s in samples], dtype=float)
    conf = np.array([s['conf'] for s in samples], dtype=float)
    acc = np.array([s['acc'] for s in samples], dtype=float)

    print(f"\n{'='*60}")
    print(f"  {label}  (n={len(samples)})")
    print(f"{'='*60}")
    print(f"  avg_acc={acc.mean():.3f}  avg_conf={conf.mean():.3f}")

    # ── 1. 直接 Spearman 相关 ──────────────────────────────────────────────
    print("\n[1] 直接 Spearman 相关")
    for name, x in [('question_pop', qpop), ('gt_pop', gt_pop), ('coo', coo)]:
        r_acc, p_acc = spearmanr(x, acc)
        r_conf, p_conf = spearmanr(x, conf)
        print(f"  {name:15s} vs acc : r={r_acc:+.3f}  p={p_acc:.3e}")
        print(f"  {name:15s} vs conf: r={r_conf:+.3f}  p={p_conf:.3e}")

    # ── 2. 偏相关：控制 question_pop，看 coo 与 acc 的关系 ─────────────────
    print("\n[2] 偏相关分析")
    r_pc, p_pc = partial_spearman(coo, acc, qpop)
    print(f"  coo vs acc  (控制 question_pop): r={r_pc:+.3f}  p={p_pc:.3e}")

    r_pc2, p_pc2 = partial_spearman(qpop, conf, coo)
    print(f"  question_pop vs conf (控制 coo): r={r_pc2:+.3f}  p={p_pc2:.3e}")

    # 反向验证
    r_pc3, p_pc3 = partial_spearman(qpop, acc, coo)
    print(f"  question_pop vs acc  (控制 coo): r={r_pc3:+.3f}  p={p_pc3:.3e}  [预期: 弱]")

    r_pc4, p_pc4 = partial_spearman(coo, conf, qpop)
    print(f"  coo vs conf (控制 question_pop): r={r_pc4:+.3f}  p={p_pc4:.3e}  [预期: 弱]")

    # ── 3. 分组控制实验 ────────────────────────────────────────────────────
    print("\n[3] 分组控制实验")

    def group_experiment(control_var, control_name, test_var, test_name, target, target_name, n_bins=4):
        """
        将 control_var 分成 n_bins 个四分位区间，
        在每个区间内按 test_var 中位数分高/低两组，
        比较 target 的差异（Mann-Whitney U 检验）。
        只汇报 bin1~3：bin4 内 control 变量范围过宽，控制效果不可靠。
        同时输出每个 bin 内 control 变量的 std，供判断控制质量。
        """
        percentiles = np.percentile(control_var, np.linspace(0, 100, n_bins + 1))
        results = []
        for i in range(n_bins):
            lo, hi = percentiles[i], percentiles[i + 1]
            if i == 0:
                mask = (control_var >= lo) & (control_var <= hi)
            else:
                mask = (control_var > lo) & (control_var <= hi)
            idx = np.where(mask)[0]
            if len(idx) < 10:
                continue
            tv = test_var[idx]
            tgt = target[idx]
            med = np.median(tv)
            high_idx = idx[tv > med]
            low_idx = idx[tv <= med]
            if len(high_idx) < 5 or len(low_idx) < 5:
                continue
            high_tgt = target[high_idx]
            low_tgt = target[low_idx]
            stat, p = mannwhitneyu(high_tgt, low_tgt, alternative='two-sided')
            ctrl_std = control_var[idx].std()
            results.append({
                'bin': i + 1,
                'n': len(idx),
                'high_mean': high_tgt.mean(),
                'low_mean': low_tgt.mean(),
                'diff': high_tgt.mean() - low_tgt.mean(),
                'p': p,
                'ctrl_std': ctrl_std,
            })
        if not results:
            print(f"    [skip] 分组后样本不足")
            return []
        # 只取 bin1~3
        results_13 = [r for r in results if r['bin'] <= 3]
        results_all = results
        avg_diff_13 = np.mean([r['diff'] for r in results_13]) if results_13 else float('nan')
        sig_bins_13 = sum(1 for r in results_13 if r['p'] < 0.05)
        print(f"  控制 {control_name}，比较高/低 {test_name} 对 {target_name} 的影响 (bin1~3):")
        for r in results_13:
            sig = '*' if r['p'] < 0.05 else ' '
            print(f"    bin{r['bin']} (n={r['n']:4d}, ctrl_std={r['ctrl_std']:.2f}): "
                  f"high={r['high_mean']:.3f}  low={r['low_mean']:.3f}  "
                  f"diff={r['diff']:+.3f}  p={r['p']:.3e} {sig}")
        if results_all[3:]:  # bin4 存在时提示
            r4 = results_all[3]
            print(f"    bin4 (n={r4['n']:4d}, ctrl_std={r4['ctrl_std']:.2f}): "
                  f"[不引用，控制变量范围过宽]")
        print(f"    → bin1~3 平均差异={avg_diff_13:+.3f}，{sig_bins_13}/{len(results_13)} 个区间显著")
        return results_13

    # 控制 question_pop，比较高/低 coo 对 acc 的影响（预期：差异大）
    group_experiment(qpop, 'question_pop', coo, 'coo', acc, 'acc')
    # 控制 question_pop，比较高/低 coo 对 conf 的影响（预期：差异小）
    group_experiment(qpop, 'question_pop', coo, 'coo', conf, 'conf')
    # 控制 coo，比较高/低 question_pop 对 conf 的影响（预期：差异大）
    group_experiment(coo, 'coo', qpop, 'question_pop', conf, 'conf')
    # 控制 coo，比较高/低 question_pop 对 acc 的影响（预期：差异小）
    group_experiment(coo, 'coo', qpop, 'question_pop', acc, 'acc')

    # ── 4. 过度自信来源分析（仅 acc=0 的样本）────────────────────────────
    print("\n[4] 过度自信来源分析（仅 acc=0 的样本）")
    wrong_mask = acc == 0
    n_wrong = wrong_mask.sum()
    if n_wrong < 20:
        print("  wrong 样本不足，跳过")
        return
    w_conf = conf[wrong_mask]
    w_qpop = qpop[wrong_mask]
    w_coo  = coo[wrong_mask]
    print(f"  wrong 样本数={n_wrong}  avg_conf={w_conf.mean():.3f}")

    # 按 question_pop 四分位分组，看 conf 变化
    pcts_q = np.percentile(w_qpop, [0, 25, 50, 75, 100])
    print("  按 question_pop 分组（做错样本中）：")
    for i in range(4):
        lo, hi = pcts_q[i], pcts_q[i + 1]
        m = (w_qpop >= lo) & (w_qpop <= hi) if i == 0 else (w_qpop > lo) & (w_qpop <= hi)
        if m.sum() < 5:
            continue
        print(f"    qpop∈[{lo:.0f},{hi:.0f}]  n={m.sum():5d}  "
              f"avg_conf={w_conf[m].mean():.3f}  avg_coo={w_coo[m].mean():.1f}")

    # 按 coo 四分位分组，看 conf 变化
    pcts_c = np.percentile(w_coo, [0, 25, 50, 75, 100])
    print("  按 coo 分组（做错样本中）：")
    for i in range(4):
        lo, hi = pcts_c[i], pcts_c[i + 1]
        m = (w_coo >= lo) & (w_coo <= hi) if i == 0 else (w_coo > lo) & (w_coo <= hi)
        if m.sum() < 5:
            continue
        print(f"    coo∈[{lo:.0f},{hi:.0f}]   n={m.sum():5d}  "
              f"avg_conf={w_conf[m].mean():.3f}  avg_qpop={w_qpop[m].mean():.1f}")

    # 用 Spearman 量化
    r_qpop_conf, p_qpop_conf = spearmanr(w_qpop, w_conf)
    r_coo_conf,  p_coo_conf  = spearmanr(w_coo,  w_conf)
    print(f"  Spearman(question_pop, conf | acc=0): r={r_qpop_conf:+.3f}  p={p_qpop_conf:.3e}")
    print(f"  Spearman(coo,          conf | acc=0): r={r_coo_conf:+.3f}  p={p_coo_conf:.3e}")
    print(f"  → {'question_pop 驱动过度自信' if abs(r_qpop_conf) > abs(r_coo_conf) else 'coo 驱动过度自信'}"
          f"（|r_qpop|={abs(r_qpop_conf):.3f} vs |r_coo|={abs(r_coo_conf):.3f}）")

    # ── 5. 交互效应分析 ──────────────────────────────────────────────────
    print("\n[5] 交互效应分析：question_pop 是否调节 coo→acc / coo→conf 的关系？")
    # 将 question_pop 分高低两组，分别计算 coo 与 acc/conf 的 Spearman 相关
    qpop_median = np.median(qpop)
    low_qpop_mask = qpop <= qpop_median
    high_qpop_mask = qpop > qpop_median
    for mask, label in [(low_qpop_mask, '低 question_pop'), (high_qpop_mask, '高 question_pop')]:
        n_mask = mask.sum()
        if n_mask < 30:
            print(f"  {label}: 样本不足 (n={n_mask})")
            continue
        r_coo_acc, p_coo_acc = spearmanr(coo[mask], acc[mask])
        r_coo_conf, p_coo_conf = spearmanr(coo[mask], conf[mask])
        print(f"  {label} (n={n_mask}): "
              f"Spearman(coo, acc)={r_coo_acc:+.3f} p={p_coo_acc:.1e}  |  "
              f"Spearman(coo, conf)={r_coo_conf:+.3f} p={p_coo_conf:.1e}")

    # 将 coo 分高低两组，分别计算 question_pop 与 acc/conf 的 Spearman 相关
    coo_median = np.median(coo)
    low_coo_mask = coo <= coo_median
    high_coo_mask = coo > coo_median
    for mask, label in [(low_coo_mask, '低 coo'), (high_coo_mask, '高 coo')]:
        n_mask = mask.sum()
        if n_mask < 30:
            print(f"  {label}: 样本不足 (n={n_mask})")
            continue
        r_qpop_acc, p_qpop_acc = spearmanr(qpop[mask], acc[mask])
        r_qpop_conf, p_qpop_conf = spearmanr(qpop[mask], conf[mask])
        print(f"  {label} (n={n_mask}): "
              f"Spearman(qpop, acc)={r_qpop_acc:+.3f} p={p_qpop_acc:.1e}  |  "
              f"Spearman(qpop, conf)={r_qpop_conf:+.3f} p={p_qpop_conf:.1e}")


# ─── 主程序 ───────────────────────────────────────────────────────────────────
def main():
    print("加载数据...")
    full_dict = load_popularity()
    co_occu = json.loads(open(COO_PATH).read())
    single_occr = json.loads(open(SINGLE_PATH).read())

    # 按 dataset × model 单独分析（不跨模型合并，避免不同模型做错样本集不同导致伪相关）
    print("\n" + "="*60)
    print("按 dataset × model 分析（逐模型，不合并）")
    print("="*60)
    for dataset in DATASETS:
        for model in MODELS:
            samples = collect_samples(dataset, model, full_dict, co_occu, single_occr)
            if len(samples) < 30:
                continue
            run_partial_correlation(samples, label=f'{dataset} × {model}')


if __name__ == '__main__':
    main()
