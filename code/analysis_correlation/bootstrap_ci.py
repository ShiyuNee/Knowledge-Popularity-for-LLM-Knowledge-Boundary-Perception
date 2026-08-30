"""
Bootstrap 置信区间计算
为核心偏相关结果（§4.1, §5.1, §5.2, §6.1）计算 95% CI
数据采集复用 verify_per_model.collect_samples（filter_pop_no=True）
"""

import json
import math
import re
import os
import argparse
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
N_BOOTSTRAP = 1000
SEED = 42


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


from verify_per_model import collect_samples, load_popularity, _skip_bball_qwen25, partial_spearman



def bootstrap_ci(data, statistic_fn, n_boot=N_BOOTSTRAP, seed=SEED):
    """
    对数据做 bootstrap，计算统计量的 95% percentile CI
    data: list of dicts (samples)
    statistic_fn: function(samples) -> scalar
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    if n < 30:
        return None, None, None

    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_samples = [data[i] for i in idx]
        stat = statistic_fn(boot_samples)
        if stat is not None and not np.isnan(stat):
            stats.append(stat)

    if len(stats) < n_boot * 0.5:
        return None, None, None

    stats = np.array(stats)
    point = statistic_fn(data)
    lo = np.percentile(stats, 2.5)
    hi = np.percentile(stats, 97.5)
    return point, lo, hi


def stat_gene_pop_conf_wrong(samples):
    """做错样本中 gene_pop→conf 的偏相关（控制 coo）"""
    wrong = [s for s in samples if s['acc'] == 0]
    if len(wrong) < 20:
        return None
    gp = np.array([s['gene_pop'] for s in wrong], dtype=float)
    cf = np.array([s['conf'] for s in wrong], dtype=float)
    co = np.array([s['coo'] for s in wrong], dtype=float)
    if np.std(gp) == 0 or np.std(cf) == 0 or np.std(co) == 0:
        return None
    r, _ = partial_spearman(gp, cf, co)
    return r


def stat_gene_coo_conf_wrong(samples):
    """做错样本中 gene_coo→conf 的偏相关（控制 coo）"""
    wrong = [s for s in samples if s['acc'] == 0]
    if len(wrong) < 20:
        return None
    gc = np.array([s['gene_coo'] for s in wrong], dtype=float)
    cf = np.array([s['conf'] for s in wrong], dtype=float)
    co = np.array([s['coo'] for s in wrong], dtype=float)
    if np.std(gc) == 0 or np.std(cf) == 0 or np.std(co) == 0:
        return None
    r, _ = partial_spearman(gc, cf, co)
    return r


def stat_gene_pop_conf_correct(samples):
    """做对样本中 gene_pop→conf 的 Spearman"""
    correct = [s for s in samples if s['acc'] == 1]
    if len(correct) < 20:
        return None
    gp = np.array([s['gene_pop'] for s in correct], dtype=float)
    cf = np.array([s['conf'] for s in correct], dtype=float)
    co = np.array([s['coo'] for s in correct], dtype=float)
    if np.std(gp) == 0 or np.std(cf) == 0:
        return None
    r, _ = partial_spearman(gp, cf, co)
    return r


def stat_qpop_conf_wrong(samples):
    """做错样本中 question_pop→conf 的偏相关（控制 coo）"""
    wrong = [s for s in samples if s['acc'] == 0]
    if len(wrong) < 20:
        return None
    qp = np.array([s['question_pop'] for s in wrong], dtype=float)
    cf = np.array([s['conf'] for s in wrong], dtype=float)
    co = np.array([s['coo'] for s in wrong], dtype=float)
    if np.std(qp) == 0 or np.std(cf) == 0 or np.std(co) == 0:
        return None
    r, _ = partial_spearman(qp, cf, co)
    return r


def stat_gt_coo_acc(samples):
    """gt_coo→acc 的 Spearman"""
    if len(samples) < 30:
        return None
    co = np.array([s['coo'] for s in samples], dtype=float)
    ac = np.array([s['acc'] for s in samples], dtype=float)
    if np.std(co) == 0 or np.std(ac) == 0:
        return None
    r, _ = spearmanr(co, ac)
    return r


def stat_qpop_conf(samples):
    """question_pop→conf 的 Spearman"""
    if len(samples) < 30:
        return None
    qp = np.array([s['question_pop'] for s in samples], dtype=float)
    cf = np.array([s['conf'] for s in samples], dtype=float)
    if np.std(qp) == 0 or np.std(cf) == 0:
        return None
    r, _ = spearmanr(qp, cf)
    return r


def format_ci(point, lo, hi):
    if point is None:
        return "N/A"
    return f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n-bootstrap', type=int, default=N_BOOTSTRAP,
        help='number of bootstrap resamples per statistic (paper setting: 1000)',
    )
    parser.add_argument(
        '--output', default='code/analysis_correlation/bootstrap_results.json',
        help='JSON file for machine-readable confidence intervals',
    )
    args = parser.parse_args()

    print("加载数据...")
    full_dict = load_popularity()
    co_occu = json.loads(open(COO_PATH).read())
    single_occr = json.loads(open(SINGLE_PATH).read())
    rng = np.random.default_rng(SEED)

    # 我们要 bootstrap 的核心统计量
    analyses = [
        ('gene_pop→conf (wrong)', stat_gene_pop_conf_wrong),
        ('gene_coo→conf (wrong)', stat_gene_coo_conf_wrong),
        ('gene_pop→conf (correct)', stat_gene_pop_conf_correct),
        ('qpop→conf (wrong)', stat_qpop_conf_wrong),
        ('gt_coo→acc', stat_gt_coo_acc),
        ('qpop→conf (all)', stat_qpop_conf),
    ]

    results = {}

    for dataset in DATASETS:
        results[dataset] = {}
        for model in MODELS:
            if dataset == 'basketball' and model.startswith('Qwen2.5'):
                continue

            samples = collect_samples(dataset, model, full_dict, co_occu, single_occr,
                                          filter_pop_no=True)
            if len(samples) < 50:
                continue

            label = f"{dataset} × {model}"
            print(f"\n{'='*60}")
            print(f"  {label} (n={len(samples)})")
            print(f"{'='*60}")

            results[dataset][model] = {}

            for name, stat_fn in analyses:
                point, lo, hi = bootstrap_ci(
                    samples, stat_fn, n_boot=args.n_bootstrap,
                    seed=rng.integers(0, 100000),
                )
                ci_str = format_ci(point, lo, hi)
                results[dataset][model][name] = (point, lo, hi)
                print(f"  {name:<30s}: {ci_str}")

    # 汇总输出
    print("\n\n" + "="*80)
    print("  汇总：核心统计量的 95% Bootstrap CI")
    print("="*80)

    # §4.2 style table: gene_pop→conf (correct vs wrong)
    print("\n[§4.2] gene_pop→conf (partial, controlling for coo)")
    print(f"{'Dataset':<12} {'Model':<16} {'Correct':>28} {'Incorrect':>28} {'Ratio':>8}")
    print("-" * 100)
    for dataset in ['movies', 'songs']:
        for model in MODELS:
            if model not in results.get(dataset, {}):
                continue
            r = results[dataset][model]
            pc, lc, hc = r.get('gene_pop→conf (correct)', (None, None, None))
            pw, lw, hw = r.get('gene_pop→conf (wrong)', (None, None, None))
            if pc is not None and pw is not None:
                ratio = pw / pc if pc != 0 else float('inf')
                c_str = f"{pc:+.3f} [{lc:+.3f},{hc:+.3f}]"
                w_str = f"{pw:+.3f} [{lw:+.3f},{hw:+.3f}]"
                print(f"{dataset:<12} {model:<16} {c_str:>28} {w_str:>28} {ratio:>7.1f}x")

    # §6.1 style table: gene_pop→conf and gene_coo→conf (wrong) across datasets
    print("\n[§6.1] Incorrect samples — gene_pop→conf vs gene_coo→conf")
    print(f"{'Dataset':<12} {'Model':<16} {'gene_pop→conf':>28} {'gene_coo→conf':>28}")
    print("-" * 100)
    for dataset in DATASETS:
        for model in MODELS:
            if model not in results.get(dataset, {}):
                continue
            r = results[dataset][model]
            pp, lp, hp = r.get('gene_pop→conf (wrong)', (None, None, None))
            pc, lc, hc = r.get('gene_coo→conf (wrong)', (None, None, None))
            if pp is not None:
                p_str = f"{pp:+.3f} [{lp:+.3f},{hp:+.3f}]"
            else:
                p_str = "N/A"
            if pc is not None:
                c_str = f"{pc:+.3f} [{lc:+.3f},{hc:+.3f}]"
            else:
                c_str = "N/A"
            print(f"{dataset:<12} {model:<16} {p_str:>28} {c_str:>28}")

    # §6.2 style: qpop→conf
    print("\n[§6.2] question_pop→conf (all samples)")
    print(f"{'Dataset':<12} {'Model':<16} {'qpop→conf':>28}")
    print("-" * 70)
    for dataset in DATASETS:
        for model in MODELS:
            if model not in results.get(dataset, {}):
                continue
            r = results[dataset][model]
            pq, lq, hq = r.get('qpop→conf (all)', (None, None, None))
            if pq is not None:
                q_str = f"{pq:+.3f} [{lq:+.3f},{hq:+.3f}]"
                print(f"{dataset:<12} {model:<16} {q_str:>28}")

    # §4.1 style: gt_coo→acc
    print("\n[§4.1] gt_coo→acc (Spearman)")
    print(f"{'Dataset':<12} {'Model':<16} {'gt_coo→acc':>28}")
    print("-" * 70)
    for dataset in DATASETS:
        for model in MODELS:
            if model not in results.get(dataset, {}):
                continue
            r = results[dataset][model]
            pa, la, ha = r.get('gt_coo→acc', (None, None, None))
            if pa is not None:
                a_str = f"{pa:+.3f} [{la:+.3f},{ha:+.3f}]"
                print(f"{dataset:<12} {model:<16} {a_str:>28}")

    native_results = {
        dataset: {
            model: {
                name: [None if value is None else float(value) for value in interval]
                for name, interval in analyses_by_model.items()
            }
            for model, analyses_by_model in models.items()
        }
        for dataset, models in results.items()
    }
    output_path = os.path.join(BASE, args.output) if not os.path.isabs(args.output) else args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as stream:
        json.dump({
            'n_bootstrap': args.n_bootstrap,
            'seed': SEED,
            'results': native_results,
        }, stream, indent=2, ensure_ascii=False)
    print(f"\nBootstrap results saved to {output_path}")


if __name__ == '__main__':
    main()
