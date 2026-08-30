"""
对比分析：过滤 vs 不过滤 Wikidata 查不到的 gt_pop/gene_pop 样本

核心对比 Table 2 (做错样本 gene_pop/gene_coo → conf) 和 Table 3 (做对 vs 做错 asymmetry)
"""
import json, math, re, os, sys
import numpy as np
from scipy.stats import spearmanr, pearsonr, rankdata

# 直接从 verify_per_model 导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_per_model import (
    collect_samples, load_popularity, partial_spearman,
    DATASETS, MODELS, PATTERN, SINGLE_OCC_THRESHOLD,
    Q_SINGLE_LO, Q_SINGLE_HI
)

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES_DIR = os.path.join(BASE, 'res')
COO_PATH = os.path.join(RES_DIR, 'cooccurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')
SINGLE_PATH = os.path.join(RES_DIR, 'single_occurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')

def run_comparison():
    print("加载数据...")
    full_dict = load_popularity()
    co_occu = json.loads(open(COO_PATH).read())
    single_occr = json.loads(open(SINGLE_PATH).read())

    # 分别收集 filter=False 和 filter=True 的样本
    data_no_filter = {}
    data_filter = {}
    
    for dataset in DATASETS:
        for model in MODELS:
            if dataset == 'basketball' and model.startswith('Qwen2.5'):
                continue
            
            s1 = collect_samples(dataset, model, full_dict, co_occu, single_occr,
                                 filter_pop_no=False)
            s2 = collect_samples(dataset, model, full_dict, co_occu, single_occr,
                                 filter_pop_no=True)
            data_no_filter[(dataset, model)] = s1
            data_filter[(dataset, model)] = s2
            
            n1, n2 = len(s1), len(s2)
            n1w = sum(1 for s in s1 if s['acc'] == 0)
            n2w = sum(1 for s in s2 if s['acc'] == 0)
            print(f"  {dataset:12s} × {model:15s}: 无过滤 n={n1:5d}(wrong={n1w:5d}) | 过滤后 n={n2:5d}(wrong={n2w:5d}) | 减少 {n1-n2:5d}")

    # ══════════════════════════════════════════════════════════════════
    # 表2对比: 做错样本 gene_pop/gene_coo → conf
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("表2 对比: 做错样本 gene_pop→conf / gene_coo→conf (partial, controlling coo)")
    print("=" * 120)
    
    for label, data in [("无过滤(pop=0)", data_no_filter), ("过滤pop='No'", data_filter)]:
        print(f"\n--- {label} ---")
        print(f"  {'dataset':12s} {'model':15s} {'n_wrong':>8s} | {'gene_pop→conf':>14s} {'gene_coo→conf':>14s} | {'pop=0占比':>10s}")
        print(f"  {'-'*12} {'-'*15} {'-'*8} {'-'*14} {'-'*14} {'-'*10}")
        
        for dataset in DATASETS:
            for model in MODELS:
                if dataset == 'basketball' and model.startswith('Qwen2.5'):
                    continue
                samples = data.get((dataset, model), [])
                wrong = [s for s in samples if s['acc'] == 0]
                if len(wrong) < 50:
                    continue
                
                gene_pop = np.array([s['gene_pop'] for s in wrong], dtype=float)
                gene_coo = np.array([s['gene_coo'] for s in wrong], dtype=float)
                conf = np.array([s['conf'] for s in wrong], dtype=float)
                coo = np.array([s['coo'] for s in wrong], dtype=float)
                
                pop0_ratio = (gene_pop == 0).sum() / len(gene_pop) * 100
                
                r_gp, _ = partial_spearman(gene_pop, conf, coo) if np.std(gene_pop) > 0 else (float('nan'), 1)
                r_gc, _ = partial_spearman(gene_coo, conf, coo) if np.std(gene_coo) > 0 else (float('nan'), 1)
                
                print(f"  {dataset:12s} {model:15s} {len(wrong):8d} | {r_gp:+14.3f} {r_gc:+14.3f} | {pop0_ratio:>9.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # 表3对比: 做对 vs 做错 asymmetry (gene_pop→conf)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("表3 对比: 做对 vs 做错 gene_pop→conf (partial, controlling coo) — asymmetry")
    print("=" * 120)
    
    for label, data in [("无过滤(pop=0)", data_no_filter), ("过滤pop='No'", data_filter)]:
        print(f"\n--- {label} ---")
        print(f"  {'dataset':12s} {'model':15s} | {'Correct':>10s} {'Incorrect':>10s} | {'Ratio':>6s} | {'pop=0_corr':>10s} {'pop=0_wrong':>11s}")
        print(f"  {'-'*12} {'-'*15} {'-'*10} {'-'*10} {'-'*6} {'-'*10} {'-'*11}")
        
        for dataset in ['movies', 'songs']:  # Basketball 的 asymmetry 不是核心
            for model in MODELS:
                if dataset == 'basketball' and model.startswith('Qwen2.5'):
                    continue
                samples = data.get((dataset, model), [])
                if len(samples) < 100:
                    continue
                
                correct = [s for s in samples if s['acc'] == 1]
                wrong = [s for s in samples if s['acc'] == 0]
                
                if len(correct) < 30 or len(wrong) < 30:
                    continue
                
                # Correct
                gene_pop_c = np.array([s['gene_pop'] for s in correct], dtype=float)
                conf_c = np.array([s['conf'] for s in correct], dtype=float)
                coo_c = np.array([s['coo'] for s in correct], dtype=float)
                r_c, _ = partial_spearman(gene_pop_c, conf_c, coo_c) if np.std(gene_pop_c) > 0 else (float('nan'), 1)
                
                # Incorrect
                gene_pop_w = np.array([s['gene_pop'] for s in wrong], dtype=float)
                conf_w = np.array([s['conf'] for s in wrong], dtype=float)
                coo_w = np.array([s['coo'] for s in wrong], dtype=float)
                r_w, _ = partial_spearman(gene_pop_w, conf_w, coo_w) if np.std(gene_pop_w) > 0 else (float('nan'), 1)
                
                pop0_c = (gene_pop_c == 0).sum() / len(gene_pop_c) * 100
                pop0_w = (gene_pop_w == 0).sum() / len(gene_pop_w) * 100
                
                ratio = abs(r_w / r_c) if abs(r_c) > 0.01 else float('inf')
                
                print(f"  {dataset:12s} {model:15s} | {r_c:+10.3f} {r_w:+10.3f} | {ratio:>5.1f}× | {pop0_c:>9.1f}% {pop0_w:>10.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # 汇总: 关键数字对比
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("汇总: 过滤前后关键指标变化")
    print("=" * 120)
    
    print(f"\n  {'dataset':12s} {'model':15s} | {'gene_pop→conf(wrong)':>22s} | {'asymmetry ratio':>16s} | {'样本减少':>8s}")
    print(f"  {'-'*12} {'-'*15} {'-'*22} {'-'*16} {'-'*8}")
    
    for dataset in DATASETS:
        for model in MODELS:
            if dataset == 'basketball' and model.startswith('Qwen2.5'):
                continue
            
            s1 = data_no_filter.get((dataset, model), [])
            s2 = data_filter.get((dataset, model), [])
            
            wrong1 = [s for s in s1 if s['acc'] == 0]
            wrong2 = [s for s in s2 if s['acc'] == 0]
            
            if len(wrong1) < 50 or len(wrong2) < 50:
                continue
            
            gene_pop1 = np.array([s['gene_pop'] for s in wrong1], dtype=float)
            gene_pop2 = np.array([s['gene_pop'] for s in wrong2], dtype=float)
            conf1 = np.array([s['conf'] for s in wrong1], dtype=float)
            conf2 = np.array([s['conf'] for s in wrong2], dtype=float)
            coo1 = np.array([s['coo'] for s in wrong1], dtype=float)
            coo2 = np.array([s['coo'] for s in wrong2], dtype=float)
            
            r1, _ = partial_spearman(gene_pop1, conf1, coo1) if np.std(gene_pop1) > 0 else (float('nan'), 1)
            r2, _ = partial_spearman(gene_pop2, conf2, coo2) if np.std(gene_pop2) > 0 else (float('nan'), 1)
            
            # Asymmetry ratio
            correct1 = [s for s in s1 if s['acc'] == 1]
            correct2 = [s for s in s2 if s['acc'] == 1]
            
            ratio1 = "N/A"
            ratio2 = "N/A"
            if len(correct1) >= 30:
                gp_c1 = np.array([s['gene_pop'] for s in correct1], dtype=float)
                cf_c1 = np.array([s['conf'] for s in correct1], dtype=float)
                coo_c1 = np.array([s['coo'] for s in correct1], dtype=float)
                rc1, _ = partial_spearman(gp_c1, cf_c1, coo_c1) if np.std(gp_c1) > 0 else (float('nan'), 1)
                if abs(rc1) > 0.01:
                    ratio1 = f"{abs(r1/rc1):.1f}×"
            if len(correct2) >= 30:
                gp_c2 = np.array([s['gene_pop'] for s in correct2], dtype=float)
                cf_c2 = np.array([s['conf'] for s in correct2], dtype=float)
                coo_c2 = np.array([s['coo'] for s in correct2], dtype=float)
                rc2, _ = partial_spearman(gp_c2, cf_c2, coo_c2) if np.std(gp_c2) > 0 else (float('nan'), 1)
                if abs(rc2) > 0.01:
                    ratio2 = f"{abs(r2/rc2):.1f}×"
            
            n_diff = len(s1) - len(s2)
            print(f"  {dataset:12s} {model:15s} | {r1:+.3f} → {r2:+.3f} ({r2-r1:+.3f}) | {ratio1:>7s} → {ratio2:>7s} | {n_diff:>8d}")


if __name__ == '__main__':
    run_comparison()
