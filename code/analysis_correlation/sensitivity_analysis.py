"""
敏感性分析：检验 single_occ 过滤阈值对实验结论的影响

测试内容：
1. Q_SINGLE_LO 变动（20, 50, 100, 200）→ quality_v2→acc 偏相关（表4）是否稳定
2. SINGLE_OCC_THRESHOLD 变动（3000, 6000, 10000, inf）→ 样本量和核心相关是否稳定
3. 不同阈值下 Basketball 样本保留率变化
"""
import json, math, re, os
import numpy as np
from scipy.stats import spearmanr, pearsonr, rankdata

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES_DIR = os.path.join(BASE, 'res')

POP_PATH = os.path.join(RES_DIR, 'gt_gene_entity_popularity_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.jsonl')
COO_PATH = os.path.join(RES_DIR, 'cooccurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')
SINGLE_PATH = os.path.join(RES_DIR, 'single_occurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')

DATASETS = ['movies', 'songs', 'basketball']
MODELS = ['qwen2', 'llama8b', 'chatgpt', 'Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']

PATTERN = {
    'movies': 'Who is the director of the movie ',
    'songs': 'Who is the performer of the song ',
    'basketball': 'Where is the birthplace of the basketball player '
}

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

def load_data():
    pop_data = read_jsonl(POP_PATH)
    full_dict = {}
    for d in pop_data:
        full_dict.update(d)
    co_occu = json.loads(open(COO_PATH, 'r', encoding='utf-8').read())
    single_occr = json.loads(open(SINGLE_PATH, 'r', encoding='utf-8').read())
    return full_dict, co_occu, single_occr

def collect_samples(dataset, model, full_dict, co_occu, single_occr,
                    single_occ_threshold=6000):
    """采样方法，阈值可变"""
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
        
        # ---- 用可变的 SINGLE_OCC_THRESHOLD 过滤 ----
        if single_occ_threshold is not None:
            if dataset in ['movies', 'songs']:
                if (single_occr[question_entity.lower()] > single_occ_threshold or
                        single_occr[gene_entity.lower()] > single_occ_threshold or
                        single_occr[ref.lower()] > single_occ_threshold):
                    continue
            else:
                if single_occr[question_entity.lower()] > single_occ_threshold:
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


# ════════════════════════════════════════════════════════════════
# 敏感性分析 1: Q_SINGLE_LO → quality_v2→acc 偏相关（表4）
# ════════════════════════════════════════════════════════════════
def analyze_table4_sensitivity(all_samples, q_single_lo, q_single_hi=6000, label=""):
    """计算表4的 quality_v2→acc (控制gt_coo) 偏相关"""
    results = []
    for dataset in DATASETS:
        for model in MODELS:
            if dataset == 'basketball' and model.startswith('Qwen2.5'):
                continue
            samples = all_samples.get((dataset, model), [])
            if len(samples) < 100:
                continue
            
            q_samples = [s for s in samples if q_single_lo <= s['q_single'] < q_single_hi]
            if len(q_samples) < 50:
                continue
            
            quality = np.array([s['quality_v2'] for s in q_samples], dtype=float)
            coo = np.array([s['coo'] for s in q_samples], dtype=float)
            acc = np.array([s['acc'] for s in q_samples], dtype=float)
            
            r_q, p_q = partial_spearman(quality, acc, coo) if np.std(quality) > 0 else (float('nan'), 1)
            r_coo, p_coo = spearmanr(coo, acc) if np.std(coo) > 0 else (float('nan'), 1)
            
            results.append({
                'dataset': dataset,
                'model': model,
                'n': len(q_samples),
                'r_quality2acc(coo_ctrl)': r_q,
                'p_quality2acc(coo_ctrl)': p_q,
                'r_coo_acc': r_coo,
                'p_coo_acc': p_coo,
            })
    return results


# ════════════════════════════════════════════════════════════════
# 敏感性分析 2: SINGLE_OCC_THRESHOLD → 样本量和基础相关
# ════════════════════════════════════════════════════════════════
def analyze_single_occ_sensitivity(all_samples, label=""):
    """在不同阈值下，统计各 dataset×model 的样本量"""
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            if dataset == 'basketball' and model.startswith('Qwen2.5'):
                continue
            samples = all_samples.get((dataset, model), [])
            n = len(samples)
            if n < 50:
                continue
            
            conf = np.array([s['conf'] for s in samples])
            acc = np.array([s['acc'] for s in samples])
            gene_pop = np.array([s['gene_pop'] for s in samples], dtype=float)
            gene_coo = np.array([s['gene_coo'] for s in samples], dtype=float)
            
            mask_wrong = (acc == 0)
            n_wrong = mask_wrong.sum()
            
            r_gp, _ = (spearmanr(gene_pop[mask_wrong], conf[mask_wrong]) 
                       if n_wrong >= 10 and np.std(gene_pop[mask_wrong]) > 0 
                       else (float('nan'), 1))
            r_gc, _ = (spearmanr(gene_coo[mask_wrong], conf[mask_wrong])
                       if n_wrong >= 10 and np.std(gene_coo[mask_wrong]) > 0
                       else (float('nan'), 1))
            
            rows.append({
                'dataset': dataset,
                'model': model,
                'n': n,
                'n_wrong': n_wrong,
                'acc': acc.mean(),
                'r_genePop_conf_wrong': r_gp,
                'r_geneCoo_conf_wrong': r_gc,
            })
    return rows


# ════════════════════════════════════════════════════════════════
# 运行敏感性分析
# ════════════════════════════════════════════════════════════════
def main():
    print("=" * 100)
    print("Sensitivity Analysis: single_occ Filter Thresholds")
    print("=" * 100)
    
    # 加载数据
    print("\n[1/4] 加载数据...")
    full_dict, co_occu, single_occr = load_data()
    print(f"  共现字典: {len(co_occu)} question entities")
    print(f"  单文档计数: {len(single_occr)} entities")
    
    # ──── 实验 A: Q_SINGLE_LO 敏感性 ────
    print("\n" + "=" * 100)
    print("实验 A: Q_SINGLE_LO 阈值对 quality_v2→acc 偏相关的影响")
    print("  对应论文 §4.2, Table 4")
    print("=" * 100)
    
    lo_thresholds = [0, 20, 50, 100, 200]
    
    # 先用默认 SINGLE_OCC_THRESHOLD=6000 收集样本
    default_samples = {}
    for dataset in DATASETS:
        for model in MODELS:
            key = (dataset, model)
            s = collect_samples(dataset, model, full_dict, co_occu, single_occr,
                               single_occ_threshold=6000)
            if len(s) >= 50:
                default_samples[key] = s
    
    for lo in lo_thresholds:
        print(f"\n--- Q_SINGLE_LO = {lo} ---")
        results = analyze_table4_sensitivity(default_samples, q_single_lo=lo)
        
        # 汇总：显著为正的比例
        n_sig_pos = sum(1 for r in results if r['r_quality2acc(coo_ctrl)'] > 0 and r['p_quality2acc(coo_ctrl)'] < 0.05)
        n_sig_neg = sum(1 for r in results if r['r_quality2acc(coo_ctrl)'] < 0 and r['p_quality2acc(coo_ctrl)'] < 0.05)
        n_total = len(results)
        
        print(f"  {'dataset':12s} {'model':20s} {'n':>6s} | {'quality_v2→acc(ctrl_coo)':>25s} {'p':>10s} | {'coo→acc':>10s} {'p':>10s}")
        print(f"  {'-'*12} {'-'*20} {'-'*6} {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
        
        for r in results:
            r_str = f"{r['r_quality2acc(coo_ctrl)']:+.3f}"
            if r['p_quality2acc(coo_ctrl)'] < 0.001:
                r_str += "***"
            elif r['p_quality2acc(coo_ctrl)'] < 0.01:
                r_str += "**"
            elif r['p_quality2acc(coo_ctrl)'] < 0.05:
                r_str += "*"
            
            coo_str = f"{r['r_coo_acc']:+.3f}"
            if r['p_coo_acc'] < 0.001:
                coo_str += "***"
            elif r['p_coo_acc'] < 0.01:
                coo_str += "**"
            elif r['p_coo_acc'] < 0.05:
                coo_str += "*"
            
            print(f"  {r['dataset']:12s} {r['model']:20s} {r['n']:6d} | {r_str:>25s} {r['p_quality2acc(coo_ctrl)']:>10.2e} | {coo_str:>10s} {r['p_coo_acc']:>10.2e}")
        
        print(f"\n  → quality_v2→acc 为正且显著的: {n_sig_pos}/{n_total}")
        print(f"  → quality_v2→acc 为负且显著的: {n_sig_neg}/{n_total}")
    
    # ──── 实验 B: SINGLE_OCC_THRESHOLD 敏感性 ────
    print("\n" + "=" * 100)
    print("实验 B: SINGLE_OCC_THRESHOLD 对样本量和核心相关的影响")
    print("  对应 Paper §4-§5 的核心 Spearman 分析")
    print("=" * 100)
    
    occ_thresholds = [3000, 6000, 10000, None]  # None = 不过滤
    
    for threshold in occ_thresholds:
        label = f"SINGLE_OCC_THRESHOLD = {threshold}" if threshold is not None else "SINGLE_OCC_THRESHOLD = None (不过滤)"
        print(f"\n--- {label} ---")
        
        samples = {}
        for dataset in DATASETS:
            for model in MODELS:
                key = (dataset, model)
                s = collect_samples(dataset, model, full_dict, co_occu, single_occr,
                                   single_occ_threshold=threshold)
                if len(s) >= 50:
                    samples[key] = s
        
        results = analyze_single_occ_sensitivity(samples)
        
        print(f"  {'dataset':12s} {'model':20s} {'n':>6s} {'n_wrong':>8s} {'acc':>6s} | {'r(gene_pop,conf)_wrong':>22s} {'r(gene_coo,conf)_wrong':>22s}")
        print(f"  {'-'*12} {'-'*20} {'-'*6} {'-'*8} {'-'*6} {'-'*22} {'-'*22}")
        
        for r in results:
            gp_str = f"{r['r_genePop_conf_wrong']:+.3f}"
            gc_str = f"{r['r_geneCoo_conf_wrong']:+.3f}"
            print(f"  {r['dataset']:12s} {r['model']:20s} {r['n']:6d} {r['n_wrong']:8d} {r['acc']:>.3f} | {gp_str:>22s} {gc_str:>22s}")
    
    # ──── 实验 C: Basketball 样本保留率 ────
    print("\n" + "=" * 100)
    print("实验 C: Basketball q_single 分布细节（不同LO阈值下的保留率）")
    print("=" * 100)
    
    print(f"\n  {'Q_LO':>6s} {'movies保留':>12s} {'songs保留':>12s} {'basketball保留':>16s}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*16}")
    
    for lo in lo_thresholds:
        cnts = {}
        for dataset in ['movies', 'songs', 'basketball']:
            total = 0
            kept = 0
            for model in MODELS:
                s = default_samples.get((dataset, model), [])
                total += len(s)
                kept += sum(1 for x in s if x['q_single'] >= lo)
            cnts[dataset] = (total, kept)
        
        m = cnts['movies']
        sg = cnts['songs']
        bb = cnts['basketball']
        print(f"  {lo:>6d} {m[1]/max(m[0],1)*100:>11.1f}% {sg[1]/max(sg[0],1)*100:>11.1f}% {bb[1]/max(bb[0],1)*100:>15.1f}%")

if __name__ == '__main__':
    main()
