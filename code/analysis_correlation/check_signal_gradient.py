"""Check signal gradient across datasets: which variable drives confidence on wrong samples?"""
import json, math, re, os, sys
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_per_model import collect_samples, load_popularity, DATASETS, MODELS

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
RES_DIR = os.path.join(BASE, 'res')
COO_PATH = os.path.join(RES_DIR, 'cooccurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')
SINGLE_PATH = os.path.join(RES_DIR, 'single_occurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')

full_dict = load_popularity()
co_occu = json.loads(open(COO_PATH).read())
single_occr = json.loads(open(SINGLE_PATH).read())

print("=" * 80)
print("Signal gradient analysis: which variable drives confidence on wrong samples?")
print("=" * 80)

for dataset in ['movies', 'songs', 'basketball']:
    all_wrong = {'gene_pop': [], 'gene_coo': [], 'qpop': [], 'conf': []}
    
    for model in MODELS:
        if dataset == 'basketball' and model.startswith('Qwen2.5'):
            continue
        samples = collect_samples(dataset, model, full_dict, co_occu, single_occr, filter_pop_no=True)
        wrong = [s for s in samples if s['acc'] == 0]
        for s in wrong:
            all_wrong['gene_pop'].append(s['gene_pop'])
            all_wrong['gene_coo'].append(s['gene_coo'])
            all_wrong['qpop'].append(s['question_pop'])
            all_wrong['conf'].append(s['conf'])
    
    gp = np.array(all_wrong['gene_pop'], dtype=float)
    gc = np.array(all_wrong['gene_coo'], dtype=float)
    qp = np.array(all_wrong['qpop'], dtype=float)
    cf = np.array(all_wrong['conf'], dtype=float)
    
    print(f"\n=== {dataset.upper()} (wrong samples, pooled) n={len(gp)} ===")
    
    for name, arr in [('gene_pop', gp), ('gene_coo', gc), ('qpop', qp)]:
        mean = arr.mean()
        std = arr.std()
        cv = std / mean if mean > 0 else float('inf')
        nz = (arr > 0).mean() * 100
        p10 = np.percentile(arr, 10)
        p90 = np.percentile(arr, 90)
        print(f"  {name:12s}: mean={mean:8.1f} median={np.median(arr):6.0f} CV={cv:.3f} nonzero={nz:5.1f}% P10={p10:6.0f} P90={p90:6.0f}")
    
    print(f"  --- Spearman with conf (wrong samples) ---")
    for name, arr in [('gene_pop', gp), ('gene_coo', gc), ('qpop', qp)]:
        if np.std(arr) > 0:
            r, p = spearmanr(arr, cf)
            print(f"  Spearman({name:12s}, conf) = {r:+.3f}  p={p:.2e}")

# Per-model breakdown for Table 5 context
print("\n" + "=" * 80)
print("Per-model: qpop->conf on incorrect samples (Basketball)")
print("=" * 80)
for model in MODELS:
    if model.startswith('Qwen2.5') and True:
        # only llama8b, qwen2, chatgpt for basketball
        pass
    for dataset in ['movies', 'songs', 'basketball']:
        if dataset == 'basketball' and model.startswith('Qwen2.5'):
            continue
        samples = collect_samples(dataset, model, full_dict, co_occu, single_occr, filter_pop_no=True)
        wrong = [s for s in samples if s['acc'] == 0]
        if len(wrong) < 20:
            continue
        qp = np.array([s['question_pop'] for s in wrong], dtype=float)
        cf = np.array([s['conf'] for s in wrong], dtype=float)
        gp = np.array([s['gene_pop'] for s in wrong], dtype=float)
        gc = np.array([s['gene_coo'] for s in wrong], dtype=float)
        
        r_qp, _ = spearmanr(qp, cf) if np.std(qp) > 0 else (0, 1)
        r_gp, _ = spearmanr(gp, cf) if np.std(gp) > 0 else (0, 1)
        r_gc, _ = spearmanr(gc, cf) if np.std(gc) > 0 else (0, 1)
        print(f"  {dataset:12s} x {model:15s} (n_wrong={len(wrong):5d}): qpop={r_qp:+.3f}  gene_pop={r_gp:+.3f}  gene_coo={r_gc:+.3f}")