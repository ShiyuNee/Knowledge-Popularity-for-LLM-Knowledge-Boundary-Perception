"""
数据探索：q_single 的分布，为敏感性分析做准备
"""
import json, os
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 1. 加载 single_occurrence 数据
SINGLE_PATH = os.path.join(BASE, 'res', 'single_occurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')
single_occr = json.loads(open(SINGLE_PATH, 'r', encoding='utf-8').read())
values = np.array(list(single_occr.values()))

print("=" * 60)
print("single_occr 全局分布")
print("=" * 60)
print(f"  总实体数:         {len(values):>8}")
print(f"  mean:             {values.mean():>8.1f}")
print(f"  median:           {np.median(values):>8.1f}")
print(f"  min:              {values.min():>8}")
print(f"  max:              {values.max():>8}")
print(f"  < 50:             {(values < 50).sum():>8} ({(values < 50).sum()/len(values)*100:.1f}%)")
print(f"  50 ~ 100:          {((values >= 50) & (values < 100)).sum():>8} ({((values >= 50) & (values < 100)).sum()/len(values)*100:.1f}%)")
print(f"  100 ~ 1000:         {((values >= 100) & (values < 1000)).sum():>8} ({((values >= 100) & (values < 1000)).sum()/len(values)*100:.1f}%)")
print(f"  1000 ~ 6000:        {((values >= 1000) & (values < 6000)).sum():>8} ({((values >= 1000) & (values < 6000)).sum()/len(values)*100:.1f}%)")
print(f"  >= 6000:            {(values >= 6000).sum():>8} ({(values >= 6000).sum()/len(values)*100:.1f}%)")
print(f"  >= 10000:           {(values >= 10000).sum():>8} ({(values >= 10000).sum()/len(values)*100:.1f}%)")
print(f"  >= 50000:           {(values >= 50000).sum():>8} ({(values >= 50000).sum()/len(values)*100:.1f}%)")

# 2. 加载样本数据，按 dataset × model 统计 q_single 分布
DATASETS = ['movies', 'songs', 'basketball']
MODELS = ['qwen2', 'llama8b', 'chatgpt', 'Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']
PATTERN = {
    'movies': 'Who is the director of the movie ',
    'songs': 'Who is the performer of the song ',
    'basketball': 'Where is the birthplace of the basketball player '
}
RES_DIR = os.path.join(BASE, 'res')

import re
import sys
sys.path.insert(0, os.path.join(BASE, 'code', 'analysis_correlation'))

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

print("\n\n" + "=" * 80)
print("按 dataset × model 统计 q_single 分布（在现有过滤前）")
print("=" * 80)

for dataset in DATASETS:
    for model in MODELS:
        fpath = os.path.join(RES_DIR, dataset, f'{dataset}_{model}_temperature1.jsonl')
        if not os.path.exists(fpath):
            continue
        
        data = [json.loads(l) for l in open(fpath, 'r', encoding='utf-8') if l.strip()]
        q_singles = []
        for item in data:
            if not item.get('Res') or item['Res'] is None:
                continue
            if item.get('popularity') == 'No':
                continue
            q_entity = item['question'].replace(PATTERN[dataset], '').lower()
            if q_entity.lower() in single_occr:
                q_singles.append(single_occr[q_entity.lower()])
        
        qs = np.array(q_singles)
        n_total = len(qs)
        n_lt50 = (qs < 50).sum()
        n_50_100 = ((qs >= 50) & (qs < 100)).sum()
        n_100_6000 = ((qs >= 100) & (qs < 6000)).sum()
        n_ge6000 = (qs >= 6000).sum()
        n_lt6000 = (qs < 6000).sum()
        
        # Sample retention rate under current filters
        # movies/songs: filter gene+ref single_occ too, but let's just show q_single impact
        print(f"\n{dataset:12s} {model:20s} n_total={n_total:>6d}")
        print(f"  {'q_single < 50':22s} {n_lt50:>5d} ({n_lt50/max(n_total,1)*100:5.1f}%)")
        print(f"  {'50 ≤ q_single < 100':22s} {n_50_100:>5d} ({n_50_100/max(n_total,1)*100:5.1f}%)")
        print(f"  {'100 ≤ q_single < 6000':22s} {n_100_6000:>5d} ({n_100_6000/max(n_total,1)*100:5.1f}%)")
        print(f"  {'q_single ≥ 6000':22s} {n_ge6000:>5d} ({n_ge6000/max(n_total,1)*100:5.1f}%)")
        print(f"  {'→ 保留 (50~6000)':22s} {n_50_100+n_100_6000:>5d} ({(n_50_100+n_100_6000)/max(n_total,1)*100:5.1f}%)")
