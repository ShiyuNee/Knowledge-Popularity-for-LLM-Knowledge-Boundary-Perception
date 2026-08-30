"""
统计 gt_pop 和 gene_pop 中 Wikidata 查不到（popularity='No' 或不在字典中）的比例
"""
import json, re, os

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES_DIR = os.path.join(BASE, 'res')
POP_PATH = os.path.join(RES_DIR, 'gt_gene_entity_popularity_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.jsonl')

DATASETS = ['movies', 'songs', 'basketball']
MODELS = ['qwen2', 'llama8b', 'chatgpt', 'Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']
SINGLE_OCC_THRESHOLD = 6000

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

def load_popularity():
    pop_data = read_jsonl(POP_PATH)
    full_dict = {}
    for d in pop_data:
        full_dict.update(d)
    return full_dict

# 加载数据
full_dict = load_popularity()

# 先统计 full_dict 中的总体情况
total_entities = len(full_dict)
no_count = sum(1 for v in full_dict.values() 
               if isinstance(v, dict) and v.get('popularity') == 'No')
zero_count = sum(1 for v in full_dict.values() 
                 if isinstance(v, dict) and v.get('popularity') == 0)

print("=" * 70)
print(f"full_dict 总体统计")
print(f"  总实体数: {total_entities}")
print(f"  popularity='No' (Wikidata 查不到): {no_count} ({no_count/max(total_entities,1)*100:.1f}%)")
print(f"  popularity=0 (查到了但 sitelinks=0): {zero_count} ({zero_count/max(total_entities,1)*100:.1f}%)")
print(f"  有有效 popularity (>0): {total_entities - no_count - zero_count} ({(total_entities-no_count-zero_count)/max(total_entities,1)*100:.1f}%)")

# 加载 single_occr 做过滤
SINGLE_PATH = os.path.join(RES_DIR, 'single_occurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')
COO_PATH = os.path.join(RES_DIR, 'cooccurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')
single_occr = json.loads(open(SINGLE_PATH, 'r', encoding='utf-8').read())
co_occu = json.loads(open(COO_PATH, 'r', encoding='utf-8').read())

print("\n" + "=" * 70)
print("逐 dataset × model 统计 gt_pop 和 gene_pop 置0情况")
print("=" * 70)

for dataset in DATASETS:
    for model in MODELS:
        fpath = os.path.join(RES_DIR, dataset, f'{dataset}_{model}_temperature1.jsonl')
        if not os.path.exists(fpath):
            continue
        
        data = read_jsonl(fpath)
        n_total = 0
        n_gt_no = 0       # gt 在 wikidata 查不到
        n_gt_zero = 0     # gt 查到了但 sitelinks=0
        n_gt_not_in_dict = 0  # gt 不在 full_dict 中
        n_gene_no = 0     # gene 在 wikidata 查不到
        n_gene_zero = 0   # gene 查到了但 sitelinks=0
        n_gene_not_in_dict = 0  # gene 不在 full_dict 中
        n_gt_pop0_in_sample = 0  # 最终样本中 gt_pop=0 的
        n_gene_pop0_in_sample = 0  # 最终样本中 gene_pop=0 的
        n_kept = 0
        
        for item in data:
            if not item.get('Res') or item['Res'] is None:
                continue
            if item.get('popularity') == 'No':
                continue
            
            question_entity = item['question'].replace(PATTERN[dataset], '').lower()
            ref = remove_punctuation_edges(item['reference'][0], dataset)
            gene_entity = remove_punctuation_edges(item['Res'], dataset)
            
            # 检查是否在 co_occu 和 single_occr 中
            if question_entity not in co_occu:
                continue
            if question_entity.lower() not in single_occr:
                continue
            if gene_entity.lower() not in single_occr:
                continue
            if ref.lower() not in single_occr:
                continue
            
            # SINGLE_OCC 过滤
            if dataset in ['movies', 'songs']:
                if (single_occr[question_entity.lower()] > SINGLE_OCC_THRESHOLD or
                        single_occr[gene_entity.lower()] > SINGLE_OCC_THRESHOLD or
                        single_occr[ref.lower()] > SINGLE_OCC_THRESHOLD):
                    continue
            else:
                if single_occr[question_entity.lower()] > SINGLE_OCC_THRESHOLD:
                    continue
            
            n_total += 1
            
            # ---- gt_pop 来源分析 ----
            ref_pop_info = full_dict.get(ref, {})
            if ref == '' or ref not in full_dict:
                n_gt_not_in_dict += 1
                gt_pop_final = 0
            elif isinstance(ref_pop_info, dict):
                rp = ref_pop_info.get('popularity', 0)
                if rp == 'No' or rp is None:
                    n_gt_no += 1
                    gt_pop_final = 0
                elif rp == 0:
                    n_gt_zero += 1
                    gt_pop_final = 0
                else:
                    gt_pop_final = int(rp)
            else:
                gt_pop_final = int(ref_pop_info) if ref_pop_info else 0
            
            # ---- gene_pop 来源分析 ----
            gene_pop_info = full_dict.get(gene_entity, {})
            if gene_entity == '' or gene_entity not in full_dict:
                n_gene_not_in_dict += 1
                gene_pop_final = 0
            elif isinstance(gene_pop_info, dict):
                gp = gene_pop_info.get('popularity', 0)
                if gp == 'No' or gp is None:
                    n_gene_no += 1
                    gene_pop_final = 0
                elif gp == 0:
                    n_gene_zero += 1
                    gene_pop_final = 0
                else:
                    gene_pop_final = int(gp)
            else:
                gene_pop_final = int(gene_pop_info) if gene_pop_info else 0
            
            if gt_pop_final == 0:
                n_gt_pop0_in_sample += 1
            if gene_pop_final == 0:
                n_gene_pop0_in_sample += 1
            n_kept += 1
        
        if n_kept == 0:
            continue
        
        print(f"\n{dataset:12s} × {model:20s}  (n={n_kept})")
        print(f"  gt_pop=0:  {n_gt_pop0_in_sample:>5d} / {n_kept} ({n_gt_pop0_in_sample/n_kept*100:.1f}%)")
        print(f"    ├─ 不在 full_dict 中:          {n_gt_not_in_dict:>5d}")
        print(f"    ├─ Wikidata 查不到 (pop='No'):  {n_gt_no:>5d}")
        print(f"    └─ 查到了但 sitelinks=0:         {n_gt_zero:>5d}")
        print(f"  gene_pop=0: {n_gene_pop0_in_sample:>5d} / {n_kept} ({n_gene_pop0_in_sample/n_kept*100:.1f}%)")
        print(f"    ├─ 不在 full_dict 中:          {n_gene_not_in_dict:>5d}")
        print(f"    ├─ Wikidata 查不到 (pop='No'):  {n_gene_no:>5d}")
        print(f"    └─ 查到了但 sitelinks=0:         {n_gene_zero:>5d}")
