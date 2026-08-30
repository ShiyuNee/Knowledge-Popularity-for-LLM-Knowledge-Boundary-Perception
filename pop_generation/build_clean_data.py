"""
构建 clean_data_for_pop_generation/ 数据

从 res/ 文件 + gene_pop + cooccurrence 数据中提取:
  - question_entity: 从 question 字段提取（如 "The Intouchables"）
  - gene_entity: 从 Res 字段获取（如 "Eric Toledano"）
  - question_pop: res/ 文件中的 popularity 字段
  - gene_pop: 从 gt_gene_entity_popularity 文件查找
  - coo_pop: 从 cooccurrence 文件查找

输出格式 (每行一个 JSON):
  {"question_entity": "The Intouchables", "gene_entity": "Eric Toledano",
   "question_pop": 60, "gene_pop": 15, "coo_pop": 9}

Usage:
    python -u build_clean_data.py

数据源:
    res/{dataset}/{dataset}_{model}_temperature1.jsonl
    res/gt_gene_entity_popularity_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.jsonl
    res/cooccurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json

输出:
    pop_generation/data/clean_data_for_pop_generation/{dataset}_{model}_temperature1.jsonl
"""

import os
import json
import re

# ─── 路径配置 ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(PROJECT_ROOT, 'res')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'clean_data_for_pop_generation')

# Gene popularity 数据（所有模型的生成实体都从同一个 pop 字典查）
GENE_POP_PATH = os.path.join(RES_DIR, 'gt_gene_entity_popularity_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.jsonl')

# Co-occurrence 数据
COO_POP_PATH = os.path.join(RES_DIR, 'cooccurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')

# 问题前缀（用于提取 question_entity）
QUESTION_PATTERNS = {
    'movies':     'Who is the director of the movie ',
    'songs':      'Who is the performer of the song ',
    'basketball': 'Where is the birthplace of the basketball player ',
}

# 需要处理的模型（与 res/ 中的文件名对应）
MODELS = ['llama8b', 'qwen2', 'chatgpt', 'Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']
DATASETS = ['movies', 'songs', 'basketball']


def remove_punctuation_edges(s, name='movies'):
    """
    清理生成的实体名称（与原始 read.py 逻辑保持一致）。
    - 去除换行
    - 去除括号内容
    - basketball 数据集中取逗号前部分
    - 短名称（<=20字符）取逗号前部分
    - 去除首尾非字母数字字符
    """
    s = s.replace('\n', '')
    s = s.split('(')[0].strip()
    if name in ['basketball']:
        s = s.split(',')[0].strip()
    else:
        if len(s) <= 20:
            s = s.split(',')[0].strip()
    s = re.sub(r'^[^\w]+|[^\w]+$', '', s)
    s = s.strip()
    return s


def load_gene_pop(path):
    """加载 gene entity popularity 数据（JSONL 格式，每行一个 dict，合并为一个大 dict）。"""
    gene_pop = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                gene_pop.update(d)
    return gene_pop


def load_coo_pop(path):
    """加载 co-occurrence 数据（JSON 格式，两层嵌套 dict）。"""
    with open(path, 'r', encoding='utf-8') as f:
        coo = json.load(f)
    return coo


def build_clean_data(dataset, model, res_data, gene_pop_dict, coo_dict):
    """
    为一个 dataset+model 构建 clean data。

    Args:
        dataset: 数据集名称 (movies/songs/basketball)
        model: 模型简称
        res_data: res/ 文件中的数据列表
        gene_pop_dict: gene entity popularity 字典
        coo_dict: co-occurrence 字典

    Returns:
        list[dict]: clean data 列表
    """
    pattern = QUESTION_PATTERNS[dataset]
    clean_data = []
    missing_gene = 0
    missing_coo = 0

    for item in res_data:
        # 提取 question_entity
        q = item['question']
        if q.startswith(pattern):
            question_entity = q[len(pattern):].rstrip('?').strip()
        else:
            question_entity = q.rstrip('?').strip()

        # gene_entity: 保留原始值（用于 coo 查询），同时生成清理版本（用于 gene_pop 查询）
        gene_entity_raw = item['Res'].strip()
        if gene_entity_raw.startswith(': '):
            gene_entity_raw = gene_entity_raw[2:]
        gene_entity_clean = remove_punctuation_edges(gene_entity_raw, dataset)

        # question_pop: 直接从 res/ 文件中的 popularity 字段获取
        question_pop = item.get('popularity', None)

        # gene_pop: 从 gene_pop 字典中查找（用清理后的名称）
        gene_info = gene_pop_dict.get(gene_entity_clean, {})
        if isinstance(gene_info, dict) and gene_info.get('popularity', 'No') != 'No':
            gene_pop = gene_info['popularity']
        else:
            # 尝试精确匹配原始名称
            gene_info = gene_pop_dict.get(gene_entity_raw, {})
            if isinstance(gene_info, dict) and gene_info.get('popularity', 'No') != 'No':
                gene_pop = gene_info['popularity']
            else:
                gene_pop = 'No'
                missing_gene += 1

        # coo_pop: 从 co-occurrence 字典中查找（用小写）
        coo_inner = coo_dict.get(question_entity.lower(), {})
        coo_pop = coo_inner.get(gene_entity_raw.lower(), None)
        if coo_pop is None:
            # 尝试用清理后的名称
            coo_pop = coo_inner.get(gene_entity_clean.lower(), 0)
            if coo_pop is None:
                coo_pop = 0
                missing_coo += 1

        clean_data.append({
            'question_entity': question_entity,
            'gene_entity': gene_entity_raw,
            'question_pop': question_pop,
            'gene_pop': gene_pop,
            'coo_pop': coo_pop,
        })

    total = len(clean_data)
    if missing_gene > 0:
        print(f'  WARNING: {missing_gene}/{total} items missing gene_pop')
    if missing_coo > 0:
        print(f'  WARNING: {missing_coo}/{total} items missing coo_pop (set to 0)')

    return clean_data


def main():
    # 加载全局数据
    print('Loading gene popularity data...')
    gene_pop_dict = load_gene_pop(GENE_POP_PATH)
    print(f'  Loaded {len(gene_pop_dict)} gene entities')

    print('Loading co-occurrence data...')
    coo_dict = load_coo_pop(COO_POP_PATH)
    print(f'  Loaded {len(coo_dict)} question entities')

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for dataset in DATASETS:
        for model in MODELS:
            # 构建 res/ 文件路径
            res_path = os.path.join(RES_DIR, dataset, f'{dataset}_{model}_temperature1.jsonl')
            if not os.path.exists(res_path):
                print(f'SKIP: {res_path} not found')
                continue

            # 加载 res/ 数据
            print(f'\nProcessing: {dataset}/{model}')
            res_data = []
            with open(res_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        res_data.append(json.loads(line))
            print(f'  Loaded {len(res_data)} samples from {res_path}')

            # 构建 clean data
            clean_data = build_clean_data(dataset, model, res_data, gene_pop_dict, coo_dict)

            # 输出
            model_name = model.lower().replace('.', '')
            out_path = os.path.join(OUTPUT_DIR, f'{dataset}_{model_name}_temperature1.jsonl')
            with open(out_path, 'w', encoding='utf-8') as f:
                for item in clean_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f'  Written {len(clean_data)} lines to {out_path}')

            # 统计
            valid_gene = sum(1 for item in clean_data if item['gene_pop'] != 'No')
            valid_coo = sum(1 for item in clean_data if isinstance(item['coo_pop'], (int, float)) and item['coo_pop'] > 0)
            print(f'  Valid gene_pop: {valid_gene}/{len(clean_data)}')
            print(f'  Non-zero coo_pop: {valid_coo}/{len(clean_data)}')


if __name__ == '__main__':
    main()