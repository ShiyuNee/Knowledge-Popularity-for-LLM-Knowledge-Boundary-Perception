"""
_plot_common.py — 论文绘图公共模块

提供:
  - 统一的 matplotlib 学术样式设置
  - 公共的数据加载函数 (collect_samples 等)
  - 统一的路径配置、常量定义
  - 统一的配色方案

用法:
  from _plot_common import setup_style, load_data, collect_samples, ...
"""

import json, os, math, re
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# 路径配置
# ═══════════════════════════════════════════════════════════════════════════════
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
RES_DIR = os.path.join(BASE, 'res')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUT_DIR, exist_ok=True)

POP_PATH = os.path.join(RES_DIR, 'gt_gene_entity_popularity_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.jsonl')
COO_PATH = os.path.join(RES_DIR, 'cooccurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')
SINGLE_PATH = os.path.join(RES_DIR, 'single_occurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')

# ═══════════════════════════════════════════════════════════════════════════════
# 数据集与模型常量
# ═══════════════════════════════════════════════════════════════════════════════
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

MODEL_LABELS = {
    'llama8b': 'Llama-3-8B',
    'qwen2': 'Qwen2-7B',
    'chatgpt': 'ChatGPT',
    'Qwen2.5-7B': 'Qwen2.5-7B',
    'Qwen2.5-14B': 'Qwen2.5-14B',
    'Qwen2.5-32B': 'Qwen2.5-32B',
}

DATASET_LABELS = {
    'movies': 'Movies',
    'songs': 'Songs',
    'basketball': 'Basketball',
}

# ═══════════════════════════════════════════════════════════════════════════════
# 配色方案 — 色盲友好、论文级
# ═══════════════════════════════════════════════════════════════════════════════
COLORS = {
    'movies': '#1f77b4',
    'songs': '#ff7f0e',
    'basketball': '#2ca02c',
    'focused': '#2171B5',     # 蓝
    'diffuse': '#CB181D',     # 红
    'correct': '#2166ac',     # 深蓝
    'incorrect': '#b2182b',   # 深红
    'gene_pop': '#762a83',
    'gene_coo': '#1b7837',
}


# ═══════════════════════════════════════════════════════════════════════════════
# matplotlib 学术样式
# ═══════════════════════════════════════════════════════════════════════════════
def setup_style():
    """设置论文级 matplotlib 全局样式 (ACL/EMNLP 规范)"""
    import matplotlib
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 3.0,
        'ytick.major.size': 3.0,
        'xtick.minor.size': 0,
        'ytick.minor.size': 0,
        'font.size': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


# ═══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════════════
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


def load_cooccurrence():
    return json.loads(open(COO_PATH).read())


def load_single_occurrence():
    return json.loads(open(SINGLE_PATH).read())


def collect_samples(dataset, model, full_dict, co_occu, single_occr,
                    filter_pop_no=True, filter_single_occ=True):
    """
    收集单个 dataset × model 的样本。

    与 verify_per_model.py 的 collect_samples 完全一致，确保数据可复现。
    """
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
        if filter_single_occ:
            if dataset in ['movies', 'songs']:
                if (single_occr[question_entity.lower()] > SINGLE_OCC_THRESHOLD or
                        single_occr[gene_entity.lower()] > SINGLE_OCC_THRESHOLD or
                        single_occr[ref.lower()] > SINGLE_OCC_THRESHOLD):
                    continue
            else:
                if single_occr[question_entity.lower()] > SINGLE_OCC_THRESHOLD:
                    continue
        # popularity
        ref_pop_info = full_dict.get(ref, {})
        ref_pop = ref_pop_info.get('popularity', 0) if isinstance(ref_pop_info, dict) else ref_pop_info
        gt_pop_missing = (ref_pop == 'No' or ref_pop is None or ref not in full_dict)
        gene_pop_info = full_dict.get(gene_entity, {})
        gene_pop = gene_pop_info.get('popularity', 0) if isinstance(gene_pop_info, dict) else gene_pop_info
        gene_pop_missing = (gene_pop == 'No' or gene_pop is None or gene_entity not in full_dict)
        if filter_pop_no and (gt_pop_missing or gene_pop_missing):
            continue
        # confidence
        if 'gpt' in model.lower():
            probs = [math.exp(t) for t in item['Log_p']['token_logprobs']]
        else:
            probs = item['Log_p']['token_probs']
        conf = sum(probs) / len(probs)
        # co-occurrence & single occurrence
        gt_coo = co_occu[question_entity].get(ref.lower(), 0)
        gene_coo = co_occu[question_entity].get(gene_entity.lower(), 0)
        q_single = single_occr[question_entity.lower()]
        gt_single = single_occr.get(ref.lower(), 0)
        quality_v2 = gt_coo / q_single if q_single > 0 else 0
        samples.append({
            'question_pop': item['popularity'],
            'gt_pop': int(ref_pop) if ref_pop and not gt_pop_missing else 0,
            'gene_pop': int(gene_pop) if gene_pop and not gene_pop_missing else 0,
            'coo': gt_coo,
            'gene_coo': gene_coo,
            'q_single': q_single,
            'gt_single': gt_single,
            'quality_v2': quality_v2,
            'conf': conf,
            'acc': item['has_answer'],
        })
    return samples


def load_all_data(datasets=None, models=None, filter_pop_no=True):
    """
    一键加载所有 dataset × model 的样本。

    Returns:
        dict: {(dataset, model): [sample, ...]}
    """
    if datasets is None:
        datasets = DATASETS
    if models is None:
        models = MODELS
    print("加载数据...")
    full_dict = load_popularity()
    co_occu = load_cooccurrence()
    single_occr = load_single_occurrence()

    all_data = {}
    for dataset in datasets:
        for model in models:
            samples = collect_samples(dataset, model, full_dict, co_occu, single_occr,
                                      filter_pop_no=filter_pop_no)
            all_data[(dataset, model)] = samples
            n = len(samples)
            n_wrong = sum(1 for s in samples if s['acc'] == 0)
            n_right = n - n_wrong
            print(f"  {dataset:12s} × {model:15s}: n={n:5d}  做对={n_right:5d}  做错={n_wrong:5d}")
    return all_data


def save_figure(fig, name, formats=None):
    """保存图片到 output 目录，默认同时保存 PDF 和 PNG"""
    if formats is None:
        formats = ['pdf', 'png']
    for fmt in formats:
        path = os.path.join(OUT_DIR, f'{name}.{fmt}')
        fig.savefig(path, format=fmt, bbox_inches='tight', dpi=300)
        print(f"  保存: {path}")