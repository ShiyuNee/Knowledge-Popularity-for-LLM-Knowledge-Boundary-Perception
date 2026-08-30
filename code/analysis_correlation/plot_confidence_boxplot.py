"""
绘制 ChatGPT 在每个数据集上正确/错误样本的 confidence 箱形图
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES_DIR = os.path.join(BASE, 'res')

POP_PATH = os.path.join(RES_DIR, 'gt_gene_entity_popularity_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.jsonl')
COO_PATH = os.path.join(RES_DIR, 'cooccurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')
SINGLE_PATH = os.path.join(RES_DIR, 'single_occurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')

DATASETS = ['movies', 'songs', 'basketball']
PATTERN = {
    'movies': 'Who is the director of the movie ',
    'songs': 'Who is the performer of the song ',
    'basketball': 'Where is the birthplace of the basketball player '
}

SINGLE_OCC_THRESHOLD = 6000


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]


def remove_punctuation_edges(s, name='movies'):
    import re
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


def collect_confidence_by_correctness(dataset, model, full_dict, co_occu, single_occr):
    """收集正确和错误样本的 confidence"""
    res_path = os.path.join(RES_DIR, dataset, f'{dataset}_{model}_temperature1.jsonl')
    if not os.path.exists(res_path):
        return [], []

    model_res = read_jsonl(res_path)
    correct_conf = []
    wrong_conf = []

    for item in model_res:
        if not item.get('Res') or item['Res'] is None:
            continue
        if item.get('popularity') == 'No':
            continue

        question_entity = item['question'].replace(PATTERN[dataset], '').lower()
        ref = remove_punctuation_edges(item['reference'][0], dataset)
        gene_entity = remove_punctuation_edges(item['Res'], dataset)

        # 检查统计文件
        if question_entity not in co_occu:
            continue
        if question_entity.lower() not in single_occr:
            continue
        if gene_entity.lower() not in single_occr:
            continue
        if ref.lower() not in single_occr:
            continue

        # 过滤 single_occurrence 异常大的样本
        if dataset in ['movies', 'songs']:
            if (single_occr[question_entity.lower()] > SINGLE_OCC_THRESHOLD or
                    single_occr[gene_entity.lower()] > SINGLE_OCC_THRESHOLD or
                    single_occr[ref.lower()] > SINGLE_OCC_THRESHOLD):
                continue
        else:
            if single_occr[question_entity.lower()] > SINGLE_OCC_THRESHOLD:
                continue

        # 提取 confidence
        if 'gpt' in model.lower() or 'chat' in model.lower():
            import math
            probs = [math.exp(t) for t in item['Log_p']['token_logprobs']]
        else:
            probs = item['Log_p']['token_probs']
        conf = sum(probs) / len(probs)

        # 根据 correctness 分类
        acc = 1 if item.get('has_answer', False) else 0
        if acc == 1:
            correct_conf.append(conf)
        else:
            wrong_conf.append(conf)

    return correct_conf, wrong_conf


def plot_confidence_boxplot(model='chatgpt'):
    """绘制指定模型在所有数据集上的 confidence 箱形图"""
    full_dict = load_popularity()
    co_occu = json.load(open(COO_PATH))
    single_occr = json.load(open(SINGLE_PATH))

    # 收集数据
    data_by_dataset = {}
    stats_by_dataset = {}

    for dataset in DATASETS:
        correct_conf, wrong_conf = collect_confidence_by_correctness(
            dataset, model, full_dict, co_occu, single_occr
        )
        data_by_dataset[dataset] = {
            'Correct': correct_conf,
            'Incorrect': wrong_conf
        }

        # 计算统计信息
        stats_by_dataset[dataset] = {
            'Correct': {
                'mean': np.mean(correct_conf) if correct_conf else 0,
                'std': np.std(correct_conf) if correct_conf else 0,
                'median': np.median(correct_conf) if correct_conf else 0,
                'count': len(correct_conf)
            },
            'Incorrect': {
                'mean': np.mean(wrong_conf) if wrong_conf else 0,
                'std': np.std(wrong_conf) if wrong_conf else 0,
                'median': np.median(wrong_conf) if wrong_conf else 0,
                'count': len(wrong_conf)
            }
        }

    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Confidence Distribution by Correctness ({model})', fontsize=14, fontweight='bold')

    colors = ['#2ecc71', '#e74c3c']  # 绿色表示正确，红色表示错误

    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx]
        data = data_by_dataset[dataset]

        # 准备箱形图数据
        box_data = [data['Correct'], data['Incorrect']]
        labels = ['Correct', 'Incorrect']

        # 绘制箱形图
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                        widths=0.6, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='black', markeredgecolor='black', markersize=6))

        # 设置颜色
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # 设置标题和标签
        dataset_title = dataset.capitalize()
        ax.set_title(f'{dataset_title}\n(n_correct={stats_by_dataset[dataset]["Correct"]["count"]}, '
                     f'n_wrong={stats_by_dataset[dataset]["Incorrect"]["count"]})',
                     fontsize=11)
        ax.set_ylabel('Confidence', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # 添加均值和方差标注
        for i, label in enumerate(labels):
            stats = stats_by_dataset[dataset][label]
            ax.text(i + 1, 0.02, f'μ={stats["mean"]:.3f}\nσ={stats["std"]:.3f}',
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(BASE, 'code', 'analysis_correlation', 'paper_figures',
                               f'confidence_boxplot_{model}.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved figure to: {output_path}')

    # 打印统计信息
    print(f'\n=== Confidence Statistics for {model} ===\n')
    for dataset in DATASETS:
        print(f'{dataset.upper()}:')
        for label in ['Correct', 'Incorrect']:
            stats = stats_by_dataset[dataset][label]
            print(f'  {label:10s}: mean={stats["mean"]:.4f}, std={stats["std"]:.4f}, '
                  f'median={stats["median"]:.4f}, n={stats["count"]}')
        # 计算差异
        correct_mean = stats_by_dataset[dataset]['Correct']['mean']
        wrong_mean = stats_by_dataset[dataset]['Incorrect']['mean']
        gap = correct_mean - wrong_mean
        print(f'  Gap (C-W): {gap:.4f}')
        print()

    plt.show()


if __name__ == '__main__':
    # 为 ChatGPT 绘制
    plot_confidence_boxplot('chatgpt')

    # 也可以为其他模型绘制
    # for model in ['llama8b', 'qwen2', 'Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']:
    #     plot_confidence_boxplot(model)
