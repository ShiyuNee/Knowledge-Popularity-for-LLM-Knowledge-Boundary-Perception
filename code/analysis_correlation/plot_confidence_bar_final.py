"""
使用 plot_bar() 的样式绘制 ChatGPT confidence 柱状图
直接替换数据，保持原有颜色和样式
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def collect_confidence_stats(dataset, model, full_dict, co_occu, single_occr):
    """收集正确和错误样本的 confidence 统计信息"""
    res_path = os.path.join(RES_DIR, dataset, f'{dataset}_{model}_temperature1.jsonl')
    if not os.path.exists(res_path):
        return None, None

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

    # 计算统计信息
    if correct_conf:
        correct_stats = {
            'mean': np.mean(correct_conf),
            'std': np.std(correct_conf),
            'n': len(correct_conf)
        }
    else:
        correct_stats = None

    if wrong_conf:
        wrong_stats = {
            'mean': np.mean(wrong_conf),
            'std': np.std(wrong_conf),
            'n': len(wrong_conf)
        }
    else:
        wrong_stats = None

    return correct_stats, wrong_stats


def plot_confidence_bar():
    """使用 plot_bar() 的样式，替换为 confidence 数据"""

    # 加载数据
    full_dict = load_popularity()
    co_occu = json.load(open(COO_PATH))
    single_occr = json.load(open(SINGLE_PATH))
    model = 'chatgpt'

    # 收集三个数据集的数据
    means_correct = []
    stds_correct = []
    means_wrong = []
    stds_wrong = []

    for dataset in DATASETS:
        correct_stats, wrong_stats = collect_confidence_stats(
            dataset, model, full_dict, co_occu, single_occr
        )

        if correct_stats:
            means_correct.append(correct_stats['mean'])
            stds_correct.append(correct_stats['std'])
        else:
            means_correct.append(0)
            stds_correct.append(0)

        if wrong_stats:
            means_wrong.append(wrong_stats['mean'])
            stds_wrong.append(wrong_stats['std'])
        else:
            means_wrong.append(0)
            stds_wrong.append(0)

    # 使用 plot_bar() 的颜色方案
    color_bar = {
        'Correct': '#e0e0e0',    # 浅灰（Vanilla LLM 颜色）
        'Incorrect': '#3b83f6',  # 蓝色（PT-RAG 颜色）
    }

    # 数据准备 - 转换为百分比形式（乘以100）
    model_info_0 = {
        'Correct': tuple(m * 100 for m in means_correct),
        'Incorrect': tuple(m * 100 for m in means_wrong),
    }

    # 标注数据
    left_annotations = {
        'Correct': tuple(f'{m*100:.2f}' for m in means_correct),
    }
    right_annotations = {
        'Incorrect': tuple(f'{m*100:.2f}' for m in means_wrong),
    }

    plt.rcParams['font.family'] = 'DejaVu Sans'

    # 使用 plot_bar 的双子图布局
    fig2, axs2 = plt.subplots(ncols=2, figsize=(15, 6))
    plt.subplots_adjust(
        top=0.95,
        bottom=0.15,
        left=0.05,
        right=0.95,
        hspace=0.25,
        wspace=0.2
    )

    x_label = ("Movies", "Songs", "Basketball")
    x = np.arange(len(x_label))
    width = 0.35

    # 第一个子图：Correct
    ax_left = axs2[0]

    ax_left.set_ylabel('Confidence (%)', fontsize=20)
    ax_left.set_xticks(x + width / 2, x_label, fontsize=18)
    ax_left.set_title('Correct Samples', fontsize=22, pad=8, fontweight='bold')
    y_max_left = 100
    ax_left.set_ylim(0, y_max_left * 1.1)
    ax_left.tick_params(axis='y', labelsize=16)
    ax_left.grid(axis='y', alpha=0.5, zorder=0)

    rects = ax_left.bar(
        x,
        model_info_0['Correct'],
        width,
        label='Correct',
        color=color_bar['Correct'],
        edgecolor='white',
        zorder=2
    )

    for i, rect in enumerate(rects):
        height = rect.get_height()
        display_str = left_annotations['Correct'][i]
        ax_left.text(
            rect.get_x() + rect.get_width() / 2,
            height + y_max_left * 0.015,
            display_str,
            ha='center', va='bottom', fontsize=15
        )

    # 第二个子图：Incorrect
    ax_right = axs2[1]

    ax_right.set_ylabel('Confidence (%)', fontsize=20)
    ax_right.set_xticks(x + width / 2, x_label, fontsize=18)
    ax_right.set_title('Incorrect Samples', fontsize=22, pad=8, fontweight='bold')
    y_max_right = 100
    ax_right.set_ylim(0, y_max_right * 1.1)
    ax_right.tick_params(axis='y', labelsize=16)
    ax_right.grid(axis='y', alpha=0.5, zorder=0)

    rects = ax_right.bar(
        x,
        model_info_0['Incorrect'],
        width,
        label='Incorrect',
        color=color_bar['Incorrect'],
        edgecolor='white',
        zorder=2
    )

    for i, rect in enumerate(rects):
        height = rect.get_height()
        display_str = right_annotations['Incorrect'][i]
        ax_right.text(
            rect.get_x() + rect.get_width() / 2,
            height + y_max_right * 0.015,
            display_str,
            ha='center', va='bottom', fontsize=15
        )

    # 美化边框
    for ax in [ax_left, ax_right]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#555555')
        ax.spines['bottom'].set_color('#555555')

    # 共享图例
    handles = [plt.Rectangle((0,0),1,1, color=color_bar[m], ec='white') for m in ['Correct', 'Incorrect']]
    labels = ['Correct', 'Incorrect']
    fig2.legend(handles, labels, loc='lower center', ncol=2, fontsize=18)

    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # 保存图片
    output_path = os.path.join(BASE, 'code', 'analysis_correlation', 'paper_figures',
                               f'confidence_bar_final_chatgpt.pdf')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f'Saved figure to: {output_path}')

    # 同时保存 PNG
    output_path_png = output_path.replace('.pdf', '.png')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f'Saved figure to: {output_path_png}')

    # 打印统计信息
    print(f'\n=== Confidence Statistics for ChatGPT ===\n')
    for i, dataset in enumerate(DATASETS):
        print(f'{dataset.upper()}:')
        print(f'  Correct  : μ={means_correct[i]:.4f}, σ={stds_correct[i]:.4f}')
        print(f'  Incorrect: μ={means_wrong[i]:.4f}, σ={stds_wrong[i]:.4f}')
        gap = means_correct[i] - means_wrong[i]
        print(f'  Gap (C-W): {gap:.4f}')
        print()

    plt.close(fig2)


if __name__ == '__main__':
    plot_confidence_bar()
