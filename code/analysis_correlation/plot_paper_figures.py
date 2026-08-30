"""
学术论文图表绘制脚本
生成4张核心图表，用于EMNLP投稿

Figure 1: 不对称性分析 — 正确/错误子集中 gene_pop→conf 和 gene_coo→conf 的对比
Figure 2: 边界条件 — 三个数据集上偏相关系数的对比
Figure 3: 缩放分析 — Qwen2.5 7B→14B→32B 的多指标趋势
Figure 4: 校准实验 — 不同特征组合的 AUROC 对比

输出: ./paper_figures/figure_{1,2,3,4}.pdf 和 .png
"""

import json
import math
import re
import os
import numpy as np
from scipy.stats import spearmanr, rankdata, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ─── 路径配置 ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES_DIR = os.path.join(BASE, 'res')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_figures')
os.makedirs(OUT_DIR, exist_ok=True)

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

# ─── Matplotlib 学术风格设置 ─────────────────────────────────────────────────
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8

# 色盲友好配色 (from ColorBrewer Set2 + custom)
COLORS = {
    'movies': '#1f77b4',      # 蓝色
    'songs': '#ff7f0e',       # 橙色
    'basketball': '#2ca02c',  # 绿色
    'correct': '#2166ac',     # 深蓝
    'incorrect': '#b2182b',   # 深红
    'gene_pop': '#762a83',    # 紫色
    'gene_coo': '#1b7837',    # 深绿
    'qwen7b': '#d73027',      # 红色
    'qwen14b': '#fc8d59',     # 橙红
    'qwen32b': '#fee090',     # 黄色
}

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

# ─── 工具函数 ─────────────────────────────────────────────────────────────────
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


def partial_spearman(x, y, z):
    """控制 z 后，x 与 y 的偏相关（rank 线性残差法）"""
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


def collect_samples(dataset, model, full_dict, co_occu, single_occr, apply_filter=True):
    """收集一个 dataset × model 的所有有效样本"""
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
        if apply_filter:
            if dataset in ['movies', 'songs']:
                if (single_occr[question_entity.lower()] > SINGLE_OCC_THRESHOLD or
                        single_occr[gene_entity.lower()] > SINGLE_OCC_THRESHOLD or
                        single_occr[ref.lower()] > SINGLE_OCC_THRESHOLD):
                    continue
            else:
                if single_occr[question_entity.lower()] > SINGLE_OCC_THRESHOLD:
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
        acc = item.get('has_answer', 0) if 'has_answer' in item else (1 if item.get('correct', False) else 0)
        q_single = single_occr[question_entity.lower()]
        quality_v2 = gt_coo / q_single if q_single > 0 else 0
        samples.append({
            'question_pop': question_pop,
            'gt_pop': ref_pop,
            'gene_pop': gene_pop,
            'gt_coo': gt_coo,
            'gene_coo': gene_coo,
            'conf': conf,
            'acc': acc,
            'q_single': q_single,
            'quality_v2': quality_v2,
        })
    return samples


def _ece(acc_arr, conf_arr, n_bins=10):
    acc_arr = np.asarray(acc_arr, dtype=float)
    conf_arr = np.asarray(conf_arr, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(acc_arr)
    for i in range(n_bins):
        mask = (conf_arr >= bins[i]) & (conf_arr < bins[i + 1]) if i < n_bins - 1 else (conf_arr >= bins[i]) & (conf_arr <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(acc_arr[mask].mean() - conf_arr[mask].mean())
    return ece


# ─── 数据预计算 ───────────────────────────────────────────────────────────────
def precompute_all_stats():
    """预计算所有需要的统计数据"""
    full_dict = load_popularity()
    co_occu = json.loads(open(COO_PATH).read())
    single_occr = json.loads(open(SINGLE_PATH).read())

    # Figure 1 & 2: 偏相关系数 (正确/错误子集)
    fig12_stats = {}  # (dataset, model) -> {correct_gene_pop, incorrect_gene_pop, correct_gene_coo, incorrect_gene_coo}
    # Figure 3: 缩放分析
    fig3_stats = {}  # dataset -> {model -> {acc, conf, ece, coo_acc, qpop_conf, overconf, acc_gap, conf_gap}}
    # Figure 4: 校准实验 (需要单独运行 classification_experiment.py)
    # 这里先用论文中的数据硬编码

    for dataset in DATASETS:
        fig3_stats[dataset] = {}
        for model in MODELS:
            samples = collect_samples(dataset, model, full_dict, co_occu, single_occr)
            if len(samples) < 30:
                continue
            s_arr = np.array(samples)
            qpop = np.array([s['question_pop'] for s in samples], dtype=float)
            gt_pop = np.array([s['gt_pop'] for s in samples], dtype=float)
            gene_pop = np.array([s['gene_pop'] for s in samples], dtype=float)
            gt_coo = np.array([s['gt_coo'] for s in samples], dtype=float)
            gene_coo = np.array([s['gene_coo'] for s in samples], dtype=float)
            conf = np.array([s['conf'] for s in samples], dtype=float)
            acc = np.array([s['acc'] for s in samples], dtype=float)

            # Figure 1 & 2: 偏相关 (控制 gt_coo 后 gene_pop→conf; 控制 gene_pop 后 gene_coo→conf)
            # 实际上论文中控制的是 coo (对于 gene_pop→conf) 和 question_pop (对于 gene_coo→conf)
            # 但根据论文描述，偏相关控制的是 "coo" (即 gt_coo)
            # 让我用控制 gt_coo 后的 partial spearman
            r_corr_gp, _ = partial_spearman(gene_pop[acc == 1], conf[acc == 1], gt_coo[acc == 1]) if (acc == 1).sum() >= 20 else (0, 1)
            r_incor_gp, _ = partial_spearman(gene_pop[acc == 0], conf[acc == 0], gt_coo[acc == 0]) if (acc == 0).sum() >= 20 else (0, 1)
            r_corr_gc, _ = partial_spearman(gene_coo[acc == 1], conf[acc == 1], gt_coo[acc == 1]) if (acc == 1).sum() >= 20 else (0, 1)
            r_incor_gc, _ = partial_spearman(gene_coo[acc == 0], conf[acc == 0], gt_coo[acc == 0]) if (acc == 0).sum() >= 20 else (0, 1)

            # 更准确地：论文中 gene_pop→conf (partial, controlling for coo) 中的 coo 是 gt_coo 还是 gene_coo?
            # 从论文 §4.2 来看: "controlling for coo" 应该指的是控制 gt_coo
            # 但 gene_coo→conf 的 partial correlation 控制的是什么？
            # 从论文 §4.3 来看: gene_coo→conf (partial) 应该控制的是 question_pop
            # 让我重新计算：
            # gene_pop→conf (控制 gt_coo)
            r_corr_gp2, _ = partial_spearman(gene_pop[acc == 1], conf[acc == 1], gt_coo[acc == 1]) if (acc == 1).sum() >= 20 else (0, 1)
            r_incor_gp2, _ = partial_spearman(gene_pop[acc == 0], conf[acc == 0], gt_coo[acc == 0]) if (acc == 0).sum() >= 20 else (0, 1)
            # gene_coo→conf (控制 question_pop)
            r_corr_gc2, _ = partial_spearman(gene_coo[acc == 1], conf[acc == 1], qpop[acc == 1]) if (acc == 1).sum() >= 20 else (0, 1)
            r_incor_gc2, _ = partial_spearman(gene_coo[acc == 0], conf[acc == 0], qpop[acc == 0]) if (acc == 0).sum() >= 20 else (0, 1)

            fig12_stats[(dataset, model)] = {
                'correct_gene_pop': r_corr_gp2,
                'incorrect_gene_pop': r_incor_gp2,
                'correct_gene_coo': r_corr_gc2,
                'incorrect_gene_coo': r_incor_gc2,
                'n_correct': (acc == 1).sum(),
                'n_incorrect': (acc == 0).sum(),
            }

            # Figure 3 stats
            avg_acc = acc.mean()
            avg_conf = conf.mean()
            overconf = (conf > acc).mean()
            ece_val = _ece(acc, conf)
            r_coo_acc, _ = spearmanr(gt_coo, acc)
            r_qpop_conf, _ = spearmanr(qpop, conf)
            # acc gap: high coo vs low coo
            coo_med = np.median(gt_coo)
            acc_gap = acc[gt_coo > coo_med].mean() - acc[gt_coo <= coo_med].mean()
            # conf gap: high qpop vs low qpop
            qpop_med = np.median(qpop)
            conf_gap = conf[qpop > qpop_med].mean() - conf[qpop <= qpop_med].mean()

            fig3_stats[dataset][model] = {
                'avg_acc': avg_acc,
                'avg_conf': avg_conf,
                'overconf': overconf,
                'ece': ece_val,
                'coo_acc': r_coo_acc,
                'qpop_conf': r_qpop_conf,
                'acc_gap': acc_gap,
                'conf_gap': conf_gap,
            }

    return fig12_stats, fig3_stats


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: 不对称性柱状图
# ═══════════════════════════════════════════════════════════════════════════════
def plot_figure1(fig12_stats):
    """Figure 1: 正确/错误子集中 gene_pop→conf 和 gene_coo→conf 的偏相关系数对比"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    # 数据集和模型配置
    datasets = ['movies', 'songs']
    models_plot = ['llama8b', 'qwen2', 'Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']
    model_labels = [MODEL_LABELS[m] for m in models_plot]
    x = np.arange(len(models_plot))
    width = 0.35

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        correct_gp = []
        incorrect_gp = []
        correct_gc = []
        incorrect_gc = []

        for model in models_plot:
            key = (dataset, model)
            if key in fig12_stats:
                correct_gp.append(fig12_stats[key]['correct_gene_pop'])
                incorrect_gp.append(fig12_stats[key]['incorrect_gene_pop'])
                correct_gc.append(fig12_stats[key]['correct_gene_coo'])
                incorrect_gc.append(fig12_stats[key]['incorrect_gene_coo'])
            else:
                correct_gp.append(0)
                incorrect_gp.append(0)
                correct_gc.append(0)
                incorrect_gc.append(0)

        correct_gp = np.array(correct_gp)
        incorrect_gp = np.array(incorrect_gp)
        correct_gc = np.array(correct_gc)
        incorrect_gc = np.array(incorrect_gc)

        # gene_pop
        bars1 = ax.bar(x - width/2, correct_gp, width, label='Correct', color=COLORS['correct'], edgecolor='white', linewidth=0.5)
        bars2 = ax.bar(x + width/2, incorrect_gp, width, label='Incorrect', color=COLORS['incorrect'], edgecolor='white', linewidth=0.5)

        # gene_coo (叠加在上方? 不，用另一组颜色?)
        # 实际上论文中是分开的两个表。这里我们画两组图：左列 gene_pop，右列 gene_coo
        # 但我已经用了1x2布局，每个子图一个数据集
        # 让我改为：每个子图画 gene_pop 和 gene_coo 的对比，用不同颜色的bar

    # 重新设计: 1x2 -> 2x2? 不，太大了
    # 改为：Figure 1 只画 Movies 的 gene_pop 不对称性，Figure 2 画边界条件
    # 或者：Figure 1 用 grouped bar，每个模型有4个bar (correct/incorrect × gene_pop/gene_coo)
    # 但这样太拥挤了

    # 让我重新设计 Figure 1: 一个图展示 Movies 中 gene_pop 的不对称性
    # 和 Songs 中 gene_coo 的不对称性 (各选一个最显著的)

    plt.close(fig)
    # 重新绘制 Figure 1
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)

    for idx, (dataset, signal_name, signal_key) in enumerate([
        ('movies', 'Generated-entity popularity → Confidence\n(gene_pop → conf | controlling for gt_coo)', 'gene_pop'),
        ('songs', 'Generated-entity co-occurrence → Confidence\n(gene_coo → conf | controlling for question_pop)', 'gene_coo'),
    ]):
        ax = axes[idx]
        correct_vals = []
        incorrect_vals = []
        for model in models_plot:
            key = (dataset, model)
            if key in fig12_stats:
                if signal_key == 'gene_pop':
                    correct_vals.append(fig12_stats[key]['correct_gene_pop'])
                    incorrect_vals.append(fig12_stats[key]['incorrect_gene_pop'])
                else:
                    correct_vals.append(fig12_stats[key]['correct_gene_coo'])
                    incorrect_vals.append(fig12_stats[key]['incorrect_gene_coo'])
            else:
                correct_vals.append(0)
                incorrect_vals.append(0)

        correct_vals = np.array(correct_vals)
        incorrect_vals = np.array(incorrect_vals)

        bars1 = ax.bar(x - width/2, correct_vals, width, label='Correct',
                       color='#74a9cf', edgecolor='black', linewidth=0.5, alpha=0.9)
        bars2 = ax.bar(x + width/2, incorrect_vals, width, label='Incorrect',
                       color='#b2182b', edgecolor='black', linewidth=0.5, alpha=0.9)

        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            if height > 0.02:
                ax.annotate(f'{height:+.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 2),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=7.5, color='#2166ac', fontweight='bold')
        for bar in bars2:
            height = bar.get_height()
            if height > 0.02:
                ax.annotate(f'{height:+.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 2),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=7.5, color='#b2182b', fontweight='bold')

        # 添加ratio标注（incorrect / correct）
        for i in range(len(correct_vals)):
            if correct_vals[i] > 0.02 and incorrect_vals[i] > 0.02:
                ratio = incorrect_vals[i] / correct_vals[i]
                mid_x = x[i]
                mid_y = max(correct_vals[i], incorrect_vals[i]) + 0.06
                ax.annotate(f'{ratio:.1f}×', xy=(mid_x, mid_y),
                           ha='center', va='bottom', fontsize=8,
                           fontweight='bold', color='#555555',
                           bbox=dict(boxstyle='round,pad=0.15', facecolor='#f0f0f0', edgecolor='gray', linewidth=0.5))

        ax.set_ylabel('Partial correlation' if idx == 0 else '')
        ax.set_title(f'{DATASET_LABELS[dataset]}: {signal_name}', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=30, ha='right', fontsize=8.5)
        ax.axhline(y=0, color='black', linewidth=0.6, linestyle='-')
        ax.set_ylim(0, 0.58)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if idx == 1:
            ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='gray', ncol=1)

    fig.suptitle('Figure 1: Asymmetric Association with Confidence', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'figure_1.pdf'), format='pdf')
    fig.savefig(os.path.join(OUT_DIR, 'figure_1.png'), format='png')
    plt.close(fig)
    print("Saved Figure 1")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: 边界条件 — 三个数据集对比
# ═══════════════════════════════════════════════════════════════════════════════
def plot_figure2(fig12_stats):
    """Figure 2: 三个数据集上 incorrect 子集的偏相关系数对比"""
    fig, ax = plt.subplots(figsize=(9, 4.5))

    datasets = ['movies', 'songs', 'basketball']
    models_plot = ['llama8b', 'qwen2', 'Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']
    model_labels = [MODEL_LABELS[m] for m in models_plot]

    # 准备数据: 每个数据集 × 每个模型 × 两个信号 (gene_pop, gene_coo)
    # 使用 grouped bar，每组 (数据集) 内有多个模型
    n_models = len(models_plot)
    n_datasets = len(datasets)
    x = np.arange(n_datasets)
    total_width = 0.75
    width = total_width / n_models

    for i, model in enumerate(models_plot):
        gp_vals = []
        gc_vals = []
        for dataset in datasets:
            key = (dataset, model)
            if key in fig12_stats:
                gp_vals.append(fig12_stats[key]['incorrect_gene_pop'])
                gc_vals.append(fig12_stats[key]['incorrect_gene_coo'])
            else:
                gp_vals.append(np.nan)
                gc_vals.append(np.nan)

        # 交错排列: gene_pop 和 gene_coo 交替
        offset = (i - n_models/2 + 0.5) * width
        # 对于每个数据集，我们画两个bar: gene_pop (稍左) 和 gene_coo (稍右)
        # 但这样每组会有 2*n_models 个bar，太拥挤
        # 改为：只画 gene_coo (因为gene_pop在basketball上不显著)
        # 或者：两个子图，左=person-name datasets, 右=all datasets

    # 重新设计：用热力图展示 incorrect 子集中的偏相关系数
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={'width_ratios': [1.2, 1]})

    # 左图: gene_pop → conf (incorrect), 三个数据集
    ax1 = axes[0]
    data_gp = []
    for model in models_plot:
        row = []
        for dataset in datasets:
            key = (dataset, model)
            if key in fig12_stats:
                row.append(fig12_stats[key]['incorrect_gene_pop'])
            else:
                row.append(np.nan)
        data_gp.append(row)
    data_gp = np.array(data_gp)

    im1 = ax1.imshow(data_gp, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.5)
    ax1.set_xticks(np.arange(len(datasets)))
    ax1.set_xticklabels([DATASET_LABELS[d] for d in datasets])
    ax1.set_yticks(np.arange(len(models_plot)))
    ax1.set_yticklabels(model_labels)
    ax1.set_title('Generated-entity popularity → Confidence\n(incorrect answers)', fontsize=11)
    for i in range(len(models_plot)):
        for j in range(len(datasets)):
            if not np.isnan(data_gp[i, j]):
                text = ax1.text(j, i, f'{data_gp[i, j]:+.2f}',
                               ha="center", va="center", color="white" if abs(data_gp[i, j]) > 0.25 else "black",
                               fontsize=9, fontweight='bold')
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Partial correlation')

    # 右图: gene_coo → conf (incorrect)
    ax2 = axes[1]
    data_gc = []
    for model in models_plot:
        row = []
        for dataset in datasets:
            key = (dataset, model)
            if key in fig12_stats:
                row.append(fig12_stats[key]['incorrect_gene_coo'])
            else:
                row.append(np.nan)
        data_gc.append(row)
    data_gc = np.array(data_gc)

    im2 = ax2.imshow(data_gc, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.5)
    ax2.set_xticks(np.arange(len(datasets)))
    ax2.set_xticklabels([DATASET_LABELS[d] for d in datasets])
    ax2.set_yticks(np.arange(len(models_plot)))
    ax2.set_yticklabels([])
    ax2.set_title('Generated-entity co-occurrence → Confidence\n(incorrect answers)', fontsize=11)
    for i in range(len(models_plot)):
        for j in range(len(datasets)):
            if not np.isnan(data_gc[i, j]):
                text = ax2.text(j, i, f'{data_gc[i, j]:+.2f}',
                               ha="center", va="center", color="white" if abs(data_gc[i, j]) > 0.25 else "black",
                               fontsize=9, fontweight='bold')
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Partial correlation')

    fig.suptitle('Figure 2: Boundary Conditions — Entity Type and Relation Strength', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'figure_2.pdf'), format='pdf')
    fig.savefig(os.path.join(OUT_DIR, 'figure_2.png'), format='png')
    plt.close(fig)
    print("Saved Figure 2")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: 缩放分析
# ═══════════════════════════════════════════════════════════════════════════════
def plot_figure3(fig3_stats):
    """Figure 3: Qwen2.5 7B→14B→32B 的多指标趋势"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    qwen_models = ['Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']
    qwen_labels = ['7B', '14B', '32B']
    x_pos = np.arange(len(qwen_models))

    metrics = [
        ('avg_acc', 'Accuracy', 'Accuracy', [0, 0.5]),
        ('ece', 'ECE', 'Expected Calibration Error', [0, 0.8]),
        ('coo_acc', 'Spearman(coo, acc)', 'Co-occurrence → Accuracy\n(Spearman ρ)', [0, 0.6]),
        ('qpop_conf', 'Spearman(qpop, conf)', 'Question popularity → Confidence\n(Spearman ρ)', [0, 0.5]),
    ]

    for idx, (metric_key, ylabel, title, ylim) in enumerate(metrics):
        ax = axes[idx]
        for dataset in DATASETS:
            vals = []
            for model in qwen_models:
                if dataset in fig3_stats and model in fig3_stats[dataset]:
                    vals.append(fig3_stats[dataset][model].get(metric_key, np.nan))
                else:
                    vals.append(np.nan)
            vals = np.array(vals)
            ax.plot(x_pos, vals, marker='o', markersize=7, linewidth=2,
                    label=DATASET_LABELS[dataset], color=COLORS[dataset])
            # 添加数值标签
            for i, v in enumerate(vals):
                if not np.isnan(v):
                    ax.annotate(f'{v:.3f}', (x_pos[i], v), textcoords="offset points",
                               xytext=(0, 8), ha='center', fontsize=8, color=COLORS[dataset])

        ax.set_xticks(x_pos)
        ax.set_xticklabels(qwen_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(ylim)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        if idx == 1:
            ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='gray')

    fig.suptitle('Figure 3: Scaling Does Not Reduce Popularity Bias', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'figure_3.pdf'), format='pdf')
    fig.savefig(os.path.join(OUT_DIR, 'figure_3.png'), format='png')
    plt.close(fig)
    print("Saved Figure 3")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: 校准实验 AUROC
# ═══════════════════════════════════════════════════════════════════════════════
def plot_figure4():
    """Figure 4: 不同特征组合的 AUROC 对比"""
    # 真实 AUROC 数据 (运行 classification_experiment.py 获得, 6 models 平均)
    fig, ax = plt.subplots(figsize=(8, 5))

    datasets = ['Movies', 'Songs', 'Basketball']
    conditions = ['conf_only', 'conf+qpop', 'conf+gene_pop', 'conf+gene_coo', 'conf+all']
    condition_labels = ['Confidence\nonly', '+Question\npop.', '+Generated\npop.', '+Co-occurrence', 'All\nfeatures']

    auroc_data = {
        'Movies':     [0.893, 0.899, 0.898, 0.946, 0.962],
        'Songs':      [0.884, 0.888, 0.883, 0.928, 0.933],
        'Basketball': [0.739, 0.739, 0.777, 0.759, 0.780],
    }

    x = np.arange(len(conditions))
    width = 0.25

    for i, dataset in enumerate(datasets):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, auroc_data[dataset], width, label=dataset,
                      color=COLORS[dataset.lower()], edgecolor='black', linewidth=0.5, alpha=0.85)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('AUROC')
    ax.set_title('Answer Correctness Prediction', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels, fontsize=9)
    ax.set_ylim(0.65, 1.0)
    ax.axhline(y=0.9, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='gray')

    fig.suptitle('Figure 4: External Signals Improve Calibration', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'figure_4.pdf'), format='pdf')
    fig.savefig(os.path.join(OUT_DIR, 'figure_4.png'), format='png')
    plt.close(fig)
    print("Saved Figure 4")


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Precomputing statistics...")
    fig12_stats, fig3_stats = precompute_all_stats()
    print(f"Computed stats for {len(fig12_stats)} dataset-model combinations")

    print("\nGenerating Figure 1: Asymmetry...")
    plot_figure1(fig12_stats)

    print("\nGenerating Figure 2: Boundary Conditions...")
    plot_figure2(fig12_stats)

    print("\nGenerating Figure 3: Scaling...")
    plot_figure3(fig3_stats)

    print("\nGenerating Figure 4: Calibration...")
    plot_figure4()

    print(f"\nAll figures saved to: {OUT_DIR}")
