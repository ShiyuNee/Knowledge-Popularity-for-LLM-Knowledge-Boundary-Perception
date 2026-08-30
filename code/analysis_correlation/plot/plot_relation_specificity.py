"""
plot_relation_specificity.py — §4.2 Relation Specificity 分层折线图（美化版）

命名约定 (与 stratified_validation.py 统一):
  - quality_v2 = gt_coo / q_single
  - focused  = high quality_v2 (= low q_single)  → quality_v2 > median
  - diffuse  = low quality_v2  (= high q_single) → quality_v2 <= median

用法:
  from plot_relation_specificity import plot_relation_specificity
  python plot_relation_specificity.py
"""

import sys, os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO_ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import mannwhitneyu
from scipy.interpolate import PchipInterpolator

from _plot_common import (
    setup_style, load_all_data, COLORS, save_figure,
    Q_SINGLE_LO, Q_SINGLE_HI,
)

# ══════════════════════════════════════════════════════════════════════════════
# gt_coo 分箱定义
# ══════════════════════════════════════════════════════════════════════════════
COO_BINS = [
    (1,  1,     '1'),
    (3,  3,     '3'),
    (4,  5,     '4'),
    (6,  7,     '6'),
    (8,  10,   '8'),
    (11, 15,   '11'),
    (16, 20,   '16'),
    (21, 30,   '21'),
    (31, 50,   '31'),
    (51, 99999, '>50'),
]

# ── 柔和配色 ──────────────────────────────────────────────────────────────────
C_FOCUSED = '#6B9FCA'   # 低饱和柔蓝
C_DIFFUSE  = '#E8A17A'  # 低饱和柔橙


def compute_stratified_accuracy(samples):
    """
    在每个 gt_coo 区间内，按 quality_v2 中位数分 focused / diffuse，
    返回每个 bin 的统计字典列表。
    """
    filtered = [s for s in samples
                if Q_SINGLE_LO <= s['q_single'] < Q_SINGLE_HI]
    if len(filtered) < 50:
        return []

    coo     = np.array([s['coo']        for s in filtered], dtype=float)
    quality = np.array([s['quality_v2'] for s in filtered], dtype=float)
    acc     = np.array([s['acc']        for s in filtered], dtype=float)

    results = []
    for lo, hi, label in COO_BINS:
        bin_mask = (coo >= lo) & (coo <= hi)
        if bin_mask.sum() < 20:
            continue

        q_med        = np.median(quality[bin_mask])
        focused_mask = bin_mask & (quality > q_med)
        diffuse_mask = bin_mask & (quality <= q_med)

        n_f, n_d = focused_mask.sum(), diffuse_mask.sum()
        if n_f < 5 or n_d < 5:
            continue

        acc_f = acc[focused_mask].mean()
        acc_d = acc[diffuse_mask].mean()
        _, p  = mannwhitneyu(acc[focused_mask], acc[diffuse_mask],
                             alternative='two-sided')

        results.append({
            'label':      label,
            'acc_focused': acc_f,
            'acc_diffuse': acc_d,
            'n_focused':   n_f,
            'n_diffuse':   n_d,
            'diff':        acc_f - acc_d,
            'p':           p,
        })

    return results


def plot_relation_specificity(
    dataset='movies',
    models=None,
    all_data=None,
    save=True,
    return_fig=False,
):
    """
    绘制 Relation Specificity 折线图（美化版）。

    - 低饱和柔和配色（柔蓝 / 柔橙）
    - Focused 实线 + 圆形标记 / Diffuse 长虚线 + 菱形标记
    - PCHIP 平滑曲线 + 半透明填充带
    - 图例放置于子图内部左上角
    """
    setup_style()

    from _plot_common import MODEL_LABELS

    if models is None:
        models = ['llama8b', 'chatgpt']
    n_models = len(models)

    if all_data is None:
        all_data = load_all_data(datasets=[dataset], models=models)

    # ── 计算每个模型的分层准确率 ──────────────────────────────────────────
    all_results = {}
    for model in models:
        samples = all_data.get((dataset, model), [])
        if not samples:
            continue
        results = compute_stratified_accuracy(samples)
        if results:
            all_results[model] = results

    if not all_results:
        print(f"[WARN] 无有效数据: {dataset}")
        return

    # ── 绘图 ─────────────────────────────────────────────────────────────
    # figsize=(宽度, 高度)：控制整个图的长宽比例
    #   宽度 = 4.2 * 子图数，高度 = 3.8
    #   增大高度值→图变高，增大宽度值→图变宽
    fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 4.2))
    if n_models == 1:
        axes = [axes]

    y_ranges = {
        'llama8b':     (0.2,  1.0),
        'qwen2':       (0.0,  1.0),
        'chatgpt':     (0.8, 1.0),
        'Qwen2.5-7B':  (0.0,  1.0),
        'Qwen2.5-14B': (0.0,  1.0),
        'Qwen2.5-32B': (0.0,  1.0),
    }

    for ax_idx, (ax, model) in enumerate(zip(axes, models)):
        if model not in all_results:
            continue
        results = all_results[model]

        # x 轴位置（等距排列，标签用 bin 名）
        n_pts = len(results)
        x_idx = np.arange(n_pts, dtype=float)  # 0, 1, 2, ...
        x_labels = [r['label'] for r in results]

        acc_f = np.array([r['acc_focused'] for r in results])
        acc_d = np.array([r['acc_diffuse'] for r in results])

        # PCHIP 平滑（等距 x）
        x_smooth = np.linspace(x_idx[0], x_idx[-1], 300)
        yf = np.clip(PchipInterpolator(x_idx, acc_f)(x_smooth), 0, 1)
        yd = np.clip(PchipInterpolator(x_idx, acc_d)(x_smooth), 0, 1)

        # 半透明填充带
        ax.fill_between(x_smooth, yd, yf, color=C_FOCUSED, alpha=0.14, zorder=1)

        # Focused：实线 + 圆形标记
        ax.plot(x_smooth, yf, linewidth=2.6, color=C_FOCUSED,
                solid_capstyle='round', solid_joinstyle='round',
                zorder=3)
        ax.scatter(x_idx, acc_f, s=90, color=C_FOCUSED,
                   edgecolors='white', linewidths=1.6,
                   marker='o', zorder=5, clip_on=False)

        # Diffuse：长虚线 + 菱形标记
        ax.plot(x_smooth, yd, linewidth=2.6, color=C_DIFFUSE,
                linestyle=(0, (6, 2)),
                solid_capstyle='round', solid_joinstyle='round',
                zorder=3)
        ax.scatter(x_idx, acc_d, s=90, color=C_DIFFUSE,
                   edgecolors='white', linewidths=1.6,
                   marker='D', zorder=5, clip_on=False)

        # ── Δ 差值标注已移除 ──────────────────────────────────────────

        # ── x 轴刻度（等距 + 标签）─────────────────────────────────
        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_labels, fontsize=15)
        ax.set_xlim(-0.3, n_pts - 0.7)
        ax.set_xlabel(r'$RPop_{GT}$', fontsize=18, labelpad=6)

        # ── y 轴范围 ─────────────────────────────────────────────────
        ylo, yhi = y_ranges.get(model, (0.0, 1.0))
        ax.set_ylim(ylo, yhi)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.tick_params(axis='y', labelsize=15)
        if ax_idx == 0:
            ax.set_ylabel('Accuracy', fontsize=18, labelpad=6)

        ax.set_title(MODEL_LABELS[model], fontsize=17,
                     fontweight='semibold', pad=10)

        # ── 共享图例（仅在第一个子图上方居中）───────────────────────
        if ax_idx == 0:
            h_f = matplotlib.lines.Line2D(
                [], [], color=C_FOCUSED, linewidth=2.6,
                marker='o', markersize=8, markeredgecolor='white',
                markeredgewidth=1.6, label='Focused')
            h_d = matplotlib.lines.Line2D(
                [], [], color=C_DIFFUSE, linewidth=2.6,
                linestyle=(0, (6, 2)),
                marker='D', markersize=8, markeredgecolor='white',
                markeredgewidth=1.6, label='Diffuse')
            fig.legend(
                handles=[h_f, h_d],
                fontsize=20, loc='upper center',
                ncol=2, columnspacing=1.5,
                frameon=False,
                handlelength=2.2, handleheight=0.9,
                bbox_to_anchor=(0.60, 1.05),
            )
        ax.grid(False)
        # 水平网格（与左图视觉等距对齐，不显示刻度标签）
        yticks = np.linspace(ylo, yhi, 5)  # 5个点=4段，含端点
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{v:.2f}' for v in yticks])
        for yt in yticks:
            ax.axhline(yt, color='#CCCCCC', linewidth=0.6,
                       linestyle='--', alpha=0.7, zorder=0)
        # 垂直网格：均匀分布
        for k in range(1, 6):
            xv = -0.3 + (n_pts - 0.7 + 0.3) * k / 6
            ax.axvline(xv, color='#CCCCCC', linewidth=0.6,
                       linestyle='--', alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_linewidth(0.6)
        ax.spines['right'].set_linewidth(0.6)
        ax.spines['bottom'].set_linewidth(0.6)
        ax.spines['left'].set_linewidth(0.6)
        ax.spines['top'].set_color('#AAAAAA')
        ax.spines['right'].set_color('#AAAAAA')
        ax.spines['bottom'].set_color('#AAAAAA')
        ax.spines['left'].set_color('#AAAAAA')

    # subplots_adjust 控制子图在画布中的位置和间距
    #   left/right/bottom/top: 子图区域在画布中的边界（0~1）
    #   wspace: 两个子图之间的水平间距（值越大间距越大）
    fig.subplots_adjust(left=0.08, right=1.1,
                        bottom=0.17, top=0.82,
                        wspace=0.12)

    if save:
        model_tag = '+'.join(models)
        fig_name  = f'figure_relation_specificity_{dataset}_{model_tag}'
        save_figure(fig, fig_name)

    if return_fig:
        return fig
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 跨数据集折线图（美化版）
# ══════════════════════════════════════════════════════════════════════════════
def plot_cross_dataset_metrics(
    models=None,
    all_data=None,
    save=True,
    return_fig=False,
):
    """
    折线图：x 轴 = 3 个数据集，多条指标曲线。

    风格统一：与 plot_relation_specificity 一致的柔和学术风，
    低饱和配色 + 差异化线型/标记 + 大字体 + 手动虚线网格。
    """
    setup_style()

    from _plot_common import DATASETS, MODEL_LABELS, DATASET_LABELS

    if models is None:
        from _plot_common import MODELS
        models = [m for m in MODELS if not m.startswith('Qwen2.5')]
    if all_data is None:
        all_data = load_all_data()

    datasets  = DATASETS
    n_models  = len(models)
    n_datasets = len(datasets)

    # ── 计算指标 ──────────────────────────────────────────────────────────
    stats = {}
    for model in models:
        stats[model] = {}
        for dataset in datasets:
            samples = all_data.get((dataset, model), [])
            if not samples:
                continue
            acc_arr  = np.array([s['acc'] for s in samples], dtype=float)
            acc_mean = acc_arr.mean()

            wrong = [s for s in samples if s['acc'] == 0]
            if len(wrong) < 10:
                stats[model][dataset] = {
                    'acc': acc_mean,
                    'ratio_pop':   np.nan,
                    'ratio_coo':   np.nan,
                    'ratio_diff':  np.nan,
                    'ratio_union': np.nan,
                    'gene_pop_cv': np.nan,
                }
                continue

            w_gene_pop = np.array([s['gene_pop'] for s in wrong], dtype=float)
            w_gt_pop   = np.array([s['gt_pop']   for s in wrong], dtype=float)
            w_gene_coo = np.array([s['gene_coo'] for s in wrong], dtype=float)
            w_gt_coo   = np.array([s['coo']      for s in wrong], dtype=float)

            mask_pop = w_gene_pop > w_gt_pop
            mask_coo = w_gene_coo > w_gt_coo

            gp_mean = w_gene_pop.mean()
            gp_cv   = w_gene_pop.std() / gp_mean if gp_mean > 0 else np.nan

            stats[model][dataset] = {
                'acc':         acc_mean,
                'ratio_pop':   mask_pop.mean(),
                'ratio_coo':   mask_coo.mean(),
                'ratio_diff':  (mask_pop & ~mask_coo).mean(),
                'ratio_union': (mask_pop | mask_coo).mean(),
                'gene_pop_cv': gp_cv,
            }

    # ── 指标配置（低饱和度五色 + 差异化线型/标记）────────────────────────
    METRIC_CFG = [
        ('acc',         'Accuracy',
         '#5A8FC0', '-',          'o',  2.6, 10),
        ('ratio_pop',   r'$Pop_{Ge}>Pop_{GT}$',
         '#D97E5A', (0, (6, 2)),  's',  2.2, 10),
        ('ratio_coo',   r'$RPop_{Ge}>RPop_{GT}$',
         '#5FAF7A', (0, (3, 2)),  '^',  2.2, 11),
        ('ratio_union', 'Either Condition',
         '#9B7FBF', (0, (5, 2)),  'D',  1.8, 9),
        ('ratio_diff',  'Entity Popularity Only',
         '#C8A44A', (0, (2, 2)),  '*',  1.8, 14),
    ]

    # ── 绘图：单模型 ────────────────────────────────────────────────────
    model = models[0]
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.2))

    x_pos = np.arange(n_datasets)
    legend_handles = []

    for mk, ml, mc, ms, mrk, lw, ms_size in METRIC_CFG:
        vals = np.array([
            stats.get(model, {}).get(ds, {}).get(mk, np.nan)
            for ds in datasets
        ])
        valid = ~np.isnan(vals)

        ax.plot(
            x_pos[valid], vals[valid],
            marker=mrk, markersize=ms_size,
            linewidth=lw, color=mc, linestyle=ms,
            markeredgecolor='white', markeredgewidth=1.6,
            solid_capstyle='round', solid_joinstyle='round',
            clip_on=False, zorder=4,
        )

        h, = ax.plot([], [],
                     marker=mrk, markersize=ms_size,
                     linewidth=lw, color=mc, linestyle=ms,
                     markeredgecolor='white', markeredgewidth=1.6,
                     label=ml)
        legend_handles.append(h)

    # ── 坐标轴样式 ───────────────────────────────────────────────────────
    ax.set_xticks(x_pos)
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets], fontsize=15)
    ax.set_xlim(-0.35, n_datasets - 0.65)
    ax.set_ylim(0.2, 1.0)
    yticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.tick_params(axis='y', labelsize=15)

    ax.set_title(MODEL_LABELS[model], fontsize=17,
                 fontweight='semibold', pad=10)

    # 手动虚线网格
    ax.grid(False)
    for yt in yticks:
        ax.axhline(yt, color='#CCCCCC', linewidth=0.6,
                   linestyle='--', alpha=0.7, zorder=0)
    for xv in x_pos:
        ax.axvline(xv, color='#CCCCCC', linewidth=0.6,
                   linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    # 细灰边框
    for spine_name in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_linewidth(0.6)
        ax.spines[spine_name].set_color('#AAAAAA')

    ax.set_ylabel('Ratio / Accuracy', fontsize=18, labelpad=6)

    # 右侧竖排图例
    metric_labels = [cfg[1] for cfg in METRIC_CFG]
    ax.legend(
        legend_handles, metric_labels,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        frameon=False,
        fontsize=15,
        handlelength=2.2, handleheight=0.9,
    )

    fig.subplots_adjust(left=0.15, right=0.68,
                        bottom=0.17, top=0.88)

    if save:
        model_tag = '+'.join(models)
        save_figure(fig, f'figure_cross_dataset_metrics_{model_tag}')

    if return_fig:
        return fig
    plt.close(fig)


def plot_bar():
    """绘制 ChatGPT 和 Llama-3-8B 的 confidence 柱状图，左右排列，共用图例"""
    import matplotlib.pyplot as plt
    import numpy as np
    import json
    import os

    # 数据路径
    BASE = str(REPO_ROOT)
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

    MODEL_LABELS = {
        'chatgpt': 'ChatGPT',
        'llama8b': 'Llama-3-8B',
    }

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

            if question_entity not in co_occu:
                continue
            if question_entity.lower() not in single_occr:
                continue
            if gene_entity.lower() not in single_occr:
                continue
            if ref.lower() not in single_occr:
                continue

            if dataset in ['movies', 'songs']:
                if (single_occr[question_entity.lower()] > SINGLE_OCC_THRESHOLD or
                        single_occr[gene_entity.lower()] > SINGLE_OCC_THRESHOLD or
                        single_occr[ref.lower()] > SINGLE_OCC_THRESHOLD):
                    continue
            else:
                if single_occr[question_entity.lower()] > SINGLE_OCC_THRESHOLD:
                    continue

            if 'gpt' in model.lower() or 'chat' in model.lower():
                import math
                probs = [math.exp(t) for t in item['Log_p']['token_logprobs']]
            else:
                probs = item['Log_p']['token_probs']
            conf = sum(probs) / len(probs)

            acc = 1 if item.get('has_answer', False) else 0
            if acc == 1:
                correct_conf.append(conf)
            else:
                wrong_conf.append(conf)

        if correct_conf:
            correct_stats = {'mean': np.mean(correct_conf), 'std': np.std(correct_conf)}
        else:
            correct_stats = None

        if wrong_conf:
            wrong_stats = {'mean': np.mean(wrong_conf), 'std': np.std(wrong_conf)}
        else:
            wrong_stats = None

        return correct_stats, wrong_stats

    # 加载数据
    full_dict = load_popularity()
    co_occu = json.load(open(COO_PATH))
    single_occr = json.load(open(SINGLE_PATH))

    models = ['chatgpt', 'Qwen2.5-7B']
    n_models = len(models)

    MODEL_LABELS = {
        'chatgpt': 'ChatGPT',
        'Qwen2.5-7B': 'Qwen2.5-7B-Instruct',
    }

    # 收集所有模型的数据
    all_data = {}
    for model in models:
        means_correct = []
        stds_correct = []
        means_wrong = []
        stds_wrong = []

        for dataset in DATASETS:
            correct_stats, wrong_stats = collect_confidence_stats(
                dataset, model, full_dict, co_occu, single_occr
            )

            if correct_stats:
                means_correct.append(correct_stats['mean'] * 100)
                stds_correct.append(correct_stats['std'] * 100)
            else:
                means_correct.append(0)
                stds_correct.append(0)

            if wrong_stats:
                means_wrong.append(wrong_stats['mean'] * 100)
                stds_wrong.append(wrong_stats['std'] * 100)
            else:
                means_wrong.append(0)
                stds_wrong.append(0)

        all_data[model] = {
            'means_correct': means_correct,
            'stds_correct': stds_correct,
            'means_wrong': means_wrong,
            'stds_wrong': stds_wrong,
        }

    # 颜色方案
    color_bar = {
        'Correct': '#e0e0e0',    # 浅灰
        'Incorrect': '#3b83f6',   # 蓝色
    }

    plt.rcParams['font.family'] = 'DejaVu Sans'

    # 创建两个子图，左右排列，间距小
    fig, axes = plt.subplots(1, n_models, figsize=(5.0 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    x_label = ("Movies", "Songs", "Basketball")
    x = np.arange(len(x_label))
    width = 0.35

    methods = ['Correct', 'Incorrect']

    for ax_idx, (ax, model) in enumerate(zip(axes, models)):
        data = all_data[model]
        means_correct = data['means_correct']
        means_wrong = data['means_wrong']

        model_info_0 = {
            'Correct': tuple(means_correct),
            'Incorrect': tuple(means_wrong),
        }

        stds_correct = data['stds_correct']
        stds_wrong = data['stds_wrong']

        if ax_idx == 0:
            ax.set_ylabel('Confidence (%)', fontsize=16)
        ax.set_xticks(x + width / 2, x_label, fontsize=13)
        ax.set_title(MODEL_LABELS[model], fontsize=17, fontweight='semibold', pad=8)
        ax.set_ylim(0, 115)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(axis='y', alpha=0.5, zorder=0)

        multiplier = 0
        for method_idx, method in enumerate(methods):
            offset = width * multiplier
            rects = ax.bar(
                x + offset,
                model_info_0[method],
                width,
                color=color_bar[method],
                edgecolor='white',
                zorder=2
            )
            multiplier += 1

            for i, rect in enumerate(rects):
                height = rect.get_height()
                std_val = stds_correct[i] if method == 'Correct' else stds_wrong[i]
                # 美观的标注：均值在柱子上方，标准差在均值下方小字显示
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    height + 5,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    color='#333333'
                )
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    height + 1.5,
                    f'±{std_val:.1f}',
                    ha='center', va='bottom', fontsize=8,
                    color='#666666'
                )

        # 美化边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#555555')
        ax.spines['bottom'].set_color('#555555')

    # 共享图例在上方居中
    handles = [plt.Rectangle((0,0),1,1, color=color_bar[m], ec='white') for m in methods]
    labels = ['Correct', 'Incorrect']
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=14,
               frameon=False, bbox_to_anchor=(0.5, 1.05))

    # 调整布局，间距小
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.14, top=0.86, wspace=0.12)

    # 保存
    output_dir = os.path.join(BASE, 'code', 'analysis_correlation', 'paper_figures')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'confidence_bar_two_models.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f'Saved to: {output_path}')

    plt.show()


def print_gene_answer_stats():
    """打印做对和做错时 gene answer 的分布统计"""
    import json
    import os
    import numpy as np

    BASE = str(REPO_ROOT)
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

    MODEL_LABELS = {
        'chatgpt': 'ChatGPT',
        'Qwen2.5-7B': 'Qwen2.5-7B-Instruct',
        'llama8b': 'Llama-3-8B',
    }

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

    def collect_gene_stats(dataset, model, full_dict, co_occu, single_occr):
        """收集做对和做错样本的 gene answer 统计信息"""
        res_path = os.path.join(RES_DIR, dataset, f'{dataset}_{model}_temperature1.jsonl')
        if not os.path.exists(res_path):
            return None, None

        model_res = read_jsonl(res_path)
        correct_gene_pop = []
        correct_gene_coo = []
        wrong_gene_pop = []
        wrong_gene_coo = []

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

            if dataset in ['movies', 'songs']:
                if (single_occr[question_entity.lower()] > SINGLE_OCC_THRESHOLD or
                        single_occr[gene_entity.lower()] > SINGLE_OCC_THRESHOLD or
                        single_occr[ref.lower()] > SINGLE_OCC_THRESHOLD):
                    continue
            else:
                if single_occr[question_entity.lower()] > SINGLE_OCC_THRESHOLD:
                    continue

            # 获取 gene_pop
            gene_pop_info = full_dict.get(gene_entity, {})
            gene_pop = gene_pop_info.get('popularity', 0) if isinstance(gene_pop_info, dict) else gene_pop_info
            if gene_pop == 'No' or gene_pop is None:
                gene_pop = 0

            # 获取 gene_coo
            gene_coo = co_occu.get(question_entity, {}).get(gene_entity.lower(), 0)

            acc = 1 if item.get('has_answer', False) else 0
            if acc == 1:
                correct_gene_pop.append(gene_pop)
                correct_gene_coo.append(gene_coo)
            else:
                wrong_gene_pop.append(gene_pop)
                wrong_gene_coo.append(gene_coo)

        def calc_stats(arr):
            if not arr:
                return {'median': 0, 'mean': 0, 'std': 0, 'cv': 0, 'n': 0}
            arr = np.array(arr)
            median = np.median(arr)
            mean = arr.mean()
            std = arr.std()
            cv = std / mean if mean > 0 else 0
            return {'median': median, 'mean': mean, 'std': std, 'cv': cv, 'n': len(arr)}

        correct_stats = {
            'pop': calc_stats(correct_gene_pop),
            'coo': calc_stats(correct_gene_coo),
        }
        wrong_stats = {
            'pop': calc_stats(wrong_gene_pop),
            'coo': calc_stats(wrong_gene_coo),
        }

        return correct_stats, wrong_stats

    # 加载数据
    full_dict = load_popularity()
    co_occu = json.load(open(COO_PATH))
    single_occr = json.load(open(SINGLE_PATH))

    models = ['llama8b', 'qwen2', 'chatgpt', 'Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']

    MODEL_LABELS = {
        'llama8b': 'Llama-3-8B',
        'qwen2': 'Qwen2-7B',
        'chatgpt': 'ChatGPT',
        'Qwen2.5-7B': 'Qwen2.5-7B',
        'Qwen2.5-14B': 'Qwen2.5-14B',
        'Qwen2.5-32B': 'Qwen2.5-32B',
    }

    # 收集所有数据
    all_results = {}
    for model in models:
        all_results[model] = {}
        for dataset in DATASETS:
            # Basketball 排除 Qwen2.5 模型
            if dataset == 'basketball' and model.startswith('Qwen2.5'):
                all_results[model][dataset] = None
                continue

            correct_stats, wrong_stats = collect_gene_stats(
                dataset, model, full_dict, co_occu, single_occr
            )
            all_results[model][dataset] = (correct_stats, wrong_stats)

    # 打印文本表格
    print("\n" + "=" * 100)
    print("Gene Answer 分布统计 (按 Correctness 分组) - Median")
    print("=" * 100)
    print(f"{'Model':<20} {'Dataset':<12} {'Correctness':<10} {'N':>6} | {'Pop Median':>10} {'Pop CV':>8} | {'Coo Median':>10} {'Coo CV':>8}")
    print("-" * 100)

    for model in models:
        for dataset in DATASETS:
            result = all_results[model][dataset]
            if result is None:
                continue
            correct_stats, wrong_stats = result
            if correct_stats and wrong_stats:
                model_name = MODEL_LABELS.get(model, model)
                # Correct
                print(f"{model_name:<20} {dataset:<12} {'Correct':<10} {correct_stats['pop']['n']:>6} | "
                      f"{correct_stats['pop']['median']:>10.2f} {correct_stats['pop']['cv']:>8.3f} | "
                      f"{correct_stats['coo']['median']:>10.2f} {correct_stats['coo']['cv']:>8.3f}")
                # Incorrect
                print(f"{model_name:<20} {dataset:<12} {'Incorrect':<10} {wrong_stats['pop']['n']:>6} | "
                      f"{wrong_stats['pop']['median']:>10.2f} {wrong_stats['pop']['cv']:>8.3f} | "
                      f"{wrong_stats['coo']['median']:>10.2f} {wrong_stats['coo']['cv']:>8.3f}")
                print("-" * 100)

    # 打印 LaTeX 表格
    print("\n\n% ===== LaTeX TABLE =====")
    print("\\begin{table*}[t]")
    print("\\centering")
    print("\\small")
    print("\\setlength{\\tabcolsep}{4pt}")
    print("\\begin{tabular}{ll c cc cc c cc cc}")
    print("\\toprule")
    print("& & \\multicolumn{5}{c}{Correct} & \\multicolumn{5}{c}{Incorrect} \\\\")
    print("\\cmidrule(lr){3-7}\\cmidrule(lr){8-12}")
    print("Model & Dataset & N & $Pop_{Ge}$ & CV & $RPop_{Ge}$ & CV & N & $Pop_{Ge}$ & CV & $RPop_{Ge}$ & CV \\\\")
    print("\\midrule")

    for model in models:
        model_name = MODEL_LABELS.get(model, model)
        is_first = True
        for dataset in DATASETS:
            result = all_results[model][dataset]
            if result is None:
                continue
            correct_stats, wrong_stats = result
            if not correct_stats or not wrong_stats:
                continue

            dataset_label = dataset.capitalize()
            model_label = model_name if is_first else ""

            print(f"{model_label} & {dataset_label} & "
                  f"{correct_stats['pop']['n']} & {correct_stats['pop']['median']:.1f} & {correct_stats['pop']['cv']:.2f} & "
                  f"{correct_stats['coo']['median']:.1f} & {correct_stats['coo']['cv']:.2f} & "
                  f"{wrong_stats['pop']['n']} & {wrong_stats['pop']['median']:.1f} & {wrong_stats['pop']['cv']:.2f} & "
                  f"{wrong_stats['coo']['median']:.1f} & {wrong_stats['coo']['cv']:.2f} \\\\")
            is_first = False
        if model != models[-1]:
            print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Gene answer statistics by correctness. $Pop_{Ge}$ = median popularity of generated answer; $RPop_{Ge}$ = median co-occurrence between question and generated answer; CV = coefficient of variation.}")
    print("\\label{tab:gene-answer-stats}")
    print("\\end{table*}")


def print_unique_entity_stats():
    """计算做错时生成的 unique 实体比例"""
    import json
    import os

    BASE = str(REPO_ROOT)
    RES_DIR = os.path.join(BASE, 'res')

    DATASETS = ['movies', 'songs', 'basketball']
    PATTERN = {
        'movies': 'Who is the director of the movie ',
        'songs': 'Who is the performer of the song ',
        'basketball': 'Where is the birthplace of the basketball player '
    }

    MODEL_LABELS = {
        'llama8b': 'Llama-3-8B',
        'qwen2': 'Qwen2-7B',
        'chatgpt': 'ChatGPT',
        'Qwen2.5-7B': 'Qwen2.5-7B',
        'Qwen2.5-14B': 'Qwen2.5-14B',
        'Qwen2.5-32B': 'Qwen2.5-32B',
    }

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

    def has_answer(a, b):
        """判断两个答案是否相同（双向判断）"""
        a_norm = a.lower().strip()
        b_norm = b.lower().strip()
        return a_norm in b_norm or b_norm in a_norm

    def count_unique_entities(dataset, model):
        """计算做错时的 unique entity 数量和比例"""
        res_path = os.path.join(RES_DIR, dataset, f'{dataset}_{model}_temperature1.jsonl')
        if not os.path.exists(res_path):
            return None, None, None

        model_res = read_jsonl(res_path)
        wrong_answers = []

        for item in model_res:
            if not item.get('Res') or item['Res'] is None:
                continue

            acc = 1 if item.get('has_answer', False) else 0
            if acc == 0:  # 只收集错误答案
                gene_entity = remove_punctuation_edges(item['Res'], dataset)
                wrong_answers.append(gene_entity)

        if not wrong_answers:
            return 0, 0, 0.0

        # 使用 has_answer 进行聚类，找出 unique entities
        unique_entities = []
        for ans in wrong_answers:
            is_new = True
            for existing in unique_entities:
                if has_answer(ans, existing):
                    is_new = False
                    break
            if is_new:
                unique_entities.append(ans)

        n_wrong = len(wrong_answers)
        n_unique = len(unique_entities)
        ratio = n_unique / n_wrong if n_wrong > 0 else 0.0

        return n_wrong, n_unique, ratio

    models = ['llama8b', 'qwen2', 'chatgpt', 'Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']

    print("\n" + "=" * 80)
    print("Unique Entity 统计 (做错时)")
    print("=" * 80)
    print(f"{'Model':<20} {'Dataset':<12} {'Wrong N':>10} {'Unique N':>10} {'Ratio':>10}")
    print("-" * 80)

    for model in models:
        for dataset in DATASETS:
            # Basketball 排除 Qwen2.5 模型
            if dataset == 'basketball' and model.startswith('Qwen2.5'):
                continue

            n_wrong, n_unique, ratio = count_unique_entities(dataset, model)
            if n_wrong is not None:
                model_name = MODEL_LABELS.get(model, model)
                print(f"{model_name:<20} {dataset:<12} {n_wrong:>10} {n_unique:>10} {ratio:>10.3f}")

    print("-" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("绘制 Relation Specificity 分层折线图（美化版）")
    print("=" * 60)

    plot_relation_specificity(dataset='movies', models=['llama8b', 'chatgpt'])

    # 跨数据集折线图
    plot_cross_dataset_metrics(models=['chatgpt'])

    # 打印 gene answer 分布统计
    print_gene_answer_stats()

    # 打印 unique entity 统计
    print_unique_entity_stats()

    print("\n完成!")
