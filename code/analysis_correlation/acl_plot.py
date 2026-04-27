import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def plot_find_wikiid_ratio_for_hallucination_data():
    # 数据
    data = {
        'Movies': [78.2, 92.4, 66.0],
        'Songs': [91.8, 91.2, 92.2],
        'Basketball': [99.2, 99.8, 99.7]
    }

    # 模型名称
    models = ['Llama3-8B', 'Qwen2-7B', 'ChatGPT']

    # 数据集名称
    datasets = list(data.keys())

    # 颜色设置（每个模型一个颜色）
    colors = ['#E68F8B', '#F4E896', '#A7E0D4']

    # 设置柱状图的位置
    num_datasets = len(datasets)
    num_models = len(models)
    bar_width = 0.02  # 每个数据集内柱子宽度（更窄）
    gap_width = 0.05 # 数据集之间的间隔

    # 计算每个数据集的起始位置
    x = np.arange(0, num_datasets * (num_models * bar_width + gap_width), num_models * bar_width + gap_width)

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(models):
        # 提取每个模型在所有数据集上的结果
        model_scores = [data[dataset][i] for dataset in datasets]
        # 绘制柱状图，每个数据集内的柱子没有间隔
        ax.bar(x + i * bar_width, model_scores, width=bar_width, color=colors[i], label=model, edgecolor='black', linewidth=0.5)

    # 添加标签、标题和图例
    ax.set_xlabel('Datasets', fontsize=12)
    ax.set_ylabel('Ratio (%)', fontsize=12)
    ax.set_title('Proportion of Model-Hallucinated Entities Found in Wikidata', fontsize=14, pad=20)
    ax.set_xticks(x + (num_models - 1) * bar_width / 2)  # 将x轴标签放在每个数据集的中间
    ax.set_xticklabels(datasets, fontsize=12)
    ax.legend(fontsize=10, title_fontsize=12)

    # 设置y轴刻度为虚线
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)

    # 设置y轴范围
    ax.set_ylim(0, 100)

    # 设置边框不可见
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 显示图形
    plt.tight_layout()
    plt.show()

def plot_pop_acc_conf_honesty(span_pop, span_acc, span_conf, align, spearman_acc_pop, spearman_conf_pop, spearman_gap_pop):
    span_pop=np.array(span_pop)
    span_acc=np.array(span_acc)
    span_conf=np.array(span_conf)
    align=np.array(align)

    # 设置图像大小和分辨率
    plt.figure(figsize=(8, 6))

    colors = {
    "accuracy": "#4a90e2",  # 深饱和蓝色
    "confidence": "#50c878",  # 深饱和绿色
    "gap": "#e27d78",  # 深饱和橙色
    }

    # 绘制曲线和数据点
    plt.ylim(0, 1)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.plot(span_pop, span_acc, label='Accuracy', color=colors["accuracy"], linestyle='-', linewidth=2, marker='o', markersize=6)
    plt.plot(span_pop, span_conf, label='Confidence', color=colors["confidence"], linestyle='-', linewidth=2, marker='^', markersize=6)
    plt.plot(span_pop, align, label='Gap', color=colors["gap"], linestyle='-', linewidth=2, marker='s', markersize=6)


    # 在合适的位置标注 Spearman 系数
    plt.text(0.7, 0.85, f"ρ(Accuracy) = {spearman_acc_pop:.2f}",
            transform=plt.gca().transAxes, fontsize=12, color=colors["accuracy"], ha='left', va='center')
    plt.text(0.7, 0.78, f"ρ(Confidence) = {spearman_conf_pop:.2f}",
            transform=plt.gca().transAxes, fontsize=12, color=colors["confidence"], ha='left', va='center')
    plt.text(0.7, 0.71, f"ρ(Gap) = {spearman_gap_pop:.2f}",
            transform=plt.gca().transAxes, fontsize=12, color=colors["gap"], ha='left', va='center')

    # 添加标题和坐标轴标签
    plt.title(f'{model_name_capital[model]} on {dataset_name_capital[dataset]}', fontsize=16, fontweight='bold')
    plt.xlabel('Question Popularity', fontsize=14, fontname='Times New Roman')
    plt.ylabel('Values', fontsize=14, fontname='Times New Roman')

    # 添加网格
    plt.grid(alpha=0.5, linestyle='--')

    # 添加图例
    plt.legend(fontsize=12, loc='best', frameon=True)

    # 调整刻度字体大小和样式
    plt.xticks(fontsize=12, fontname='Times New Roman')
    plt.yticks(fontsize=12, fontname='Times New Roman')

    # 提高边距布局
    plt.tight_layout()

    # 显示图表
    plt.show()

def plot_find_wikiid_ratio_for_hallucination_data_with_hatch():
    # 数据
    co_occ_lt_0 = {
        'Movies': [48.1, 55.7, 68.9],
        'Songs': [52.9, 49.4, 68.7],
        'Basketball': [60.4, 68.2, 63.4]
    }
    # found_in_wiki = {
    #     'Movies': [78.2, 92.4, 66.0],
    #     'Songs': [91.8, 91.2, 92.2],
    #     'Basketball': [99.2, 99.8, 99.7]
    # }
    co_occ_lt_gt = {
        'Movies': [10.2, 9.50, 27.2],
        'Songs': [16.2, 10.0, 37.4],
        'Basketball': [33.2, 42.3, 38.5]
    }
    
    # 模型名称
    models = ['Llama3-8B', 'Qwen2-7B', 'ChatGPT']

    # 数据集名称
    datasets = list(co_occ_lt_0.keys())

    # 颜色设置（每个模型一个颜色）
    colors = ['#6D9EEB', '#DCD3E9', '#FFC002']

    # 设置柱状图的位置
    num_datasets = len(datasets)
    num_models = len(models)
    bar_width = 0.015  # 每个数据集内柱子宽度
    gap_width = 0.08   # 数据集之间的间隔

    # 计算每个数据集的起始位置
    x = np.arange(0, num_datasets * (num_models * bar_width + gap_width), num_models * bar_width + gap_width)
    
    plt.rcParams['hatch.linewidth'] = 0.4
    fig, ax = plt.subplots(figsize=(10, 6))

    # **第一层：绘制 co-occ > 0 的柱子**
    for i, model in enumerate(models):
        model_scores = [co_occ_lt_gt[dataset][i] for dataset in datasets]
        ax.bar(x + i * bar_width, model_scores, width=bar_width, color=colors[i], 
               edgecolor='black', linewidth=0.5)

    # **第二层：绘制 found_in_wiki 的柱子**
    for i, model in enumerate(models):
        model_scores = [co_occ_lt_0[dataset][i] for dataset in datasets]
        ax.bar(x + (i + num_models) * bar_width, model_scores, width=bar_width, color=colors[i],
               edgecolor='black', linewidth=0.5)

    # **第三层：绘制带 hatch 的透明柱子**
    for i, model in enumerate(models):
        model_scores = [co_occ_lt_0[dataset][i] for dataset in datasets]
        ax.bar(x + (i + num_models) * bar_width, model_scores, width=bar_width, facecolor='none',
               edgecolor='gray', hatch='///')

    # **修改图例**
    legend_handles = []
    # 颜色图例（模型）
    for i, model in enumerate(models):
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=colors[i], label=model, hatch='///'))
    legend_handles.append(Patch(facecolor='white', edgecolor='black', label="R(Q,G)>R(Q,GT)", hatch=''))
    legend_handles.append(Patch(facecolor='white', edgecolor='black', label="R(Q,G) > 0", hatch='///'))




    ax.legend(handles=legend_handles, fontsize=10, title="Legend", title_fontsize=12)

    # 添加标签、标题
    plt.ylim(0, 85)
    ax.set_xlabel('Datasets', fontsize=12)
    ax.set_ylabel('Ratio (%)', fontsize=12)
    ax.set_xticks(x + (2*num_models - 1) * bar_width / 2)
    ax.set_xticklabels(datasets, fontsize=12)

    # 设置y轴刻度为虚线
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)

    # 移除边框
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_wrong_gt_gene_right_gene_pop():
    # 数据
    gt_pop = {
        'Movies': [27.305, 29.568, 27.308],
        'Songs': [57.977, 57.557, 47.223],
        'Basketball': [102.294, 101.384, 98.138]
    }

    gene_pop = {
        'Movies': [36.123, 40.915, 31.12],
        'Songs': [50.092, 66.698, 53.91],
        'Basketball': [154.811, 189.06, 172.469]
    }

    right_gene_pop = {
        'Movies': [41.012, 47.305, 36.661],
        'Songs': [72.773, 85.228, 74.472],
        'Basketball': [178.91, 212.229, 138.498]
    }

    # 模型名称
    models = ['Llama3-8B', 'Qwen2-7B', 'ChatGPT']

    # 数据集名称
    datasets = list(gt_pop.keys())

    # 颜色设置（每个模型一个颜色）
    colors = ['#6D9EEB', '#DCD3E9', '#FFC002']

    # 设置柱状图的位置
    num_datasets = len(datasets)
    num_models = len(models)
    bar_width = 0.01  # 每个数据集内柱子宽度（更窄）
    gap_width = 0.08   # 数据集之间的间隔

    # 计算每个数据集的起始位置
    x = np.arange(0, num_datasets * (num_models * bar_width + gap_width), num_models * bar_width + gap_width)
    plt.rcParams['hatch.linewidth'] = 0.4
    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(10, 6))

    # **第一层：绘制普通柱子**
    for i, model in enumerate(models):
        model_scores = [gt_pop[dataset][i] for dataset in datasets]
        ax.bar(x + i * bar_width, model_scores, width=bar_width, color=colors[i], edgecolor='black', linewidth=0.5)

    for i, model in enumerate(models):
        model_scores = [gene_pop[dataset][i] for dataset in datasets]
        ax.bar(x + (i + num_models) * bar_width, model_scores, width=bar_width, color=colors[i], edgecolor='black', linewidth=0.5)

    for i, model in enumerate(models):
        model_scores = [right_gene_pop[dataset][i] for dataset in datasets]
        ax.bar(x + (i + 2*num_models) * bar_width, model_scores, width=bar_width, color=colors[i], edgecolor='black', linewidth=0.5)

    # **第二层：绘制透明柱子 + 灰色 hatch**
    for i, model in enumerate(models):
        model_scores = [gene_pop[dataset][i] for dataset in datasets]
        ax.bar(x + (i + num_models) * bar_width, model_scores, width=bar_width, facecolor='none',
            edgecolor='gray', hatch='///')

    for i, model in enumerate(models):
        model_scores = [right_gene_pop[dataset][i] for dataset in datasets]
        ax.bar(x + (i + 2*num_models) * bar_width, model_scores, width=bar_width, facecolor='none',
            edgecolor='gray', hatch='\\\\\\')

    # **修改图例**
    legend_handles = []
    # 颜色图例（模型）
    for i, model in enumerate(models):
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=colors[i], label=model))

    # 添加 hatch 图例
    legend_handles.append(Patch(facecolor='white', edgecolor='black', label="Acc=0, GT Pop", hatch=''))
    legend_handles.append(Patch(facecolor='white', edgecolor='black', label="Acc=0, Gene Pop", hatch='///'))
    legend_handles.append(Patch(facecolor='white', edgecolor='black', label="Acc=1, Gene Pop", hatch='\\\\\\'))

    # 设置图例
    ax.legend(handles=legend_handles, fontsize=10, title="Legend", title_fontsize=12, loc='upper left')

    # 添加标签、标题
    ax.set_xlabel('Datasets', fontsize=12)
    ax.set_ylabel('Popularity', fontsize=12)
    ax.set_xticks(x + (3*num_models - 1) * bar_width / 2)  # 将x轴标签放在每个数据集的中间
    ax.set_xticklabels(datasets, fontsize=12)

    # 设置y轴刻度为虚线
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)

    # 设置边框不可见
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 显示图形
    plt.tight_layout()
    plt.show()

def plot_tank(data1, data2):
    def remove_outliers(data):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return [x for x in data if lower <= x <= upper]

    # 去除离群点后的数据
    clean_data1 = remove_outliers(data1)
    clean_data2 = remove_outliers(data2)

    # 画图
    plt.boxplot([clean_data1, clean_data2], labels=['List 1', 'List 2'])
    plt.title('Boxplot After Removing Outliers')
    plt.ylabel('Values')
    plt.grid(True)
    plt.show()

# def plot_wrong_gt_gene_right_gene_pop():
#     # 数据
#     gt_pop = {
#         'Movies': [27.305, 29.568, 27.308],
#         'Songs': [57.977, 57.557, 47.223],
#         'Basketball': [102.294, 101.384, 98.138]
#     }

#     gene_pop = {
#         'Movies': [36.123, 40.915, 31.12],
#         'Songs': [50.092, 66.698, 53.91],
#         'Basketball': [154.811, 189.06, 172.469]
#     }

#     right_gene_pop = {
#         'Movies': [41.012, 47.305, 36.661],
#         'Songs': [72.773, 85.228, 74.472],
#         'Basketball': [178.91, 212.229, 138.498]
#     }

#     # 模型名称
#     models = ['Llama3-8B', 'Qwen2-7B', 'ChatGPT']
#     datasets = list(gt_pop.keys())
    
#     # 颜色和hatch设置
#     colors = ['#6D9EEB', '#DCD3E9', '#FFC002']
#     hatches = ['', '///', '\\\\\\']
#     pop_types = ['GT Pop', 'Gene Pop (Acc=0)', 'Gene Pop (Acc=1)']
    
#     # 布局参数
#     bar_width = 0.2       # 单个柱子宽度
#     group_width = 0.6     # 每组（三个hatch）总宽度
#     dataset_gap = 0.4     # 数据集之间的间隔
    
#     # 创建坐标轴
#     fig, ax = plt.subplots(figsize=(12, 7))
    
#     # 生成x轴位置
#     x_base = 0
#     x_ticks = []
#     x_labels = []
    
#     for dataset_idx, dataset in enumerate(datasets):
#         # 每组数据的起始位置
#         x_start = x_base + dataset_idx * (len(pop_types)*len(models)*bar_width + dataset_gap)
        
#         # 绘制每个模型的三根柱子
#         for model_idx, model in enumerate(models):
#             # 计算当前模型的x位置
#             model_x = x_start + model_idx * (len(pop_types)*bar_width)
            
#             # 绘制三种pop类型
#             for pop_idx, (pop_data, hatch) in enumerate(zip(
#                 [gt_pop, gene_pop, right_gene_pop], 
#                 hatches
#             )):
#                 value = pop_data[dataset][model_idx]
#                 rect = ax.bar(
#                     x=model_x + pop_idx*bar_width,
#                     height=value,
#                     width=bar_width,
#                     color=colors[model_idx],
#                     edgecolor='black',
#                     linewidth=0.5,
#                     hatch=hatch,
#                     zorder=2
#                 )
                
#         # 记录坐标轴位置
#         x_ticks.append(x_start + group_width/2 - bar_width)
#         x_labels.append(dataset)
#         x_base = x_start + group_width - bar_width*2  # 更新基准位置

#     # 设置坐标轴
#     ax.set_xticks(x_ticks)
#     ax.set_xticklabels(x_labels, fontsize=12)
#     ax.set_ylabel('Popularity', fontsize=12)
#     ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    
#     # 创建复合图例
#     legend_handles = []
    
#     # 模型颜色图例
#     for model_idx, model in enumerate(models):
#         legend_handles.append(
#             Patch(facecolor=colors[model_idx], edgecolor='black', label=model)
#         )
    
#     # Pop类型hatch图例
#     for pop_idx, pop_type in enumerate(pop_types):
#         legend_handles.append(
#             Patch(facecolor='white', edgecolor='black', 
#                  hatch=hatches[pop_idx], label=pop_type)
#         )
    
#     # 添加图例
#     ax.legend(
#         handles=legend_handles, 
#         ncol=2, 
#         fontsize=10,
#         title="Legend:",
#         title_fontsize=12,
#         loc='upper left',
#         framealpha=0.9
#     )
    
#     # 美化布局
#     plt.tight_layout()
#     for spine in ax.spines.values():
#         spine.set_visible(False)
#     plt.show()

def plot_NMI():
    # 数据
    categories = ['Acc.', 'R-Pop', 'P(R|Q)', 'P(R|A)', 'NMI']
    # values_movies = [94.23, 15.41, 0.162, 0.107, 0.034] 
    # values_songs = [69.57, 23.35, 0.457, 0.080, 0.032]  
    # values_basketball = [34.96, 5.01, 0.204, 0.002, 0.002]

    values_movies = [94.23, 15.41, 0.033, 0.047, 0.034]
    values_songs = [69.57, 23.35, 0.061, 0.020, 0.032]  
    values_basketball = [34.96, 5.0, 0.075, 0.00013, 0.002]

    # 指标的最大刻度值（用于归一化）
    # scales = [100, 25, 0.5, 0.12, 0.035]
    scales = [100, 25, 0.1, 0.05, 0.035]

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图

    # 归一化数据
    values_movies_scaled = [v / s for v, s in zip(values_movies, scales)] + [values_movies[0] / scales[0]]
    values_songs_scaled = [v / s for v, s in zip(values_songs, scales)] + [values_songs[0] / scales[0]]
    values_basketball_scaled = [v / s for v, s in zip(values_basketball, scales)] + [values_basketball[0] / scales[0]]

    # 配色方案
    colors = {
        "Movies": "#1f77b4",  # 蓝色
        "Songs": "#ff7f0e",   # 橙色
        "Basketball": "#d62728"  # 红色
    }

    # 画雷达图
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values_movies_scaled, color=colors["Movies"], alpha=0.3, label="Movies")
    ax.fill(angles, values_songs_scaled, color=colors["Songs"], alpha=0.3, label="Songs")
    ax.fill(angles, values_basketball_scaled, color=colors["Basketball"], alpha=0.3, label="Basketball")
    
    ax.plot(angles, values_movies_scaled, color=colors["Movies"], linewidth=2)
    ax.plot(angles, values_songs_scaled, color=colors["Songs"], linewidth=2)
    ax.plot(angles, values_basketball_scaled, color=colors["Basketball"], linewidth=2)

    # 设置轴标签，并向外偏移
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        label.set_horizontalalignment('center')
        label.set_y(0.01)  # 调整标签位置，远离中心

    # 去除内部刻度标签
    ax.set_yticklabels([])

    # 调整数值标注的偏移量
    movies_offsets = [-0.15, 0.07, -0.05, 0.02, 0.03]  # 适配不同数值的偏移
    songs_offsets = [-0.25, 0.07, 0.05, 0.02, -0.1]  # 适配不同数值的偏移
    basketball_offsets = [0.1, 0.07, -0.02, 0.02, -0.1]  # 适配不同数值的偏移
    for i, angle in enumerate(angles[:-1]):
        ha = 'left' if angle < np.pi / 2 or angle > 3 * np.pi / 2 else 'right'
        ax.text(angle, values_movies_scaled[i] + movies_offsets[i], f"{values_movies[i]:.3g}",
                ha=ha, color=colors["Movies"], fontsize=10, fontweight='bold', 
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        ax.text(angle, values_songs_scaled[i] + songs_offsets[i] / 2, f"{values_songs[i]:.3g}",
                ha=ha, color=colors["Songs"], fontsize=10, fontweight='bold', 
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        ax.text(angle, values_basketball_scaled[i] - basketball_offsets[i] / 2, f"{values_basketball[i]:.3g}",
                ha=ha, color=colors["Basketball"], fontsize=10, fontweight='bold', 
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # 美化雷达图
    ax.spines['polar'].set_visible(False)  # 去掉外框
    ax.grid(color='gray', linestyle='dotted', linewidth=0.7)  # 设置网格样式

    # 添加图例
    plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1), fontsize=10, frameon=False)

    # 显示图像
    plt.show()

def plot_line_between_pop_and_acc_conf_align(span_pop, span_acc, span_conf, align, 
                                             spearman_acc_pop, spearman_conf_pop, spearman_gap_pop,
                                             model, dataset, y_lim_bottom, y_lim_upper):
    model_name_capital = {
        'llama8b': 'Llama3-8B',
        'qwen2': 'Qwen2-7B',
        'chatgpt': 'ChatGPT',
        'Qwen2.5-7B': 'Qwen2.5-7B',
        'Qwen2.5-14B': 'Qwen2.5-14B',
        'Qwen2.5-32B': 'Qwen2.5-32B'
    }

    dataset_name_capital = {
        'movies': 'Movies',
        'songs': 'Songs',
        'basketball': 'Basketball'
    }
    span_pop=np.array(span_pop)
    span_acc=np.array(span_acc)
    span_conf=np.array(span_conf)
    align=np.array(align)

    # 设置图像大小和分辨率
    plt.figure(figsize=(8, 6))

    colors = {
    "accuracy": "#4a90e2",  # 深饱和蓝色
    "confidence": "#50c878",  # 深饱和绿色
    "gap": "#e27d78",  # 深饱和橙色
    }

    # 绘制曲线和数据点
    plt.ylim(y_lim_bottom, y_lim_upper)
    # plt.yticks(y_ticks)
    plt.plot(span_pop, span_acc, label='Accuracy', color=colors["accuracy"], linestyle='-', linewidth=2, marker='o', markersize=10)
    plt.plot(span_pop, span_conf, label='Confidence', color=colors["confidence"], linestyle='-', linewidth=2, marker='^', markersize=10)
    plt.plot(span_pop, align, label='Alignment', color=colors["gap"], linestyle='-', linewidth=2, marker='s', markersize=10)


    # 在合适的位置标注 Spearman 系数
    # plt.text(0.7, 0.85, f"ρ(Accuracy) = {spearman_acc_pop:.2f}",
    #         transform=plt.gca().transAxes, fontsize=12, color=colors["accuracy"], ha='left', va='center')
    # plt.text(0.7, 0.78, f"ρ(Confidence) = {spearman_conf_pop:.2f}",
    #         transform=plt.gca().transAxes, fontsize=12, color=colors["confidence"], ha='left', va='center')
    # plt.text(0.7, 0.71, f"ρ(Honesty) = {spearman_gap_pop:.2f}",
    #         transform=plt.gca().transAxes, fontsize=12, color=colors["gap"], ha='left', va='center')

    # 添加标题和坐标轴标签
    plt.title(f'{model_name_capital[model]} on {dataset_name_capital[dataset]}', fontsize=20, fontweight='bold')
    plt.xlabel('Question Popularity', fontsize=24, fontname='Times New Roman')
    plt.ylabel('Values', fontsize=24, fontname='Times New Roman')

    # 添加网格
    plt.grid(alpha=0.5, linestyle='--')

    # 添加图例
    plt.legend(fontsize=18, loc='best', frameon=True)

    # 调整刻度字体大小和样式
    plt.xticks(fontsize=22, fontname='Times New Roman')
    plt.yticks(fontsize=22, fontname='Times New Roman')

    # 提高边距布局
    plt.tight_layout()

    # 显示图表
    # plt.show()
    out_path=f'/Users/shiyuni/Documents/research/conference/EMNLP2026/knowledge_popularity/figs/question_pop_{model}_{dataset}.png'
    plt.savefig(out_path)

def plot_spearman_gene_entity_acc():
    # 数据
    datasets = ['Movies', 'Songs', 'Basketball']
    models = ['Llama3-8B', 'Qwen2-7B', 'ChatGPT']
    g_pop = {
        'Movies': [0.100, 0.087, 0.083],
        'Songs': [0.257, 0.188, 0.218],
        'Basketball': [0.116, 0.116, -0.164]
    }
    co_occ = {
        'Movies': [0.637, 0.756, 0.208],
        'Songs': [0.621, 0.666, 0.351],
        'Basketball': [0.245, 0.106, 0.293]
    }

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(10, 6))

    # 设置宽度
    width = 0.3
    x = np.arange(len(datasets))

    # 为每个模型创建条形图
    for i, model in enumerate(models):
        ax.bar(x - width + i * width, [g_pop[dataset][i] for dataset in datasets], width, label=f'{model} - G-Pop')
        ax.bar(x - width + i * width, [co_occ[dataset][i] for dataset in datasets], width, bottom=[g_pop[dataset][i] for dataset in datasets], label=f'{model} - Co-Occ')

    # 设置标签和标题
    ax.set_xlabel('Datasets')
    ax.set_ylabel('Spearman Correlation Coefficient')
    ax.set_title('Spearman Correlation Coefficients for G-Pop and Co-Occ by Models')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)

    # 设置图例
    ax.legend()

    # 显示图表
    plt.tight_layout()
    plt.show()

def few_shot_prediction_acc():
    # zero_shot = [85.95, 81.4, 95.84, 78.87, 86.07, 80.05, 67.69, 68.08, 78.7]
    # three_shot = [85.55, 82.14, 95.89, 79.42, 85.84, 79.18, 68.64, 67.41, 78.67]
    # five_shot = [86.32, 81.33, 0, 79.42, 84.35, 0, 68.18, 68.1, 0]
    # ten_shot = [86.41, 81.47, 0, 78.01, 83.18, 0, 68.68, 68.49, 0]
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter

    shots = np.array(['0', '3', '5', '10'])
    llama8b = np.array([77.5, 77.87, 77.97, 77.7])
    qwen2 = np.array([78.52, 78.46, 77.93, 77.71])
    chatgpt = np.array([84.86, 84.58, np.nan, np.nan])  # 避免绘制 0.0

    # 自定义 y 轴非线性变换
    def scale_func(y):
        return np.where(y < 79, y, 79 + (y - 79) / 3)  # 79 以上数据压缩

    def inverse_scale_func(y):
        return np.where(y < 79, y, 79 + (y - 79) * 3)  # 逆变换恢复原始值

    # 创建 figure 和 axis
    fig, ax = plt.subplots()

    # 绘制曲线
    ax.plot(shots, llama8b, marker='o', label='LLaMA 8B')
    ax.plot(shots, qwen2, marker='s', label='Qwen2')
    ax.plot(shots[:2], chatgpt[:2], marker='^', label='ChatGPT')  # 只绘制非 nan 部分

    # 手动应用 y 轴非线性缩放
    yticks = np.array([77, 78, 79, 80, 81, 82, 83, 84, 85, 87])  # 增加 87，避免 ChatGPT 贴边
    ax.set_yticks(scale_func(yticks))  # 映射 y 轴刻度
    ax.set_yticklabels([f"{int(inverse_scale_func(y))}" for y in yticks])  # 逆映射显示刻度

    # 设置 y 轴范围，确保 ChatGPT 最高值显示
    ax.set_ylim(scale_func(77), scale_func(103))

    # 添加网格和标签
    ax.set_xlabel("Shots")
    ax.set_ylabel("Prediction Accuracy (%)")
    ax.legend()
    ax.grid(True)

    # 显示图像
    plt.show()


def filter_outliers(data, lower_percentile=10, upper_percentile=90):
    """
    过滤掉指定百分位数之外的离群点
    
    参数:
    data: 原始数据列表/数组
    lower_percentile: 下百分位数 (默认10%)
    upper_percentile: 上百分位数 (默认90%)
    
    返回:
    过滤后的数据
    """
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return data[(data >= lower_bound) & (data <= upper_bound)]

def plot_seaborn_boxplots(data_list, labels=None, title="多组数据箱线图", figsize=(8, 5)):
    """
    改进的紧凑型箱线图 - 更窄更美观
    
    参数:
    data_list: 包含多个数值列表的列表，如 [list1, list2, list3]
    labels: 每个箱线图的标签列表，如 ["组A", "组B", "组C"]
    title: 图表标题
    figsize: 图表尺寸，默认(8,5)
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    # 将数据转换为DataFrame格式
    filtered_data = [filter_outliers(np.array(data)) for data in data_list]
    df = pd.DataFrame()
    for i, data in enumerate(data_list):
        group_name = labels[i] if labels else f"组{i+1}"
        temp_df = pd.DataFrame({'数值': data, '组别': group_name})
        df = pd.concat([df, temp_df])
    
    # 设置风格
    sns.set_style("whitegrid")
    plt.figure(figsize=figsize)
    
    # 绘制更窄的箱线图
    ax = sns.boxplot(
        x='组别', 
        y='数值', 
        data=df, 
        palette="husl",  # 更鲜艳的颜色
        width=0.4,       # 控制箱体宽度
        linewidth=1.5,   # 边框线宽
        fliersize=4      # 离群点大小
    )
    
    # 美化图表
    plt.title(title, fontsize=14, pad=10)
    plt.xlabel("")  # 去掉x轴标签
    plt.ylabel("Popularity", fontsize=14)
    
    # 调整网格线和边框
    ax.grid(True, linestyle=':', alpha=0.6)
    sns.despine(offset=10, trim=True)  # 移除顶部和右侧边框
    
    # 调整标签字体
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()  # 自动调整子图参数
    plt.show()







if __name__ == '__main__':
    # plot_find_wikiid_ratio_for_hallucination_data()
    # plot_find_wikiid_ratio_for_hallucination_data_with_hatch()
    # plot_NMI()
    plot_wrong_gt_gene_right_gene_pop()
    # plot_spearman_gene_entity_acc()
    # plot_find_wikiid_ratio_for_hallucination_data_with_hatch()
    # few_shot_prediction_acc()
