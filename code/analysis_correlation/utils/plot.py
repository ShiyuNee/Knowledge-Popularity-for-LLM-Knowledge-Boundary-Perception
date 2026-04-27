import json
import jsonlines
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from utils.utils import *
from utils.compute import *
import string
from matplotlib.pyplot import MultipleLocator
from collections import Counter
import pandas as pd

def read_json(path):
    qa_data = []
    f = open(path, 'r', encoding='utf-8')
    for line in f.readlines():
        qa_data.append(json.loads(line))
    return qa_data

def write_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f'write jsonl to: {path}')
    f.close()

def plot_confidence_sample_ratio(data):
    # 分区
    bins = np.arange(0, 1.1, 0.1)
    indices = np.digitize(data, bins) - 1 # 0 到 10
    counts = Counter(indices)

    # 计算每个区间的数据占比
    total_count = len(data)
    percentages = [counts[i] / total_count * 100 for i in range(len(bins) - 1)]

    # 绘制柱状图
    interval_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
    plt.figure(figsize=(10, 6))
    plt.bar(interval_labels, percentages, color='skyblue', edgecolor='black')

    # 添加标签和标题
    plt.xlabel("Confidence Span")
    plt.ylabel("Ratio (%)")
    plt.title("Data ration for each confidence span")
    plt.xticks(rotation=45)
    plt.tight_layout()  # 调整布局以防止标签重叠
    plt.show()

def plot_confidence_acc(conf_data, acc_data):
    """
    Plots the average accuracy for confidence intervals.

    Parameters:
        conf_data (list or numpy array): List of confidence values (between 0 and 1).
        acc_data (list or numpy array): List of accuracy values (0 or 1).

    Returns:
        None
    """
    # Define 10 intervals between 0 and 1
    conf_data = np.array(conf_data)
    acc_data = np.array(acc_data)
    bins = np.linspace(0, 1, 11)
    
    # Assign each confidence value to a bin
    indices = np.digitize(conf_data, bins) # 在哪个bin中, bin_idx, 1到11
    print(indices)
    
    # Calculate average accuracy for each bin
    average_acc = [
        acc_data[indices == i].mean() if np.sum(indices == i) > 0 else 0
        for i in range(1, len(bins))
    ]
    
    # Calculate bin centers for plotting
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.bar(bin_centers, average_acc, width=0.09, edgecolor="black", alpha=0.7)
    plt.xlabel("Confidence Intervals")
    plt.ylabel("Average Accuracy")
    plt.title("Average Accuracy per Confidence Interval")
    plt.xticks(bin_centers, labels=[f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)])
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.show()

def compute_average_acc_fixed_intervals(popularity_list, acc_list, num_intervals=10):
    """
    Divides the popularity list into fixed-size intervals and calculates the average acc for each interval.

    Parameters:
        popularity_list (list or numpy array): List of popularity values.
        acc_list (list or numpy array): List of accuracy values (0 or 1).
        num_intervals (int): Number of intervals to divide the popularity range into.

    Returns:
        tuple: (interval_bounds, average_acc) where
            - interval_bounds: List of tuples representing the intervals.
            - average_acc: List of average accuracies for each interval.
    """
    # Define equal-sized intervals based on the range of popularity values
    min_val, max_val = np.min(popularity_list), np.max(popularity_list)
    bins = np.linspace(min_val, max_val, num_intervals + 1) # num_intervals+1个边界值, num_intervals个区间
  

    # Assign each popularity value to a bin
    indices = np.digitize(popularity_list, bins, right=False) # 从1开始
    print([np.sum(indices == i) for i in range(1, len(bins))])
    # Calculate average accuracy for each bin
    average_acc = [
        acc_list[indices == i].mean() if np.sum(indices == i) > 0 else 0
        for i in range(1, len(bins))
    ]

    # Create interval bounds
    interval_bounds = [(bins[i], bins[i+1]) for i in range(len(bins) - 1)]

    return interval_bounds, average_acc

def plot_popularity_acc_fixed_intervals(popularity_list, acc_list, num_intervals=5):
    """
    Plots the average accuracy for popularity intervals with fixed-size ranges.

    Parameters:
        popularity_list (list or numpy array): List of popularity values.
        acc_list (list or numpy array): List of accuracy values (0 or 1).
        num_intervals (int): Number of intervals to divide the popularity range into.

    Returns:
        None
    """
    # Compute average accuracy for each interval
    popularity_list = np.array(popularity_list)
    acc_list = np.array(acc_list)
    interval_bounds, average_acc = compute_average_acc_fixed_intervals(popularity_list, acc_list, num_intervals)
    
    # Create labels for each interval
    interval_labels = [
        f"{bounds[0]:.2f}-{bounds[1]:.2f}" for bounds in interval_bounds
    ]
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_intervals), average_acc, tick_label=interval_labels, width=0.8, alpha=0.7, edgecolor="black")
    plt.xlabel("Popularity Intervals")
    plt.ylabel("Average Risk")
    plt.title("Average Risk per Fixed Popularity Interval")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

def compute_average_acc(popularity_list, acc_list, num_intervals=10):
    """
    Divides the popularity list into intervals (fixed sample count) and calculates the average acc for each interval.

    Parameters:
        popularity_list (list or numpy array): List of popularity values.
        acc_list (list or numpy array): List of accuracy values (0 or 1).
        num_intervals (int): Number of intervals to divide the popularity list into.

    Returns:
        tuple: (interval_bounds, average_acc) where
            - interval_bounds: List of tuples representing the intervals.
            - average_acc: List of average accuracies for each interval.
    """
    # Sort the popularity list and divide it into intervals
    sorted_indices = np.argsort(popularity_list) # 返回排序后的索引
    sorted_popularity = np.array(popularity_list)[sorted_indices]
    sorted_acc = np.array(acc_list)[sorted_indices]

    # Divide into intervals
    interval_size = len(popularity_list) // num_intervals
    intervals = [
        sorted_popularity[i * interval_size : (i + 1) * interval_size]
        for i in range(num_intervals)
    ]

    # Calculate average accuracy for each interval
    average_acc = [
        sorted_acc[i * interval_size : (i + 1) * interval_size].mean()
        for i in range(num_intervals)
    ]

    # Get the interval bounds based on popularity
    interval_bounds = [
        (interval[0], interval[-1]) if len(interval) > 0 else (None, None)
        for interval in intervals
    ]

    return interval_bounds, average_acc, intervals

def plot_popularity_acc(popularity_list, acc_list, num_intervals=5, name='Acc'):
    """
    Plots the average accuracy for popularity intervals(fixed sample count).

    Parameters:
        popularity_list (list or numpy array): List of popularity values.
        acc_list (list or numpy array): List of accuracy values (0 or 1).
        num_intervals (int): Number of intervals to divide the popularity list into.

    Returns:
        None
    """
    # Compute average accuracy for each interval
    interval_bounds, average_acc, intervals = compute_average_acc(popularity_list, acc_list, num_intervals)
    
    # Create labels for each interval
    interval_labels = [
        f"{bounds[0]:.2f}-{bounds[1]:.2f}" if bounds[0] is not None else "Empty"
        for bounds in interval_bounds
    ]
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_intervals), average_acc, tick_label=interval_labels, width=0.8, alpha=0.7, edgecolor="black")
    plt.xlabel("GT Popularity Intervals")
    plt.ylabel(f"Average {name}")
    plt.title(f"Average {name} per Fixed Sample Interval")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
    return intervals

def plot_line(x, y, num_intervals = 5):
    interval_size = len(x) // num_intervals

    # 初始化均值列表
    x_means = []
    y_means = []

    # 遍历每个区间计算均值
    for i in range(num_intervals):
        start = i * interval_size
        end = start + interval_size if i < num_intervals - 1 else len(x)
        x_means.append(np.mean(x[start:end]))
        y_means.append(np.mean(y[start:end]))

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(x_means, y_means, marker='o', linestyle='-', color='b', label='Mean Values per Interval')

    # 添加 y=x 的虚线
    # max_val = max(max(x_means), max(y_means))
    # plt.plot([0, max_val], [0, max_val], linestyle='--', color='gray', label='y = x')

    # 标注每个点的坐标
    for x_mean, y_mean in zip(x_means, y_means):
        plt.text(x_mean, y_mean, f'({x_mean:.1f}, {y_mean:.1f})', fontsize=9, ha='center', va='bottom')

    # 设置标题和标签
    plt.title('Sorted Mean question Popularity and Corresponding Gene Popularoty in 10 (Fixed Sample Count) Intervals')
    plt.xlabel('Sorted Mean Question Popularity (Top k%)')
    plt.ylabel('Corresponding Gene Popularity')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_scaler(popularity, confidence, acc):
    # 定义颜色：红色代表 acc=0，绿色代表 acc=1
    filter_mask = np.array(confidence) >= 0.8

# 筛选数据
    confidence = np.array(confidence)[filter_mask]
    popularity = np.array(popularity)[filter_mask]
    acc = np.array(acc)[filter_mask]

    colors = ['red' if a == 0 else 'green' for a in acc]


    # 绘制散点图
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(popularity, confidence, s=5, alpha=0.5, c=colors)

    # 添加图例：创建隐藏的点用于图例标注
    plt.scatter([], [], color='red', label='Acc = 0')  # 红色代表 acc=0
    plt.scatter([], [], color='green', label='Acc = 1')  # 绿色代表 acc=1
    plt.legend(fontsize=12, title='Accuracy', title_fontsize=14)  # 添加图例

    # 添加标题和标签
    plt.title('Scatter Plot of Popularity vs Confidence', fontsize=16)
    plt.xlabel('Popularity', fontsize=14)
    plt.ylabel('Confidence', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 显示图像
    plt.show()

def plot_popularity_for_acc_in_confidence_interval(confidence, popularity, acc):
    """
    统计相同confidence水平下, 做错的部分生成entity的popularity水平和做对部分的popularity水平
    """
    # # 转换为DataFrame
    data = pd.DataFrame({"confidence": confidence, "acc": acc, "popularity": popularity})

    # 按confidence分成10等份
    data['confidence_bin'] = pd.qcut(data['confidence'], q=10, duplicates='drop')

    # 计算每个等份中acc=0和acc=1的平均popularity
    avg_popularity = data.groupby(['confidence_bin', 'acc'])['popularity'].mean().unstack()

    # 计算每个等份中acc=0和acc=1的样本数量
    counts = data.groupby(['confidence_bin', 'acc']).size().unstack()

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(avg_popularity.index.astype(str), avg_popularity[0], label='acc = 0', marker='o')
    plt.plot(avg_popularity.index.astype(str), avg_popularity[1], label='acc = 1', marker='o')

    # 在每个点上标注样本数量
    for i, bin_label in enumerate(avg_popularity.index):
        # 检查acc=0是否存在，并标注
        if not np.isnan(avg_popularity.loc[bin_label, 0]):
            plt.text(i, avg_popularity.loc[bin_label, 0], f'n={counts.loc[bin_label, 0]}', 
                    ha='center', va='bottom', fontsize=10, color='blue')
        # 检查acc=1是否存在，并标注
        if not np.isnan(avg_popularity.loc[bin_label, 1]):
            plt.text(i, avg_popularity.loc[bin_label, 1], f'n={counts.loc[bin_label, 1]}', 
                    ha='center', va='bottom', fontsize=10, color='green')

    # 设置图表标题和标签
    plt.title('Average Popularity by Confidence Bins and Accuracy', fontsize=14)
    plt.xlabel('Confidence Bins', fontsize=12)
    plt.ylabel('Average Popularity', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()

    # 显示图表
    plt.tight_layout()
    plt.show()

    """
    统计相同confidence水平下, 做错的部分生成entity的popularity水平和做对部分的popularity水平，绘制箱型图。
    """
    # 转换为DataFrame
#     data = pd.DataFrame({"confidence": confidence, "acc": acc, "popularity": popularity})

#     # 按confidence分成10等份
#     data['confidence_bin'] = pd.qcut(data['confidence'], q=10, duplicates='drop')

#     # 创建图形
#     plt.figure(figsize=(12, 6))
    
#     # 分别绘制 acc = 0 和 acc = 1 的箱型图
#     for acc_value in [0, 1]:
#         subset = data[data['acc'] == acc_value]
        
#         # 按 confidence_bin 分组，绘制箱型图
#         popularity_by_bin = [subset[subset['confidence_bin'] == bin_label]['popularity'] for bin_label in subset['confidence_bin'].unique()]
        
#         # 绘制箱型图
#         plt.boxplot(popularity_by_bin, positions=np.arange(len(popularity_by_bin)) + (acc_value - 0.5) * 0.2, 
#                     widths=0.2, patch_artist=True, boxprops=dict(facecolor='blue' if acc_value == 0 else 'green', alpha=0.5))
    
#     # 添加图例和标题
#     plt.title('Popularity Distribution by Confidence Bins and Accuracy', fontsize=14)
#     plt.xlabel('Confidence Bins', fontsize=12)
#     plt.ylabel('Popularity', fontsize=12)
#     plt.xticks(ticks=np.arange(10), labels=data['confidence_bin'].cat.categories, rotation=45)
#     plt.legend(['acc = 0', 'acc = 1'])
#     plt.grid()

#     # 显示图表
#     plt.tight_layout()
#     plt.show()

def plot_gap_sample_distribution(data, intervals):
    # 拆分为正数和负数
    positive_data = [x for x in data if x > 0]
    negative_data = [x for x in data if x < 0]

    # 绘制直方图
    plt.hist(negative_data, bins=intervals, color='blue', alpha=0.7, edgecolor='black', label='Negative')
    plt.hist(positive_data, bins=intervals, color='orange', alpha=0.7, edgecolor='black', label='Positive')

    # 添加零值单独显示
    if 0 in data:
        plt.axvline(0, color='gray', linestyle='dashed', linewidth=1, label='Zero')

    # 设置图表标题和标签
    plt.title('Sample Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

    # 显示图表
    plt.show()

def plot_3D(conf, acc, question_pop, ans_pop):
    # 将数据转化为numpy数组，方便计算
    conf = np.array(conf)
    question_pop = np.array(question_pop)
    ans_pop = np.array(ans_pop)
    acc = np.array(acc)

    # 创建一个3D图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制三维散点图，acc用来着色
    sc = ax.scatter(question_pop, ans_pop, conf, c=acc, cmap='coolwarm', s=5)

    # 设置轴标签
    ax.set_xlabel('Question Pop')
    ax.set_ylabel('Ans Pop')
    ax.set_zlabel('Conf')

    # 添加颜色条
    plt.colorbar(sc)

    # 显示图形
    plt.show()