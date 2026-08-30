"""
计算 (RPop_GT, Pop_Q, Pop_GT) 分别与 confidence 和 calibration score 的偏相关系数。

框架与 tab:acc-corr 完全一致：
  三变量互控，每个因子控制其余两个后与目标变量的偏 Spearman 相关系数。

目标变量：
  1. conf = 平均 token probability（样本中已有）
  2. cal = 1 - |acc - conf|（校准分数，越高越校准）

与 acc 表的区别：acc 表用 {coo, question_pop, gt_pop}（知识属性，即 GT 相关），
这里用户要求同样的三变量集 {coo, question_pop, gt_pop}。
"""

import json
import numpy as np
from scipy.stats import pearsonr, rankdata
import os
import sys

# 复用 verify_per_model 的基础设施
sys.path.insert(0, os.path.dirname(__file__))
from verify_per_model import (
    DATASETS, MODELS, load_popularity, collect_samples, _skip_bball_qwen25,
    multivariate_partial, three_var_partial,
    COO_PATH, SINGLE_PATH,
)


def compute_conf_and_cal_partial(all_data):
    """计算 {coo, question_pop, gt_pop} → conf 和 → cal 的三变量互控偏相关"""
    
    ACC_FACTORS = ['coo', 'question_pop', 'gt_pop']
    
    # ── Table 1: → conf ──
    print("\n" + "=" * 100)
    print("Table: 三变量互控偏相关 — {gt_coo, qpop, gt_pop} → confidence")
    print("  每个因子控制其余两个 | 变量集与 acc 表相同")
    print("=" * 100)
    print(f"{'dataset':12s} {'model':15s} {'n':>5s} | {'RPop_GT→conf':>13s} {'Pop_Q→conf':>12s} {'Pop_GT→conf':>13s}")
    print("-" * 80)
    
    for dataset in DATASETS:
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = all_data.get((dataset, model), [])
            if len(samples) < 100:
                continue
            
            res = three_var_partial(samples, ACC_FACTORS, 'conf')
            
            print(f"{dataset:12s} {model:15s} {len(samples):5d} | "
                  f"{res['coo']:+13.3f} {res['question_pop']:+12.3f} {res['gt_pop']:+13.3f}")
    
    # ── Table 2: → cal (= 1 - |acc - conf|) ──
    # 需要给每个样本加上 'cal' 字段
    print("\n" + "=" * 100)
    print("Table: 三变量互控偏相关 — {gt_coo, qpop, gt_pop} → calibration (1-|acc-conf|)")
    print("  每个因子控制其余两个 | 变量集与 acc 表相同")
    print("=" * 100)
    print(f"{'dataset':12s} {'model':15s} {'n':>5s} | {'RPop_GT→cal':>13s} {'Pop_Q→cal':>12s} {'Pop_GT→cal':>13s}")
    print("-" * 80)
    
    for dataset in DATASETS:
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = all_data.get((dataset, model), [])
            if len(samples) < 100:
                continue
            
            # 给样本加 cal 字段
            for s in samples:
                s['cal'] = 1.0 - abs(s['acc'] - s['conf'])
            
            res = three_var_partial(samples, ACC_FACTORS, 'cal')
            
            print(f"{dataset:12s} {model:15s} {len(samples):5d} | "
                  f"{res['coo']:+13.3f} {res['question_pop']:+12.3f} {res['gt_pop']:+13.3f}")
    
    # 与 acc 表对比（复现 acc 表便于 refer）
    print("\n" + "=" * 100)
    print("Reference: 三变量互控偏相关 — {gt_coo, qpop, gt_pop} → acc (复现)")
    print("=" * 100)
    print(f"{'dataset':12s} {'model':15s} {'n':>5s} | {'RPop_GT→acc':>13s} {'Pop_Q→acc':>12s} {'Pop_GT→acc':>13s}")
    print("-" * 80)
    
    for dataset in DATASETS:
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = all_data.get((dataset, model), [])
            if len(samples) < 100:
                continue
            
            res = three_var_partial(samples, ACC_FACTORS, 'acc')
            
            print(f"{dataset:12s} {model:15s} {len(samples):5d} | "
                  f"{res['coo']:+13.3f} {res['question_pop']:+12.3f} {res['gt_pop']:+13.3f}")


def main():
    print("加载数据...")
    full_dict = load_popularity()
    co_occu = json.loads(open(COO_PATH).read())
    single_occr = json.loads(open(SINGLE_PATH).read())
    
    all_data = {}
    for dataset in DATASETS:
        for model in MODELS:
            samples = collect_samples(dataset, model, full_dict, co_occu, single_occr,
                                      filter_pop_no=True)
            if not _skip_bball_qwen25(dataset, model):
                all_data[(dataset, model)] = samples
    
    compute_conf_and_cal_partial(all_data)


if __name__ == '__main__':
    main()