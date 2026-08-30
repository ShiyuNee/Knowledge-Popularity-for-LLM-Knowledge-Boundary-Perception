#!/bin/bash
# ==============================================================================
# ChatGPT QA 多次采样脚本 (Self-Consistency Baseline Step 1)
# ==============================================================================
#
# 使用 ChatGPT API 对每个问题采样 N 次（默认 10 次）。
# 运行环境: 本地（chatanywhere 可直连，无需代理）
# 使用 asyncio 异步并发加速，--max_concurrent 控制同时飞行的请求数。
#
# 采样结果将用于 self_consistency.py 的 Judge 步骤。
#
# 使用方式:
#   bash run_chatgpt_sampling.sh [num_samples] [api_model] [max_concurrent]
#   Example:
#     bash run_chatgpt_sampling.sh 10 gpt-3.5-turbo-1106 50
#
# 输出文件: ./self_consistency/{dataset}_chatgpt_sampling.jsonl
# ==============================================================================

NUM_SAMPLES=${1:-3}
API_MODEL=${2:-gpt-3.5-turbo-1106}
MAX_CONCURRENT=${3:-5}

echo "=== ChatGPT Sampling: num_samples=$NUM_SAMPLES api_model=$API_MODEL max_concurrent=$MAX_CONCURRENT ==="

for dataset in movies songs basketball
do
    source="../data/${dataset}.jsonl"
    outfile="./self_consistency/${dataset}_chatgpt_sampling.jsonl"

    if [ ! -f "$source" ]; then
        echo "Skipping (source not found): $source"
        continue
    fi

    echo "=== Sampling: dataset=$dataset ==="

    python3 -u chatgpt_sampling.py \
        --source "$source" \
        --outfile "$outfile" \
        --api_model "$API_MODEL" \
        --num_samples $NUM_SAMPLES \
        --temperature 1.0 \
        --max_concurrent $MAX_CONCURRENT
done