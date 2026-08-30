#!/bin/bash
# ==============================================================================
# ChatGPT: Self-Consistency 采样 + Verbalized Confidence 自动链式执行
# ==============================================================================
#
# Step 1: 对每个问题采样 N 次（默认 3 次），用于 Self-Consistency baseline
# Step 2: 对每个问题让 ChatGPT 判断置信度（certain/uncertain），用于 Verbalized Confidence baseline
#
# 已有结果自动跳过（断点续跑），不会重复运行。
#
# 使用方式:
#   bash run_chatgpt_all.sh [num_samples] [api_model] [max_concurrent]
#   Example:
#     bash run_chatgpt_all.sh 3 gpt-3.5-turbo-1106 5
#
# 输出文件:
#   Step 1: ./self_consistency/{dataset}_chatgpt_sampling.jsonl
#   Step 2: ./verbalized_confidence/{dataset}_chatgpt_vc.jsonl
# ==============================================================================

NUM_SAMPLES=${1:-3}
API_MODEL=${2:-gpt-3.5-turbo-1106}
MAX_CONCURRENT=${3:-5}

echo "======================================================================"
echo "  ChatGPT All Baselines"
echo "  num_samples=$NUM_SAMPLES  api_model=$API_MODEL  max_concurrent=$MAX_CONCURRENT"
echo "======================================================================"

# ─── Step 1: Self-Consistency 采样 ──────────────────────────────────
echo ""
echo "======================================================================"
echo "  Step 1: ChatGPT Sampling (Self-Consistency)"
echo "======================================================================"

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

echo ""
echo "=== Step 1 Complete ==="

# ─── Step 2: Verbalized Confidence ─────────────────────────────────
echo ""
echo "======================================================================"
echo "  Step 2: ChatGPT Verbalized Confidence"
echo "======================================================================"

for dataset in movies songs basketball
do
    source="../res/${dataset}/${dataset}_chatgpt_temperature1.jsonl"
    outfile="./verbalized_confidence/${dataset}_chatgpt_vc.jsonl"

    if [ ! -f "$source" ]; then
        echo "Skipping (source not found): $source"
        continue
    fi

    echo "=== Verbalized Confidence: dataset=$dataset ==="

    python3 -u verbalized_confidence.py \
        --source "$source" \
        --outfile "$outfile" \
        --mode api \
        --api_model "$API_MODEL" \
        --max_concurrent $MAX_CONCURRENT
done

echo ""
echo "======================================================================"
echo "  All Done! Step 1 (Sampling) + Step 2 (Verbalized Confidence) Complete"
echo "======================================================================"