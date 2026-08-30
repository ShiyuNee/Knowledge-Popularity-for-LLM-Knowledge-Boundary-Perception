#!/bin/bash
# ==============================================================================
# Verbalized Confidence Baseline 运行脚本
# ==============================================================================
#
# 让模型判断自己生成的答案是否正确:
#   - 回答 "certain" → 置信度 = 1.0
#   - 回答 "uncertain" → 置信度 = 0.0
#
# 分两种运行环境:
#   1. 服务器 (VLLM): Qwen2-7B, Llama3-8B, Qwen2.5-7B/14B/32B
#   2. 本地 (API): ChatGPT (gpt-3.5-turbo-1106)
#
# 使用方式:
#   bash run_verbalized_confidence.sh [model_path_prefix]
#   Example:
#     bash run_verbalized_confidence.sh /models
#
# 注意:
#   - 服务器部分 (Section 1) 需要在有 GPU 的机器上运行
#   - ChatGPT 部分 (Section 2) 需要在能访问代理的本地机器运行
# ==============================================================================

MODEL_PREFIX=${1:-/models}

# ─── 1. 开源模型 (VLLM, 服务器上运行) ───────────────────────
# 格式: model_dir:model_short:model_name
#   model_dir  = 模型文件夹名（用于拼接 model_path）
#   model_short= res/ 文件名中的模型简称
#   model_name = 输出文件名中的小写模型名

VLLM_MODELS=(
    "Qwen2-7B-Instruct:qwen2:qwen2"
    "Meta-Llama-3-8B-Instruct:llama8b:llama3-8b"
    "Qwen2.5-7B-Instruct:Qwen2.5-7B:qwen2.5-7b"
    "Qwen2.5-14B-Instruct:Qwen2.5-14B:qwen2.5-14b"
    "Qwen2.5-32B-Instruct:Qwen2.5-32B:qwen2.5-32b"
)

for model_info in "${VLLM_MODELS[@]}"
do
    IFS=':' read -r model_dir model_short model_name <<< "$model_info"

    for dataset in movies songs basketball
    do
        source="../res/${dataset}/${dataset}_${model_short}_temperature1.jsonl"
        outfile="./verbalized_confidence/${dataset}_${model_name}_vc.jsonl"

        if [ ! -f "$source" ]; then
            echo "Skipping (source not found): $source"
            continue
        fi

        echo "=== VC VLLM: model=$model_dir dataset=$dataset ==="

        python -u verbalized_confidence.py \
            --source "$source" \
            --outfile "$outfile" \
            --mode vllm \
            --model_path "${MODEL_PREFIX}/${model_dir}" \
            --tensor_parallel 2
    done
done

# ─── 2. ChatGPT (API, 本地运行) ──────────────────────────────
# 取消注释以运行。需要本地代理 (127.0.0.1:7890)

# for dataset in movies songs basketball
# do
#     source="../res/${dataset}/${dataset}_chatgpt_temperature1.jsonl"
#     outfile="./verbalized_confidence/${dataset}_chatgpt_vc.jsonl"
#
#     if [ -f "$outfile" ]; then
#         echo "Skipping (already exists): $outfile"
#         continue
#     fi
#
#     echo "=== VC API: ChatGPT dataset=$dataset ==="
#
#     python -u verbalized_confidence.py \
#         --source "$source" \
#         --outfile "$outfile" \
#         --mode api \
#         --api_model gpt-3.5-turbo-1106 \
#         --batch_size 5
# done