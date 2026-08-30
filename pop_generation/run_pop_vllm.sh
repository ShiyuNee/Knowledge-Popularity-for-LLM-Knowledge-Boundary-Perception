#!/bin/bash
# ==============================================================================
# Popularity Generation for Qwen2.5 models using VLLM
# ==============================================================================
#
# 为 Qwen2.5-7B/14B/32B 生成 LLM 自感知流行度 (1-10)
#
# 优化: 每个模型只初始化一次 VLLM，内部自动遍历 3 数据集 × 3 类型 × 4 shot = 36 个任务
# 总共只需运行 3 次（3 个模型各一次）
#
# 运行前确保已构建 few-shot 数据:
#   python build_clean_data.py
#
# 使用方式:
#   bash run_pop_vllm.sh [model_path_prefix] [source_prefix] [output_prefix]
#   Example:
#     bash run_pop_vllm.sh /models ../res ./llm_pop_generation
#
# 输出路径: {output_prefix}/{dataset}/{dataset}_{model_name}_{gene_type}_pop_{n_shot}.jsonl
# ==============================================================================

MODEL_PREFIX=${1:-../models}
SOURCE_PREFIX=${2:-../res}
OUTPUT_PREFIX=${3:-./llm_pop_generation}

# ─── Step 0: 构建 clean_data（如果尚未生成）─────────────────────
CLEAN_DATA_DIR="./data/clean_data_for_pop_generation"
if [ ! -d "$CLEAN_DATA_DIR" ] || [ -z "$(ls -A $CLEAN_DATA_DIR 2>/dev/null)" ]; then
    echo "=== Building clean_data_for_pop_generation ==="
    python -u build_clean_data.py
else
    echo "=== clean_data already exists, skipping build ==="
fi

# ─── Step 1: 逐模型运行（每个模型只初始化一次 VLLM）───────────────

for model in Qwen2.5-7B-Instruct Qwen2.5-14B-Instruct Qwen2.5-32B-Instruct
do
    echo ""
    echo "=========================================================================="
    echo "  Processing model: $model"
    echo "  (internal: 3 datasets × 3 gene_types × 4 n_shots = 36 tasks)"
    echo "=========================================================================="

    python -u run_pop_vllm.py \
        --model_path "${MODEL_PREFIX}/${model}" \
        --source_prefix "$SOURCE_PREFIX" \
        --output_prefix "$OUTPUT_PREFIX" \
        --temperature 0.0 \
        --tensor_parallel 2

    echo ""
    echo "=== Finished: $model ==="
    echo ""
done