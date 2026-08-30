#!/bin/bash
# ==============================================================================
# QA Generation for Qwen2.5 models using VLLM
# ==============================================================================
#
# 为 Qwen2.5-7B/14B/32B 生成 QA 结果
# 每个模型只初始化一次 VLLM，内部自动遍历 3 个数据集 (movies/songs/basketball)
#
# 两种运行模式:
#   模式 1 — 贪心 (temperature=0): 每题生成 1 条结果，输出到 ../res/{dataset}/
#   模式 2 — 采样 (temperature=1, n=10): 每题生成 10 条结果，输出到 ../baselines/self_consistency/
#
# 使用方式:
#   # 贪心模式（主实验）
#   bash run_vllm.sh greedy
#
#   # 采样模式（Self-Consistency）
#   bash run_vllm.sh sampling
#
#   # 两者都跑
#   bash run_vllm.sh all
# ==============================================================================

MODEL_PREFIX=${1:-../models}
MODE=${2:-all}  # greedy / sampling / all

DATASETS="movies songs basketball"
DATA_PREFIX="../data"
RES_PREFIX="../res"
SC_PREFIX="../baselines/self_consistency"
TENSOR_PARALLEL=${TENSOR_PARALLEL:-2}

echo "======================================================================"
echo "  QA Generation for Qwen2.5 models"
echo "  MODE: $MODE"
echo "  MODEL_PREFIX: $MODEL_PREFIX"
echo "  TENSOR_PARALLEL: $TENSOR_PARALLEL"
echo "======================================================================"

# ─── Step 1: 贪心模式 (temperature=0) ──────────────────────────────
run_greedy() {
    for model in Qwen2.5-7B-Instruct Qwen2.5-14B-Instruct Qwen2.5-32B-Instruct
    do
        echo ""
        echo "=========================================================================="
        echo "  [GREEDY] Processing model: $model"
        echo "  Output: $RES_PREFIX/{dataset}/"
        echo "=========================================================================="

        python -u run_vllm.py \
            --model_path "${MODEL_PREFIX}/${model}" \
            --data_prefix "$DATA_PREFIX" \
            --output_prefix "$RES_PREFIX" \
            --temperature 0.0 \
            --num_samples 1 \
            --tensor_parallel $TENSOR_PARALLEL

        echo ""
        echo "=== [GREEDY] Finished: $model ==="
        echo ""
    done
}

# ─── Step 2: 采样模式 (temperature=1, n=10) ───────────────────────
run_sampling() {
    for model in Qwen2.5-7B-Instruct Qwen2.5-14B-Instruct Qwen2.5-32B-Instruct
    do
        echo ""
        echo "=========================================================================="
        echo "  [SAMPLING] Processing model: $model"
        echo "  Output: $SC_PREFIX/{dataset}_{model}_sampling.jsonl"
        echo "=========================================================================="

        python -u run_vllm.py \
            --model_path "${MODEL_PREFIX}/${model}" \
            --data_prefix "$DATA_PREFIX" \
            --output_prefix "$SC_PREFIX" \
            --temperature 1.0 \
            --num_samples 10 \
            --tensor_parallel $TENSOR_PARALLEL

        echo ""
        echo "=== [SAMPLING] Finished: $model ==="
        echo ""
    done
}

# ─── 执行 ─────────────────────────────────────────────────────────
case "$MODE" in
    greedy)
        run_greedy
        ;;
    sampling)
        run_sampling
        ;;
    all)
        run_greedy
        run_sampling
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: bash run_vllm.sh [model_prefix] [greedy|sampling|all]"
        exit 1
        ;;
esac

echo ""
echo "======================================================================"
echo "  All tasks completed!"
echo "======================================================================"