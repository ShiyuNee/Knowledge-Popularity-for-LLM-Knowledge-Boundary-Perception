#!/bin/bash
# ==============================================================================
# Self-Consistency Baseline: Judge + Compute (Step 2+3)
# ==============================================================================
#
# 使用 Qwen2.5-32B-Instruct (VLLM) 判断每个采样答案与原始答案的一致性，
# 然后计算 Self-Consistency 置信度。
#
# 只加载一次模型，批量处理所有 source:original:outfile 数据对。
#
# 采样文件位置: ./self_consistency/sampling_res/{dataset}/
# 原始答案位置: ../../res/{dataset}/
#
# 使用方式:
#   bash run_self_consistency.sh [model_path_prefix]
#   Example:
#     bash run_self_consistency.sh /models
#
# 注意: 需要在有 GPU 的服务器上运行
# ==============================================================================

MODEL_PREFIX=${1:-/models}
PROJECT_PREFIX=".."
SAMPLING_DIR="./self_consistency/sampling_res"
JUDGE_MODEL="${MODEL_PREFIX}/Qwen2.5-32B-Instruct"
OUTPUT_DIR="./self_consistency/consis_judge_res"
mkdir -p "$OUTPUT_DIR"

# ─── 生成 pairs.jsonl ─────────────────────────────────────────────
# 格式: 每行 {"source": "...", "original": "...", "outfile": "..."}

PAIRS_FILE="./self_consistency/pairs.jsonl"
> "$PAIRS_FILE"

# 定义所有模型+数据集组合
# 格式: "dataset:sampling_filename:original_filename:output_name"

MODELS=(
    # ── movies ──
    "movies:movies_Qwen2.5-7B_sampling:movies_Qwen2.5-7B_temperature1:qwen2.5-7b"
    "movies:movies_Qwen2.5-14B_sampling:movies_Qwen2.5-14B_temperature1:qwen2.5-14b"
    "movies:movies_Qwen2.5-32B_sampling:movies_Qwen2.5-32B_temperature1:qwen2.5-32b"
    "movies:movies_chatgpt_sampling:movies_chatgpt_temperature1:chatgpt"
    "movies:movie_qwen2_temperature1_10:movies_qwen2_temperature1:qwen2"
    "movies:movie_llama8b_temperature1_10:movies_llama8b_temperature1:llama3-8b"

    # ── songs ──
    "songs:songs_Qwen2.5-7B_sampling:songs_Qwen2.5-7B_temperature1:qwen2.5-7b"
    "songs:songs_Qwen2.5-14B_sampling:songs_Qwen2.5-14B_temperature1:qwen2.5-14b"
    "songs:songs_Qwen2.5-32B_sampling:songs_Qwen2.5-32B_temperature1:qwen2.5-32b"
    "songs:songs_chatgpt_sampling:songs_chatgpt_temperature1:chatgpt"
    "songs:songs_qwen2_temperature1_10:songs_qwen2_temperature1:qwen2"
    "songs:songs_llama8b_temperature1_10:songs_llama8b_temperature1:llama3-8b"

    # ── basketball ──
    "basketball:basketball_Qwen2.5-7B_sampling:basketball_Qwen2.5-7B_temperature1:qwen2.5-7b"
    "basketball:basketball_Qwen2.5-14B_sampling:basketball_Qwen2.5-14B_temperature1:qwen2.5-14b"
    "basketball:basketball_Qwen2.5-32B_sampling:basketball_Qwen2.5-32B_temperature1:qwen2.5-32b"
    "basketball:basketball_chatgpt_sampling:basketball_chatgpt_temperature1:chatgpt"
    "basketball:basketball_qwen2_temperature1_10:basketball_qwen2_temperature1:qwen2"
    "basketball:basketball_llama8b_temperature1_10:basketball_llama8b_temperature1:llama3-8b"
)

# 过滤出文件实际存在的数据对
count=0
for entry in "${MODELS[@]}"
do
    IFS=':' read -r dataset sampling_name original_name output_name <<< "$entry"

    sampling_file="${SAMPLING_DIR}/${dataset}/${sampling_name}.jsonl"
    original_file="${PROJECT_PREFIX}/res/${dataset}/${original_name}.jsonl"
    outfile="${OUTPUT_DIR}/${dataset}_${output_name}_sc.jsonl"

    if [ -f "$sampling_file" ] && [ -f "$original_file" ]; then
        # 写入 pairs.jsonl（JSON 格式）
        echo "{\"source\": \"${sampling_file}\", \"original\": \"${original_file}\", \"outfile\": \"${outfile}\"}" >> "$PAIRS_FILE"
        count=$((count + 1))
    else
        [ ! -f "$sampling_file" ] && echo "  SKIP (sampling not found): $sampling_file"
        [ ! -f "$original_file" ] && echo "  SKIP (original not found): $original_file"
    fi
done

echo "======================================================================"
echo "  Self-Consistency Judge + Compute"
echo "  Judge model: $JUDGE_MODEL"
echo "  Valid pairs: $count"
echo "======================================================================"

if [ "$count" -eq 0 ]; then
    echo "No valid pairs found. Exiting."
    exit 1
fi

# ─── 运行批量 Judge + Compute ─────────────────────────────────────
python -u self_consistency.py \
    --pairs_file "$PAIRS_FILE" \
    --judge_model_path "$JUDGE_MODEL" \
    --tensor_parallel 2
