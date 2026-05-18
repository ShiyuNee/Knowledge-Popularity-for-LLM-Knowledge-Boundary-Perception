# for model in Qwen2.5-32B-Instruct
# do
#     # 根据 model 设置 model_name
#     if [ "$model" = "Qwen2.5-7B-Instruct" ]; then
#         model_name="Qwen2.5-7B"
#     elif [ "$model" = "Qwen2.5-14B-Instruct" ]; then
#         model_name="Qwen2.5-14B"
#     elif [ "$model" = "Qwen2.5-32B-Instruct" ]; then
#         model_name="Qwen2.5-32B"
#     fi

#     for dataset in movie
#     do
#         python -u run_vllm.py \
#             --source ./data/${dataset}/${dataset}.jsonl \
#             --type qa \
#             --ra none \
#             --outfile ./res/${dataset}/${dataset}_${model_name}_greedy_vllm.jsonl \
#             --model_path ../models/$model \
#             --task nq \
#             --max_new_tokens 64 \
#             --temperature 0 \
#             --tensor_parallel 1
#     done
# done

for model in Qwen2.5-7B-Instruct Qwen2.5-14B-Instruct Qwen2.5-32B-Instruct
do
    # 根据 model 设置 model_name
    if [ "$model" = "Qwen2.5-7B-Instruct" ]; then
        model_name="Qwen2.5-7B"
    elif [ "$model" = "Qwen2.5-14B-Instruct" ]; then
        model_name="Qwen2.5-14B"
    elif [ "$model" = "Qwen2.5-32B-Instruct" ]; then
        model_name="Qwen2.5-32B"
    fi

    for dataset in movie songs basketball
    do
        python -u run_vllm.py \
            --source ./data/${dataset}/${dataset}.jsonl \
            --type qa \
            --ra none \
            --outfile ./res/${dataset}/${dataset}_${model_name}_sampling_vllm.jsonl \
            --model_path ../models/$model \
            --task nq \
            --max_new_tokens 64 \
            --sampling \
            --temperature 1.0 \
            --num_return_sequences 10 \
            --tensor_parallel 2
    done
done