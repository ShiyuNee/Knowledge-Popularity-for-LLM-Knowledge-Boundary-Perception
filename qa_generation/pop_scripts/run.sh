for model in Qwen2.5-32B-Instruct
do
    # 根据 model 设置 model_name
    if [ "$model" = "Qwen2.5-7B-Instruct" ]; then
        model_name="Qwen2.5-7B"
    elif [ "$model" = "Qwen2.5-14B-Instruct" ]; then
        model_name="Qwen2.5-14B"
    elif [ "$model" = "Qwen2.5-32B-Instruct" ]; then
        model_name="Qwen2.5-32B"
    fi

    for dataset in movie basketball
    do
        python -u run_nq.py \
            --source ./data/${dataset}/${dataset}.jsonl \
            --type qa \
            --ra none \
            --outfile ./res/${dataset}/${dataset}_${model_name}_greedy.jsonl \
            --model_path ../models/$model \
            --batch_size 64 \
            --task nq \
            --max_new_tokens 64 \
            --temperature 1
    done
done

# 