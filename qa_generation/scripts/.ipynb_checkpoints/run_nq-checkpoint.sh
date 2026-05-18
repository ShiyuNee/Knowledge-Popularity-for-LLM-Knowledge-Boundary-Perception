
# python -u run_nq.py \
#     --source ../share/datasets/nq/nq-mini.jsonl \
#     --type qa_cot \
#     --ra none \
#     --outfile ./res/nq/nq_mini_llama3_test.jsonl \
#     --model_path ../models/llama2-7B-chat \
#     --batch_size 36 \
#     --task nq \
#     --max_new_tokens 256 \
#     --hidden_idx_mode first,last,avg \
#     --need_layers mid \

need_layers=mid
for model in llama2-7B-chat llama3_8b_instruct Qwen2-7B-Instruct llama2-13b-chat 
do
    for dataset in nq hq
    do
        for data_type in test
        do
            python -u run_nq.py \
            --source ../share/datasets/${dataset}/${dataset}-${data_type}.jsonl \
            --type qa_prior \
            --ra none \
            --outfile ./res/${dataset}_${data_type}_${model}_tokens_verbalized_conf.jsonl \
            --model_path ../models/$model \
            --batch_size 16 \
            --task nq \
            --max_new_tokens 64 \
            --temperature 1.0 \
            # --hidden_states 1 \
            # --hidden_idx_mode first,last,avg \
            # --need_layers ${need_layers}
        done
    done
done

# for dataset in test dev train
# do
#     python -u run_nq.py \
#         --source ../share/datasets/hq/hq-${dataset}.jsonl \
#         --type qa_more \
#         --ra none \
#         --outfile ./res/hq/hq_${dataset}_llama8b_10_answers.jsonl \
#         --model_path ../models/llama3_8b_instruct \
#         --batch_size 30 \
#         --task nq \
#         --max_new_tokens 128
# done