for mode in train test dev
do
    python -u run_nq.py \
        --source ./res/nq/llama2-chat-7b/cot/nq_${mode}_llama7b_tokens_cot_mid_layer.jsonl \
        --type qa_extract \
        --ra none \
        --outfile ./res/nq/nq_${mode}_llama7b_extract.jsonl \
        --model_path ../models/Qwen2-7B-Instruct \
        --batch_size 32 \
        --task nq \
        --max_new_tokens 64 
done