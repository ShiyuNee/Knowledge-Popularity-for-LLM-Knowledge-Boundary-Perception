# python run_mmlu.py --source ../datasets/mmlu/ --type qa --ra none --outfile ./res/mmlu/zero-shot-output-punish/ --n_shot 0 --model_path ../models/llama2-7B-chat --batch_size 1 --task mmlu --with_answer 0 --output_states 1 --hidden_states 1

# deepspeed --num_gpus 1 run_mmlu.py --source ../datasets/mmlu/ --type qa --ra none --outfile ./res/mmlu/zero-shot-hidden/ --n_shot 0 --model_path ../models/llama2-7B-chat --batch_size 2 --task mmlu --max_new_tokens 1 

# type=mc_qa
# task=tq
# source=../datasets/mmlu/
# outfile=./res/mmlu/llama3_8b_instruct/zero-shot-cot/

# task=tq
# source=./truthfulqa
# outfile=./res/tq/zero-shot-punish/

# type=mc_qa
# task=tq
# for chat_mode in 3-gene-1-gt-choose-gt-gpt4o 3-gene-1-none-choose-none-gpt4o 
# do
#     for model_path in "../models/llama2-7B-chat" "../models/llama3_8b_instruct" "../models/Qwen2-7B-Instruct"
#     do
#         for dataset in hq
#         do
#             if [ "$model_path" == "../models/llama3_8b_instruct" ]; then
#                 data_mode=${chat_mode}
#                 outfile="./res/${dataset}-mc/llama3_8b_instruct/zero-shot-${data_mode}/"
                
#             elif [ "$model_path" == "../models/llama2-7B-chat" ]; then
#                 data_mode=${chat_mode}
#                 outfile="./res/${dataset}-mc/llama2-chat-7b/zero-shot-${data_mode}/"
                
#             elif [ "$model_path" == "../models/Qwen2-7B-Instruct" ]; then
#                 data_mode=${chat_mode}
#                 outfile="./res/${dataset}-mc/qwen2-7b-instruct/zero-shot-${data_mode}/"
#             fi
#             source=../share/datasets/${dataset}-mc/
#             python run_mmlu.py \
#                 --source $source \
#                 --data_mode $data_mode \
#                 --type $type \
#                 --ra none \
#                 --outfile $outfile \
#                 --n_shot 0 \
#                 --model_path ${model_path} \
#                 --batch_size 18 \
#                 --task $task \
#                 --max_new_tokens 64 \
#                 --hidden_states 1 \
#                 --need_layers mid \
#                 --hidden_idx_mode first,last,avg,ans
#         done
#     done
# done


# for model_path in "../models/llama2-7B-chat"
# do
#     for dataset in hq
#     do
#         if [ "$model_path" == "../models/llama3_8b_instruct" ]; then
#         outfile="./res/${dataset}-mc/llama3_8b_instruct/zero-shot-none/"
#         elif [ "$model_path" == "../models/llama2-7B-chat" ]; then
#             outfile="./res/${dataset}-mc/llama2-chat-7b/zero-shot-none/"
#         elif [ "$model_path" == "../models/Qwen2-7B-Instruct" ]; then
#             outfile="./res/${dataset}-mc/qwen2-7b-instruct/zero-shot-none/"
#         fi
#         source=../share/datasets/${dataset}-mc/
#         python run_mmlu.py \
#             --source $source \
#             --type $type \
#             --ra none \
#             --outfile $outfile \
#             --n_shot 0 \
#             --model_path $model_path \
#             --batch_size 32 \
#             --task $task \
#             --max_new_tokens 64 \
#             --hidden_states 1 \
#             --need_layers mid \
#             --hidden_idx_mode first,last,avg,ans
#     done
# done


# source=../datasets/mmlu/
# type=mc_qa_cot
# task=mmlu
# outfile=./res/mmlu/llama2-chat-13b/zero-shot-cot/

# python run_mmlu.py \
#     --source $source \
#     --data_mode test \
#     --type $type \
#     --ra none \
#     --outfile $outfile \
#     --n_shot 0 \
#     --model_path ../models/llama2-13b-chat \
#     --batch_size 18 \
#     --task $task \
#     --max_new_tokens 256 \
#     --hidden_states 1 \
#     --need_layers mid \
#     --hidden_idx_mode first,last,avg,ans

type=mc_qa
task=tq
for chat_mode in 7-gene-1-gt-choose-gt-gpt4o
do
    for model_path in "../models/llama3_8b_instruct" "../models/llama2-7B-chat" "../models/Qwen2-7B-Instruct" "../models/llama2-13b-chat"
    do
        for dataset in nq hq
        do
            if [ "$model_path" == "../models/llama3_8b_instruct" ]; then
                data_mode=${chat_mode}
                outfile="../share/res/${dataset}-mc/llama3-8b-instruct/zero-shot-${data_mode}/"
                
            elif [ "$model_path" == "../models/llama2-7B-chat" ]; then
                data_mode=${chat_mode}
                outfile="../share/res/${dataset}-mc/llama2-chat-7b/zero-shot-${data_mode}/"
                
            elif [ "$model_path" == "../models/Qwen2-7B-Instruct" ]; then
                data_mode=${chat_mode}
                outfile="../share/res/${dataset}-mc/qwen2/zero-shot-${data_mode}/"
            elif [ "$model_path" == "../models/llama2-13b-chat" ]; then
                data_mode=${chat_mode}
                outfile="../share/res/${dataset}-mc/llama2-chat-13b/zero-shot-${data_mode}/"
            fi
            source=../share/datasets/${dataset}-mc/
            python run_mmlu.py \
                --source $source \
                --data_mode $data_mode \
                --type $type \
                --ra none \
                --outfile $outfile \
                --n_shot 0 \
                --model_path ${model_path} \
                --batch_size 18 \
                --task $task \
                --max_new_tokens 64 \
                --hidden_states 1 \
                --need_layers mid \
                --hidden_idx_mode first,last,avg,ans
        done
    done
done