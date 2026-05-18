#!/bin/bash
model_name=chatgpt
for dataset in movies songs basketball
do
    for gene_type in question gene coo
    do
        if [ "$gene_type" = "coo" ]; then
            type="qa_coo_rank_diverse"
        else
            type="qa_pop_rank_diverse"
        fi
        
        for n_shot in 0 3
        do
            python -u run_api.py \
            --source ./data/clean_data_for_pop_generation/${dataset}_${model_name}_temperature1.jsonl \
            --type ${type} \
            --outfile ./llm_pop_generation/${dataset}/${dataset}_${model_name}_${gene_type}_pop_${n_shot}.jsonl \
            --model gpt-3.5-turbo-1106 \
            --batch_size 3 \
            --gene_type ${gene_type} \
            --dataset_name ${dataset} \
            --temperature 0.0 \
            --n_shot ${n_shot}
        done
    done
done
