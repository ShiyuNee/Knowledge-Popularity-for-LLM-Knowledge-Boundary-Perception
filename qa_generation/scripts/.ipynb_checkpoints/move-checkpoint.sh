for dataset in nq-mc hq-mc
do
    for mode in zero-shot-3-gene-1-gt-choose-gt-gpt4o zero-shot-3-gene-1-none-choose-none-gpt4o
    do
        mv ./res/${dataset}/qwen2-7b-instruct/${mode}/ ../share/res/${dataset}/qwen2/mid_layer/
    done
done