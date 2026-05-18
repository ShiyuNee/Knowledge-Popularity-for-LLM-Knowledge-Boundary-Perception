import os
from tqdm import tqdm
import json
import logging
import argparse
from utils.llm_api import get_llm_result
from utils.prompt import get_prompt
from utils.data_api import QADataset


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/source/nq.json')
    parser.add_argument('--response', type=str, default='')
    parser.add_argument('--type', type=str, choices=['qa_coo_rank_diverse', 'qa_pop_rank_diverse'], default='qa_pop_rank_diverse')
    parser.add_argument('--outfile', type=str, default='data/qa/chatgpt-nq-none.json')     
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--logprobs', type=bool, default=True)
    parser.add_argument('--usechat', action='store_true')
    parser.add_argument('--prompt_name', type=str, default='song')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--n_shot', type=int, choices=[0, 3, 5, 10])
    parser.add_argument('--gene_type', type=str, default='question', choices=['question', 'gene', 'coo'])
    parser.add_argument('--dataset_name', type=str, default='movies')
    args = parser.parse_args()

    return args



def batch_process(all_data, batch_size):
    """将数据分批"""
    for i in range(0, len(all_data), batch_size):
        yield all_data[i:i + batch_size]

def main():
    args = get_args()
    begin = 0
    if os.path.exists(args.outfile):
        with open(args.outfile, 'r', encoding='utf-8') as outfile:
            for line in outfile:
                if line.strip():
                    begin += 1
        outfile = open(args.outfile, 'a', encoding='utf-8')
    else:
        outfile = open(args.outfile, 'w', encoding='utf-8')

    get_prompt_dataset = QADataset(args)
    all_prompts = get_prompt_dataset.prompts
    all_samples = get_prompt_dataset.data
    batch_size = args.batch_size  # 每次处理的样本数量
    num_output = 0
    buffer = []
    buffer_size = 10  # 文件写入缓冲区大小

    try:
        for batch_start in tqdm(range(begin, len(all_prompts), batch_size), desc="Processing batches"):
            batch_prompts = all_prompts[batch_start:batch_start + batch_size]
            batch_samples = all_samples[batch_start:batch_start + batch_size]

            responses = get_llm_result(batch_prompts, batch_samples, args)

            buffer.extend(responses)
            num_output += len(responses)

            if len(buffer) >= buffer_size:
                outfile.writelines([json.dumps(res) + "\n" for res in buffer])
                buffer = []

        if buffer:
            outfile.writelines([json.dumps(res) + "\n" for res in buffer])
    except Exception as e:
        logging.exception(e)
    finally:
        print(args.outfile, " has output %d line(s)." % num_output)
        outfile.close()

if __name__ == '__main__':
    main()
