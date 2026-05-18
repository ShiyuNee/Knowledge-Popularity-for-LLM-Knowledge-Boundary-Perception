import os
import json
import math
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.utils import load_source, write_jsonl, has_answer, deal_judge_new
from utils.prompt import get_prompt


ra_dict = {
    'none':         'none',
    'sparse':       {'sparse_ctxs': 1},
    'dense':        {'dense_ctxs': 1},
    'chatgpt':      {'gen_ctxs': 100},
    'sparse+dense': {'dense_ctxs': 5, 'sparse_ctxs': 5},
    'gold':         {'gold_ctxs': 1},
    'strong':       {'strong_ctxs': 10},
    'weak':         {'weak_ctxs': 10},
    'rand':         {'rand_ctxs': 10},
    'dpr':          {'dpr_ctx': 1},
    'extract':      {'dpr_ctx': 1},
    'dpr_wrong':    {'dpr_ctx_wrong': 1},
}

prompt_dict = {
    'qa': {
        'none': 'Answer the following question based on your internal knowledge with one or few words. Provide only one name, with no additional or irrelevant text.\nQuestion: {question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words.\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
}


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--source',          type=str,   default='data/source/nq.json')
    parser.add_argument('--response',        type=str,   default='')
    parser.add_argument('--usechat',         action='store_true')

    parser.add_argument(
        '--type',
        type=str,
        choices=[
            'qa',
            'qa_evidence',
            'qa_cot',
            'qa_more',
            'qa_extract',
            'qa_prior',
            'qa_post'
        ],
        default='qa'
    )

    parser.add_argument(
        '--ra',
        type=str,
        default='none',
        choices=ra_dict.keys()
    )

    parser.add_argument(
        '--outfile',
        type=str,
        default='data/qa/vllm-nq-none.jsonl'
    )

    parser.add_argument('--idx',             type=str,   default='')
    parser.add_argument('--model_path',      type=str,   default='')
    parser.add_argument('--task',            type=str,   default='nq')

    parser.add_argument('--max_new_tokens',  type=int,   default=64)
    parser.add_argument('--temperature',     type=float, default=0.0)

    parser.add_argument('--sampling',        action='store_true')

    parser.add_argument(
        '--tensor_parallel',
        type=int,
        default=1,
        help='vLLM tensor_parallel_size'
    )

    # 新增：生成数量
    parser.add_argument(
        '--num_return_sequences',
        type=int,
        default=1,
        help='number of responses generated per prompt'
    )

    args = parser.parse_args()

    args.ra = ra_dict[args.ra]
    args.model_name = args.model_path.split('/')[-1].replace('_', '-').lower()

    return args


def build_log_p(token_ids, logprobs_list):
    """
    从 vLLM 返回的 logprobs 还原 Log_p 字典。
    vLLM 0.3.x: output.logprobs -> List[Dict[int, float]]
    """

    token_probs = []
    token_entropy = []

    for token_id, lp_dict in zip(token_ids, logprobs_list):

        lp = lp_dict.get(token_id, min(lp_dict.values()))

        token_probs.append(round(math.exp(lp), 6))

        probs_vals = [math.exp(v) for v in lp_dict.values()]
        s = sum(probs_vals)

        probs_norm = [p / s for p in probs_vals] if s > 0 else probs_vals

        entropy = -sum(
            p * math.log2(p + 1e-12)
            for p in probs_norm
            if p > 0
        )

        token_entropy.append(round(entropy, 6))

    return {
        'tokens': token_ids,
        'token_probs': token_probs,
        'token_entropy': token_entropy,
    }


def _build_user_content(sample, args) -> str:
    """
    构建 user message 的纯文本内容
    """

    paras = ""
    ref_key = 'question'

    prompt = prompt_dict[args.type]['none']

    if args.ra != 'none':

        ra_setting = args.ra

        i = 0
        doc = []

        for k, v in ra_setting.items():

            v = min(v, len(sample[k]))

            for j in range(v):
                doc.append(("Passage-%d" % i) + sample[k][j])
                i += 1

        paras = '\n'.join(doc)

        prompt = prompt_dict[args.type]['ra']

    tail = prompt_dict[args.type]['tail'] if not args.usechat else ""

    prediction = sample['Res'] if 'post' in args.type else ""

    if args.task == 'mmlu' or args.task == 'tq':

        user_content = prompt.format(
            question=sample[ref_key],
            paras=paras,
            prediction=prediction,
            subject=args.subject,
        ) + tail

    else:

        user_content = prompt.format(
            question=sample[ref_key],
            paras=paras,
            prediction=prediction,
        ) + tail

    return user_content


def get_prompt(sample, args, tokenizer: AutoTokenizer) -> str:
    """
    使用 apply_chat_template 构建 prompt
    """

    system_prompt = (
        "You are Qwen, created by Alibaba Cloud. "
        "You are a helpful assistant"
    )

    user_content = _build_user_content(sample, args)

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_content
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return prompt


def main():

    args = get_args()

    print(args)

    # ============================================================
    # 读取数据
    # ============================================================

    all_data = load_source(args.source)

    # ============================================================
    # 断点续跑
    # ============================================================

    begin = 0

    if os.path.exists(args.outfile):

        with open(args.outfile, 'r', encoding='utf-8') as f:

            for line in f:
                if line.strip():
                    begin += 1

        outfile = open(args.outfile, 'a', encoding='utf-8')

    else:

        outfile = open(args.outfile, 'w', encoding='utf-8')

    remaining_data = all_data[begin:]

    if not remaining_data:

        print('All data already processed.')

        outfile.close()

        return

    print(
        f'Total: {len(all_data)}, '
        f'already done: {begin}, '
        f'remaining: {len(remaining_data)}'
    )

    # ============================================================
    # tokenizer
    # ============================================================

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # ============================================================
    # vLLM
    # ============================================================

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel,
        dtype='float16',
        tokenizer_mode="slow",
        max_model_len=2048,
    )

    # ============================================================
    # sampling params
    # ============================================================

    sampling_params = SamplingParams(
        temperature=args.temperature if args.sampling else 0.0,
        max_tokens=args.max_new_tokens,
        logprobs=1,
        top_p=1.0,
        top_k=-1,

        # 核心：控制生成数量
        n=args.num_return_sequences,
    )

    # ============================================================
    # prompts
    # ============================================================

    prompts = [
        get_prompt(sample, args, tokenizer)
        for sample in remaining_data
    ]

    print(f'prompts: {prompts[:2]}')

    # ============================================================
    # generate
    # ============================================================

    outputs = llm.generate(prompts, sampling_params)

    # ============================================================
    # evaluate
    # ============================================================

    acc = 0

    for i, (sample, output) in enumerate(
        tqdm(
            zip(remaining_data, outputs),
            total=len(remaining_data)
        )
    ):

        responses = []

        # output.outputs:
        # n=1 -> len=1
        # n=10 -> len=10

        for out in output.outputs:

            res_text = out.text.strip()

            token_ids = list(out.token_ids)

            log_p = (
                build_log_p(token_ids, out.logprobs)
                if out.logprobs else {
                    'tokens': token_ids,
                    'token_probs': [],
                    'token_entropy': []
                }
            )

            # correctness
            if 'prior' in args.type or 'post' in args.type:
                correct = deal_judge_new(res_text)
            else:
                correct = has_answer(sample['reference'], res_text)

            responses.append({
                'Res': res_text,
                'Log_p': log_p,
                'has_answer': correct,
            })

        # ========================================================
        # 单输出：兼容原格式
        # ========================================================

        if args.num_return_sequences == 1:

            res_sample = {
                'question': sample['question'],
                'reference': sample['reference'],

                'Res': responses[0]['Res'],
                'Log_p': responses[0]['Log_p'],
                'has_answer': responses[0]['has_answer'],

                'qa_prompt': prompts[i],
            }

            acc += int(responses[0]['has_answer'])

        # ========================================================
        # 多输出：保存 response list
        # ========================================================

        else:

            res_sample = {
                'question': sample['question'],
                'reference': sample['reference'],

                'responses': responses,

                'qa_prompt': prompts[i],
            }

            # 至少一个正确算正确
            acc += int(
                any(r['has_answer'] for r in responses)
            )

        outfile.write(
            json.dumps(
                res_sample,
                ensure_ascii=False
            ) + '\n'
        )

        outfile.flush()

    outfile.close()

    total = len(remaining_data)

    print(
        f'Accuracy: {acc}/{total} = {acc / total:.4f}'
    )


if __name__ == '__main__':
    main()