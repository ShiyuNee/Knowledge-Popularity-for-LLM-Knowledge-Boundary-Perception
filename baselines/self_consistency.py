"""
Self-Consistency Baseline: 使用 LLM Judge 判断一致性并计算置信度

整体流程:
  Step 1: 对每个问题多次采样（10 次）
    - Qwen2.5 模型：在服务器上用 VLLM 采样（复用 qa_generation/run_vllm.py --sampling）
    - ChatGPT：在本地用 baselines/chatgpt_sampling.py 采样

  Step 2: 使用 Qwen2.5-32B-Instruct (VLLM) 判断每个采样答案与原始答案是否一致
    - 输入：原始答案 + 采样答案 → Judge → yes/no

  Step 3: 计算自一致置信度 SC = (一致答案数) / (总采样数)

本脚本负责 Step 2 和 Step 3。

支持两种运行方式:
  1. 单任务模式: --source + --original + --outfile
  2. 批量模式: --pairs_file，一次性加载模型，依次处理所有数据对

输入格式（采样结果文件）:
  - 新格式 (chatgpt/Qwen2.5): 每行含 responses 列表
  - 旧格式 (qwen2/llama8b): 扁格式，同一问题出现 N 次，自动聚合
  - 原始贪心答案文件: 每行含 question, reference, Res, Log_p, has_answer 等

Usage:
    # 批量模式（推荐，只加载一次模型）:
    python -u self_consistency.py \
        --pairs_file pairs.jsonl \
        --judge_model_path /models/Qwen2.5-32B-Instruct \
        --tensor_parallel 2

    # 单任务模式:
    python -u self_consistency.py \
        --source ./sampling.jsonl \
        --original ./original.jsonl \
        --outfile ./output_sc.jsonl \
        --judge_model_path /models/Qwen2.5-32B-Instruct \
        --tensor_parallel 2 \
        --mode judge_and_compute

pairs.jsonl 格式（每行一个 JSON 对象）:
    {"source": "path/to/sampling.jsonl", "original": "path/to/original.jsonl", "outfile": "path/to/output.jsonl"}
"""

import os
import json
import argparse
from collections import OrderedDict
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ─── Judge Prompt ────────────────────────────────────────────────
# 让 Judge 模型判断两个答案是否语义一致（指同一实体或表达相同含义）

JUDGE_PROMPT_TEMPLATE = (
    "Determine whether the following two answers to the same question are consistent "
    "(i.e., they refer to the same entity or convey the same meaning). "
    "If they are consistent, answer 'yes'. If not, answer 'no'. "
    "Provide only 'yes' or 'no' without any other words.\n"
    "Question: {question}\n"
    "Answer 1: {answer1}\n"
    "Answer 2: {answer2}\n"
    "Response: "
)

MODEL_SYSTEM_PROMPT = 'You are a helpful assistant.'


# ─── 辅助函数 ─────────────────────────────────────────────────────

def load_jsonl(path):
    """读取 JSONL 文件，返回 list[dict]"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_and_aggregate_sampling(path):
    """
    读取采样结果文件，自动检测格式并聚合为 responses 格式。

    新格式（chatgpt/Qwen2.5）: 每行含 responses 列表，直接返回
    旧格式（qwen2/llama8b）: 同一问题出现 N 次（扁格式），聚合为 responses 格式
    """
    raw = load_jsonl(path)
    if not raw:
        return raw

    # 已经是 responses 格式，直接返回
    if 'responses' in raw[0]:
        return raw

    # 扁格式：按 question 分组聚合
    groups = OrderedDict()
    for item in raw:
        q = item['question']
        if q not in groups:
            groups[q] = {
                'question': q,
                'reference': item.get('reference', []),
                'qa_prompt': item.get('qa_prompt', ''),
                'responses': [],
            }
        groups[q]['responses'].append({
            'Res': item['Res'],
            'Log_p': item.get('Log_p', None),
            'has_answer': item.get('has_answer', False),
        })

    result = list(groups.values())
    print(f'Aggregated {len(raw)} flat rows into {len(result)} questions '
          f'(~{len(raw) // max(len(result), 1)} samples per question)')
    return result


def write_jsonl(data, path):
    """写入 JSONL 文件"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'Written {len(data)} lines to {path}')


def clean_answer(answer):
    """清理答案文本（去除模型输出中的前缀等）"""
    if not answer:
        return ''
    answer = answer.strip()
    # Qwen2 模型有时输出 ": xxx" 格式，去除开头的 ": "
    if answer.startswith(': '):
        answer = answer[2:]
    return answer


# ─── Step 2: LLM Judge ──────────────────────────────────────────
# 使用 Qwen2.5-32B-Instruct (VLLM) 逐对判断采样答案与原始答案是否一致

def run_judge(source_data, original_data, llm, tokenizer, outfile):
    """
    使用 Judge 模型判断每个采样答案与原始答案是否一致。

    Args:
        source_data: 采样结果列表（含 'responses' 字段）
        original_data: 原始贪心结果列表（含 'Res' 字段）
        llm: 已初始化的 VLLM 实例
        tokenizer: 已初始化的 tokenizer
        outfile: Judge 结果输出路径

    Returns:
        list[dict]: 每个样本含 question, reference, original_answer,
                    sampled_answers, judge_results (bool 列表)
    """
    assert len(source_data) == len(original_data), \
        f'Source ({len(source_data)}) and original ({len(original_data)}) must have same length'

    sampling_params = SamplingParams(temperature=0.0, max_tokens=10, top_p=1.0, top_k=-1)

    # ── 断点续跑 ──────────────────────────────────────────────
    if os.path.exists(outfile):
        existing = load_jsonl(outfile)
        if len(existing) == len(source_data):
            all_complete = True
            for ex, sample in zip(existing, source_data):
                n_sampled = len(sample['responses']) if 'responses' in sample else 1
                if len(ex.get('judge_results', [])) != n_sampled:
                    all_complete = False
                    break
            if all_complete:
                print(f'All judge results already exist in {outfile}. Skipping.')
                return existing
        print(f'Found incomplete judge results in {outfile}, regenerating...')

    # ── 构建 Judge prompts ────────────────────────────────────
    all_prompts = []
    prompt_meta = []       # (sample_idx, response_idx)

    for idx, (sample, orig) in enumerate(zip(source_data, original_data)):
        question = sample.get('question', orig.get('question', ''))
        original_answer = clean_answer(orig['Res'])

        if 'responses' in sample:
            sampled_answers = [clean_answer(r['Res']) for r in sample['responses']]
        else:
            sampled_answers = [clean_answer(sample['Res'])]

        for r_idx, sampled_ans in enumerate(sampled_answers):
            prompt_text = JUDGE_PROMPT_TEMPLATE.format(
                question=question,
                answer1=original_answer,
                answer2=sampled_ans,
            )
            messages = [
                {'role': 'system', 'content': MODEL_SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt_text},
            ]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_prompts.append(formatted)
            prompt_meta.append((idx, r_idx))

    print(f'Total judge prompts: {len(all_prompts)}')

    # ── 初始化结果数据结构 ────────────────────────────────────
    result_data = []
    for i, (sample, orig) in enumerate(zip(source_data, original_data)):
        question = sample.get('question', orig.get('question', ''))
        original_answer = clean_answer(orig['Res'])
        sampled_answers = [clean_answer(r['Res']) for r in sample['responses']] if 'responses' in sample else [clean_answer(sample['Res'])]
        result_data.append({
            'question': question,
            'reference': sample.get('reference', orig.get('reference', [])),
            'original_answer': original_answer,
            'sampled_answers': sampled_answers,
            'judge_results': [],
        })

    # ── 批量推理 ──────────────────────────────────────────────
    outputs = llm.generate(all_prompts, sampling_params)

    # ── 解析 Judge 结果 ──────────────────────────────────────
    for (s_idx, r_idx), output in zip(prompt_meta, outputs):
        res_text = output.outputs[0].text.strip().lower()
        is_consistent = 'yes' in res_text
        result_data[s_idx]['judge_results'].append(is_consistent)

    # ── 写入结果 ──────────────────────────────────────────────
    write_jsonl(result_data, outfile)
    return result_data


# ─── Step 3: 计算 Self-Consistency 置信度 ───────────────────────
# SC = 与原始答案一致的采样数 / 总采样数

def compute_confidence(judged_data, outfile):
    """
    根据 Judge 结果计算自一致置信度。

    Args:
        judged_data: Judge 结果列表（每项含 judge_results: list[bool]）
        outfile: 置信度结果输出路径

    Returns:
        list[dict]: 每项含 sc_confidence 字段
    """
    results = []
    for item in judged_data:
        judge_results = item.get('judge_results', [])
        n_total = len(judge_results)
        n_consistent = sum(judge_results)
        confidence = n_consistent / n_total if n_total > 0 else 0.0

        results.append({
            'question': item.get('question', ''),
            'reference': item.get('reference', []),
            'original_answer': item.get('original_answer', ''),
            'sampled_answers': item.get('sampled_answers', []),
            'n_sampled': n_total,
            'n_consistent': n_consistent,
            'sc_confidence': round(confidence, 4),
        })

    write_jsonl(results, outfile)

    # 打印统计信息
    if results:
        avg_conf = sum(r['sc_confidence'] for r in results) / len(results)
        print(f'Average self-consistency confidence: {avg_conf:.4f} over {len(results)} samples')

    return results


# ─── 批量处理 ─────────────────────────────────────────────────────

def run_batch(pairs, judge_model_path, tensor_parallel):
    """
    批量处理多个 source:original:outfile 对，只加载一次模型。

    Args:
        pairs: list of (source, original, outfile) 元组
        judge_model_path: Judge 模型路径
        tensor_parallel: VLLM 张量并行数
    """
    # ── 只初始化一次模型和 tokenizer ──────────────────────────
    print(f'Loading judge model: {judge_model_path} ...')
    tokenizer = AutoTokenizer.from_pretrained(judge_model_path, trust_remote_code=True, use_fast=False)
    llm = LLM(
        model=judge_model_path,
        tensor_parallel_size=tensor_parallel,
        dtype='float16',
        tokenizer_mode='slow',
        max_model_len=1024,
    )
    print('Judge model loaded.')

    for i, (source, original, outfile) in enumerate(pairs):
        print(f'\n{"="*70}')
        print(f'  [{i+1}/{len(pairs)}] Processing: {os.path.basename(source)}')
        print(f'  source:  {source}')
        print(f'  original: {original}')
        print(f'  outfile:  {outfile}')
        print(f'{"="*70}')

        # 检查文件
        if not os.path.exists(source):
            print(f'SKIP: source file not found: {source}')
            continue
        if not os.path.exists(original):
            print(f'SKIP: original file not found: {original}')
            continue

        # 检查最终输出是否已完成
        if os.path.exists(outfile):
            existing = load_jsonl(outfile)
            source_data = load_and_aggregate_sampling(source)
            if len(existing) == len(source_data):
                print(f'SKIP: output already complete: {outfile}')
                continue

        source_data = load_and_aggregate_sampling(source)
        original_data = load_jsonl(original)
        print(f'Questions: {len(source_data)}, Original samples: {len(original_data)}')

        if len(source_data) != len(original_data):
            print(f'ERROR: question count mismatch ({len(source_data)} vs {len(original_data)}), skipping.')
            continue

        judge_outfile = outfile.replace('.jsonl', '_judged.jsonl')
        judged_data = run_judge(source_data, original_data, llm, tokenizer, judge_outfile)
        compute_confidence(judged_data, outfile)

    print(f'\n{"="*70}')
    print(f'  All {len(pairs)} tasks completed!')
    print(f'{"="*70}')


# ─── Main ────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description='Self-consistency baseline: LLM Judge + confidence computation'
    )
    # 单任务模式参数
    parser.add_argument(
        '--source', type=str, default='',
        help='采样结果文件路径（单任务模式）'
    )
    parser.add_argument(
        '--original', type=str, default='',
        help='原始贪心答案文件路径（单任务模式，Judge 模式必需）'
    )
    parser.add_argument(
        '--outfile', type=str, default='',
        help='输出文件路径（单任务模式）'
    )
    # 批量模式参数
    parser.add_argument(
        '--pairs_file', type=str, default='',
        help='批量模式：JSONL 文件路径，每行 {"source": "...", "original": "...", "outfile": "..."}'
    )
    # 通用参数
    parser.add_argument(
        '--judge_model_path', type=str,
        default='/models/Qwen2.5-32B-Instruct',
        help='Judge 模型路径 (default: /models/Qwen2.5-32B-Instruct)'
    )
    parser.add_argument(
        '--tensor_parallel', type=int, default=2,
        help='VLLM 张量并行数 (default: 2)'
    )
    parser.add_argument(
        '--mode', type=str, choices=['judge', 'compute', 'judge_and_compute'],
        default='judge_and_compute',
        help='运行模式（仅单任务模式有效）: judge=仅Judge, compute=仅计算置信度, judge_and_compute=两步都跑'
    )
    return parser.parse_args()


def main():
    args = get_args()
    print(f'Args: {args}')

    if args.pairs_file:
        # ── 批量模式：只加载一次模型，处理所有数据对 ──────
        pairs = []
        for line in load_jsonl(args.pairs_file):
            # 兼容 JSONL 中每行是 dict 的情况
            if isinstance(line, dict):
                pairs.append((line['source'], line['original'], line['outfile']))
            else:
                print(f'Invalid line in pairs_file: {line}')
                continue
        print(f'Loaded {len(pairs)} pairs from {args.pairs_file}')
        run_batch(pairs, args.judge_model_path, args.tensor_parallel)

    elif args.source:
        # ── 单任务模式 ────────────────────────────────────
        if args.mode in ('judge', 'judge_and_compute'):
            if not args.original:
                raise ValueError('--original is required for judge mode')

            source_data = load_and_aggregate_sampling(args.source)
            original_data = load_jsonl(args.original)
            print(f'Source: {len(source_data)} questions, Original: {len(original_data)} samples')

            judge_outfile = args.outfile.replace('.jsonl', '_judged.jsonl')
            judged_data = run_judge(
                source_data, original_data,
                args.judge_model_path, args.tensor_parallel,
                judge_outfile
            )

        if args.mode == 'compute':
            judged_data = load_jsonl(args.source)

        if args.mode in ('compute', 'judge_and_compute'):
            compute_confidence(judged_data, args.outfile)

    else:
        raise ValueError('Either --pairs_file or --source must be provided')


if __name__ == '__main__':
    main()