"""
Verbalized Confidence Baseline: 让模型判断自己答案的正确性

对每个问题，让模型判断其自身生成的答案是否正确：
  - 回答 "certain" → 置信度 = 1.0
  - 回答 "uncertain" → 置信度 = 0.0

支持两种运行模式:
  1. VLLM 模式：用于 Qwen2 等开源模型（在服务器上运行）
  2. API 模式：用于 ChatGPT（在本地通过代理调用 API）

Usage:
    # VLLM 模式 (Qwen2 模型, 服务器上运行)
    python -u verbalized_confidence.py \
        --source ../res/movies/movies_Qwen2.5-7B_temperature1.jsonl \
        --outfile ./verbalized_confidence/movies_qwen2.5-7b_vc.jsonl \
        --mode vllm \
        --model_path /models/Qwen2.5-7B-Instruct \
        --tensor_parallel 2

    # API 模式 (ChatGPT, 本地运行)
    python -u verbalized_confidence.py \
        --source ../res/movies/movies_chatgpt_temperature1.jsonl \
        --outfile ./verbalized_confidence/movies_chatgpt_vc.jsonl \
        --mode api \
        --api_model gpt-3.5-turbo-1106 \
        --batch_size 5
"""

import os
import json
import re
import argparse
import time
import asyncio
from tqdm import tqdm


# ─── Verbalized Confidence Prompt ──────────────────────────────
# 让模型判断自己生成的答案是否正确
# certain → 置信度 1.0, uncertain → 置信度 0.0

VERBALIZED_PROMPT_TEMPLATE = (
    "Judge whether the following answer (this is your self-generated answer) about "
    "the question is correct. If you are sure the answer is correct, say certain. "
    "If not, please say uncertain. Just give your judgement without any other words.\n"
    "Question: {question}\n"
    "Answer: {answer}."
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
    if answer.startswith(': '):
        answer = answer[2:]
    return answer


def parse_verbalized_confidence(text):
    """
    解析模型输出的 verbalized confidence。

    Args:
        text: 模型输出文本

    Returns:
        str: 'certain', 'uncertain', 或 None（无法解析）
    """
    if text is None:
        return None
    text_lower = text.strip().lower()
    if 'certain' in text_lower and 'uncertain' not in text_lower:
        return 'certain'
    elif 'uncertain' in text_lower:
        return 'uncertain'
    else:
        return None


# ─── VLLM 模式（服务器上运行，用于 Qwen2 等开源模型）──────────

def run_vllm(source_data, model_path, tensor_parallel, outfile, resume=True):
    """
    使用 VLLM 运行 Verbalized Confidence。

    Args:
        source_data: QA 结果数据列表
        model_path: 模型路径
        tensor_parallel: VLLM 张量并行数
        outfile: 输出路径
        resume: 是否断点续跑
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel,
        dtype='float16',
        tokenizer_mode='slow',
        max_model_len=1024,
    )
    # 贪心解码，只需输出 certain/uncertain
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16, top_p=1.0, top_k=-1)

    # ── 断点续跑 ──────────────────────────────────────────────
    begin = 0
    if resume and os.path.exists(outfile):
        existing = load_jsonl(outfile)
        begin = len(existing)
    else:
        existing = []

    remaining_data = source_data[begin:]
    if not remaining_data:
        print('All data already processed.')
        return

    print(f'Total: {len(source_data)}, already done: {begin}, remaining: {len(remaining_data)}')

    # ── 构建 Prompts ──────────────────────────────────────────
    formatted_prompts = []
    for item in remaining_data:
        question = item['question']
        answer = clean_answer(item['Res'])

        user_content = VERBALIZED_PROMPT_TEMPLATE.format(question=question, answer=answer)
        messages = [
            {'role': 'system', 'content': MODEL_SYSTEM_PROMPT},
            {'role': 'user', 'content': user_content},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(prompt)

    # 打印样例
    if formatted_prompts:
        print(f'--- Sample prompt ---')
        print(formatted_prompts[0][:400])

    # ── 批量推理 ──────────────────────────────────────────────
    outputs = llm.generate(formatted_prompts, sampling_params)

    # ── 解析 & 写入 ──────────────────────────────────────────
    results = list(existing)
    for item, output in zip(remaining_data, outputs):
        res_text = output.outputs[0].text.strip()
        vc_label = parse_verbalized_confidence(res_text)
        vc_confidence = 1.0 if vc_label == 'certain' else (0.0 if vc_label == 'uncertain' else None)

        results.append({
            'question': item['question'],
            'reference': item.get('reference', []),
            'Res': item['Res'],
            'has_answer': item.get('has_answer', None),
            'popularity': item.get('popularity', None),
            'vc_response': res_text,
            'vc_label': vc_label,
            'vc_confidence': vc_confidence,
        })

    write_jsonl(results, outfile)

    # ── 统计 ──────────────────────────────────────────────────
    valid = [r for r in results if r['vc_confidence'] is not None]
    if valid:
        avg_conf = sum(r['vc_confidence'] for r in valid) / len(valid)
        certain_ratio = sum(1 for r in valid if r['vc_confidence'] == 1.0) / len(valid)
        print(f'Average verbalized confidence: {avg_conf:.4f}')
        print(f'Certain ratio: {certain_ratio:.4f} ({len(valid)}/{len(results)} valid)')


# ─── API 模式（本地运行，用于 ChatGPT）— asyncio 版 ─────────────

def _reorder_outfile(outfile):
    """
    读取输出文件，按 _idx 字段排序后重写（去除 _idx 字段）。
    兼容旧格式（无 _idx）：旧条目按文件中的行号作为索引。
    """
    if not os.path.exists(outfile):
        return
    with open(outfile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if not lines:
        return

    results = []
    for line_no, line in enumerate(lines):
        if line.strip():
            r = json.loads(line)
            # 旧格式条目没有 _idx，按行号补上（旧版本按序写入，行号即原始索引）
            if '_idx' not in r:
                r['_idx'] = line_no
            results.append(r)

    if not results:
        return

    # 检查是否已经有序且无需排序
    need_sort = False
    for i, r in enumerate(results):
        if r['_idx'] != i:
            need_sort = True
            break

    if need_sort:
        results.sort(key=lambda x: x['_idx'])

    # 重写文件，去除 _idx
    with open(outfile, 'w', encoding='utf-8') as f:
        for r in results:
            r.pop('_idx', None)
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    if need_sort:
        print(f'Reordered {len(results)} results in {outfile}')
    else:
        print(f'Already in order, {len(results)} results in {outfile}')


def run_api(source_data, api_model, max_concurrent, outfile, resume=True):
    """
    使用 ChatGPT API 运行 Verbalized Confidence（异步并发版）。

    Args:
        source_data: QA 结果数据列表
        api_model: ChatGPT 模型名称
        max_concurrent: 最大并发请求数
        outfile: 输出路径
        resume: 是否断点续跑
    """
    asyncio.run(_run_api_async(source_data, api_model, max_concurrent, outfile, resume))


async def _run_api_async(source_data, api_model, max_concurrent, outfile, resume=True):
    """异步实现：全并发 API 调用 + 完成即写入 + 断点续跑 + 最终排序"""

    import openai

    # ── API 配置（openai v1.x+ 异步客户端）─────────────────
    # 通过环境变量配置：export OPENAI_API_KEY="sk-..."（可选 OPENAI_BASE_URL）
    client = openai.AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1"),
    )

    # ── 断点续跑：用 (question, Res) 去重 ─────────────────
    done_keys = set()
    if resume and os.path.exists(outfile):
        with open(outfile, 'r', encoding='utf-8') as fin:
            for line in fin:
                if line.strip():
                    try:
                        r = json.loads(line)
                        # 用 question+Res 组合作为唯一标识（同一问题同一答案只判一次）
                        key = (r.get('question', ''), r.get('Res', ''))
                        done_keys.add(key)
                    except json.JSONDecodeError:
                        pass
        print(f'Found {len(done_keys)} completed results in {outfile}')

    # 过滤出未完成的样本
    remaining = []
    for i, item in enumerate(source_data):
        key = (item.get('question', ''), item.get('Res', ''))
        if key not in done_keys:
            remaining.append((i, item))

    if not remaining:
        print('All samples already completed.')
        _reorder_outfile(outfile)
        return

    print(f'Total: {len(source_data)}, already done: {len(done_keys)}, remaining: {len(remaining)}')

    # 打开文件追加写入
    os.makedirs(os.path.dirname(outfile) if os.path.dirname(outfile) else '.', exist_ok=True)
    f = open(outfile, 'a', encoding='utf-8')

    # ── 异步 API 请求 ────────────────────────────────────────
    semaphore = asyncio.Semaphore(max_concurrent)

    async def call_api(messages):
        """单次异步 API 调用，带信号量 + 指数退避重试"""
        async with semaphore:
            for attempt in range(10):
                try:
                    res = await client.chat.completions.create(
                        model=api_model,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=16,
                    )
                    return res.choices[0].message.content.strip()
                except openai.RateLimitError as e:
                    wait = min(5 * (2 ** attempt), 60)
                    print(f'\nRateLimitError (attempt {attempt+1}): retrying in {wait}s...')
                    await asyncio.sleep(wait)
                except openai.APIStatusError as e:
                    print(f'\nAPIStatusError (attempt {attempt+1}): retrying in 5s...')
                    await asyncio.sleep(5)
                except openai.APIConnectionError as e:
                    print(f'\nAPIConnectionError (attempt {attempt+1}): retrying in 5s...')
                    await asyncio.sleep(5)
                except Exception as e:
                    print(f'\nUnexpected error (attempt {attempt+1}): {e}')
                    if attempt < 9:
                        await asyncio.sleep(5)
                    else:
                        return None
            return None

    async def process_one(idx, item):
        """处理单个样本"""
        question = item['question']
        answer = clean_answer(item['Res'])
        user_content = VERBALIZED_PROMPT_TEMPLATE.format(question=question, answer=answer)
        messages = [{"role": "user", "content": user_content}]

        res_text = await call_api(messages)
        vc_label = parse_verbalized_confidence(res_text)
        vc_confidence = 1.0 if vc_label == 'certain' else (0.0 if vc_label == 'uncertain' else None)

        return idx, {
            '_idx': idx,  # 原始索引，用于最终排序
            'question': item['question'],
            'reference': item.get('reference', []),
            'Res': item['Res'],
            'has_answer': item.get('has_answer', None),
            'popularity': item.get('popularity', None),
            'vc_response': res_text,
            'vc_label': vc_label,
            'vc_confidence': vc_confidence,
        }

    # ── 全并发 + 完成即写入 ─────────────────────────────────
    tasks = [process_one(idx, item) for idx, item in remaining]
    all_results = []

    pbar = tqdm(total=len(tasks), desc="ChatGPT verbalized confidence")
    for coro in asyncio.as_completed(tasks):
        idx, result = await coro
        # 完成即写入
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
        f.flush()
        all_results.append(result)
        pbar.update(1)

    pbar.close()
    f.close()

    # ── 统计 ──────────────────────────────────────────────────
    valid = [r for r in all_results if r['vc_confidence'] is not None]
    if valid:
        avg_conf = sum(r['vc_confidence'] for r in valid) / len(valid)
        certain_ratio = sum(1 for r in valid if r['vc_confidence'] == 1.0) / len(valid)
        print(f'Average verbalized confidence: {avg_conf:.4f}')
        print(f'Certain ratio: {certain_ratio:.4f}')

    # ── 最终排序：按原始顺序重排输出文件 ──
    _reorder_outfile(outfile)


# ─── Main ────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description='Verbalized confidence baseline'
    )
    parser.add_argument(
        '--source', type=str, required=True,
        help='QA 结果文件路径（如 res/movies/movies_Qwen2.5-7B_temperature1.jsonl）'
    )
    parser.add_argument(
        '--outfile', type=str, required=True,
        help='输出文件路径'
    )
    parser.add_argument(
        '--mode', type=str, choices=['vllm', 'api'], default='vllm',
        help='推理模式: vllm=开源模型(服务器), api=ChatGPT(本地)'
    )
    # VLLM 参数
    parser.add_argument(
        '--model_path', type=str, default='',
        help='模型路径 (VLLM 模式必需, 如 /models/Qwen2.5-7B-Instruct)'
    )
    parser.add_argument(
        '--tensor_parallel', type=int, default=2,
        help='VLLM 张量并行数 (default: 2)'
    )
    # API 参数
    parser.add_argument(
        '--api_model', type=str, default='gpt-3.5-turbo-1106',
        help='ChatGPT 模型名称 (default: gpt-3.5-turbo-1106)'
    )
    parser.add_argument(
        '--max_concurrent', type=int, default=5,
        help='最大并发 API 请求数 (default: 5)'
    )
    return parser.parse_args()


def main():
    args = get_args()
    print(f'Args: {args}')

    source_data = load_jsonl(args.source)
    print(f'Loaded {len(source_data)} samples from {args.source}')

    if args.mode == 'vllm':
        if not args.model_path:
            raise ValueError('--model_path is required for vllm mode')
        run_vllm(source_data, args.model_path, args.tensor_parallel, args.outfile)
    elif args.mode == 'api':
        run_api(source_data, args.api_model, args.max_concurrent, args.outfile)


if __name__ == '__main__':
    main()