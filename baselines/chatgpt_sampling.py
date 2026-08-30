"""
ChatGPT QA 多次采样脚本 (用于 Self-Consistency baseline) — asyncio 版

对每个问题，使用 ChatGPT API 采样 N 次（默认 10 次），
输出格式与 qa_generation/run_vllm.py --sampling 一致：
每个样本包含 'responses' 列表（每个元素含 Res, Log_p=None, has_answer）。

所有 API 调用全并发，通过 --max_concurrent 信号量控制最大并发请求数。
结果完成即写入（不保序），跑完后自动按原始顺序重排。
断点续跑：已有结果的问题自动跳过。

运行环境：本地

Usage:
    # 采样 10 次（用于 Self-Consistency）
    python -u chatgpt_sampling.py \
        --source ../data/movies.jsonl \
        --outfile ./self_consistency/movies_chatgpt_sampling.jsonl \
        --api_model gpt-3.5-turbo-1106 \
        --num_samples 10 \
        --max_concurrent 5

    # 仅采样 1 次
    python -u chatgpt_sampling.py \
        --source ../data/songs.jsonl \
        --outfile ./self_consistency/songs_chatgpt_sampling_1shot.jsonl \
        --api_model gpt-3.5-turbo-1106 \
        --num_samples 1
"""

import os
import json
import argparse
import asyncio
from tqdm import tqdm

# ─── API 配置 ──────────────────────────────────────────────────────
# 通过环境变量配置 API（不要在此处硬编码密钥）：
#   export OPENAI_API_KEY="sk-..."
#   export OPENAI_BASE_URL="https://api.openai.com/v1"   # 可选；也可指向兼容代理端点
# 如需代理，取消以下注释并填写代理地址：
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'

import openai

# openai v1.x+ 异步客户端
_client = openai.AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1"),
)


# ─── QA Prompt（与 qa_generation 一致）───────────────────────────
QA_PROMPT_TEMPLATE = (
    "Answer the following question based on your internal knowledge "
    "with one or few words. Provide only one name, with no additional "
    "or irrelevant text.\n"
    "Question: {question}\n"
    "Answer: "
)


# ─── 辅助函数 ─────────────────────────────────────────────────────

def load_jsonl(path):
    """读取 JSONL 文件，返回 list[dict]"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def has_answer(reference, text):
    """
    检查生成的文本是否包含 reference 中的答案。
    逻辑与 qa_generation/utils/utils.py 一致。
    """
    if not text:
        return False
    text_lower = text.lower()
    for ref in reference:
        ref_lower = ref.lower().strip()
        if ref_lower and ref_lower in text_lower:
            return True
    return False


def reorder_outfile(outfile):
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


async def call_api(semaphore, messages, api_model, temperature=1.0, max_tokens=64):
    """
    单次异步 API 调用，带信号量限制 + 重试机制。
    """
    async with semaphore:
        for attempt in range(10):
            try:
                res = await _client.chat.completions.create(
                    model=api_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return res.choices[0].message.content.strip()
            except openai.RateLimitError as e:
                wait = min(5 * (2 ** attempt), 60)  # 指数退避，最长 60s
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


async def process_one_question(semaphore, idx, item, api_model, num_samples, temperature):
    """
    对单个问题采样 N 次（N 次采样全部并发）。
    返回 (idx, result)，idx 为原始数据索引。
    """
    question = item['question']
    reference = item.get('reference', [])

    user_content = QA_PROMPT_TEMPLATE.format(question=question)
    messages = [{"role": "user", "content": user_content}]

    # N 次采样全部并发发起
    tasks = [
        call_api(semaphore, messages, api_model, temperature=temperature)
        for _ in range(num_samples)
    ]
    results = await asyncio.gather(*tasks)

    responses = []
    for res_text in results:
        correct = has_answer(reference, res_text) if res_text else False
        responses.append({
            'Res': res_text if res_text else '',
            'Log_p': None,
            'has_answer': correct,
        })

    result = {
        '_idx': idx,  # 原始索引，用于最终排序
        'question': question,
        'reference': reference,
        'responses': responses,
        'qa_prompt': user_content,
    }
    return idx, result


async def run_async(all_data, outfile, api_model, num_samples, temperature, max_concurrent):
    """异步主流程：并发采样 + 完成即写入 + 断点续跑 + 最终排序"""

    # ── 断点续跑：读取已有结果，收集已完成的 question ──
    done_questions = set()  # 用 question 文本做去重
    if os.path.exists(outfile):
        with open(outfile, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        done_questions.add(r['question'])
                    except json.JSONDecodeError:
                        pass
        print(f'Found {len(done_questions)} completed results in {outfile}')

    # 过滤出未完成的问题
    remaining = [(i, item) for i, item in enumerate(all_data) if item['question'] not in done_questions]

    if not remaining:
        print('All questions already completed.')
        reorder_outfile(outfile)
        return

    print(f'Total: {len(all_data)}, already done: {len(done_questions)}, remaining: {len(remaining)}')

    # 打开文件追加写入
    os.makedirs(os.path.dirname(outfile) or '.', exist_ok=True)
    f_out = open(outfile, 'a', encoding='utf-8')

    # ── 并发采样 ──
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        process_one_question(semaphore, idx, item, api_model, num_samples, temperature)
        for idx, item in remaining
    ]

    acc = 0
    total = 0

    pbar = tqdm(total=len(tasks), desc="ChatGPT sampling")
    for coro in asyncio.as_completed(tasks):
        idx, result = await coro
        total += 1

        # 完成即写入，不管顺序
        f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
        f_out.flush()

        if result.get('responses') and any(r['has_answer'] for r in result['responses']):
            acc += 1
        pbar.update(1)

    pbar.close()
    f_out.close()

    print(f'{outfile} written. Accuracy (at least 1 hit): {acc}/{total} = {acc/total:.4f}' if total > 0 else 'No results.')
    print(f'Each sample has {num_samples} responses.')

    # ── 最终排序：按原始顺序重排输出文件 ──
    reorder_outfile(outfile)


def main():
    args = get_args()
    print(f'Args: {args}')

    all_data = load_jsonl(args.source)
    print(f'Loaded {len(all_data)} samples from {args.source}')

    asyncio.run(run_async(
        all_data, args.outfile, args.api_model,
        args.num_samples, args.temperature, args.max_concurrent,
    ))


def get_args():
    parser = argparse.ArgumentParser(
        description='ChatGPT QA sampling for self-consistency baseline (asyncio version)'
    )
    parser.add_argument(
        '--source', type=str, required=True,
        help='源数据路径（JSONL），含 question, reference 字段，如 ../data/movies.jsonl'
    )
    parser.add_argument(
        '--outfile', type=str, required=True,
        help='输出路径（JSONL），如 ./self_consistency/movies_chatgpt_sampling.jsonl'
    )
    parser.add_argument(
        '--api_model', type=str, default='gpt-3.5-turbo-1106',
        help='ChatGPT 模型名称 (default: gpt-3.5-turbo-1106)'
    )
    parser.add_argument(
        '--num_samples', type=int, default=10,
        help='每个问题的采样次数 (default: 10)'
    )
    parser.add_argument(
        '--temperature', type=float, default=1.0,
        help='采样温度 (default: 1.0)'
    )
    parser.add_argument(
        '--max_concurrent', type=int, default=5,
        help='全局最大并发 API 请求数 (default: 5)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()