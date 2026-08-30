"""
QA Generation for Qwen2.5 models using VLLM (单模型全量批量版)

对单个模型一次初始化 VLLM，内部遍历所有数据集，生成 QA 结果。
支持两种模式:
  - 贪心 (temperature=0.0, --num_samples 1): 每题生成 1 条结果，输出到 res/
  - 采样 (temperature=1.0, --num_samples 10): 每题生成 10 条结果，输出到 baselines/self_consistency/

输出格式:
  - 贪心模式: 与原有 res/ 文件完全一致 (question, reference, Res, Log_p, has_answer, qa_prompt, popularity)
  - 采样模式: 与 chatgpt_sampling.py 一致 (question, reference, responses列表, qa_prompt)

运行环境: 服务器 (需要 GPU + VLLM)

Usage:
    # 贪心模式 (1 条/题) — 结果写入 res/{dataset}/
    python -u run_vllm.py \
        --model_path /models/Qwen2.5-7B-Instruct \
        --data_prefix ../data \
        --output_prefix ../res \
        --temperature 0.0 \
        --tensor_parallel 2

    # 采样模式 (10 条/题) — 结果写入 baselines/self_consistency/
    python -u run_vllm.py \
        --model_path /models/Qwen2.5-7B-Instruct \
        --data_prefix ../data \
        --output_prefix ../baselines/self_consistency \
        --temperature 1.0 \
        --num_samples 10 \
        --tensor_parallel 2
"""

import os
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.utils import has_answer, load_source


# ─── 配置常量 ──────────────────────────────────────────────────────

DATASETS = ['movies', 'songs', 'basketball']

# QA Prompt（与原有 run_nq.py / prompt.py 保持一致）
QA_PROMPT_TEMPLATE = (
    "Answer the following question based on your internal knowledge "
    "with one or few words. Provide only one name, with no additional "
    "or irrelevant text.\n"
    "Question: {question}\n"
    "Answer: "
)

MODEL_SYSTEM_PROMPT = 'You are a helpful assistant.'


# ─── 辅助函数 ─────────────────────────────────────────────────────

def get_model_name(model_path: str) -> str:
    """从模型路径提取模型名（如 Qwen2.5-7B-Instruct → Qwen2.5-7B）"""
    name = os.path.basename(model_path.rstrip('/'))
    # Qwen2.5-7B-Instruct → Qwen2.5-7B
    if name.endswith('-Instruct'):
        name = name[:-len('-Instruct')]
    return name


def build_output_filename(dataset: str, model_name: str, temperature: float) -> str:
    """
    输出文件名规则（与已有文件保持一致）:
      - temperature=0.0: {dataset}_{model_name}_temperature0.jsonl
      - temperature=1.0: {dataset}_{model_name}_temperature1.jsonl
    """
    temp_str = '1' if temperature > 0 else '0'
    return f"{dataset}_{model_name}_temperature{temp_str}.jsonl"


# ─── 主流程 ────────────────────────────────────────────────────────

def run_all_datasets(model_path, data_prefix, output_prefix, temperature,
                     num_samples, tensor_parallel, max_model_len):
    """一次 VLLM 初始化，遍历所有数据集"""

    model_name = get_model_name(model_path)
    print(f'\n{"="*70}')
    print(f'  Model: {model_name}  (path: {model_path})')
    print(f'  Temperature: {temperature}, Samples/question: {num_samples}')
    print(f'  Datasets: {DATASETS}')
    print(f'{"="*70}\n')

    # ── 初始化 VLLM（只做一次）────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel,
        dtype='float16',
        tokenizer_mode='slow',
        max_model_len=max_model_len,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=64,
        top_p=1.0,
        top_k=-1,
        n=num_samples,           # VLLM 原生支持 n>1 采样
        logprobs=1,              # 保留 token 概率信息
    )

    # ── 逐数据集处理 ──────────────────────────────────────────────
    for dataset in DATASETS:
        source_path = os.path.join(data_prefix, f'{dataset}.jsonl')
        if not os.path.exists(source_path):
            print(f'  [SKIP] {source_path} not found')
            continue

        out_filename = build_output_filename(dataset, model_name, temperature)

        # 输出路径：采样模式 (n>1) 平铺到 output_prefix 下，贪心模式 (n=1) 按数据集分子目录
        if num_samples > 1:
            # 采样文件命名: {dataset}_{model_name}_sampling.jsonl
            out_filename = f"{dataset}_{model_name}_sampling.jsonl"
            out_dir = output_prefix
        else:
            out_dir = os.path.join(output_prefix, dataset)
        out_path = os.path.join(out_dir, out_filename)

        # 创建输出目录
        os.makedirs(out_dir, exist_ok=True)

        # ── 断点续跑：检查已有结果 ──
        all_data = load_source(source_path)
        done_questions = set()
        if os.path.exists(out_path):
            with open(out_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            r = json.loads(line)
                            # 采样模式下用 (question, sample_idx) 去重
                            # 贪心模式下用 question 去重
                            done_questions.add(r['question'])
                        except json.JSONDecodeError:
                            pass
            print(f'  [{dataset}] Found {len(done_questions)} existing results')

        # 过滤出未完成的问题
        # 注意：采样模式下原文件是每题一行包含 responses 列表
        # 这里简单处理：如果已有结果数 >= 数据量，则跳过
        remaining_data = []
        for item in all_data:
            if item['question'] not in done_questions:
                remaining_data.append(item)

        if not remaining_data:
            print(f'  [{dataset}] All done, skipping.')
            continue

        print(f'  [{dataset}] Total: {len(all_data)}, Remaining: {len(remaining_data)}')

        # ── 构建 Prompts ──────────────────────────────────────
        formatted_prompts = []
        for item in remaining_data:
            user_content = QA_PROMPT_TEMPLATE.format(question=item['question'])
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
            print(f'  --- Sample prompt ---')
            print(formatted_prompts[0][:400])
            print()

        # ── VLLM 批量推理 ─────────────────────────────────────
        # VLLM 的 n=num_samples 会为每个 prompt 生成 n 个回复
        outputs = llm.generate(formatted_prompts, sampling_params)

        # ── 写入结果 ──────────────────────────────────────────
        f_out = open(out_path, 'a', encoding='utf-8')
        acc = 0
        total = len(remaining_data)

        for item, output in zip(remaining_data, outputs):
            question = item['question']
            reference = item.get('reference', [])
            user_content = QA_PROMPT_TEMPLATE.format(question=question)

            if num_samples == 1:
                # 贪心模式：单条结果，与原有格式一致
                res_text = output.outputs[0].text.strip()
                # 清理可能的 ": " 前缀
                if res_text.startswith(': '):
                    res_text = res_text[2:]

                # 提取 Log_p（token 概率）
                logprobs_list = output.outputs[0].logprobs
                token_probs = []
                token_ids = []
                token_entropies = []
                if logprobs_list:
                    for pos_logprobs in logprobs_list:
                        if pos_logprobs:
                            # 取实际生成的 token
                            generated_token = list(pos_logprobs.keys())[0]
                            token_ids.append(generated_token)
                            prob = pos_logprobs[generated_token].logprob
                            token_probs.append(round(prob, 6))
                            # entropy 计算较复杂，这里简化为 -log2(p)
                            import math
                            token_entropies.append(round(-math.log2(max(prob, 1e-10)), 4))

                correct = has_answer(reference, res_text)

                result = {
                    'question': question,
                    'reference': reference,
                    'Res': res_text,
                    'Log_p': {
                        'tokens': token_ids,
                        'token_probs': token_probs,
                        'token_entropy': token_entropies,
                    },
                    'has_answer': correct,
                    'qa_prompt': user_content,
                    'popularity': item.get('popularity', None),
                }
                if correct:
                    acc += 1

                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            else:
                # 采样模式：多条结果，与 chatgpt_sampling 格式一致
                responses = []
                for sample_output in output.outputs:
                    res_text = sample_output.text.strip()
                    if res_text.startswith(': '):
                        res_text = res_text[2:]
                    correct = has_answer(reference, res_text)
                    responses.append({
                        'Res': res_text,
                        'Log_p': None,  # 采样模式不保留详细 logprobs
                        'has_answer': correct,
                    })

                result = {
                    'question': question,
                    'reference': reference,
                    'responses': responses,
                    'qa_prompt': user_content,
                }
                if any(r['has_answer'] for r in responses):
                    acc += 1

                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')

        f_out.flush()
        f_out.close()

        if total > 0:
            print(f'  [{dataset}] Accuracy: {acc}/{total} = {acc/total:.4f}')
        print(f'  [{dataset}] Results saved to: {out_path}')

    print(f'\n{"="*70}')
    print(f'  All datasets completed for model: {model_name}')
    print(f'{"="*70}\n')


def get_args():
    parser = argparse.ArgumentParser(
        description='QA Generation for Qwen2.5 models using VLLM (batch mode)'
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='模型路径, 如 /models/Qwen2.5-7B-Instruct'
    )
    parser.add_argument(
        '--data_prefix', type=str, default='../data',
        help='数据目录, 包含 movies.jsonl, songs.jsonl, basketball.jsonl (default: ../data)'
    )
    parser.add_argument(
        '--output_prefix', type=str, default='../res',
        help='输出目录前缀, 结果写入 {output_prefix}/{dataset}/ (default: ../res)'
    )
    parser.add_argument(
        '--temperature', type=float, default=0.0,
        help='采样温度 (0.0=贪心, 1.0=采样) (default: 0.0)'
    )
    parser.add_argument(
        '--num_samples', type=int, default=1,
        help='每题采样次数, 仅 temperature>0 时有效 (default: 1)'
    )
    parser.add_argument(
        '--tensor_parallel', type=int, default=2,
        help='VLLM 张量并行数 (default: 2)'
    )
    parser.add_argument(
        '--max_model_len', type=int, default=2048,
        help='VLLM 最大模型长度 (default: 2048)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(f'Args: {args}')

    run_all_datasets(
        model_path=args.model_path,
        data_prefix=args.data_prefix,
        output_prefix=args.output_prefix,
        temperature=args.temperature,
        num_samples=args.num_samples,
        tensor_parallel=args.tensor_parallel,
        max_model_len=args.max_model_len,
    )