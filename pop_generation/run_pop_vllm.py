"""
Popularity generation for Qwen2.5 models using VLLM (单模型全量批量版)

为单个模型的一次 VLLM 初始化中，处理所有 dataset × gene_type × n_shot 组合，
避免重复加载模型。

每个模型只需运行一次，内部自动遍历:
  3 datasets × 3 gene_types × 4 n_shots = 36 个任务

所有 prompt 一次性送入 VLLM 生成，结果分别写入对应输出文件。

运行前确保已运行 build_clean_data.py 生成 clean_data_for_pop_generation/ 数据。

Usage:
    # 运行一个模型（自动处理所有组合）
    python -u run_pop_vllm.py \
        --model_path /models/Qwen2.5-7B-Instruct \
        --source_prefix ../res \
        --output_prefix ./llm_pop_generation \
        --tensor_parallel 2

    # 运行另一个模型
    python -u run_pop_vllm.py \
        --model_path /models/Qwen2.5-14B-Instruct \
        --source_prefix ../res \
        --output_prefix ./llm_pop_generation \
        --tensor_parallel 2
"""

import os
import json
import re
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ─── 配置常量 ──────────────────────────────────────────────────────

DATASETS = ['movies', 'songs', 'basketball']
GENE_TYPES = ['question', 'gene', 'coo']
N_SHOTS = [0, 3, 5, 10]

PROMPT_TEMPLATES = {
    'qa_pop_rank_diverse': (
        "Rate how familiar you are with the {entity_type} '{entity_name}'. "
        "The familiarity is rated on a scale from 1 to 10, where 10 means you are "
        "highly familiar with it, and 1 means you have little to no knowledge about it. "
        "Your answer needs to be a precise integer. Provide only the number, "
        "without any additional explanation.\n"
        "Number: "
    ),
    'qa_coo_rank_diverse': (
        "Rate how familiar you are with the relationship between the "
        "{question_entity_type} '{question_entity_name}' and the "
        "{gene_entity_type} '{gene_entity_name}'. "
        "The familiarity is rated on a scale from 1 to 10, where 10 means you are "
        "highly familiar with their relationship, and 1 means you know little to "
        "nothing about it. Your answer needs to be a precise integer. Provide only "
        "the number, without any additional explanation.\n"
        "Number: "
    ),
}

DATASET_ENTITY_NAMES = {
    'movies':     {'question': 'movie',       'gene': 'director',          'coo': ['movie', 'director']},
    'songs':      {'question': 'song',        'gene': 'performer',         'coo': ['song', 'performer']},
    'basketball': {'question': 'basketball player', 'gene': 'city',        'coo': ['basketball player', 'city']},
}

QUESTION_PATTERNS = {
    'movies':     'Who is the director of the movie ',
    'songs':      'Who is the performer of the song ',
    'basketball': 'Where is the birthplace of the basketball player ',
}

MODEL_SYSTEM_PROMPT = 'You are a helpful assistant.'

# Few-shot 数据路径（固定使用 llama8b 的 clean_data）
FEWSHOT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'data', 'clean_data_for_pop_generation')
FEWSHOT_MODEL = 'llama8b'

# 模型名称映射: model_path basename → res/ 文件名中的简称
MODEL_SHORT_NAMES = {
    'Qwen2.5-7B-Instruct': 'Qwen2.5-7B',
    'Qwen2.5-14B-Instruct': 'Qwen2.5-14B',
    'Qwen2.5-32B-Instruct': 'Qwen2.5-32B',
    'Qwen2-7B-Instruct': 'qwen2',
    'Meta-Llama-3-8B-Instruct': 'llama8b',
}


# ─── 辅助函数 ─────────────────────────────────────────────────────

def load_source(path):
    """读取 JSONL 文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_entities_from_res(data, dataset_name):
    """从 res/ 格式数据中提取 question_entity 和 gene_entity。"""
    pattern = QUESTION_PATTERNS[dataset_name]
    for item in data:
        q = item['question']
        if q.startswith(pattern):
            item['question_entity'] = q[len(pattern):].rstrip('?').strip()
        else:
            item['question_entity'] = q.rstrip('?').strip()

        gene = item['Res'].strip()
        if gene.startswith(': '):
            gene = gene[2:]
        item['gene_entity'] = gene
    return data


def assign_levels(data, field, num_levels=10):
    """将 popularity 值均匀分为 1-10 个 level（与原始 data_api.py 一致）。"""
    field_values = [item[field] for item in data if isinstance(item.get(field), (int, float))]
    unique_values = sorted(set(field_values))
    step = len(unique_values) // num_levels
    boundaries = [unique_values[i * step] for i in range(1, num_levels)]
    boundaries.append(float('inf'))

    for item in data:
        value = item.get(field)
        if isinstance(value, (int, float)):
            level = 1
            for b in boundaries:
                if value <= b:
                    break
                level += 1
            item[field + '_level'] = level
        else:
            item[field + '_level'] = None


def load_fewshot_examples(dataset_name, gene_type, n_shot):
    """从 llama8b 的 clean_data 中选取 few-shot 示例。

    选取规则:
      - 3-shot:  level 2, 5, 8 各 1 个
      - 5-shot:  level 1, 3, 5, 7, 9 各 1 个
      - 10-shot: level 1~10 各 1 个
    """
    fewshot_path = os.path.join(FEWSHOT_DATA_DIR,
                                 f'{dataset_name}_{FEWSHOT_MODEL}_temperature1.jsonl')
    if not os.path.exists(fewshot_path):
        print(f'  WARNING: Few-shot data not found: {fewshot_path}')
        return []

    data = load_source(fewshot_path)
    valid_data = [d for d in data
                  if d.get('question_pop') != 'No' and d.get('gene_pop') != 'No'
                  and isinstance(d.get('question_pop'), (int, float))
                  and isinstance(d.get('gene_pop'), (int, float))]

    field = f'{gene_type}_pop'
    assign_levels(valid_data, field, num_levels=10)

    if n_shot == 3:
        need_levels = [2, 5, 8]
    elif n_shot == 5:
        need_levels = [1, 3, 5, 7, 9]
    elif n_shot == 10:
        need_levels = list(range(1, 11))
    else:
        return []

    grouped = {}
    for item in valid_data:
        level = item.get(field + '_level')
        if level is not None:
            grouped.setdefault(level, []).append(item)

    examples = []
    for lv in need_levels:
        if lv in grouped and grouped[lv]:
            examples.append(grouped[lv].pop(0))

    return examples


def build_prompt(sample, gene_type, dataset_name, few_shot_examples):
    """构建 user content（不含 chat template）。

    Prompt 结构: base_prompt(去掉\nNumber:) + few_shot + base_prompt(保留\nNumber:)
    """
    need_names = DATASET_ENTITY_NAMES[dataset_name]

    if gene_type in ('question', 'gene'):
        template_key = 'qa_pop_rank_diverse'
        base_prompt = PROMPT_TEMPLATES[template_key].format(
            entity_type=need_names[gene_type],
            entity_name=sample[gene_type + '_entity'],
        )
    else:  # coo
        template_key = 'qa_coo_rank_diverse'
        base_prompt = PROMPT_TEMPLATES[template_key].format(
            question_entity_type=need_names['coo'][0],
            question_entity_name=sample['question_entity'],
            gene_entity_type=need_names['coo'][1],
            gene_entity_name=sample['gene_entity'],
        )

    # Few-shot 部分
    few_shot_str = ''
    if few_shot_examples:
        few_shot_str = '\nHere are some examples:\n'
        for ex in few_shot_examples:
            if gene_type == 'question':
                few_shot_str += f'The {need_names["question"]}: {ex["question_entity"]}\n'
                few_shot_str += f'Number: {ex["question_pop_level"]}\n'
            elif gene_type == 'gene':
                few_shot_str += f'The {need_names["gene"]}: {ex["gene_entity"]}\n'
                few_shot_str += f'Number: {ex["gene_pop_level"]}\n'
            elif gene_type == 'coo':
                few_shot_str += (
                    f'The {need_names["coo"][0]}: {ex["question_entity"]}; '
                    f'The {need_names["coo"][1]}: {ex["gene_entity"]}\n'
                )
                few_shot_str += f'Number: {ex["coo_pop_level"]}\n'

    # 组合
    base_no_tail = base_prompt.rstrip()
    if base_no_tail.endswith('\nNumber:'):
        base_no_tail = base_no_tail[:-len('\nNumber:')]
    return base_no_tail + few_shot_str + base_prompt


def parse_pop_value(text):
    """从 LLM 输出中提取 1-10 的整数值。"""
    if text is None:
        return None
    m = re.search(r'(-?\d+)', text)
    if m:
        val = int(m.group(1))
        return max(1, min(10, val))
    return None


def get_output_path(output_prefix, dataset, model_name, gene_type, n_shot):
    """构建输出文件路径。"""
    return os.path.join(output_prefix, dataset,
                         f'{dataset}_{model_name}_{gene_type}_pop_{n_shot}.jsonl')


def count_existing_lines(path):
    """统计已有输出文件的行数（用于断点续跑）。"""
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


# ─── 单个任务: 构建所有 prompt ────────────────────────────────────

def build_task_prompts(source_data, gene_type, dataset_name, n_shot):
    """为单个 (dataset, gene_type, n_shot) 任务构建所有 prompt。

    Returns:
        list[str]: user content 列表
    """
    few_shot_examples = []
    if n_shot > 0:
        few_shot_examples = load_fewshot_examples(dataset_name, gene_type, n_shot)

    prompts = []
    for sample in source_data:
        user_content = build_prompt(sample, gene_type, dataset_name, few_shot_examples)
        prompts.append(user_content)
    return prompts


# ─── Main ────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description='Popularity generation: 单模型全量批量版 (一次 VLLM 初始化处理所有组合)'
    )
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路径 (如 /models/Qwen2.5-7B-Instruct)')
    parser.add_argument('--source_prefix', type=str, default='../res',
                        help='res/ 目录路径 (default: ../res)')
    parser.add_argument('--output_prefix', type=str, default='./llm_pop_generation',
                        help='输出目录路径 (default: ./llm_pop_generation)')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_new_tokens', type=int, default=16)
    parser.add_argument('--tensor_parallel', type=int, default=2)
    return parser.parse_args()


def main():
    args = get_args()
    print(f'Args: {args}')

    # ── 确定模型名称 ──────────────────────────────────────────
    model_basename = os.path.basename(args.model_path)
    model_short = MODEL_SHORT_NAMES.get(model_basename, model_basename)
    # 输出文件名用的模型名 (小写, 去点): Qwen2.5-7B → qwen2.5-7b
    model_name = model_short.lower().replace('.', '')

    print(f'Model: {model_basename} → short={model_short} → name={model_name}')

    # ── 加载所有数据集 ────────────────────────────────────────
    # 提前加载，避免重复读取
    dataset_cache = {}  # dataset_name → list[dict]
    for dataset in DATASETS:
        source_path = os.path.join(args.source_prefix, dataset,
                                    f'{dataset}_{model_short}_temperature1.jsonl')
        if not os.path.exists(source_path):
            print(f'WARNING: {source_path} not found, skipping dataset={dataset}')
            continue
        data = load_source(source_path)
        # 提取实体
        if data and 'question_entity' not in data[0] and 'question' in data[0]:
            data = extract_entities_from_res(data, dataset)
        dataset_cache[dataset] = data
        print(f'  {dataset}: {len(data)} samples loaded')

    if not dataset_cache:
        print('ERROR: No dataset loaded. Check --source_prefix and model name.')
        return

    # ── 初始化 VLLM (只初始化一次) ───────────────────────────
    print('\nInitializing VLLM...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel,
        dtype='float16',
        tokenizer_mode='slow',
        max_model_len=1024,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        top_p=1.0,
        top_k=-1,
    )
    print('VLLM initialized.\n')

    # ── 遍历所有任务组合 ──────────────────────────────────────
    total_tasks = 0
    skipped_tasks = 0

    for dataset in DATASETS:
        if dataset not in dataset_cache:
            continue
        source_data = dataset_cache[dataset]

        for gene_type in GENE_TYPES:
            for n_shot in N_SHOTS:
                total_tasks += 1
                outfile = get_output_path(args.output_prefix, dataset,
                                           model_name, gene_type, n_shot)

                # ── 断点续跑 ──────────────────────────────────
                begin = count_existing_lines(outfile)
                remaining_data = source_data[begin:]

                if not remaining_data:
                    skipped_tasks += 1
                    print(f'  [SKIP] {dataset}/{gene_type}/{n_shot}-shot: already done ({begin} lines)')
                    continue

                print(f'\n=== Task: {dataset} / {gene_type} / {n_shot}-shot ===')
                print(f'  Total: {len(source_data)}, done: {begin}, remaining: {len(remaining_data)}')

                # ── 构建 Prompt ────────────────────────────────
                user_contents = build_task_prompts(remaining_data, gene_type, dataset, n_shot)

                # Apply chat template
                formatted_prompts = []
                for user_content in user_contents:
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
                    print(f'  Sample prompt (first 200 chars): {user_contents[0][:200]}')

                # ── 生成 ──────────────────────────────────────
                outputs = llm.generate(formatted_prompts, sampling_params)

                # ── 解析 & 写入 ───────────────────────────────
                os.makedirs(os.path.dirname(outfile), exist_ok=True)
                num_valid = 0
                with open(outfile, 'a', encoding='utf-8') as f:
                    for user_content, output in zip(user_contents, outputs):
                        res_text = output.outputs[0].text.strip()
                        pop_value = parse_pop_value(res_text)
                        result = {
                            'qa_prompt': user_content,
                            'Res': res_text,
                            'pop_value': pop_value,
                        }
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        if pop_value is not None:
                            num_valid += 1

                print(f'  Written {len(remaining_data)} lines to {outfile}')
                print(f'  Valid pop_value: {num_valid}/{len(remaining_data)}')

    # ── 总结 ──────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'Model: {model_basename}')
    print(f'Total tasks: {total_tasks}, skipped (already done): {skipped_tasks}')
    print(f'New tasks completed: {total_tasks - skipped_tasks}')


if __name__ == '__main__':
    main()