"""
计算 §5.2 需要的完整数据：
三个因素 (gene_pop, gene_coo, qpop) 在 correct/incorrect 子集上的偏相关，
使用生成答案属性间互控：
  - gene_pop→conf: 控制 gene_coo
  - gene_coo→conf: 控制 gene_pop
  - qpop→conf: 控制 gene_coo + gene_pop

使用 filter_pop_no=True 来过滤 Wikidata 查不到 popularity 的样本，
与 §5.1 的数据口径一致。
"""

import json, math, re, os
import numpy as np
from scipy.stats import spearmanr, pearsonr, rankdata

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES_DIR = os.path.join(BASE, 'res')

POP_PATH = os.path.join(RES_DIR, 'gt_gene_entity_popularity_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.jsonl')
COO_PATH = os.path.join(RES_DIR, 'cooccurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')
SINGLE_PATH = os.path.join(RES_DIR, 'single_occurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json')

DATASETS = ['movies', 'songs', 'basketball']
MODELS = ['llama8b', 'qwen2', 'chatgpt', 'Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']
PATTERN = {
    'movies': 'Who is the director of the movie ',
    'songs': 'Who is the performer of the song ',
    'basketball': 'Where is the birthplace of the basketball player '
}
SINGLE_OCC_THRESHOLD = 6000


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]


def remove_punctuation_edges(s, name='movies'):
    s = s.replace('\n', '')
    s = s.split('(')[0].strip()
    if name in ['basketball']:
        s = s.split(',')[0].strip()
    else:
        if len(s) <= 20:
            s = s.split(',')[0].strip()
    s = re.sub(r'^[^\w]+|[^\w]+$', '', s)
    return s.strip()


def partial_spearman(x, y, z):
    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)
    rz = rankdata(z).astype(float)
    rz_c = rz - rz.mean()
    if np.dot(rz_c, rz_c) == 0:
        return 0, 1
    beta_x = np.dot(rz_c, rx) / np.dot(rz_c, rz_c)
    res_x = rx - beta_x * rz_c
    beta_y = np.dot(rz_c, ry) / np.dot(rz_c, rz_c)
    res_y = ry - beta_y * rz_c
    return pearsonr(res_x, res_y)


def multivariate_partial(x, y, controls):
    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    if len(controls) == 0:
        return pearsonr(rx_c, ry_c)
    Z = np.column_stack([rankdata(c).astype(float) for c in controls])
    Z = Z - Z.mean(axis=0)
    try:
        beta_x = np.linalg.lstsq(Z, rx_c, rcond=None)[0]
        beta_y = np.linalg.lstsq(Z, ry_c, rcond=None)[0]
        res_x = rx_c - Z @ beta_x
        res_y = ry_c - Z @ beta_y
        return pearsonr(res_x, res_y)
    except:
        return 0, 1


def load_popularity():
    pop_data = read_jsonl(POP_PATH)
    full_dict = {}
    for d in pop_data:
        full_dict.update(d)
    return full_dict


def collect_samples(dataset, model, full_dict, co_occu, single_occr,
                    filter_pop_no=True):
    res_path = os.path.join(RES_DIR, dataset, f'{dataset}_{model}_temperature1.jsonl')
    if not os.path.exists(res_path):
        return []
    model_res = read_jsonl(res_path)
    samples = []
    for item in model_res:
        if not item.get('Res') or item['Res'] is None:
            continue
        if item.get('popularity') == 'No':
            continue
        question_entity = item['question'].replace(PATTERN[dataset], '').lower()
        ref = remove_punctuation_edges(item['reference'][0], dataset)
        gene_entity = remove_punctuation_edges(item['Res'], dataset)

        if question_entity not in co_occu:
            continue
        if question_entity.lower() not in single_occr:
            continue
        if gene_entity.lower() not in single_occr:
            continue
        if ref.lower() not in single_occr:
            continue
        if dataset in ['movies', 'songs']:
            if (single_occr[question_entity.lower()] > SINGLE_OCC_THRESHOLD or
                    single_occr[gene_entity.lower()] > SINGLE_OCC_THRESHOLD or
                    single_occr[ref.lower()] > SINGLE_OCC_THRESHOLD):
                continue
        else:
            if single_occr[question_entity.lower()] > SINGLE_OCC_THRESHOLD:
                continue

        question_pop = item['popularity']
        ref_pop_info = full_dict.get(ref, {})
        ref_pop = ref_pop_info.get('popularity', 0) if isinstance(ref_pop_info, dict) else ref_pop_info
        gt_pop_missing = (ref_pop == 'No' or ref_pop is None or ref not in full_dict)
        if gt_pop_missing:
            ref_pop = 0

        gene_pop_info = full_dict.get(gene_entity, {})
        gene_pop = gene_pop_info.get('popularity', 0) if isinstance(gene_pop_info, dict) else gene_pop_info
        gene_pop_missing = (gene_pop == 'No' or gene_pop is None or gene_entity not in full_dict)
        if gene_pop_missing:
            gene_pop = 0

        if filter_pop_no and (gt_pop_missing or gene_pop_missing):
            continue

        if 'gpt' in model.lower():
            probs = [math.exp(t) for t in item['Log_p']['token_logprobs']]
        else:
            probs = item['Log_p']['token_probs']
        conf = sum(probs) / len(probs)

        gt_coo = co_occu[question_entity].get(ref.lower(), 0)
        gene_coo = co_occu[question_entity].get(gene_entity.lower(), 0)
        q_single = single_occr[question_entity.lower()]

        samples.append({
            'question_pop': question_pop,
            'gt_pop': int(ref_pop) if ref_pop else 0,
            'gene_pop': int(gene_pop) if gene_pop else 0,
            'coo': gt_coo,
            'gene_coo': gene_coo,
            'q_single': q_single,
            'conf': conf,
            'acc': item['has_answer'],
        })
    return samples


def _skip_bball_qwen25(dataset, model):
    return dataset == 'basketball' and model.startswith('Qwen2.5')


def calc_partial(data, factor_key, target_key='conf', control_key=None, control_keys=None):
    """计算偏相关。
    单控制: control_key
    多控制: control_keys (list)
    无控制: 两者都不提供
    """
    if len(data) < 30:
        return float('nan')
    arr = np.array([s[factor_key] for s in data], dtype=float)
    tgt = np.array([s[target_key] for s in data], dtype=float)
    if np.std(arr) == 0 or np.std(tgt) == 0:
        return float('nan')
    if control_key is not None:
        ctrl = np.array([s[control_key] for s in data], dtype=float)
        r, _ = partial_spearman(arr, tgt, ctrl)
        return r
    elif control_keys is not None:
        ctrls = [np.array([s[k] for s in data], dtype=float) for k in control_keys]
        r, _ = multivariate_partial(arr, tgt, ctrls)
        return r
    else:
        r, _ = spearmanr(arr, tgt)
        return r


def main():
    full_dict = load_popularity()
    with open(COO_PATH) as f:
        co_occu = json.load(f)
    with open(SINGLE_PATH) as f:
        single_occr = json.load(f)

    all_data = {}
    for dataset in DATASETS:
        for model in MODELS:
            if _skip_bball_qwen25(dataset, model):
                continue
            samples = collect_samples(dataset, model, full_dict, co_occu, single_occr,
                                      filter_pop_no=True)
            all_data[(dataset, model)] = samples

    print("=" * 140)
    print("§5.2 完整数据: 三因素 (gene_pop, gene_coo, qpop) × Correct/Incorrect × Ratio")
    print("偏相关 (生成答案属性间互控): gene_pop 控制 gene_coo, gene_coo 控制 gene_pop, qpop 控制 gene_coo+gene_pop")
    print("=" * 140)

    for dataset in DATASETS:
        print(f"\n{'='*140}")
        print(f"  Dataset: {dataset.upper()}")
        print(f"{'='*140}")
        print(f"{'Factor':15s} {'Model':15s} | {'n_correct':>10s} {'n_wrong':>10s} | "
              f"{'Correct':>10s} {'Incorrect':>10s} {'Ratio':>8s}")
        print("-" * 100)

        for factor in ['gene_pop', 'gene_coo', 'question_pop']:
            factor_label = f"{factor}→conf"
            for model in MODELS:
                if _skip_bball_qwen25(dataset, model):
                    continue
                samples = all_data.get((dataset, model), [])
                right = [s for s in samples if s['acc'] == 1]
                wrong = [s for s in samples if s['acc'] == 0]

                # Compute with appropriate controls
                if factor == 'gene_pop':
                    r_correct = calc_partial(right, factor, control_key='gene_coo')
                    r_incorrect = calc_partial(wrong, factor, control_key='gene_coo')
                elif factor == 'gene_coo':
                    r_correct = calc_partial(right, factor, control_key='gene_pop')
                    r_incorrect = calc_partial(wrong, factor, control_key='gene_pop')
                elif factor == 'question_pop':
                    r_correct = calc_partial(right, factor, control_keys=['gene_coo', 'gene_pop'])
                    r_incorrect = calc_partial(wrong, factor, control_keys=['gene_coo', 'gene_pop'])

                if np.isnan(r_correct) or np.isnan(r_incorrect):
                    ratio_str = "—"
                elif abs(r_correct) < 0.001:
                    ratio_str = "∞" if r_incorrect > 0.001 else "—"
                elif r_correct * r_incorrect <= 0:
                    ratio_str = "—"
                else:
                    ratio = r_incorrect / r_correct
                    ratio_str = f"{ratio:.1f}×"

                print(f"{factor_label:15s} {model:15s} | "
                      f"{len(right):10d} {len(wrong):10d} | "
                      f"{r_correct:+10.3f} {r_incorrect:+10.3f} {ratio_str:>8s}")
            print()

    # Markdown format
    print("\n\n" + "=" * 140)
    print("MARKDOWN TABLE FORMAT")
    print("=" * 140)

    for dataset in DATASETS:
        print(f"\n**{dataset.capitalize()} (partial correlation; gene_pop controls gene_coo, gene_coo controls gene_pop, qpop controls gene_coo+gene_pop):**\n")
        print(f"| Factor | Model | Correct | Incorrect | Ratio |")
        print(f"|--------|-------|:-------:|:---------:|:-----:|")
        for factor in ['gene_pop', 'gene_coo', 'question_pop']:
            factor_label = f"{factor}→conf"
            for model in MODELS:
                if _skip_bball_qwen25(dataset, model):
                    continue
                samples = all_data.get((dataset, model), [])
                right = [s for s in samples if s['acc'] == 1]
                wrong = [s for s in samples if s['acc'] == 0]

                if factor == 'gene_pop':
                    r_correct = calc_partial(right, factor, control_key='gene_coo')
                    r_incorrect = calc_partial(wrong, factor, control_key='gene_coo')
                elif factor == 'gene_coo':
                    r_correct = calc_partial(right, factor, control_key='gene_pop')
                    r_incorrect = calc_partial(wrong, factor, control_key='gene_pop')
                elif factor == 'question_pop':
                    r_correct = calc_partial(right, factor, control_keys=['gene_coo', 'gene_pop'])
                    r_incorrect = calc_partial(wrong, factor, control_keys=['gene_coo', 'gene_pop'])

                if np.isnan(r_correct) or np.isnan(r_incorrect):
                    ratio_str = "—"
                    correct_str = f"{r_correct:+.3f}" if not np.isnan(r_correct) else "—"
                    incorrect_str = f"{r_incorrect:+.3f}" if not np.isnan(r_incorrect) else "—"
                elif abs(r_correct) < 0.001:
                    ratio_str = "∞" if r_incorrect > 0.001 else "—"
                    correct_str = f"{r_correct:+.3f}"
                    incorrect_str = f"**{r_incorrect:+.3f}**" if abs(r_incorrect) > abs(r_correct) and r_incorrect > 0 else f"{r_incorrect:+.3f}"
                elif r_correct * r_incorrect <= 0:
                    ratio_str = "—"
                    correct_str = f"{r_correct:+.3f}"
                    incorrect_str = f"{r_incorrect:+.3f}"
                else:
                    ratio = r_incorrect / r_correct
                    ratio_str = f"{ratio:.1f}×"
                    correct_str = f"{r_correct:+.3f}"
                    incorrect_str = f"**{r_incorrect:+.3f}**" if abs(r_incorrect) > abs(r_correct) else f"{r_incorrect:+.3f}"

                print(f"| {factor_label} | {model} | {correct_str} | {incorrect_str} | {ratio_str} |")
        print()


if __name__ == '__main__':
    main()