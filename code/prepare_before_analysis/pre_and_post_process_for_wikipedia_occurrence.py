import json
from collections import defaultdict
import sys
sys.path.append('../')
from my_utils.utils import read_json, write_jsonl, remove_punctuation_edges
import os
import pandas as pd
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

def get_all_entities_for_wikipedia(models_names, out_path):
    """
    得到数据中所有出现过的entity, 为查找entity在wikipedia出过的文档的个数作准备
    """
    pattern = {
        'movies': 'Who is the director of the movie ',
        'songs': 'Who is the performer of the song ',
        'basketball': 'Where is the birthplace of the basketball player '
    }
    for dataset in ['movies', 'songs', 'basketball']:
        base_out_path=f'../../res/{dataset}'
        all_entity = []
        for model in models_names:
            path = f'../../res/{dataset}/{dataset}_{model}_temperature1.jsonl'
            question_template = pattern[dataset]
            data = read_json(path)

            for item in data:
                if 'reference' not in item:
                    item['reference'] = item['answer']
                if 'Res' not in item:
                    item['Res'] = item['response'][0]
                all_entity.append(item['question'].replace(question_template, ''))
                all_entity.append(remove_punctuation_edges(item['Res'], dataset))
                all_entity += ([remove_punctuation_edges(t, dataset) for t in item['reference']])
            all_entity = list(set(all_entity))
        
        outfile=os.path.join(base_out_path, dataset + '_' + out_path)
        print(f'datasets: {dataset}')
        print(len(all_entity))
        write_jsonl(all_entity, outfile)

def get_relevant_entity():
    """
    得到每个问题的entity和其对应的ground truth/gene entity,用于寻找需要的relation popularity
    """
    pattern = {
        'movies': 'Who is the director of the movie ',
        'songs': 'Who is the performer of the song ',
        'basketball': 'Where is the birthplace of the basketball player '
    }
    all_entity = defaultdict(list)
    for dataset in ['movies', 'songs', 'basketball']:
        for model in ['Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']:
            path = f'../../res/{dataset}/{dataset}_{model}_temperature1.jsonl'
            replace_pattern = pattern[dataset]
            data = read_json(path)

            for item in data:
                if 'reference' not in item:
                    item['reference'] = item['answer']
                if 'Res' not in item:
                    item['Res'] = item['response'][0]
                q_entity = item['question'].replace(replace_pattern, '').lower()
                g_entity = [remove_punctuation_edges(item['Res'], dataset)]
                gt_entity = [remove_punctuation_edges(t, dataset) for t in item['reference']]
                g_entity = [item.lower() for item in g_entity]
                gt_entity = [item.lower() for item in gt_entity]
                if q_entity == '' or q_entity == " ":
                    continue
                all_entity[q_entity].extend(g_entity)
                all_entity[q_entity].extend(gt_entity)
    cnt = 0
    for key in all_entity:
        all_entity[key] = list(set(all_entity[key]))
        cnt += 1
    print(cnt)
    with open('../../res/relevant_entity_qwen2.5_7B_14B_32B.json', 'w') as f:
        json.dump(all_entity, f)


def get_cooccurance(key_value_path, entity_doc_path, out_path):
    my_dict = json.loads(open(key_value_path).read())
    # 假设你有一个大文件 'entities_in_docs.jsonl'，格式为 {'entity': [docid1, docid2, ...]}
    path = entity_doc_path
    data = pd.read_parquet(path).to_numpy()
    print(f'len all the entity: {len(my_dict)}')
    print(f'len entity-docid: {len(data)}')

    # 用来存储每个实体的文档ID
    entity_to_docs = defaultdict(set)

    # 逐行读取文件，并构建 entity_to_docs 映射
    for item in data:
        entity = item[0]
        docs = item[1]
        entity_to_docs[entity].update(docs)  # 将 docs 集合加入实体对应的文档集合中
    key_docnum = {}
    for key, value in entity_to_docs.items():
        key_docnum[key] = len(value)
    
    # 结果字典，用来存储每个 key 和它的 value 中实体共同出现的文档数
    result = {key: defaultdict(int) for key in my_dict}

    # 对每个 key 和它的 value 中的每个实体，计算共同出现的文档数
    for key, value_entities in my_dict.items():
        key_docs = entity_to_docs[key]  # 获取 key 对应的文档ID集合
        for value in value_entities:
            if value in entity_to_docs:
                value_docs = entity_to_docs[value]  # 获取 value 对应的文档ID集合
                # 计算 key 和 value 共同出现的文档ID集合的大小
                common_docs = key_docs.intersection(value_docs)
                result[key][value] = len(common_docs)
                print(f'value doc: {len(value_docs)}, key doc: {len(key_docs)}, common: {len(common_docs)}')

    # 将结果保存为 JSON 文件
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)


def merge_cooccurrence_json_overwrite(file1, file2, output_file):
    """
    合并两个 cooccurrence JSON 文件，遇到重复的 entity-pair 时使用 file2 的值覆盖 file1 的值。
    """
    with open(file1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    with open(file2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)

    merged = {}

    # 先拷贝 file1 的内容
    for key, subdict in data1.items():
        merged[key] = dict(subdict)

    # 用 file2 覆盖或新增
    for key, subdict in data2.items():
        if key not in merged:
            merged[key] = dict(subdict)
        else:
            for subkey, count in subdict.items():
                merged[key][subkey] = count  # 直接覆盖（不相加）

    with open(output_file, 'w', encoding='utf-8') as fout:
        json.dump(merged, fout, ensure_ascii=False, indent=2)

    print(f"Merged (overwrite) saved to {output_file}")

if __name__ == '__main__':
    # 统计所有的entity, 后续在wikipedia中统计每个entity在wikipedia的哪些文档中出现过
    # model_names=['Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']
    # out_path='Qwen2.5-7B-14B-32B-all-entity.jsonl'
    # get_all_entities_for_wikipedia(model_names, out_path)
    # 调用search_entity_in_wikipedia.py,在wikipedia中统计上面的每个实体在哪些doc中出现过

    # 得到每个问题的entity和其对应的ground truth/gene entity,用于寻找需要的relation popularity
    # get_relevant_entity()

    # 通过entity-docid的文件，在wikipedia中统计entity的共现
    # key_value_path = '../../res/relevant_entity_qwen2.5_7B_14B_32B.json'
    # entity_doc_path = '../../res/all_entity_docid_qwen2.5-7b-14b-32b.parquet'
    # out_path = '../../res/cooccurrence_qwen2.5_7B_14B_32B.json'
    # get_cooccurance(key_value_path, entity_doc_path, out_path)

    merge_cooccurrence_json_overwrite(
    "../../res/cooccurance_qwen2_llama3_chatgpt.json",
    "../../res/cooccurrence_qwen2.5_7B_14B_32B.json",
    "../../res/cooccurrence_qwen2_llama3_chatgpt_qwen2.5_7B_14B_32B.json"
    )