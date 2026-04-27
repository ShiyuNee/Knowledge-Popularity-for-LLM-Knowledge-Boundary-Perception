import pandas as pd
import json

def get_entity_doc_counts():
    """
    统计每个实体在 wikipedia 上出现的文档数（即 entity-docid 文件中 doc_ids 列表的长度）。
    输出格式为 {entity: doc_count} 的 JSON 文件。
    """
    df = pd.read_parquet("../../res/all_entity_docid_qwen2.5-7b-14b-32b.parquet")

    result = {}

    for _, row in df.iterrows():
        entity = row["entity"]
        doc_ids = row["doc_ids"]

        result[entity] = len(doc_ids) if doc_ids is not None else 0

    # 写入 json
    with open("../../res/single_occurrence_qwen2.5_7B_14B_32B.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def merge_dicts(dict1, dict2):
    """
    递归合并两个字典：
    - 如果 key 不冲突，直接添加
    - 如果 key 冲突：
        - 如果都是 dict，则递归合并
        - 否则用 dict2 覆盖 dict1
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value)
            else:
                result[key] = value  # 覆盖
        else:
            result[key] = value
    
    return result


def merge_json_files(file1, file2, output_file):
    with open(file1, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)
    
    with open(file2, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)
    
    merged = merge_dicts(data1, data2)
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(merged, f_out, ensure_ascii=False, indent=4)


# 使用示例
merge_json_files('../../res/single_occurrence_qwen2.5_7B_14B_32B.json', '../../res/single_occurrence.json', '../../res/single_occurrence_qwen2_llama3_chargpt_qwen2.5_7b_14b_32b.json')