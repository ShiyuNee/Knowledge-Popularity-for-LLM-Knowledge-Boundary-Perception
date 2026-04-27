import pandas as pd
import json
from collections import defaultdict
import concurrent.futures
from tqdm import tqdm
import ahocorasick
import pyarrow as pa
import pyarrow.parquet as pq

# 读取JSONL文件的函数
def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f.readlines()]

# 将 doc_id 转换为整数
def convert_doc_id_to_int(doc_id):
    try:
        return int(doc_id)
    except ValueError:
        raise ValueError(f"Invalid doc_id: {doc_id}, cannot convert to int.")

# 构建Aho-Corasick自动机
def build_automaton(all_entity):
    A = ahocorasick.Automaton()
    for idx, entity in enumerate(all_entity):
        A.add_word(entity, (idx, entity))
    A.make_automaton()
    return A

# 处理单个parquet文件的函数，使用Aho-Corasick自动机进行高效匹配
def process_parquet_file_optimized(file_path, automaton):
    data = pd.read_parquet(file_path).to_numpy()
    doc_entity_map = defaultdict(list)

    for item in tqdm(data, desc=f"Processing {file_path}", ncols=100):
        doc_id = convert_doc_id_to_int(item[0])
        doc_text = item[-1].lower()
        for end_index, (idx, entity) in automaton.iter(doc_text):
            doc_entity_map[entity].append(doc_id)

    return doc_entity_map

# 按行写入 Parquet 文件
def write_to_parquet_line_by_line(all_entity, all_entity_doc, output_path):
    # 创建 Arrow 表的 schema
    schema = pa.schema([("entity", pa.string()), ("doc_ids", pa.list_(pa.int32()))])
    writer = None

    try:
        for entity in tqdm(all_entity, desc="Writing to Parquet", ncols=100):
            doc_ids = all_entity_doc.get(entity, [])
            table = pa.Table.from_pydict({"entity": [entity], "doc_ids": [doc_ids]}, schema=schema)

            if writer is None:
                writer = pq.ParquetWriter(output_path, schema)
            writer.write_table(table)

    finally:
        if writer:
            writer.close()
        print(f"Data written to {output_path}")

# 主函数
def main():
    # 读取所有实体, 统计每个实体在哪些文档中出现
    all_entity = []
    for dataset in ['movies', 'songs', 'basketball']:
        entity_path = f"./Qwen2.5/{dataset}_Qwen2.5-7B-14B-32B-all-entity.jsonl"
        all_entity.extend(read_json(entity_path))
    all_entity = [item.lower() for item in all_entity]
    all_entity = list(set(all_entity))
    print(f'all entity cnt: {len(all_entity)}')

    # 构建 Aho-Corasick 自动机
    automaton = build_automaton(all_entity)

    # 处理的parquet文件路径, 这些是wikipedia的切分数据
    parquet_files = [f'./data/train-000{i:02}-of-00041.parquet' for i in range(41)]

    # 并行处理 parquet 文件
    max_workers = 41  # 调整线程池大小
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(lambda file: process_parquet_file_optimized(file, automaton), parquet_files),
            total=len(parquet_files),
            desc="Processing Parquet Files"
        ))

    # 合并结果到 all_entity_doc 字典
    all_entity_doc = defaultdict(list)
    for result in results:
        for name, doc_ids in result.items():
            all_entity_doc[name].extend(doc_ids)

    # 按行写入到 Parquet 文件
    out_path = './all_entity_docid_qwen2.5-7b-14b-32b.parquet'
    write_to_parquet_line_by_line(all_entity, all_entity_doc, out_path)

if __name__ == "__main__":
    main()
