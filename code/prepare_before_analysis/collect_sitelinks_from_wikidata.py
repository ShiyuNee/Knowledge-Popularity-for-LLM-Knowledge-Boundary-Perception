from SPARQLWrapper import SPARQLWrapper, JSON
import re
from tqdm import tqdm
import time
import os
import json
import random
import math
import sys
sys.path.append('../')
from my_utils.utils import read_json, write_jsonl, remove_punctuation_edges


def get_all_entities(data):
    all_entities = []
    for item in data:
        all_entities.append(item['Res']) # 模型生成答案实体
        all_entities += item['reference'] # ground-truth答案实体
    all_entities = ["" if item is None else item for item in all_entities]
    res = list(set(all_entities))
    print(len(res))
    return sorted(res)

def get_id_and_sitelinks_count(entity_name):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    # 定义要查询的实体名称
    # entity_name = "Ian MacNaughton"  # 可替换为其他实体名称

    # SPARQL查询：通过实体名称获取该实体的ID和sitelink数量, 忽略大小写

    sitelink_query = f"""
    SELECT ?entity ?sitelink_count WHERE {{
    ?entity rdfs:label "{entity_name}"@en;
            wikibase:sitelinks ?sitelink_count.
    }}
    LIMIT 1
    """

    # 设置查询并指定返回格式
    sparql.setQuery(sitelink_query)
    sparql.setReturnFormat(JSON)

    # 执行查询并解析结果
    results = sparql.query().convert()

    # 解析并输出sitelink数量
    if results["results"]["bindings"]:
        entity_id = results["results"]["bindings"][0]["entity"]["value"].split('/')[-1]
        sitelink_count = int(results["results"]["bindings"][0]["sitelink_count"]["value"])
        print(f"Entity ID for '{entity_name}': {entity_id}")
        print(f"Sitelink count: {sitelink_count}")
        return entity_id, sitelink_count
    else:
        print(f"No sitelinks found for entity '{entity_name}'.")
        return 'No', 'No'

def get_popularity_for_all_entities():
    """
    访问wikidata获得entity popularity
    """
    dataset_pop = {
        'movies': 'director',
        'songs': 'performer',
        'basketball': 'birthplace',
    }
    for dataset in ['movies', 'songs', 'basketball']:
        for model_name in ['Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']:
            data = read_json(f'../../res/{dataset}/{dataset}_{model_name}_temperature1.jsonl')
            total_entities = get_all_entities(data) # 得到所有生成答案实体和gt答案实体
            res = []
            begin = 0
            outfile = f'../../res/gt_gene_entity_popularity.jsonl'

            if os.path.exists(outfile):
                f = open(outfile, 'r', encoding='utf-8')
                for line in f.readlines():
                    if line != "":
                        begin += 1
                f.close()
                print(begin)
                f = open(outfile, 'a', encoding='utf-8')
            else:
                f = open(outfile, 'w', encoding='utf-8')

            begin = begin + 1
            exist_data = read_json(outfile)
            exist_entities = [list(item.keys())[0] for item in exist_data] if len(exist_data) > 0 else []
            retry_cnt = 0
            for item in tqdm(total_entities):
                item = remove_punctuation_edges(item, dataset)
                print(item)
                if item in exist_entities:
                    continue
                retry_cnt = 0
                while True:
                    try:
                        if retry_cnt >= 1 or len(item.split()) >= 20:
                            print('xxxxxxxxxxxxxxxxxxxx')
                            temp_res = {item: {'id': 'No', 'popularity': 'No'}}
                            res.append(temp_res)
                            f.write(json.dumps(temp_res) + "\n")
                            break
                        else:
                            wiki_id, link_cnt = get_id_and_sitelinks_count(item)
                            temp_res = {item: {'id': wiki_id, 'popularity': link_cnt}}
                            res.append(temp_res)
                            f.write(json.dumps(temp_res) + "\n")
                            # sleep_time = random.randint(2, 5)
                            # time.sleep(sleep_time)
                            break
                    except:
                        retry_cnt += 1
                        print('Request fail, retry.')
                        sleep_time = random.randint(1, 5)
                        time.sleep(sleep_time)
        f.close()

get_popularity_for_all_entities()
