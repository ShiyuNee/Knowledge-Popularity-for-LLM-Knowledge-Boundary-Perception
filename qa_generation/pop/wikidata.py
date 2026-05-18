from SPARQLWrapper import SPARQLWrapper, JSON
import re
from tqdm import tqdm
import time
import os
import json
import random
# from utils.plot import *
from scipy.stats import spearmanr
# from sklearn.metrics import roc_auc_score
import numpy as np

def select_conf_thre(acc_list, conf):
    """
    选择合适的confidence阈值,来将conf二值化,从而与acc_list对应
    acc_list和conf都是dev数据上的
    """
    acc_array = np.array(acc_list)
    conf_array = np.array(conf)
    
    # 构造所有二值化的结果矩阵 (threshold rows x data columns)
    binary_conf_matrix = (conf_array[:, None] > conf_array).astype(int).T
    
    # 计算每个阈值的对齐度
    align_scores = (binary_conf_matrix == acc_array).mean(axis=1)
    
    # 找到最佳阈值和对齐度
    best_idx = np.argmax(align_scores)
    best_thre = conf_array[best_idx]
    best_align = align_scores[best_idx]
    
    return best_thre, best_align

def read_json(path):
    qa_data = []
    f = open(path, 'r', encoding='utf-8')
    for line in f.readlines():
        qa_data.append(json.loads(line))
    return qa_data

def write_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f'write jsonl to: {path}')
    f.close()

# 设置Wikidata的SPARQL查询端点
def get_id_and_sitelinks_count(entity_name):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    # 定义要查询的实体名称
    # entity_name = "Ian MacNaughton"  # 可替换为其他实体名称

    # SPARQL查询：通过实体名称获取该实体的ID和sitelink数量
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
        print(results["results"]["bindings"])
        entity_id = results["results"]["bindings"][0]["entity"]["value"].split('/')[-1]
        sitelink_count = int(results["results"]["bindings"][0]["sitelink_count"]["value"])
        print(f"Entity ID for '{entity_name}': {entity_id}")
        print(f"Sitelink count: {sitelink_count}")
        return entity_id, sitelink_count
    else:
        print(f"No sitelinks found for entity '{entity_name}'.")
        return 'No', 'No'

def get_all_entities_for_greedy_llama8b(data):
    all_entities = []
    for item in data:
        all_entities.append(item['Res'])
        all_entities += item['reference']
    res = list(set(all_entities))
    print(len(res))
    return sorted(res)

def remove_punctuation_edges(s):
    s = s.replace('\n', '')
    s = s.split('(')[0]
    s = re.sub(r'^[^\w]+|[^\w]+$', '', s)
    s = s.strip()
    return s

def get_popularity_for_all_entities():
    data = read_json('./data/res/movies/movie_1_llama8b_temperature1.jsonl')
    total_entities = get_all_entities_for_greedy_llama8b(data)
    res = []
    begin = 0
    outfile = './data/res/movies/ditector_popularity.jsonl'

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
        item = remove_punctuation_edges(item)
        print(item)
        if item in exist_entities:
            continue
        retry_cnt = 0
        while True:
            try:
                print(len(item.split()))
                if retry_cnt > 2 or len(item.split()) >= 20:
                    print('xxxxxxxxxxxxxxxxxxxx')
                    temp_res = {item: {'wiki_id': 'No', 'popularity': 'No'}}
                    res.append(temp_res)
                    f.write(json.dumps(temp_res) + "\n")
                    break
                else:
                    wiki_id, link_cnt = get_id_and_sitelinks_count(item)
                    temp_res = {item: {'wiki_id': wiki_id, 'popularity': link_cnt}}
                    res.append(temp_res)
                    f.write(json.dumps(temp_res) + "\n")
                    sleep_time = random.randint(3, 6)
                    time.sleep(sleep_time)
                    break
            except:
                retry_cnt += 1
                print('Request fail, retry.')
                sleep_time = random.randint(30, 60)
                time.sleep(sleep_time)
    write_jsonl(res, 'director_full.jsonl')
    f.close()
    
# def popularity_post_process(popularity_data, model_res):


class Postprocessor:
    def __init__(self, popularity_data, model_res) -> None:
        self.popularity_data = popularity_data
        self.model_res = model_res
        full_entities_dict = {}
        for d in self.popularity_data:
            full_entities_dict.update(d)
        self.full_entities_dict = full_entities_dict

    def get_correlation_between_gene_gt_entity(self):
        """
        得到生成的entity与ground truth entity popularity之间的correlation
        """
        # 将所有entity合并成一个大字典
        all_conf = []
        all_right_conf = []
        all_wrong_conf = []

        all_gt_popularity = []
        all_gene_popularity = []
        all_acc = []

        all_question_popularity = []
        all_wrong_question_popularity = []
        all_right_question_popularity = []

        gene_right_popularity = []
        gene_wrong_popularity = []
        gene_wrong_gt_popularity = []

        for item in self.model_res:
            gene_entity = remove_punctuation_edges(item['Res'])
            ref = remove_punctuation_edges(item['reference'][0]) # 做对的, 用gene_entity来表示ref_entity, 否则就用第一个ref
            if self.full_entities_dict[gene_entity]['popularity'] == "No" or self.full_entities_dict[ref]['popularity'] == "No":
                continue
            gene_pop = self.full_entities_dict[gene_entity]['popularity']
            ref_pop = self.full_entities_dict[ref]['popularity']
            all_question_popularity.append(item['popularity'])
            all_conf.append(sum(item['Log_p']['token_probs'])/len(item['Log_p']['token_probs']))


            all_acc.append(item['has_answer'])
            if item['has_answer'] == 1:
                gene_right_popularity.append(gene_pop)
                all_gt_popularity.append(gene_pop)
                all_gene_popularity.append(gene_pop)
                all_right_question_popularity.append(item['popularity'])
                all_right_conf.append(sum(item['Log_p']['token_probs'])/len(item['Log_p']['token_probs']))
            else: # 做错的部分
                # if ref_pop > gene_pop:
                    # print(f'question: {item["question"]}, res: {item["Res"]}, res pop: {gene_pop}, ref: {item["reference"]}, ref pop: {ref_pop}')
                all_wrong_conf.append(sum(item['Log_p']['token_probs'])/len(item['Log_p']['token_probs']))
                all_gt_popularity.append(ref_pop)
                gene_wrong_popularity.append(gene_pop)
                gene_wrong_gt_popularity.append(ref_pop)
                all_gene_popularity.append(gene_pop)
                all_wrong_question_popularity.append(item['popularity'])
        large = []
        small = []
        for id, item in enumerate(gene_wrong_popularity):
            if gene_wrong_gt_popularity[id] >= item:
                large.append([gene_wrong_gt_popularity[id], item])
            else:
                small.append([gene_wrong_gt_popularity[id], item])
        print(sum([x[0] for x in small]) / len([x[0] for x in small]))
        print(sum([x[1] for x in small]) / len([x[1] for x in small]))
        # print(f'data count: {len(all_acc)}, ave acc: {sum(all_acc)/len(all_acc)}, Avg gt_popularity: {sum(all_gt_popularity)/len(all_gt_popularity)}, Avg gene_popularity: {sum(all_gene_popularity)/len(all_gene_popularity)}')
        # print(f'wrong data count: {len(gene_wrong_popularity)}, Avg gt_popularity: {sum(gene_wrong_gt_popularity)/len(gene_wrong_gt_popularity)}, Avg gene_popularity: {sum(gene_wrong_popularity)/len(gene_wrong_popularity)}, Gene popularity > gt popularity ratio: {sum(gene_larger_that_gt_ratio)/len(gene_larger_that_gt_ratio)}')
        # plot_popularity_acc(all_gene_popularity, all_acc)


        sorted_conf_indices = np.argsort(all_conf) # 返回排序后的索引
        sorted_gene_popularity = np.array(all_gene_popularity)[sorted_conf_indices]
        # sorted_gt_popularity = np.array(gene_wrong_gt_popularity)[sorted_gt_indices]
        # sorted_question_popularity = np.array(all_wrong_question_popularity)[sorted_question_indices]
        # sorted_ratio = sorted_gene_popularity > sorted_gt_popularity
        sorted_conf = np.array(all_conf)[sorted_conf_indices]
        sorted_acc = np.array(all_acc)[sorted_conf_indices]


        
        # sorted_gt_indices = np.argsort(gene_wrong_gt_popularity) # 返回排序后的索引
        # sorted_gt_popularity = np.array(gene_wrong_gt_popularity)[sorted_gt_indices]

        # sorted_gene_indices = np.argsort(gene_wrong_popularity)
        # sorted_gene_popularity = np.array(gene_wrong_popularity)[sorted_gt_indices]
        spearman_corr, p_value = spearmanr(sorted_conf, sorted_gene_popularity)
        print("Spearman 相关系数:", spearman_corr)
        print("p-value:", p_value)
        # plot_line(sorted_gt_popularity, sorted_gene_popularity)
        # plot_popularity_acc(sorted_gt_popularity, sorted_ratio, name='Gene > GT ratio')

        # thre, align = select_conf_thre(all_acc, all_conf)
        # print(f'thre: {thre}, align: {align}')
        # self.overcon_conserv(all_conf, all_acc, all_gene_popularity, thre)
        # plot_scaler(all_gene_popularity, all_conf, all_acc)
        # plot_popularity_for_acc_in_confidence_interval(sorted_conf, sorted_gene_popularity, sorted_acc)
        self.confidence_calibration(all_conf, all_gene_popularity, all_acc)

    def spelling_error(self):
        res = []
        for item in self.model_res:
            gene_entity = remove_punctuation_edges(item['Res'])
            ref = remove_punctuation_edges(item['reference'][0]) # 做对的, 用gene_entity来表示ref_entity, 否则就用第一个ref
            if self.full_entities_dict[gene_entity]['popularity'] == "No" or self.full_entities_dict[ref]['popularity'] == "No":
                continue
            if item['has_answer'] == 0:
                gene_tokens = self.full_entities_dict[gene_entity]['token_ids']
                ref_tokens = self.full_entities_dict[ref]['token_ids']
                res.append(gene_tokens[0] == ref_tokens[0])
        print(f'data cnt: {len(res)}, spelling error ratio: {sum(res)/len(res)}')

    def overcon_conserv(self, conf, acc, all_pop, thre):
        assert len(conf) == len(acc)
        overcon = []
        overcon_pop = []
        overcon_conf = []
        conserv = []
        conserv_pop = []
        conserv_conf = []
        large_conf_pop = []
        small_conf_pop = []
        large_conf = []
        small_conf = []
        for id, item in enumerate(conf):
            if item > thre:
                overcon.append(acc[id] == 0)
                large_conf_pop.append(all_pop[id])
                large_conf.append(item)
                if acc[id] == 0:
                    overcon_pop.append(all_pop[id])
                    overcon_conf.append(item)
            else:
                conserv.append(acc[id] == 1)
                small_conf_pop.append(all_pop[id])
                small_conf.append(item)
                if acc[id] == 1:
                    conserv_pop.append(all_pop[id])
                    conserv_conf.append(item)
        print(f'overconfidence: {sum(overcon)/len(conf)}, pop: {sum(overcon_pop)/len(overcon_pop)}, conf: {sum(overcon_conf)/len(overcon_conf)}')
        print(f'large conf pop: {sum(large_conf_pop)/len(large_conf_pop)}, conf: {sum(large_conf)/len(large_conf)}')
        print(f'conserve: {sum(conserv)/len(conf)}, pop: {sum(conserv_pop)/len(conserv_pop)}, conf: {sum(conserv_conf)/len(conserv_conf)}')
        print(f'small conf pop: {sum(small_conf_pop)/len(small_conf_pop)}, conf: {sum(small_conf)/len(small_conf)}')

    def normalize_with_smoothing(self, lst, constant):
        # 加1平滑
        smoothed = [x + constant for x in lst]
        # 归一化
        max_value = max(smoothed)
        normalized = [x / max_value for x in smoothed]
        return normalized
    
    def confidence_calibration(self, confidence, popularity, acc):
        best_align = 0.0
        best_para = {'constant': 0, 'alpha': 0}
        for constant in range(280):
            for alpha in [round(i * 0.01, 2) for i in range(101)]:
                norm_pop = self.normalize_with_smoothing(popularity, constant)
                
                final_score = [confidence[idx] - alpha * norm_pop[idx] for idx in range(len(confidence))]
                thre, align = select_conf_thre(acc, final_score)
                if align > best_align:
                    best_align = align
                    best_para = {'constant': constant, 'alpha': alpha}
                print(f'constant: {constant}, alpha: {alpha}, align: {align}')
        print(f'best align: {best_align}, best para: {best_para}')

        # auroc = roc_auc_score(acc, final_score)
        # print(f"AUROC: {auroc}")
        
        # plot_scaler(popularity, final_score, acc)




if __name__ == '__main__':
    popularity_data = read_json('./ditector_popularity.jsonl')
    model_res = read_json('./movie_1_llama8b_temperature1.jsonl')
    # get_popularity_for_all_entities()
    processor = Postprocessor(popularity_data, model_res)
    processor.get_correlation_between_gene_gt_entity()



