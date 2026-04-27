from SPARQLWrapper import SPARQLWrapper, JSON
import re
from tqdm import tqdm
import time
import os
import json
import random
import numpy as np
import math
from utils.plot import *
from acl_plot import plot_line_between_pop_and_acc_conf_align, plot_tank, plot_seaborn_boxplots
from scipy.stats import spearmanr
from collect import select_conf_thre
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


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

    
def extract_number(s):
    match = re.search(r'\d+', s)  # 查找连续的数字
    if match:
        return int(match.group())  # 转换为整数
    return 0  # 如果没有找到数字，返回None

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

def get_top_or_bottom_k_percent_for_acc(acc, conf, pop, gt_pop, k, acc_value=1, top=True):
    """
    从 acc 列表中筛选出值为 acc_value 的数据，
    获取对应的 conf 中前 k% 或后 k% 的元素，
    同时返回这些位置对应的 pop 值。

    参数：
        acc (list): 0/1 值列表。
        conf (list): 用于排序的列表。
        pop (list): 对应值的列表，长度应与 acc 和 conf 相同。
        k (float): 百分比值 (0 < k <= 100)。
        acc_value (int): 筛选的 acc 值 (0 或 1)。
        top (bool): 是否取前 k%。True 表示取前 k%，False 表示取后 k%。

    返回：
        tuple: (conf 的前 k% 或后 k% 的元素列表, pop 中对应的值列表)。
    """
    if not (0 < k <= 100):
        raise ValueError("k 必须在 (0, 100] 范围内。")

    if len(acc) != len(conf) or len(conf) != len(pop) or len(conf) != len(gt_pop):
        raise ValueError("acc, conf 和 pop 列表必须非空且长度相同。")

    # 筛选出 acc 等于 acc_value 的索引
    filtered_indices = [i for i, value in enumerate(acc) if value == acc_value]

    # 根据筛选后的索引提取 conf 和 pop 对应的值
    filtered_conf = [conf[i] for i in filtered_indices]
    filtered_pop = [pop[i] for i in filtered_indices]
    filtered_gt_pop = [gt_pop[i] for i in filtered_indices]

    # 计算需要的元素数量
    n = len(filtered_conf)
    count = max(1, int(n * k / 100))  # 至少取一个元素

    # 获取排序后的索引
    sorted_indices = sorted(range(n), key=lambda i: filtered_conf[i])

    if top:
        selected_indices = sorted_indices[-count:]
    else:
        selected_indices = sorted_indices[:count]

    # 根据索引获取 conf 和 pop 中对应的值
    selected_conf = [filtered_conf[i] for i in selected_indices]
    selected_pop = [filtered_pop[i] for i in selected_indices]
    selected_gt_pop = [filtered_gt_pop[i] for i in selected_indices]

    return selected_conf, selected_pop, selected_gt_pop

def ece(acc, confidence, n_bins=10):
    # 将置信度和准确度按置信度分为n_bins个区间
    bins = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    n = len(acc)  # 样本总数
    
    for i in range(n_bins):
        # 获取属于第i个区间的样本
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        
        # 找出置信度在区间[bin_lower, bin_upper)内的样本
        bin_mask = (confidence >= bin_lower) & (confidence < bin_upper)
        bin_acc = np.array(acc)[bin_mask]  # 对应样本的准确率
        bin_conf = np.array(confidence)[bin_mask]  # 对应样本的置信度
        
        # 如果该区间没有样本，则跳过
        if len(bin_acc) == 0:
            continue
        
        # 计算该区间的平均准确率和平均置信度
        avg_acc = np.mean(bin_acc)
        avg_conf = np.mean(bin_conf)
        
        # 计算该区间的误差并加权
        bin_size = len(bin_acc)
        ece += (bin_size / n) * np.abs(avg_acc - avg_conf)
    
    return ece


def calculate_partition_means(pop, acc, conf, gene_pop, gt_pop, n, k=20):
    """
    对排序后的 pop 序列进行等样本切分，并合并相邻 pop_mean 相同的区间。
    
    参数：
    - pop: 数列1 (list 或 np.array)
    - acc: 数列2 (list 或 np.array)
    - conf: 置信度 (list 或 np.array)
    - gene_pop: 基因流行度 (list 或 np.array)
    - n: 划分的目标区间数
    - k: 用于计算 top 或 bottom 百分比的参数

    返回：
    - (pop_mean, acc_mean, conf等其他统计指标的列表)
    """
    # 转换为 numpy 数组并按 pop 排序
    pop, acc, conf, gene_pop, gt_pop = map(np.array, (pop, acc, conf, gene_pop, gt_pop))
    sorted_indices = np.argsort(pop)
    pop = pop[sorted_indices]
    acc = acc[sorted_indices]
    conf = conf[sorted_indices]
    gene_pop = gene_pop[sorted_indices]
    gt_pop = gt_pop[sorted_indices]


    # 样本总数及每区间目标样本数
    total_samples = len(pop)
    samples_per_partition = max(total_samples // n, 1)  # 防止样本数过少导致无法划分

    # 初步切分
    partitions = []
    for i in range(0, total_samples, samples_per_partition):
        partitions.append({
            "pop": pop[i:i + samples_per_partition],
            "acc": acc[i:i + samples_per_partition],
            "conf": conf[i:i + samples_per_partition],
            "gene_pop": gene_pop[i:i + samples_per_partition],
            'gt_pop': gt_pop[i:i + samples_per_partition]
        })

    # 合并相邻区间
    merged_partitions = [partitions[0]]
    for i in range(1, len(partitions)):
        current_partition = partitions[i]
        prev_partition = merged_partitions[-1]
        if np.mean(current_partition["pop"]) == np.mean(prev_partition["pop"]):
            # 合并区间
            for key in current_partition:
                prev_partition[key] = np.concatenate((prev_partition[key], current_partition[key]))
        else:
            merged_partitions.append(current_partition)

    # 统计合并后的结果
    results = {
        "pop": [],
        "acc": [],
        "conf": [],
        "gap": [],
        "right_conf": [],
        "wrong_conf": [],
        "gene_right_pop": [],
        "gene_wrong_pop": [],
        "gt_right_pop": [],
        "gt_wrong_pop": [],
        "count": [],
        "NMI": []
    }

    for partition in merged_partitions:
        pop_mean = np.mean(partition["pop"])
        acc_mean = np.mean(partition["acc"])
        conf_mean = np.mean(partition["conf"])
        # gap_mean = np.mean([abs(a - c) for a, c in zip(partition["acc"], partition["conf"])])
        gap_mean = ece(partition['acc'], partition['conf'])

        right_bottom_conf, right_bottom_ans_pop, right_bottom_gt_pop = get_top_or_bottom_k_percent_for_acc(
            partition["acc"], partition["conf"], partition["gene_pop"], partition["gt_pop"], k, 1, False)
        
        wrong_top_conf, wrong_top_ans_pop, wrong_top_gt_pop = get_top_or_bottom_k_percent_for_acc(
            partition["acc"], partition["conf"], partition["gene_pop"], partition["gt_pop"], k, 0, True)
        
        # NMI = mutual_information_and_nmi(partition["gene_pop"], partition["gt_pop"], partition["pop"], 6000000)
        NMI=0
        
        results["pop"].append(pop_mean)
        results["acc"].append(acc_mean)
        results["conf"].append(conf_mean)
        results["gap"].append(gap_mean)
        results["right_conf"].append(np.mean(right_bottom_conf))
        results["wrong_conf"].append(np.mean(wrong_top_conf))
        results["gene_right_pop"].append(np.mean(right_bottom_ans_pop))
        results["gene_wrong_pop"].append(np.mean(wrong_top_ans_pop))
        results["gt_right_pop"].append(np.mean(right_bottom_gt_pop))
        results["gt_wrong_pop"].append(np.mean(wrong_top_gt_pop))
        results["count"].append(len(partition["pop"]))
        results["NMI"].append(NMI)
    # print(len(results['count']))
    # print(results['count'])
    end_idx = len(results['count'])
    if results['count'][-1] <= 20:
        end_idx = -1
        
    # print(f'NMI: {results["NMI"][:end_idx]}')
    # print(f'co-occurrence: {results["pop"][:end_idx]}')
    return (results["pop"][:end_idx], results["acc"][:end_idx], results["conf"][:end_idx], results["right_conf"][:end_idx], 
            results["wrong_conf"][:end_idx], results["gene_right_pop"][:end_idx], results["gene_wrong_pop"][:end_idx], results["gt_right_pop"][:end_idx], results["gt_wrong_pop"][:end_idx], results["gap"][:end_idx], results["NMI"][:end_idx])


# 设置Wikidata的SPARQL查询端点
def compute_auroc(conf_list, acc_list):
    auroc = roc_auc_score(acc_list, conf_list)
    return auroc

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
    
def query_entity_relations(entity_name): # 没什么必要, 限制太严格了很容易查不到
    """
    通过实体名称查询其 QID，并统计该实体作为主语或宾语关联的关系及次数。
    """
    timeout=10
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setTimeout(timeout)  # 设置超时时间

    # SPARQL 查询：获取实体 QID
    qid_query = f"""
    SELECT ?entity WHERE {{
        ?entity rdfs:label "{entity_name}"@en.
    }}
    LIMIT 1
    """
    sparql.setQuery(qid_query)
    sparql.setReturnFormat(JSON)

    qid_results = sparql.query().convert()

    # 提取实体 QID
    if not qid_results["results"]["bindings"]:
        print(f"No entity found for '{entity_name}'.")
        return "No", "No", "No"
    entity_id = qid_results["results"]["bindings"][0]["entity"]["value"].split("/")[-1]
    print(f"Entity Name: {entity_name}")
    print(f"Entity ID (QID): {entity_id}")

    # SPARQL 查询：统计关联关系及次数，并标明作为主语或宾语
    relation_query = f"""
    SELECT ?relation (COUNT(?relation) AS ?relation_count) ?role WHERE {{
        {{
            # 实体作为主语
            wd:{entity_id} ?relation ?object.
            BIND("subj" AS ?role)
        }}
        UNION
        {{
            # 实体作为宾语
            ?subject ?relation wd:{entity_id}.
            BIND("obj" AS ?role)
        }}
    }}
    GROUP BY ?relation ?role
    ORDER BY DESC(?relation_count)
    """
    sparql.setQuery(relation_query)
    sparql.setReturnFormat(JSON)
    relation_results = sparql.query().convert()

    # 输出关联关系及其次数，标注主语或宾语
    relations = []
    roles = []
    counts = []
    print("\nRelations and their counts (subj and obj):")
    for result in relation_results["results"]["bindings"]:
        relation = result["relation"]["value"].split("/")[-1]  # 提取关系 ID
        count = int(result["relation_count"]["value"])  # 获取关系次数
        role = result["role"]["value"]  # 主语还是宾语
        relations.append(relation)
        roles.append(role)
        counts.append(count)
        # print(f"  {relation} (as {role}): {count}")
    return relations, roles, counts


def get_all_entities_for_greedy_llama8b(data):
    all_entities = []
    for item in data:
        all_entities.append(item['Res'])
        all_entities += item['reference']
        if 'constraint' in item:
            if type(item['constraint']) == list:
                all_entities += item['constraint']
            else:
                all_entities.append(item['constraint'])
    all_entities = ["" if item is None else item for item in all_entities]
    res = list(set(all_entities))
    print(len(res))
    return sorted(res)

def remove_punctuation_edges(s, name='movies'):
    """
    basketball数据集可能输出 "城市, 国家", 因此需要用逗号split
    """
    s = s.replace('\n', '')
    s = s.split('(')[0].strip()
    if name in ['basketball']:
        s = s.split(',')[0].strip()
    else:
        if len(s) <= 20:
            s = s.split(',')[0].strip()
    s = re.sub(r'^[^\w]+|[^\w]+$', '', s)
    s = s.strip()
    return s

def mutual_information_and_nmi(x_counts_list, y_counts_list, xy_counts_list, N):
    """
    计算互信息 (I(X; Y)) 和归一化互信息 (NMI(X, Y))
    
    参数：
        x_counts_list: x的出现次数列表。
        y_counts_list: y的出现次数列表。
        xy_counts_list: x和y共现次数列表。
    
    返回：
        I: 互信息值。
        NMI: 归一化互信息值。
    """
    N = 6000000  # 总样本数

    mutual_information = 0.0
    H_x = 0.0  # X的边际熵
    H_y = 0.0  # Y的边际熵
    
    for movie_count, director_count, joint_count in zip(x_counts_list, y_counts_list, xy_counts_list):
        # 跳过没有共现的数据
        if joint_count == 0:
            continue
        
        # 计算概率
        P_xy = joint_count / N  # 联合概率
        P_x = movie_count / N   # 电影的边际概率
        P_y = director_count / N  # 导演的边际概率
        
        # 计算互信息增量
        if P_x * P_y > 0:
            mutual_information += P_xy * math.log(P_xy / (P_x * P_y), 2)
        
        # 计算X和Y的边际熵增量
        H_x -= P_x * math.log(P_x, 2) if P_x > 0 else 0
        H_y -= P_y * math.log(P_y, 2) if P_y > 0 else 0
    
    # 计算归一化互信息
    NMI = mutual_information / math.sqrt(H_x * H_y) if H_x > 0 and H_y > 0 else 0

    print(f"互信息 (Mutual Information): {mutual_information}")
    print(f"归一化互信息 (Normalized Mutual Information): {NMI}")
    
    return NMI

def calculate_average_dimensions(x, indices):
    """
    计算indices中的样本在x中各个维度上的平均值
    - x: [[a,b,c,d,...]]
    - indices: 需要考虑的x样本的索引
    """
    # 初始化每个维度的总和
    sum_conf = 0
    sum_pop = 0
    sum_qpop = 0
    sum_cooc = 0
    
    # 遍历所有指定索引
    for i in indices:
        # 确保索引在有效范围内
        if i < len(x):
            sum_conf += x[i][0]  # confidence
            sum_pop += x[i][1]   # popularity
            sum_qpop += x[i][2]  # question_pop
            sum_cooc += x[i][3]  # cooccurrence
    
    # 计算平均值
    count = len(indices)
    if count == 0:
        return None  # 避免除以零
    
    avg_conf = sum_conf / count
    avg_pop = sum_pop / count
    avg_qpop = sum_qpop / count
    avg_cooc = sum_cooc / count
    
    return [avg_conf, avg_pop, avg_qpop, avg_cooc]

def compute_overconf_conserv(acc, conf):
    wrong_rec = 0
    right_rec = 0
    # print(acc)
    wrong_cnt = len([item for item in acc if item == 0])
    right_cnt = len([item for item in acc if item == 1])
    for idx in range(len(acc)):
        if acc[idx] == 0 and conf[idx] == 0:
            wrong_rec += 1
        if acc[idx] == 1 and conf[idx] == 1:
            right_rec += 1
    print(f'wrong recognize: {wrong_rec/wrong_cnt}')
    print(f'right recognize: {right_rec/right_cnt}')

class Postprocessor:
    def __init__(self, popularity_data, model_res, model) -> None:
        self.popularity_data = popularity_data
        self.model_res = model_res
        full_entities_dict = {}
        for d in self.popularity_data:
            full_entities_dict.update(d)
        self.full_entities_dict = full_entities_dict
        self.model = model

    def get_correlation_between_gene_gt_entity(self, dataset, type, cooccurrence_path, single_occurrence_path):
        """
        得到生成的entity与ground truth entity popularity之间的correlation
        """
        # 将所有entity合并成一个大字典
        all_acc = []
        all_conf = []

        all_question_popularity = []
        all_gt_popularity = []
        all_gene_popularity = []

        gene_wrong_popularity = []
        gene_wrong_gt_popularity = []
        
        no_acc = []
        # 统计共现关系
        cooccurance = []
        gt_cooccurance = []

        # 模型做错时, gt ans与question的共现, gene ans与question的共现
        wrong_gt_coo = []
        wrong_gene_coo = []

        #wikipedia出现次数
        question_single_occ = []
        gt_single_occ = []
        

        pattern = {
        'movies': 'Who is the director of the movie ',
        'songs': 'Who is the performer of the song ',
        'basketball': 'Where is the birthplace of the basketball player '
        }   
        # 在wikipedia上统计的结果
        co_occu = json.loads(open(cooccurrence_path).read())
        single_occr = json.loads(open(single_occurrence_path).read())
        
        for item in self.model_res:
            if item['Res'] == "" or item['Res'] == None:
                continue
            # 数据集本身的问题, question/gt entity pop找不到, continue
            if item['popularity'] == "No":
                continue

            question_entity = item['question'].replace(pattern[dataset], '').lower()
            ref = remove_punctuation_edges(item['reference'][0], dataset) # 做对的, 用gene_entity来表示ref_entity, 否则就用第一个ref
            gene_entity = remove_punctuation_edges(item['Res'], dataset)
            # 过滤 gene/ref entity
            # if self.full_entities_dict[gene_entity]['popularity'] == "No" or self.full_entities_dict[ref]['popularity'] == "No":
            #     no_acc.append(item['has_answer'])
            #     continue
            # # 过滤single occr不合规的
            if dataset in ['movies', 'songs'] and (
                single_occr[question_entity.lower()] > 6000 or single_occr[gene_entity.lower()] > 6000 or single_occr[ref.lower()] > 6000
            ):
                continue
            
            gene_pop = self.full_entities_dict[gene_entity]['popularity'] if self.full_entities_dict[gene_entity]['popularity'] != "No" else 0
            ref_pop = self.full_entities_dict[ref]['popularity'] if self.full_entities_dict[ref]['popularity'] != "No" else 0
            all_question_popularity.append(item['popularity'])
            # 计算confidence
            if 'gpt' in self.model:
                probs = [math.exp(t) for t in item['Log_p']['token_logprobs']]
                temp_conf = sum(probs)/len(probs)
            else:
                temp_conf = sum(item['Log_p']['token_probs'])/len(item['Log_p']['token_probs'])
            # 两个从wikipedia中统计的single occurrence
            question_single_occ.append(single_occr[question_entity.lower()])
            gt_single_occ.append(single_occr[ref.lower()])

            # 基于wikidata统计的
            all_conf.append(temp_conf)
            all_acc.append(item['has_answer'])
            all_gt_popularity.append(gene_pop)
            all_gene_popularity.append(gene_pop)
            # 基于wikipedia统计的共现次数
            cooccurance.append(co_occu[question_entity][gene_entity.lower()])
            gt_cooccurance.append(co_occu[question_entity][gene_entity.lower()])
            if item['has_answer'] == 0: # 做对的不用统计，因为gene和gt entity相同
                # 统计做错部分,ref和gene entity的pop的差异
                gene_wrong_gt_popularity.append(ref_pop)
                gene_wrong_popularity.append(gene_pop)

                wrong_gt_coo.append(co_occu[question_entity][ref.lower()])
                wrong_gene_coo.append(co_occu[question_entity][gene_entity.lower()])
            # 下面这块是计算互信息的
            # N = 6000000 # 总共的文档个数
            # gt_nmi_list = []
            # for nmi_idx in range(len(question_single_occ)):
            #     P_xy = gt_cooccurance[nmi_idx] / N  # 联合概率
            #     P_x = question_single_occ[nmi_idx] / N   # 电影的边际概率
            #     P_y = gt_single_occ[nmi_idx] / N # 导演的边际概率
                
            #     # 计算互信息增量
            #     if P_x * P_y > 0 and P_xy > 0:
            #         gt_nmi_list.append(P_xy * math.log(P_xy / (P_x * P_y), 2))
            #     else:
            #         gt_nmi_list.append(0)


        wrong_gene_larger_than_gt = [gene_wrong_popularity[idx] > gene_wrong_gt_popularity[idx] for idx in range(len(gene_wrong_gt_popularity))]
        all_align = [1 - abs(all_acc[i] - all_conf[i]) for i in range(len(all_acc))]

        # print(f'total cnt: {len(all_acc)}, acc: {sum(all_acc)/len(all_acc)}, avg conf:{sum(all_conf)/len(all_conf)}, avg align: {sum(all_align)/len(all_align)}')
        # print(f'wrong gene pop: {round(sum(gene_wrong_popularity)/len(gene_wrong_popularity), 3)}, wrong gt pop: {round(sum(gene_wrong_gt_popularity)/len(gene_wrong_gt_popularity), 3)}, larger ratio: {sum(wrong_gene_larger_than_gt)/len(wrong_gene_larger_than_gt)}')
        gene_coo_larger_ratio = [wrong_gene_coo[t] < wrong_gt_coo[t] for t in range(len(wrong_gt_coo))]

        print(f'wrong gt co-occ: {round(sum(wrong_gt_coo)/len(wrong_gt_coo), 3)}, wrong gene co-occ: {round(sum(wrong_gene_coo)/len(wrong_gene_coo), 3)}, larger_ratio: {round(sum(gene_coo_larger_ratio)/len(gene_coo_larger_ratio), 3)}')
        # plot_tank(wrong_gt_coo, wrong_gene_coo)
        

        if type == 'question':
            used_pop = all_question_popularity
        elif type == 'gene':
            used_pop = all_gt_popularity
        elif type == 'coo':
            used_pop = gt_cooccurance
        else:
            raise ValueError('Specify the wrong type')
        # spearman_acc_nmi, _ = spearmanr(gt_nmi_list, all_conf)
        # print(f'spearman nmi & acc: {spearman_acc_nmi}')

        
        span_pop, span_acc, span_conf, right_conf, wrong_conf, right_gene_pop, wrong_gene_pop, right_gt_pop, wrong_gt_pop, align, _ = calculate_partition_means(used_pop, all_acc, all_conf, all_gene_popularity, all_gt_popularity, 10, 100)
        align = [1-item for item in align]
        # 分块计算相关性
        # spearman_acc_pop, _ = spearmanr(span_pop, span_acc)
        # spearman_conf_pop, _ = spearmanr(span_pop, span_conf)
        # spearman_gap_pop, _ = spearmanr(span_pop, align)

        # 不分块，直接计算相关性
        spearman_acc_pop, _ = spearmanr(used_pop, all_acc)
        spearman_conf_pop, _ = spearmanr(used_pop, all_conf)
        spearman_gap_pop, _ = spearmanr(used_pop, all_align)

        # plot
        y_lim_upper = 1.0
        y_lim_bottom = 0.0
        # plot_line_between_pop_and_acc_conf_align(span_pop, span_acc, span_conf, align, spearman_acc_pop, spearman_conf_pop, spearman_gap_pop, self.model, dataset, y_lim_bottom, y_lim_upper)

        return round(spearman_acc_pop, 3), round(spearman_conf_pop, 3), round(spearman_gap_pop, 3), round(sum(all_acc)/len(all_acc), 3), round(sum(all_conf)/len(all_conf), 3), round(sum(all_align)/len(all_align), 3)

    
    def compute_alignment(self, type, seed, test_ratio, dataset, model):
        all_conf = []
        all_acc = []
        all_question_popularity = []
        all_gene_popularity = []
        all_cooccurrence = []
        no_acc = []
        no_align = []
        no_conf = []
        no_conf_question_pop = []
        no_conf_gene_pop = []
        no_conf_question_gene_pop = []

        no_gene_pop_ratio_for_wrong = []

        pattern = {
        'movies': 'Who is the director of the movie ',
        'songs': 'Who is the performer of the song ',
        'basketball': 'Where is the birthplace of the basketball player '
        }   
        co_occu = json.loads(open('./data/cooccurance.json').read())

        llm_generate_question_pop = read_json(f'./llm_pop_generation/{dataset}/{dataset}_{model}_question_pop_3.jsonl')
        llm_generate_gene_pop = read_json(f'./llm_pop_generation/{dataset}/{dataset}_{model}_gene_pop_3.jsonl')
        llm_generate_coo = read_json(f'./llm_pop_generation/{dataset}/{dataset}_{model}_coo_pop_3.jsonl')

        for idx, item in enumerate(self.model_res):
            gene_entity = remove_punctuation_edges(item['Res'], dataset)
            question_entity = item['question'].replace(pattern[dataset], '').lower()
            if 'gpt' in self.model:
                probs = [math.exp(t) for t in item['Log_p']['token_logprobs']]
                temp_conf = sum(probs)/len(probs)
            else:
                temp_conf = sum(item['Log_p']['token_probs'])/len(item['Log_p']['token_probs'])

            if item['has_answer'] == 0:
                if self.full_entities_dict[gene_entity]['popularity'] == "No":
                    no_gene_pop_ratio_for_wrong.append(1)
                else:
                    no_gene_pop_ratio_for_wrong.append(0)
            
            question_pop = item['popularity']
            gene_pop = self.full_entities_dict[gene_entity]['popularity']
            coo_pop = co_occu[question_entity.lower()][gene_entity.lower()]

            # question_pop = extract_number(llm_generate_question_pop[item['origin_idx']]['Res'])
            # gene_pop = extract_number(llm_generate_gene_pop[item['origin_idx']]['Res'])
            # coo_pop = extract_number(llm_generate_coo[item['origin_idx']]['Res'])
            all_question_popularity.append(question_pop)
            all_gene_popularity.append(gene_pop)
            all_cooccurrence.append(coo_pop)
            all_conf.append(temp_conf)
            all_acc.append(item['has_answer'])
        # print(f'avg acc: {sum(all_acc)/len(all_acc)}')
        # no_gene_pop_ratio = sum(no_gene_pop_ratio_for_wrong)/len(no_gene_pop_ratio_for_wrong)
        # print(f'no gene pop ratio for wrong data: {1 - round(no_gene_pop_ratio, 3)}')
        filtered_align, model, thre_align, thre = self.logistic_regression(all_conf, all_gene_popularity, all_question_popularity, all_cooccurrence, all_acc, type, seed, test_ratio, dataset, model)
        return sum(thre_align) / len(thre_align)


    def logistic_regression(self, confidence, popularity, question_pop, cooccurrence, acc, type, seed=42, test_ratio=0.5, dataset='moives', model_name='llama8b'):
        case_res = []
        if type == 'conf':
            x = [[confidence[i]] for i in range(len(confidence))]
        elif type == 'conf_pop':
            x = [[popularity[i]] for i in range(len(confidence))]
        elif type == 'conf_question':
            x = [[question_pop[i]] for i in range(len(confidence))]
        elif type == 'conf_pop_question':
            x = [[cooccurrence[i]] for i in range(len(confidence))]
        elif type == 'conf_pop_question_coo':
            x = [[confidence[i], popularity[i], question_pop[i], cooccurrence[i]] for i in range(len(confidence))]
        else:
            raise ValueError('wrong type')
        cali_x = x = [[confidence[i], popularity[i], question_pop[i], cooccurrence[i]] for i in range(len(confidence))]

        indices = np.arange(len(acc))
        y = acc
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(x, y, indices, test_size=test_ratio, random_state=seed)
        print(f'acc: {sum(y_test)/len(y_test)}')

        thre, _ = select_conf_thre(y_train, [item[0] for item in X_train])
        
        model = LogisticRegression()

        # 训练模型
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)
        align = [y_pred[i] == y_test[i] for i in range(len(y_pred))]
        # print(f'y_pred: {sum(y_pred)/len(y_pred)}')
        # print(f'filtered align: {sum(align)/len(align)}')

        thre_pred = [1 if x > thre else 0 for x in [item[0] for item in X_test]]
        thre_align = [thre_pred[i] == y_test[i] for i in range(len(thre_pred))]
        print(f'conf thre: {thre}')
        print(f'thre_pred: {sum(thre_pred)/len(thre_pred)}')


        cali_path=f'./data/case_study/{dataset}/{model_name}_conf_pop_question_coo0.npy'
        cali_align = np.load(cali_path)
        cali_conf = []
        for idx in range(len(cali_align)):
            if y_test[idx] == 0:
                if cali_align[idx] == 0:
                    cali_conf.append(1)
                else:
                    cali_conf.append(0)
            else:
                if cali_align[idx] == 0:
                    cali_conf.append(0)
                else:
                    cali_conf.append(1)
        


        compute_overconf_conserv(y_test, cali_conf)
        compute_overconf_conserv(y_test, thre_pred)
        # cali_indices = []
        # cali_overconf = []
        # cali_conserv = []
        # wrong_cali_indices = []
        # print(f'cali alignment: {sum(cali_align)/len(cali_align)}')

        # for idx in range(len(thre_align)):
        #     if thre_align[idx] == 0 and cali_align[idx] == 1:
        #         cali_indices.append(test_indices[idx])
        #         if thre_pred[idx] == 1:
        #             cali_overconf.append(test_indices[idx])
        #         else:
        #             cali_conserv.append(test_indices[idx])

        #     if thre_align[idx] == 1 and cali_align[idx] == 0:
        #         wrong_cali_indices.append(test_indices[idx])
        
        # # 为什么能calibrate?因为哪个popularity低或者高
        # cali_overconf_pop=calculate_average_dimensions(cali_x, cali_overconf)
        # cali_conserv_pop=calculate_average_dimensions(cali_x, cali_conserv)
        

        # print(f'right calibration: {len(cali_indices)/len(thre_align)}')
        # print(f'overconf calibration: {len(cali_overconf)/len(thre_align)}')
        # print(f'conserv calibration: {len(cali_conserv)/len(thre_align)}')
        # print(f'wrong calibration: {len(wrong_cali_indices)/len(thre_align)}')

        # print(f'overconf calibration pop: {cali_overconf_pop}')
        # print(f'conserv calibration pop: {cali_conserv_pop}')






        # print(f'thre: {thre}')
        print(f'thre align: {sum(thre_align) / len(thre_align)}')
        return align, model, thre_align, thre

def write_xlsx_with_header(data, path, datasets, models, types):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active

    header = ["Dataset", "Model"]

    # corr部分
    metrics = ["acc_corr", "conf_corr", "gap_corr"]
    for t in types:
        for m in metrics:
            header.append(f"{m}_{t}")

    # 全局
    header.extend(["avg_acc", "avg_conf", "avg_align"])

    ws.append(header)

    idx = 0
    for dataset in datasets:
        for model in models:
            row = [dataset, model] + data[idx]
            ws.append(row)
            idx += 1

    wb.save(path)
    print(f"Saved to {path}")

def norm_pop(data, type):
    if type == 'ans':
        norm_val = [10, 50]
    else:
        norm_val = [3, 10]
    quantized_data = []
    for val in data:
        if val < 100:
            # 对小于100的数按5为间隔量化
            quantized_val = round(val / norm_val[0]) * norm_val[0]
        else:
            # 对大于等于100的数按50为间隔量化
            quantized_val = round(val / norm_val[1]) * norm_val[1]
        quantized_data.append(quantized_val)
    quantized_data = np.array(quantized_data)
    data_min = np.min(quantized_data)
    data_max = np.max(quantized_data)
    normalized_data = (quantized_data - data_min) / (data_max - data_min)
    return normalized_data


def get_results_and_plot(): # 训练+测试
    pop_name = {
        'movies': 'director',
        'songs': 'performer',
        'basketball': 'birthplace',
        'headquarter': 'headquarter',
        'citizenship': 'citizenship',
        'books': 'writer_book'
    }   
    res = []
    seeds = [0]
    # seeds = [42]
    # train_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    train_ratios = [0.5]
    datasets = ['movies', 'songs', 'basketball']
    models = ['llama8b', 'qwen2', 'chatgpt']
    for train_ratio in train_ratios:
        for id, type in enumerate(['conf']):
            temp_res = []
            for dataset in datasets:
                for model in models: 
                    cur_res = []
                    for seed in seeds:
                        print('--------------------------------------------------------')
                        print(f'dataset: {dataset}, model: {model}')
                        # popularity_data = read_json(f'./data/res/{dataset}/popularity.jsonl')
                        popularity_data = read_json(f'./data/res/gene_gt_entity_relation_limit.jsonl')
                        if dataset == 'basketball': # basketball数据集的label_imbalance太严重, 因此做一下样本均衡
                            model_res = read_json(f'./data/res/{dataset}/{dataset}_{model}_compliant_select.jsonl')
                        else:
                            model_res = read_json(f'./data/res/{dataset}/{dataset}_{model}_compliant.jsonl')
                            
                        processor = Postprocessor(popularity_data, model_res, model)
                    
                    # for id, type in enumerate(['conf']):
                        align = processor.compute_alignment(type, seed, 1-train_ratio, dataset, model)
                        cur_res.append(align)
                    cur_res = sum(cur_res)/len(cur_res)
                    temp_res.append(round(cur_res*100, 2))
            res.append(temp_res)

    # write_xlsx(res, 'align_llm_generate.xlsx')


if __name__ == '__main__':
    # get_popularity_for_all_entities()
    # get_results_and_plot()

    dataset_pop = {
        'movies': 'director',
        'songs': 'performer',
        'basketball': 'birthplace',
    }
    plot_res = []
    spearman_res = []
    datasets = ['movies', 'songs', 'basketball']
    models = ['qwen2', 'llama8b', 'chatgpt', 'Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']
    types = ['question', 'gene', 'coo']
    for dataset in datasets:
        for model in models:
            temp_spearman_acc = []
            temp_spearman_conf = []
            temp_spearman_gap = []
            corr_part = []
            avg_part = None
            for type in types:
                res_path = f'../../res/{dataset}/{dataset}_{model}_temperature1.jsonl'
                # 从wikidata统计的
                pop_path = f'../../res/gt_gene_entity_popularity_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.jsonl'
                # 从wikipedia统计的
                cooccurrence_path = f'../../res/cooccurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json'
                single_occurrence_path = f'../../res/single_occurrence_qwen2_llama3_chatgpt_qwen2.5_7b_14b_32b.json'
                popularity = read_json(pop_path)
                res_data = read_json(res_path)
                P = Postprocessor(popularity, res_data, model)
                res = P.get_correlation_between_gene_gt_entity(
                dataset, type, cooccurrence_path, single_occurrence_path
                )
                # 前3个是相关性
                corr_part.extend(res[:3])

                # 后3个是全局平均（只取一次）
                if avg_part is None:
                    avg_part = res[3:]
            spearman_res.append(corr_part + list(avg_part))
                # gt_pop = P.get_correlation_between_gene_gt_entity(dataset, type, cooccurrence_path, single_occurrence_path)
                # plot_res.append(gt_pop)
    # plot_seaborn_boxplots(plot_res, ['Movies', 'Songs', 'Basketball'], 'Ground-Truth Answer Popularity')
    
    write_xlsx_with_header(
        spearman_res,
        'spearmanr.xlsx',
        datasets,
        models,
        types
    ) 