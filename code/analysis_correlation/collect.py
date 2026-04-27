from utils.utils import read_json, write_jsonl
from sklearn.metrics import roc_auc_score
from collections import Counter
import math
import re
from utils.plot import *
from utils.preprocess import post_acc_detection, basketball_filte, get_pop_books

def cluster_elements_entropy(lst: list, n=10):
    """
    将列表中每n个元素聚类,同一个类别组成一个list
    Return:
    - [ [ ([], []) ] ], 返回每个类以及类中元素对在lst中对应的索引
    """
    result = []
    for i in range(0, len(lst), n):
        # 每隔 n 个元素分成一个子组
        group = lst[i:i + n]
        
        # 统计每个元素的出现次数及其对应的索引
        element_indices = {}
        for idx, elem in enumerate(group):
            original_idx = i + idx  # 计算元素在原始列表中的索引
            if elem in element_indices:
                element_indices[elem].append(original_idx)
            else:
                element_indices[elem] = [original_idx]
        
        # 创建包含聚类元素及其原始索引的子组
        clustered_group = [
            ([elem] * len(indices), indices) if len(indices) > 1 else ([elem], indices)
            for elem, indices in element_indices.items()
        ]
        result.append(clustered_group)
    
    return result
    
def compute_auroc(conf_list, acc_list):
    auroc = roc_auc_score(acc_list, conf_list)
    return auroc

def get_relevant_questions(data, cnt):
    """
    从模型回复中,提取各个问题
    """
    questions= []
    for item in data:
        temp_question = item['Res'].split('\n')
        temp_question = [item for item in temp_question if len(item) !=0 and 'Here are 10' not in item]
        if len(temp_question) != cnt:
            print(temp_question)
        questions += [{'question': re.sub(r"^\d+\.\s*", "", item.strip()), 'reference': ['no reference']} for item in temp_question]
    print(len(questions))
    return questions

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

def get_consistency_auroc(stable_data, consistency_data):
    """
    利用consistency代表confidence,计算auroc
    Return:
    - acc_list: stable data的acc列表
    - conf_list: 利用consistency_data估计出的confidence
    """
    print(f'function: get_consistency_auroc')
    sample_cnt = len(consistency_data) / len(stable_data)
    consistency_data = [item['consistency'] for item in consistency_data] # 0/1表示是否一致

    conf_list = [sum(consistency_data[i:i+int(sample_cnt)])/sample_cnt for i in range(0, len(consistency_data), 10)]
    acc_list = [item['has_answer'] for item in stable_data ]
    print(f'confidence: {sum(conf_list)/len(conf_list)}')
    print(f'acc: {sum(acc_list)/len(acc_list)}')
    # auroc = compute_auroc(conf_list, acc_list)
    # print(f"AUROC: {auroc:.4f}")
    return acc_list, conf_list

def get_consistency_align(dev_stable, dev_consis, test_stable, test_consis):
    """
    为consistency数据选阈值并得到其alignment
    """
    dev_acc_list, dev_conf = get_consistency_auroc(dev_stable, dev_consis)
    thre, _ = select_conf_thre(dev_acc_list, dev_conf)

    acc_list, conf = get_consistency_auroc(test_stable, test_consis)
    get_align_constant_thre(conf, acc_list)
    conf = [1 if x > thre else 0 for x in conf]

    align = [conf[t] == acc_list[t] for t in range(len(acc_list))]
    print(f'Alignment: {sum(align)/len(align)}')


def get_relevant_consistency_auroc(stable_data, rele_stable_data, consis_data, rele_consis_data):
    _, stable_conf = get_consistency_auroc(stable_data, consis_data)
    acc_list = [item['has_answer'] for item in stable_data]
    _, rele_conf = get_consistency_auroc(rele_stable_data, rele_consis_data)
    rele_cnt = 3
    rele_conf_1 = [rele_conf[i] for i in range(0, len(rele_conf), rele_cnt)]
    # rele_conf_2 = [sum(rele_conf[i:i+2])/2 for i in range(0, len(rele_conf), rele_cnt)]
    # rele_conf_3 = [sum(rele_conf[i:i+rele_cnt])/3 for i in range(0, len(rele_conf), rele_cnt)]
    total_conf = []
    for alpha in [round(i * 0.1, 1) for i in range(11)]:
        final_conf = [(1-alpha) * stable_conf[t] + alpha * rele_conf_1[t] for t in range(len(stable_conf))]
        total_conf.append(final_conf)
        auroc = compute_auroc(final_conf, acc_list)
        print(f"alpha: {alpha}, AUROC: {auroc:.4f}, Avg conf: {sum(final_conf)/len(final_conf)}")
    return acc_list, total_conf

def get_relevant_consistency_align(dev_stable, dev_rele_stable, dev_consis, dev_rele_consis, test_stable, test_rele_stable, test_consis, test_rele_consis):

    # 通过验证集选合适的alpha和阈值
    dev_acc_list, dev_conf = get_relevant_consistency_auroc(dev_stable, dev_rele_stable, dev_consis, dev_rele_consis)
    best_thre = {'thre': 0, 'align': 0, 'id': 0}
    for id, item in enumerate(dev_conf):
        thre, align = select_conf_thre(dev_acc_list, item)
        if align > best_thre['align']:
            best_thre = {'thre': thre, 'align': align, 'id': id}
    thre = best_thre['thre']
    id = best_thre['id']

    acc_list, conf = get_relevant_consistency_auroc(test_stable, test_rele_stable, test_consis, test_rele_consis)
    get_align_constant_thre(conf[id], acc_list)
    conf = [1 if x > thre else 0 for x in conf[id]]

    align = [conf[t] == acc_list[t] for t in range(len(acc_list))]
    print(sum(align)/len(align))
    
    
def get_semantic_entropy_auroc(stable_data, entropy_data):
    """
    利用semantic entropy来计算auroc
    """
    sample_cnt = len(entropy_data) / len(stable_data)
    cluster_ids = [item['cluster_id'] for item in entropy_data]
    logprobs = [sum(item['Log_p']['token_logprobs']) for item in entropy_data]
    norm_logprobs = [sum(item['Log_p']['token_logprobs'])/len(item['Log_p']['token_logprobs']) for item in entropy_data]
    clusters = cluster_elements_entropy(cluster_ids, int(sample_cnt))

    acc_list = [item['has_answer'] for item in stable_data]
    black_entropy_list = []
    grey_entropy_list = []
    grey_entropy_norm_list = []
    for item in clusters:
        # black-box 每个类的prob
        black_temp_prob = [len(t[0])/sample_cnt for t in item]
        # grey-box 每个类的prob
        grey_temp_prob = [sum([math.exp(logprobs[id]) for id in t[1]]) for t in item]
        grey_prob = [t/sum(grey_temp_prob) for t in grey_temp_prob]
        # grey-box normalized 每个类的prob
        grey_temp_prob_norm = [sum([math.exp(norm_logprobs[id]) for id in t[1]]) for t in item]
        grey_prob_norm = [t/sum(grey_temp_prob_norm) for t in grey_temp_prob_norm]

        black_entropy_list.append(sum(p * math.log(p) for p in black_temp_prob if p > 0))
        grey_entropy_list.append(sum(p * math.log(p) for p in grey_prob if p > 0))
        grey_entropy_norm_list.append(sum(p * math.log(p) for p in grey_prob_norm if p > 0))
    auroc = compute_auroc(black_entropy_list, acc_list)
    grey_auroc = compute_auroc(grey_entropy_list, acc_list)
    grey_norm_auroc = compute_auroc(grey_entropy_norm_list, acc_list)
    print(f"AUROC: {auroc:.4f}")
    print(f"Grey AUROC: {grey_auroc:.4f}")
    print(f"Normalized Grey AUROC: {grey_norm_auroc:.4f}")

def get_token_prob_auroc(stable_data):   
    acc_list = [item['has_answer'] for item in stable_data]
    ppl = [sum(item['Log_p']['token_logprobs']) for item in stable_data]
    norm_ppl = [sum(item['Log_p']['token_logprobs'])/len(item['Log_p']['token_logprobs']) for item in stable_data]
    conf = [math.exp(item) for item in ppl]
    norm_conf = [math.exp(item) for item in norm_ppl]
    auroc = compute_auroc(conf, acc_list)
    norm_auroc = compute_auroc(norm_conf, acc_list)
    print(f'avg conf: {sum(conf)/len(conf)}')
    print(f'avg norm conf: {sum(norm_conf)/len(norm_conf)}')
    print(f"AUROC: {auroc:.4f}")
    print(f"Normalized AUROC: {norm_auroc:.4f}")
    return acc_list, conf, norm_conf

def get_entropy_auroc(stable_data):
    entropy_list = []
    norm_entropy_list = []
    for idx in range(len(stable_data)):
        entropy = []
        for item in stable_data[idx]['Log_p']['top_logprobs']:
            token_entropy = sum([-math.exp(t['logprob']) * t['logprob'] for t in item])
            entropy.append(token_entropy)
        entropy_list.append(-sum(entropy))
        norm_entropy_list.append(-sum(entropy)/len(entropy))
    acc_list = [item['has_answer'] for item in stable_data]
    auroc = compute_auroc(entropy_list, acc_list)
    norm_auroc = compute_auroc(norm_entropy_list, acc_list)
    print(f"AUROC: {auroc:.4f}")
    print(f"Normalized AUROC: {norm_auroc:.4f}")

def get_token_prob_align(dev_data, test_data):
    dev_acc_list, dev_conf, dev_norm_conf = get_token_prob_auroc(dev_data)
    thre, _ = select_conf_thre(dev_acc_list, dev_conf)
    norm_thre, _ = select_conf_thre(dev_acc_list, dev_norm_conf)

    acc_list, conf, norm_conf = get_token_prob_auroc(test_data)
    get_align_constant_thre(norm_conf, acc_list)

    conf = [1 if x > thre else 0 for x in conf]
    norm_conf = [1 if x > norm_thre else 0 for x in norm_conf]

    align = [conf[t] == acc_list[t] for t in range(len(acc_list))]
    norm_align = [norm_conf[t] == acc_list[t] for t in range(len(acc_list))]
    print(f'ppl align: {sum(align)/len(align)}')
    print(f'normalized ppl align: {sum(norm_align)/len(norm_align)}')

def get_align_constant_thre(conf_list, acc_list):
    conf_list = [item > 0.5 for item in conf_list]
    align = [conf_list[idx] == acc_list[idx] for idx in range(len(acc_list))]
    overcon = [conf_list[idx] > acc_list[idx] for idx in range(len(acc_list))]
    conserv = [conf_list[idx] < acc_list[idx] for idx in range(len(acc_list))]

    print(f'Alignment for confidence threshold=0.5: {sum(align)/len(align)}')
    print(f'overcon: {sum(overcon)/len(overcon)}')
    print(f'conserv: {sum(conserv)/len(conserv)}')

def popularity_acc(data):
    popu_list = []
    wrong_list = []
    prob_list = []

    for item in data:
        popu_list.append(item['popularity'])
        wrong_list.append(1 - item['has_answer'])
        token_probs = item['Log_p']['token_probs']
        # logprobs = [math.log(t) for t in token_probs]

        prob_list.append(sum(token_probs)/len(token_probs))
    # plot_popularity_acc_fixed_intervals(popu_list, prob_list)
    plot_popularity_acc(popu_list, prob_list)
    




if __name__ == '__main__':
    stable_data = read_json('./data/res/nq/nq_test_sample_gpt4o_stable.jsonl')
    consis_data = read_json('./data/res/nq/nq_test_sample_gpt4o_consistency.jsonl')
    equal_consis_data = read_json('./data/res/nq/nq_test_sample_gpt4o_ans_gpt4o_equal_gpt4o_consistency.jsonl')
    # cluster_ids = read_json('./data/res/nq/nq_test_sample_gpt4o_entropy.jsonl')
    rele_stable_data = read_json('./data/res/nq/nq_test_sample_gpt4o_relevant_stable.jsonl')
    rele_consis_data = read_json('./data/res/nq/nq_test_sample_gpt4o_relevant_consistency.jsonl')
    acc_list, conf = get_consistency_auroc(stable_data, equal_consis_data)
    # get_semantic_entropy_auroc(stable_data, cluster_ids)
    # acc_list, conf, norm_conf = get_token_prob_auroc(stable_data)
    # get_entropy_auroc(stable_data)
    # acc_list, total_conf = get_relevant_consistency_auroc(stable_data, rele_stable_data, consis_data, rele_consis_data)

    # load dev data
    # dev_stable_data = read_json('./data/res/nq/nq_dev_sample_gpt4o_stable.jsonl')
    # dev_consis_data = read_json('./data/res/nq/nq_dev_sample_gpt4o_consistency.jsonl')
    # dev_rele_stable_data = read_json('./data/res/nq/nq_dev_sample_gpt4o_relevant_stable.jsonl')
    # dev_rele_consis_data = read_json('./data/res/nq/nq_dev_sample_gpt4o_relevant_consistency.jsonl')
    # get_consistency_align(stable_data, equal_consis_data, stable_data, equal_consis_data)
    # get_relevant_consistency_align(stable_data, rele_stable_data, consis_data, rele_consis_data, stable_data, rele_stable_data, consis_data, rele_consis_data)
    # get_token_prob_align(stable_data, stable_data)

    # plot
    # plot_confidence_sample_ratio(conf)
    # plot_confidence_acc(conf, acc_list)

    # movie data
    path = './data/res/basketball/basketball_chatgpt_temperature1.jsonl'
    data = read_json(path)
    # pop_data = read_json('./data/res/books/writer_book_popularity.jsonl')
    new_data = post_acc_detection(data)
    # new_data = basketball_filte(data)
    # new_data = get_pop_books(data, pop_data)
    # write_jsonl(new_data, path.replace('.jsonl', '_select.jsonl'))
    

