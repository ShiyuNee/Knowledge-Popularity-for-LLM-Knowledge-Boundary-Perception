import json
import jsonlines
import math
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import deal_answer, has_answer, deal_judge_new
from utils.plot import read_json, write_jsonl
from utils.compute import compute_ppl, get_confidence_ppl
import string
from matplotlib.pyplot import MultipleLocator
import random
import re
random.seed(0)

def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join([ch if ch in text and ch not in exclude else ' ' for ch in text])

def remove_pattern(text, patterns):
    text = text.lower()
    for item in patterns:
        text = text.replace(item, '')
    return text

def get_pattern_idx(sample, pattern):
    """
    记录pattern在tokens中的idx
    """
    cnt = 0
    max_len = [2, 1, 0] #最大token序列长度
    # 寻找certain/uncertain对应的索引
    if 'idx' in sample:
        sample.pop('idx')
    flag = 0 # 是否匹配上 pattern
    new_tokens = [i.strip().replace('▁', '') for i in sample['Log_p']['tokens']]
    end_idx = len(new_tokens) - 1 # 从后往前先找是否有uncertain, 再找是否有certain
    while end_idx >= 0:
        for tok_num in max_len:
            start_idx = end_idx - tok_num
            if start_idx >= 0:
                cur_text = ''.join(new_tokens[start_idx: end_idx + 1])
                if cur_text.lower() in pattern:
                    sample['idx'] = list(range(start_idx, end_idx + 1))
                    flag = 1
                    cnt += 1
                    break
        end_idx -= 1 # 结尾从后往前退
        if flag == 1:
            break
    confidence_ppl = get_confidence_ppl(sample, False)
    sample['confidence_ppl'] = confidence_ppl
    return sample
    
def change_file(path, out_path, replace_path, qa_path, ref, mode, post_path="", confidence_idx_path="", replace_idx_path=""):
    """
    替换answer中的uncertain为空
    计算EM, F1, has_answer分数(答案长度<=1就用replace_data计算分数)
    对davinci调用get_pattern_idx,获得pattern在tokens中对应的idx, 对chatgpt添加伪idx(方便后续计算)
    - ref: 数据集中reference对应的key
    - mode: davinci/chatgpt
    """
    pattern = ['uncertainty', 'certainty', 'uncertain', 'certainly', 'certain', 'unsure']
    idx_list, cnt_list = [], []
    data = read_json(path)
    qa_data = read_json(qa_path)
    replace_data = read_json(replace_path) if replace_path != '' else []
    post_data = read_json(post_path) if post_path != "" else []
    no_answer = 0
    ppl_list = []
    confidence_ppl_list = []
    # 替换答案中在pattern中存在的字符串
    for idx in range(len(data)):
        data[idx]['question'] = qa_data[idx]['question']

        if 'Res' not in data[idx] or data[idx]['Res'] == None: # 过滤不合规数据
            continue
        new_res = remove_pattern(data[idx]['Res'], pattern).strip()
        save_res = new_res
        # 关于pattern, 找Giveup, confidence_ppl
        if new_res != data[idx]['Res'].lower(): # 存在pattern
            data[idx]['Giveup'] = deal_judge_new(data[idx]['Res'])
            try:
                data[idx] = get_pattern_idx(data[idx], pattern)
            except:
                pass

        else: # 不存在pattern, 需要post
            cnt_list.append(idx)
            if len(post_data) != 0:
                data[idx]['Giveup'] = deal_judge_new(post_data[idx]['Res'])
                try:
                    data[idx]['confidence_ppl'] = get_pattern_idx(post_data[idx], pattern)['confidence_ppl']
                    data[idx]['idx'] = [-1]
                except:
                    pass
        # 关于answer, 算has_answer和ppl
        data[idx]['ppl'] = compute_ppl(data[idx])
        if len(new_res) <= 1:
            if len(post_data) != 0:
                new_res = replace_data[idx]['Res']
                data[idx]['ppl'] = compute_ppl(replace_data[idx])
            no_answer += 1
            idx_list.append(idx) # replace_idx
        data[idx]['has_answer'] = has_answer(qa_data[idx][ref], new_res)
        data[idx]['Res'] = save_res # 保存的是原始答案去除pattern, 不是replace answer
        try:
            confidence_ppl_list.append(data[idx]['confidence_ppl'])
            ppl_list.append(data[idx]['ppl'])
        except:
            print(f'fail to get ppl or confidence ppl')
    print(f'pattern no match count: {len(cnt_list)}')
    print(f'replace count: {no_answer}')
    print(f'ppl list count: {len(ppl_list)}')
    print(f'avg ppl: {sum(ppl_list) / len(ppl_list)}')
    print(f'avg confidence ppl: {sum(confidence_ppl_list) / len(confidence_ppl_list)}')

    write_jsonl(data, out_path)
    if confidence_idx_path != "":
        write_jsonl(cnt_list, confidence_idx_path) # 保存需要post处理的数据idx
        write_jsonl(idx_list, replace_idx_path) # 记下需要replace data的数据idx

def merge_post_files(qa_path, post_path):
    qa_data = read_json(qa_path)
    post_data = read_json(post_path)
    assert len(qa_data) == len(post_data)
    for idx in range(len(qa_data)):
        qa_data[idx]['Giveup'] = deal_judge_new(post_data[idx]['Res'])
    return qa_data

def post_acc_detection(data):
    acc = []
    origin_acc = []
    for item in data:
        # temp_acc = 0
        # if has_answer(item['reference'], item['Res']):
        #     temp_acc = 1
        # if sum([has_answer([item['Res']], candi_ans) for candi_ans in item['reference']]) >= 1:
        #     temp_acc = 1
        # acc.append(temp_acc)
        origin_acc.append(item['has_answer'])
    print(len(origin_acc))
    print(sum(origin_acc)/len(origin_acc))
    # print(sum(acc)/len(acc))

def basketball_filte(data):
    wrong_data = []
    right_data = []
    for item in data:
        if item['has_answer'] == 1:
            right_data.append(item)
        else:
            wrong_data.append(item)

    all_data = right_data + random.sample(wrong_data, len(right_data))
    print(f'count: {len(all_data)}')
    acc = [item['has_answer'] for item in all_data]
    print(f'acc: {sum(acc)/len(acc)}')
    return all_data
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

def get_pop_books(data, pop):
    full_entities_dict = {}
    for d in pop:
        full_entities_dict.update(d)
    for id, item in enumerate(data):
        item['popularity'] = full_entities_dict[remove_punctuation_edges(item['book'])]['popularity']
    return data


    

if __name__ == '__main__':
    path = './data/res/movies/movies_llama8b_temperature1.jsonl'
    data = read_json(path)
    post_acc_detection(data)