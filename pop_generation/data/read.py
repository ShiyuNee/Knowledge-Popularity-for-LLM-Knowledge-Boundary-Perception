import json
import re

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

def remove_punctuation_edges(s, name='movies'):
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

def merge_entity_and_pop_into_one_file():
    pattern = {
        'movies': 'Who is the director of the movie ',
        'songs': 'Who is the performer of the song ',
        'basketball': 'Where is the birthplace of the basketball player '
    }
    # NOTE: update paths before running
    pop_data = read_json(f'/path/to/gene_gt_entity_relation_limit.jsonl')
    full_entities_dict = {}
    for d in pop_data:
        full_entities_dict.update(d)

    co_occu = json.loads(open('/path/to/cooccurance.json').read())

    for dataset in ['movies', 'songs', 'basketball']:
        for model in ['llama8b', 'qwen2', 'chatgpt']:
            new_data = []
            data_path = f'/path/to/res/{dataset}/{dataset}_{model}_temperature1.jsonl'
            data = read_json(data_path)
            for item in data:
                temp_item = {}
                question_entity = item['question'].replace(pattern[dataset], "")
                gene_entity = remove_punctuation_edges(item['Res'], dataset)

                temp_item['question_entity'] = question_entity
                temp_item['gene_entity'] = item['Res']
                temp_item['question_pop'] = item['popularity']
                temp_item['gene_pop'] = full_entities_dict[gene_entity]['popularity']
                temp_item['coo_pop'] = co_occu[question_entity.lower()][gene_entity.lower()]

                new_data.append(temp_item)
            out_path = f'./{dataset}_{model}_temperature1.jsonl'
            write_jsonl(new_data, out_path)
