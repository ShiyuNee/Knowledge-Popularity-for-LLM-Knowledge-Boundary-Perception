import json
from torch.utils.data import DataLoader, Dataset, RandomSampler
from utils.prompt import get_prompt
import pandas as pd
import os
import random

def read_json(path):
    qa_data = []
    f = open(path, 'r', encoding='utf-8')
    for line in f.readlines():
        qa_data.append(json.loads(line))
    return qa_data

def assign_levels(data, field, num_levels=10):
    field_values = [item[field] for item in data if isinstance(item[field], (int, float))]
    unique_values = sorted(set(field_values))
    step = len(unique_values) // num_levels
    boundaries = [unique_values[i * step] for i in range(1, num_levels)]
    boundaries.append(float('inf'))
    levels = []
    for item in data:
        value = item[field]
        if isinstance(value, (int, float)):
            current_level = 1
            for boundary in boundaries:
                if value <= boundary:
                    break
                current_level += 1
            levels.append(current_level)
            item[field + '_level'] = current_level
        else:
            levels.append(None)
    return levels

def select_samples_from_levels(data, levels, required_levels):
    grouped_by_level = {}
    for item, level in zip(data, levels):
        if level is not None:
            grouped_by_level.setdefault(level, []).append(item)
    selected_samples = []
    for level in required_levels:
        if level in grouped_by_level and grouped_by_level[level]:
            selected_samples.append(grouped_by_level[level].pop(0))
    return selected_samples

class QADataset(Dataset):
    def __init__(self, args):
        self.data = self.read(args.source)
        self.prompts = []
        self.idxs = []
        self.args = args
        self.few_shot_examples = []
        if args.n_shot > 0:
            self.few_shot_examples = self.get_few_shot_examples()
        self.get_prompted_data()

    def read(self, path):
        qa_data = []
        f = open(path, 'r', encoding='utf-8')
        for line in f.readlines():
            qa_data.append(json.loads(line))
        return qa_data
    
    def get_prompted_data(self):
        for idx in range(len(self.data)):
            if 'info' not in self.data[idx]:
                self.idxs.append(idx)
                self.prompts.append(get_prompt(self.data[idx], self.args, self.few_shot_examples)) 
        for item in self.prompts[:5]:
            print(f'example: {item}')

    def get_few_shot_examples(self):
        use_model = 'llama8b'
        data = read_json(f'./data/clean_data_for_pop_generation/{self.args.dataset_name}_{use_model}_temperature1.jsonl')
        new_data = []
        for item in data:
            if item["question_pop"] != "No" and item["gene_pop"] != "No":
                new_data.append(item)

        field_to_rank = f"{self.args.gene_type}_pop"
        levels = assign_levels(data, field_to_rank, num_levels=10)

        if self.args.n_shot == 3:
            need_levels = [2, 5, 8]
        elif self.args.n_shot == 5:
            need_levels = [1, 3, 5, 7, 9]
        elif self.args.n_shot == 10:
            need_levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        few_shot_examples = select_samples_from_levels(data, levels, need_levels)

        return few_shot_examples

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return self.prompts[index]
