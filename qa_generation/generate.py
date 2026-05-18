from transformers import AutoTokenizer, LlamaForCausalLM
import torch

device = torch.device('cuda')
class engine:
    def __init__(self, path="../models/llama2-7B-chat"):
        self.model = LlamaForCausalLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(device)
        self.model.eval()
        
    def run(self, data):
        

inputs = tokenizer(prompt, return_tensors="pt")
print(f'inputs:{inputs.keys()}')
inputs.to(device)

# Generate
outs = model.generate(inputs.input_ids, max_length=30, output_scores=True, return_dict_in_generate=True, output_attentions=True)
print(outs.keys())
print(type(outs['attentions'][0][0]))