from transformers import AutoTokenizer, LlamaForCausalLM
import torch

model = LlamaForCausalLM.from_pretrained("../models/llama2-7B-chat")
tokenizer = AutoTokenizer.from_pretrained("../models/llama2-7B-chat")
device = torch.device('cuda')
model.to(device)

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
print(f'inputs:{inputs.keys()}')
inputs.to(device)

# Generate
outs = model.generate(inputs.input_ids, max_length=30, output_scores=True, return_dict_in_generate=True, output_attentions=True)
print(outs.keys())
print(type(outs['attentions'][0][0]))
# res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
# print(res)
