import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
from datasets import load_dataset
import json

model_name = "deepseek-ai/deepseek-math-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

ds = load_dataset("openai/gsm8k", "main")
num_samples = len(ds['train'])


output_file = "gsm8k-deepseek-traces.jsonl"

# create data file if not already created
with open(output_file, "w") as f:
    pass


for i in tqdm(range(num_samples)):
    messages = [
        {"role": "user", "content": f"{ds['train']['question'][i]}\nPlease reason step by step, and put your final answer within \\boxed{{}}."}
    ]
    

    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    
    attention_mask = torch.ones_like(input_tensor)
    
    outputs = model.generate(
        input_tensor.to(model.device),
        attention_mask=attention_mask.to(model.device), 
        max_new_tokens=512,
        do_sample=False, 
        pad_token_id=tokenizer.eos_token_id
    )

    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)

    trace = {
        'index': i,
        "problem": ds['train']['question'][i],
        "reasoning": result, 
        "answer": ds['train']['answer'][i],
    }
    
    with open(output_file, "a") as f:
        f.write(json.dumps(trace) + "\n")
        
    
