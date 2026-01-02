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

progress_bar = tqdm(total=num_samples)

output_file = "gsm8k-deepseek-traces"

# create data file if not already created
with open(output_file, "w") as f:
    pass


for i in tqdm(range(num_samples)):
    messages = [
        {"role": "user", "content": f"{ds['train']['question'][i]}\nPlease reason step by step, and put your final answer within \\boxed{{}}."}
    ]
    

    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    outputs = model.generate(input_tensor.to(model.device), max_new_tokens=100)

    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)

    trace = {
        'index': i,
        "problem": ds['train']['question'][i],
        "reasoning": result, 
    }
    
    with open(output_file, "a") as f:
        f.write(json.dumps(trace) + "\n")
        
    progress_bar.update(1)
    
