import json
# from vllm import LLM, SamplingParams
from datasets import load_dataset
from tqdm import tqdm 
import sys

def main():


    ds = load_dataset("qwedsacf/competition_math")
            
    samples_to_process = 12499
    
    output_file = "SCoTD-deepseek-math-v2"   

    llm = LLM(model="deepseek-ai/deepseek-math-7b-instruct")

    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=1024,
        top_p=0.9
    )

    progress_bar = tqdm(total=samples_to_process)

    for i in range(0, samples_to_process, 1):
        prompt = f"{ds['train']['problem'][i]} \n Please reason step by step, and put your final answer within \\\\boxed{{}}."

        outputs = llm.generate([prompt] * 6, sampling_params)

        solutions = []
        for output in outputs:
            full_response = output.outputs[0].text
            
            # check if output prompt is too short or doesn't contain an actual answer.
            while (len(full_response) < 5 or "\\\\boxed" not in full_response):
                out = llm.generate(prompt, sampling_params)
                full_response = out.outputs[0].text
            
            solutions.append(full_response)


        data_point = {
            "problem": ds["train"]["problem"][i],
            "thinking_traces": solutions,
            "correct_answer": ds["train"]["solution"][i],
            "subject": ds["train"]["type"][i],
            "level": ds["train"]["level"][i]
        }

        with open(output_file, "a") as f:
            f.write(json.dumps(data_point) + "\n")

        progress_bar.update(1)
    
main()