import os
import json
import re
import random

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

SYSTEM_PROMPT = "You are a rigorous logical reasoning assistant. For any user query, you must first output a detailed step-by-step thinking process wrapped in <think> and </think> tags, and then output the final conclusion wrapped in <answer> and </answer> tags."

def extract_math_answer(solution_text):
    # 提取 \boxed{} 里的最终答案，一般来说数学竞赛题的最终答案都会放在 \boxed{} 里
    matches = re.findall(r'\\boxed\{((?:[^{}]|{[^{}]*})*)\}', str(solution_text))
    if matches:
        return matches[-1] 
    return None
#首先是从Math里面抽了500条数据，然后从ECQA里面抽了500条数据，最后混合打乱生成1000条冷启动数据。每条数据的格式是一个包含system、user和assistant三轮对话的JSON对象，assistant的内容包含了<tool_call>和<answer>标签，分别对应模型的思考过程和最终答案，dim/competition_math_selected，yangdong/ecqa这两个都是从HFmirror上拉取的（服务器不用挂梯子），前者是数学竞赛题，后者是一个包含多选题的常识推理数据集。
def process_math_data(num_samples=500):
    dataset = load_dataset("dim/competition_math_selected", split="train")
    
    processed_data = []

    for item in dataset:
        problem = item.get('problem', '')
        solution = item.get('solution', '')
        
        final_answer = extract_math_answer(solution)
        if not final_answer:
            continue
            
        thought_process = solution.replace(f"\\boxed{{{final_answer}}}", final_answer).strip()
        
        message = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": f"<think>\n{thought_process}\n</think>\n<answer>{final_answer}</answer>"}
            ]
        }
        processed_data.append(message)
        
        if len(processed_data) >= num_samples:
            break
            
    return processed_data

def process_ecqa_data(num_samples=500):
   
    dataset = load_dataset("yangdong/ecqa", split="train")
    
    processed_data = []
    for item in dataset:
        question = item.get('q_text', '')
       
        choices = "\n".join([
            f"A. {item.get('q_op1', '')}", 
            f"B. {item.get('q_op2', '')}", 
            f"C. {item.get('q_op3', '')}", 
            f"D. {item.get('q_op4', '')}", 
            f"E. {item.get('q_op5', '')}"
        ])
        full_question = f"{question}\nOptions:\n{choices}"
        
        thought_process = item.get('taskB') or item.get('taskA_pos') or ''
        final_answer = item.get('q_ans', '')
        
        if not question or not thought_process or not final_answer:
            continue
            
        message = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_question},
                {"role": "assistant", "content": f"<think>\n{thought_process}\n</think>\n<answer>{final_answer}</answer>"}
            ]
        }
        processed_data.append(message)
        
        if len(processed_data) >= num_samples:
            break
            
    return processed_data

def main():
    math_data = process_math_data(500)
    ecqa_data = process_ecqa_data(500)
    
    # 打乱
    mixed_data = math_data + ecqa_data
    random.shuffle(mixed_data)
    
    output_file = "cold_start_sft_english_1000.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in mixed_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()