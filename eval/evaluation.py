import os
import json
import re
import torch
import pandas as pd
import ast
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# 配置
TEST_DATA_PATH = "/root/data/gsm8k/test.parquet"
BASE_MODEL_PATH = "Qwen/Qwen3-0.6B"
SFT_MODEL_PATH = "/root/workspace/ft/qwen3-0.6b-cold-start-sft"
OUTPUT_DIR = "/root/workspace/eval"

def extract_answer(text):
    """从生成文本中提取最后的数字答案"""
    # 查找最后出现的数字（可能是答案）
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    return None

def extract_ground_truth(reward_model):
    """从 reward_model 字段中提取 ground_truth"""
    if isinstance(reward_model, str):
        reward_model = json.loads(reward_model)
    return reward_model.get('ground_truth', '')

def parse_prompt(prompt_str):
    """解析 prompt 字段（来自 pandas，可能是字符串或列表）"""
    if isinstance(prompt_str, str):
        # 如果是字符串，尝试解析为 Python 列表
        try:
            prompt = ast.literal_eval(prompt_str)
        except:
            prompt = [{"role": "user", "content": prompt_str}]
    else:
        prompt = prompt_str
    
    return prompt

def evaluate_model(model_path, model_name, test_df, tokenizer, device):
    """评估单个模型"""
    print(f"\n{'='*80}")
    print(f"评估 {model_name} ({model_path})...")
    print(f"{'='*80}\n")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    model.eval()
    
    correct = 0
    total = 0
    predictions = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"推理中..."):
        try:
            # 解析输入
            prompt = parse_prompt(row['prompt'])
            ground_truth = extract_ground_truth(row['reward_model'])
            
            # 构造对话
            messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
            
            # 生成回复（使用 apply_chat_template）
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=False
                )
            
            # 解码输出
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 去除输入以获得纯回复
            generated = response[len(text):]
            
            # 提取答案
            pred_answer = extract_answer(generated)
            
            # 判断是否正确
            is_correct = pred_answer == ground_truth if pred_answer else False
            if is_correct:
                correct += 1
            total += 1
            
            predictions.append({
                'question': messages[0].get('content', '')[:100],  # 取前 100 字
                'predicted': pred_answer,
                'ground_truth': ground_truth,
                'correct': is_correct,
                'full_response': generated[:200]  # 保存前 200 字的生成文本
            })
            
        except Exception as e:
            print(f"  样本 {idx} 出错: {e}")
            total += 1
            predictions.append({
                'question': '',
                'predicted': 'ERROR',
                'ground_truth': '',
                'correct': False,
                'full_response': str(e)
            })
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n✅ {model_name} 结果:")
    print(f"   准确率: {accuracy:.2%} ({correct}/{total})")
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'predictions': predictions
    }

def main():
    print("加载测试集...")
    test_df = pd.read_parquet(TEST_DATA_PATH)
    print(f"测试集大小: {len(test_df)}")
    
    # 只用前 100 条进行快速评估
    test_df = test_df.head(100)
    print(f"使用前 100 条进行评估\n")
    
    # 加载 tokenizer（两个模型使用同一个 tokenizer）
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 评估 base 模型
    base_result = evaluate_model(BASE_MODEL_PATH, "Base Model", test_df, tokenizer, device)
    
    # 评估 SFT 模型
    sft_result = evaluate_model(SFT_MODEL_PATH, "SFT-357 Model", test_df, tokenizer, device)
    
    # 对比结果
    print(f"\n{'='*80}")
    print("对比结果:")
    print(f"{'='*80}")
    print(f"Base Model 准确率: {base_result['accuracy']:.2%}")
    print(f"SFT-357 准确率: {sft_result['accuracy']:.2%}")
    improvement = (sft_result['accuracy'] - base_result['accuracy']) / (base_result['accuracy'] + 1e-8)
    print(f"相对提升: {improvement:+.2%}")
    
    # 保存结果到 JSON
    results = {
        'test_set': TEST_DATA_PATH,
        'num_samples': len(test_df),
        'base_model': base_result,
        'sft_model': sft_result,
        'improvement_percentage': improvement * 100
    }
    
    output_file = f"{OUTPUT_DIR}/eval_results.json"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 因为 predictions 可能无法 JSON 序列化，我们只保存摘要
    summary = {
        'test_set': TEST_DATA_PATH,
        'num_samples': len(test_df),
        'base_model': {
            'model': base_result['model'],
            'accuracy': base_result['accuracy'],
            'correct': base_result['correct'],
            'total': base_result['total']
        },
        'sft_model': {
            'model': sft_result['model'],
            'accuracy': sft_result['accuracy'],
            'correct': sft_result['correct'],
            'total': sft_result['total']
        },
        'improvement_percentage': improvement * 100
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 结果已保存至: {output_file}")

if __name__ == "__main__":
    main()
