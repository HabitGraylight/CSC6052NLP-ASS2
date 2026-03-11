import os
import json
import re
import argparse
import subprocess
import sys
import torch
import pandas as pd
import ast
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from datetime import datetime

# 配置
TEST_DATA_PATH = "/root/data/gsm8k/test.parquet"
BASE_MODEL_PATH = "Qwen/Qwen3-0.6B"
SFT_MODEL_PATH = "/root/workspace/ft/qwen3-0.6b-cold-start-sft"
RL_CHECKPOINT_PATH = "/root/workspace/verl/output/qwen3_0.6b_grpo_gsm8k/global_step_550"
RL_CHECKPOINT_ROOT = os.path.dirname(RL_CHECKPOINT_PATH)
OUTPUT_DIR = "/root/workspace/eval"
DEFAULT_GPU_ID = 4

HF_WEIGHT_FILES = (
    "pytorch_model.bin",
    "model.safetensors",
    "tf_model.h5",
    "model.ckpt.index",
    "flax_model.msgpack",
)


def has_hf_weights(model_dir):
    """判断目录是否包含 HuggingFace 可识别的权重文件。"""
    if not os.path.isdir(model_dir):
        return False
    return any(os.path.exists(os.path.join(model_dir, filename)) for filename in HF_WEIGHT_FILES)


def detect_fsdp_actor_dir(checkpoint_dir):
    """检测是否为 verl 的 FSDP actor checkpoint 目录。"""
    actor_dir = os.path.join(checkpoint_dir, "actor")
    if not os.path.isdir(actor_dir):
        return None
    has_sharded_weight = any(
        name.startswith("model_world_size_") and name.endswith(".pt") for name in os.listdir(actor_dir)
    )
    if has_sharded_weight and os.path.exists(os.path.join(actor_dir, "fsdp_config.json")):
        return actor_dir
    return None


def merge_fsdp_checkpoint(actor_dir, target_dir):
    """调用 verl.model_merger 将 FSDP checkpoint 转为 HuggingFace 格式。"""
    cmd = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        actor_dir,
        "--target_dir",
        target_dir,
        "--trust-remote-code",
    ]
    print("\n检测到 FSDP checkpoint，开始自动转换为 HuggingFace 权重...")
    print("执行命令:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def resolve_model_path(model_path, auto_merge_fsdp=False):
    """解析模型路径，兼容 HF 目录与 verl FSDP checkpoint 目录。"""
    if not os.path.exists(model_path):
        # 允许直接传入 HuggingFace Hub repo id（如 Qwen/Qwen3-0.6B）
        return model_path

    candidate_dirs = [
        model_path,
        os.path.join(model_path, "actor"),
        os.path.join(model_path, "actor", "huggingface"),
        os.path.join(model_path, "actor", "huggingface_merged"),
    ]

    for candidate in candidate_dirs:
        if has_hf_weights(candidate):
            return candidate

    actor_dir = detect_fsdp_actor_dir(model_path)
    if actor_dir:
        merged_dir = os.path.join(model_path, "actor", "huggingface_merged")
        if has_hf_weights(merged_dir):
            return merged_dir

        merge_cmd = (
            f"python -m verl.model_merger merge --backend fsdp "
            f"--local_dir {actor_dir} --target_dir {merged_dir} --trust-remote-code"
        )
        if auto_merge_fsdp:
            try:
                merge_fsdp_checkpoint(actor_dir, merged_dir)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    "自动 merge FSDP checkpoint 失败，请手动执行以下命令后重试:\n"
                    f"{merge_cmd}\n"
                    f"原始错误: {e}"
                ) from e

            if has_hf_weights(merged_dir):
                return merged_dir

            raise RuntimeError(
                "已执行 checkpoint merge，但未在目标目录发现权重文件。"
                f"请检查目录: {merged_dir}"
            )

        raise FileNotFoundError(
            "检测到 RL checkpoint 是 FSDP 分片格式，不能直接使用 AutoModelForCausalLM.from_pretrained 加载。\n"
            "请先 merge 为 HuggingFace 权重，命令如下:\n"
            f"{merge_cmd}\n"
            "或者重新运行脚本并加上参数 --auto-merge-rl-checkpoint。"
        )

    raise FileNotFoundError(
        f"未找到可加载的 HuggingFace 权重文件。当前路径: {model_path}\n"
        f"期望文件之一: {', '.join(HF_WEIGHT_FILES)}"
    )


def normalize_rl_checkpoint_path(rl_checkpoint):
    """支持传入绝对路径，或类似 global_step_3000 的相对 checkpoint 名。"""
    if os.path.exists(rl_checkpoint):
        return rl_checkpoint

    candidate = os.path.join(RL_CHECKPOINT_ROOT, rl_checkpoint)
    if os.path.exists(candidate):
        return candidate

    return rl_checkpoint


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
        try:
            reward_model = json.loads(reward_model)
        except json.JSONDecodeError:
            return ""
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

def evaluate_model(model_path, model_name, test_df, tokenizer):
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

def export_eval_data_json(test_df, output_path):
    """导出本次评估使用的数据为 JSON，便于和结果一同上传。"""
    records = []
    for idx, row in test_df.iterrows():
        records.append({
            "index": int(idx),
            "prompt": parse_prompt(row["prompt"]),
            "ground_truth": extract_ground_truth(row["reward_model"]),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="Run full-test evaluation for Base/SFT/RL on GSM8K parquet.")
    parser.add_argument("--gpu-id", type=int, default=DEFAULT_GPU_ID, help="Physical GPU id to use.")
    parser.add_argument(
        "--rl-checkpoint",
        type=str,
        default=RL_CHECKPOINT_PATH,
        help="Path to RL checkpoint.",
    )
    parser.add_argument(
        "--auto-merge-rl-checkpoint",
        action="store_true",
        help="If RL checkpoint is verl FSDP shards, automatically merge to HF format before evaluation.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=f"{OUTPUT_DIR}/eval_results_3models_fulltest.json",
        help="Combined output JSON file for base/sft/rl results.",
    )
    parser.add_argument(
        "--data-json-file",
        type=str,
        default=f"{OUTPUT_DIR}/eval_data_fulltest.json",
        help="Exported test data JSON file to upload with results.",
    )
    args = parser.parse_args()

    args.rl_checkpoint = normalize_rl_checkpoint_path(args.rl_checkpoint)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"使用物理 GPU {args.gpu_id} (进程内映射为 cuda:0)")
    print(f"时间戳: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("加载测试集...")
    full_test_df = pd.read_parquet(TEST_DATA_PATH)
    print(f"完整测试集大小: {len(full_test_df)}")
    test_df = full_test_df
    print("使用完整 test parquet 进行评估\n")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    selected_models = ["base", "sft", "rl"]
    
    # 用于存储所有结果
    all_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_set': TEST_DATA_PATH,
        'num_samples': len(test_df),
        'gpu_id': args.gpu_id,
        'selected_models': selected_models,
        'models': {},
        'comparisons': {}
    }

    base_model_path = resolve_model_path(BASE_MODEL_PATH, auto_merge_fsdp=False)
    sft_model_path = resolve_model_path(SFT_MODEL_PATH, auto_merge_fsdp=False)

    if not os.path.exists(args.rl_checkpoint):
        raise FileNotFoundError(f"RL checkpoint 不存在: {args.rl_checkpoint}")
    resolved_rl_path = resolve_model_path(
        args.rl_checkpoint,
        auto_merge_fsdp=args.auto_merge_rl_checkpoint,
    )

    base_result = None
    sft_result = None
    rl_result = None

    print("\n" + "="*80)
    print("评估模型在完整 test parquet 上的准确率（固定 Base/SFT/RL）")
    print("="*80)

    base_result = evaluate_model(base_model_path, "Base Model", test_df, tokenizer)
    all_results['models']['base'] = {
        'model_path': base_model_path,
        'accuracy': base_result['accuracy'],
        'correct': base_result['correct'],
        'total': base_result['total'],
        'predictions': base_result['predictions'],
    }
    torch.cuda.empty_cache()

    sft_result = evaluate_model(sft_model_path, "SFT-357 Model", test_df, tokenizer)
    all_results['models']['sft'] = {
        'model_path': sft_model_path,
        'accuracy': sft_result['accuracy'],
        'correct': sft_result['correct'],
        'total': sft_result['total'],
        'predictions': sft_result['predictions'],
    }
    torch.cuda.empty_cache()

    rl_result = evaluate_model(resolved_rl_path, "RL Model", test_df, tokenizer)
    all_results['models']['rl'] = {
        'model_path': resolved_rl_path,
        'accuracy': rl_result['accuracy'],
        'correct': rl_result['correct'],
        'total': rl_result['total'],
        'predictions': rl_result['predictions'],
    }
    torch.cuda.empty_cache()

    sft_improvement = (sft_result['accuracy'] - base_result['accuracy']) / (base_result['accuracy'] + 1e-8)
    rl_improvement_vs_sft = (rl_result['accuracy'] - sft_result['accuracy']) / (sft_result['accuracy'] + 1e-8)
    rl_improvement_vs_base = (rl_result['accuracy'] - base_result['accuracy']) / (base_result['accuracy'] + 1e-8)

    all_results['comparisons'] = {
        'sft_vs_base_relative': sft_improvement,
        'rl_vs_sft_relative': rl_improvement_vs_sft,
        'rl_vs_base_relative': rl_improvement_vs_base,
    }

    print(f"\n{'='*80}")
    print("结果对比:")
    print(f"{'='*80}")
    print(f"Base Model 准确率:   {base_result['accuracy']:.2%} ({base_result['correct']}/{base_result['total']})")
    print(f"SFT-357 Model 准确率: {sft_result['accuracy']:.2%} ({sft_result['correct']}/{sft_result['total']})")
    print(f"RL Model 准确率:      {rl_result['accuracy']:.2%} ({rl_result['correct']}/{rl_result['total']})")
    print(f"SFT 相对提升 (vs Base): {sft_improvement:+.2%}")
    print(f"RL 相对提升 (vs SFT):   {rl_improvement_vs_sft:+.2%}")
    print(f"RL 相对提升 (vs Base):  {rl_improvement_vs_base:+.2%}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    export_eval_data_json(test_df, args.data_json_file)

    print(f"\n{'='*80}")
    print(f"✅ 三模型汇总结果已保存至: {args.output_file}")
    print(f"✅ 评测数据 JSON 已保存至: {args.data_json_file}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
