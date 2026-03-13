import argparse
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
	from datasets import load_dataset
except ImportError as exc:
	raise ImportError(
		"缺少 datasets 依赖，请先执行: pip install datasets"
	) from exc


TEST_DATASET_NAME = "cais/mmlu"
TEST_DATASET_CONFIG = "all"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
BASE_MODEL_PATH = "Qwen/Qwen3-0.6B"
SFT_MODEL_PATH = "/root/workspace/ft/qwen3-0.6b-cold-start-sft"
RL_CHECKPOINT_PATH = "/root/workspace/verl/output/qwen3_0.6b_grpo_gsm8k/global_step_550"
RL_CHECKPOINT_ROOT = "/root/workspace/verl/output/qwen3_0.6b_grpo_gsm8k"
OUTPUT_DIR = "/root/workspace/eval"
DEFAULT_GPU_ID = 4
DEFAULT_NTRAIN = 5
DEFAULT_MAX_NEW_TOKENS = 8

HF_WEIGHT_FILES = (
	"pytorch_model.bin",
	"model.safetensors",
	"tf_model.h5",
	"model.ckpt.index",
	"flax_model.msgpack",
)
CHOICE_LETTERS = ["A", "B", "C", "D"]


def has_hf_weights(model_dir):
	if not os.path.isdir(model_dir):
		return False
	return any(os.path.exists(os.path.join(model_dir, filename)) for filename in HF_WEIGHT_FILES)


def detect_fsdp_actor_dir(checkpoint_dir):
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
	if not os.path.exists(model_path):
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
			except subprocess.CalledProcessError as exc:
				raise RuntimeError(
					"自动 merge FSDP checkpoint 失败，请手动执行以下命令后重试:\n"
					f"{merge_cmd}\n"
					f"原始错误: {exc}"
				) from exc

			if has_hf_weights(merged_dir):
				return merged_dir

			raise RuntimeError(
				"已执行 checkpoint merge，但未在目标目录发现权重文件。"
				f"请检查目录: {merged_dir}"
			)

		raise FileNotFoundError(
			"检测到 RL checkpoint 是 FSDP 分片格式，不能直接加载。\n"
			"请先 merge 为 HuggingFace 权重，命令如下:\n"
			f"{merge_cmd}\n"
			"或者重新运行脚本并加上参数 --auto-merge-rl-checkpoint。"
		)

	raise FileNotFoundError(
		f"未找到可加载的 HuggingFace 权重文件。当前路径: {model_path}\n"
		f"期望文件之一: {', '.join(HF_WEIGHT_FILES)}"
	)


def normalize_rl_checkpoint_path(rl_checkpoint):
	if os.path.exists(rl_checkpoint):
		return rl_checkpoint

	candidate = os.path.join(RL_CHECKPOINT_ROOT, rl_checkpoint)
	if os.path.exists(candidate):
		return candidate

	return rl_checkpoint


def find_latest_rl_checkpoint(checkpoint_root):
	if not os.path.isdir(checkpoint_root):
		return None

	global_step_dirs = []
	for entry in os.listdir(checkpoint_root):
		full_path = os.path.join(checkpoint_root, entry)
		if not os.path.isdir(full_path) or not entry.startswith("global_step_"):
			continue
		try:
			step = int(entry.split("global_step_")[-1])
		except ValueError:
			continue
		global_step_dirs.append((step, full_path))

	if not global_step_dirs:
		return None

	global_step_dirs.sort(key=lambda item: item[0])
	return global_step_dirs[-1][1]


def resolve_rl_checkpoint_path(rl_checkpoint):
	normalized_path = normalize_rl_checkpoint_path(rl_checkpoint)
	if os.path.exists(normalized_path):
		return normalized_path

	latest_checkpoint = find_latest_rl_checkpoint(RL_CHECKPOINT_ROOT)
	if latest_checkpoint:
		print(
			"指定的 RL checkpoint 不存在，自动回退到最新可用 checkpoint: "
			f"{latest_checkpoint}"
		)
		return latest_checkpoint

	raise FileNotFoundError(
		f"RL checkpoint 不存在: {normalized_path}\n"
		f"且在目录 {RL_CHECKPOINT_ROOT} 下未找到任何可用的 global_step_* checkpoint。"
	)


def answer_to_letter(answer):
	if isinstance(answer, int):
		if 0 <= answer < len(CHOICE_LETTERS):
			return CHOICE_LETTERS[answer]
		return None

	if isinstance(answer, str):
		normalized = answer.strip().upper()
		if normalized in CHOICE_LETTERS:
			return normalized
		if normalized.isdigit():
			answer_idx = int(normalized)
			if 0 <= answer_idx < len(CHOICE_LETTERS):
				return CHOICE_LETTERS[answer_idx]
	return None


def extract_choice(text):
	if not text:
		return None

	cleaned = text.strip().upper()
	match = re.search(r"\b([ABCD])\b", cleaned)
	if match:
		return match.group(1)

	match = re.search(r"([ABCD])", cleaned[:16])
	if match:
		return match.group(1)
	return None


def format_subject(subject):
	return subject.replace("_", " ")


def format_example(example, include_answer):
	lines = [
		f"Question: {example['question']}",
		f"A. {example['choices'][0]}",
		f"B. {example['choices'][1]}",
		f"C. {example['choices'][2]}",
		f"D. {example['choices'][3]}",
		"Answer:" if not include_answer else f"Answer: {answer_to_letter(example['answer'])}",
	]
	return "\n".join(lines)


def build_prompt(example, subject_dev_examples, ntrain):
	intro = (
		f"The following are multiple choice questions about {format_subject(example['subject'])}. "
		"Choose the correct answer from A, B, C, and D. Reply with only the letter."
	)
	parts = [intro]
	for dev_example in subject_dev_examples[:ntrain]:
		parts.append(format_example(dev_example, include_answer=True))
	parts.append(format_example(example, include_answer=False))
	return "\n\n".join(parts)


def build_chat_text(tokenizer, prompt):
	messages = [{"role": "user", "content": prompt}]
	return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prepare_examples(test_dataset, dev_dataset, tokenizer, ntrain, max_samples=None, subjects=None):
	dev_by_subject = defaultdict(list)
	for item in dev_dataset:
		subject = item["subject"]
		if subjects and subject not in subjects:
			continue
		dev_by_subject[subject].append(item)

	examples = []
	for idx, item in enumerate(test_dataset):
		subject = item["subject"]
		if subjects and subject not in subjects:
			continue

		prompt = build_prompt(item, dev_by_subject[subject], ntrain)
		examples.append(
			{
				"index": idx,
				"subject": subject,
				"question": item["question"],
				"choices": item["choices"],
				"answer": answer_to_letter(item["answer"]),
				"prompt": prompt,
				"chat_text": build_chat_text(tokenizer, prompt),
			}
		)

		if max_samples is not None and len(examples) >= max_samples:
			break

	return examples


def load_model(model_path):
	use_cuda = torch.cuda.is_available()
	model_kwargs = {
		"trust_remote_code": True,
		"low_cpu_mem_usage": False,
	}
	if use_cuda:
		model_kwargs["torch_dtype"] = torch.bfloat16

	# 使用常规加载再显式 .to(device)，避免并行线程中 accelerate 的 meta tensor 分发错误。
	model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
	if use_cuda:
		model = model.to("cuda:0")
	else:
		model = model.to("cpu")
	model.eval()
	return model


def run_single_prediction(model, tokenizer, chat_text, max_new_tokens):
	inputs = tokenizer([chat_text], return_tensors="pt")
	inputs = {key: value.to(model.device) for key, value in inputs.items()}

	with torch.no_grad():
		outputs = model.generate(
			**inputs,
			max_new_tokens=max_new_tokens,
			do_sample=False,
			pad_token_id=tokenizer.pad_token_id,
			eos_token_id=tokenizer.eos_token_id,
			use_cache=True,
		)

	generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
	generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
	prediction = extract_choice(generated_text)

	del inputs, outputs, generated_tokens
	return prediction, generated_text


def evaluate_model(model_key, model_name, model_path, examples, max_new_tokens, progress_position):
	start_time = time.time()
	tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	model = load_model(model_path)

	correct = 0
	predictions = []
	subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})

	progress_bar = tqdm(
		examples,
		desc=f"{model_name} MMLU",
		position=progress_position,
		leave=True,
	)

	for example in progress_bar:
		prediction, raw_output = run_single_prediction(
			model=model,
			tokenizer=tokenizer,
			chat_text=example["chat_text"],
			max_new_tokens=max_new_tokens,
		)
		ground_truth = example["answer"]
		is_correct = prediction == ground_truth

		if is_correct:
			correct += 1

		subject_stats[example["subject"]]["total"] += 1
		if is_correct:
			subject_stats[example["subject"]]["correct"] += 1

		predictions.append(
			{
				"index": example["index"],
				"subject": example["subject"],
				"question": example["question"],
				"choices": example["choices"],
				"prediction": prediction,
				"ground_truth": ground_truth,
				"correct": is_correct,
				"raw_output": raw_output,
			}
		)

		progress_bar.set_postfix(acc=f"{correct / len(predictions):.2%}")

	elapsed_seconds = time.time() - start_time
	total = len(predictions)
	subject_results = {}
	for subject, stats in sorted(subject_stats.items()):
		subject_results[subject] = {
			"correct": stats["correct"],
			"total": stats["total"],
			"accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0,
		}

	del model
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	return model_key, {
		"model_name": model_name,
		"model_path": model_path,
		"correct": correct,
		"total": total,
		"accuracy": correct / total if total else 0.0,
		"elapsed_seconds": elapsed_seconds,
		"per_subject": subject_results,
		"predictions": predictions,
	}


def evaluate_model_worker(payload):
	return evaluate_model(
		model_key=payload["model_key"],
		model_name=payload["model_name"],
		model_path=payload["model_path"],
		examples=payload["examples"],
		max_new_tokens=payload["max_new_tokens"],
		progress_position=payload["progress_position"],
	)


def build_model_specs(args):
	rl_checkpoint = resolve_rl_checkpoint_path(args.rl_checkpoint)
	print(f"使用 RL checkpoint: {rl_checkpoint}")

	return [
		("base", "Base Model", resolve_model_path(BASE_MODEL_PATH, auto_merge_fsdp=False)),
		("sft", "SFT Model", resolve_model_path(SFT_MODEL_PATH, auto_merge_fsdp=False)),
		(
			"rl",
			"RL Model",
			resolve_model_path(rl_checkpoint, auto_merge_fsdp=args.auto_merge_rl_checkpoint),
		),
	]


def export_results(output_file, payload):
	os.makedirs(os.path.dirname(output_file), exist_ok=True)
	with open(output_file, "w", encoding="utf-8") as file_obj:
		json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def main():
	parser = argparse.ArgumentParser(description="Evaluate Base/SFT/RL models on cais/mmlu for catastrophic forgetting.")
	parser.add_argument("--gpu-id", type=int, default=DEFAULT_GPU_ID, help="Physical GPU id to use.")
	parser.add_argument("--dataset-name", type=str, default=TEST_DATASET_NAME, help="HuggingFace dataset name.")
	parser.add_argument("--dataset-config", type=str, default=TEST_DATASET_CONFIG, help="Dataset config name.")
	parser.add_argument("--split", type=str, default="test", help="Evaluation split.")
	parser.add_argument("--ntrain", type=int, default=DEFAULT_NTRAIN, help="Number of dev examples per subject.")
	parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap for debugging.")
	parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Max new tokens for answer generation.")
	parser.add_argument(
		"--rl-checkpoint",
		type=str,
		default=RL_CHECKPOINT_PATH,
		help="Path to RL checkpoint.",
	)
	parser.add_argument(
		"--auto-merge-rl-checkpoint",
		action="store_true",
		help="Automatically merge RL FSDP checkpoint into HuggingFace format if needed.",
	)
	parser.add_argument(
		"--hf-endpoint",
		type=str,
		default=HF_MIRROR_ENDPOINT,
		help="HuggingFace endpoint, default uses hf-mirror.",
	)
	parser.add_argument(
		"--subjects",
		nargs="*",
		default=None,
		help="Optional subset of subjects to evaluate.",
	)
	parser.add_argument(
		"--output-file",
		type=str,
		default=f"{OUTPUT_DIR}/eval_mmlu_results.json",
		help="Output JSON file path.",
	)
	parser.add_argument(
		"--sequential",
		action="store_true",
		help="Disable parallel inference and evaluate three models sequentially.",
	)
	args = parser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
	os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
	os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

	print(f"使用物理 GPU {args.gpu_id} (进程内映射为 cuda:0)")
	print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}")
	print(f"时间戳: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

	print(f"加载数据集: {args.dataset_name} / {args.dataset_config}")
	dev_dataset = load_dataset(args.dataset_name, args.dataset_config, split="dev")
	test_dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)

	bootstrap_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
	if bootstrap_tokenizer.pad_token is None:
		bootstrap_tokenizer.pad_token = bootstrap_tokenizer.eos_token

	subject_filter = set(args.subjects) if args.subjects else None
	examples = prepare_examples(
		test_dataset=test_dataset,
		dev_dataset=dev_dataset,
		tokenizer=bootstrap_tokenizer,
		ntrain=args.ntrain,
		max_samples=args.max_samples,
		subjects=subject_filter,
	)

	if not examples:
		raise ValueError("没有可评测的样本，请检查 split 或 subjects 参数。")

	print(f"评测样本数: {len(examples)}")
	print(f"评测学科数: {len(sorted({example['subject'] for example in examples}))}")
	print(f"并行模式: {'关闭' if args.sequential else '开启'}\n")

	model_specs = build_model_specs(args)

	all_results = {
		"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
		"task": "MMLU catastrophic forgetting evaluation",
		"dataset": {
			"name": args.dataset_name,
			"config": args.dataset_config,
			"split": args.split,
			"ntrain": args.ntrain,
			"num_examples": len(examples),
			"subjects": sorted({example["subject"] for example in examples}),
		},
		"runtime": {
			"gpu_id": args.gpu_id,
			"parallel_inference": not args.sequential,
			"max_new_tokens": args.max_new_tokens,
			"hf_endpoint": os.environ.get("HF_ENDPOINT"),
		},
		"models": {},
		"comparisons": {},
	}

	if args.sequential:
		for position, (model_key, model_name, model_path) in enumerate(model_specs):
			result_key, result_value = evaluate_model(
				model_key=model_key,
				model_name=model_name,
				model_path=model_path,
				examples=examples,
				max_new_tokens=args.max_new_tokens,
				progress_position=position,
			)
			all_results["models"][result_key] = result_value
	else:
		ctx = mp.get_context("spawn")
		payloads = [
			{
				"model_key": model_key,
				"model_name": model_name,
				"model_path": model_path,
				"examples": examples,
				"max_new_tokens": args.max_new_tokens,
				"progress_position": position,
			}
			for position, (model_key, model_name, model_path) in enumerate(model_specs)
		]
		with ProcessPoolExecutor(max_workers=len(model_specs), mp_context=ctx) as executor:
			futures = [executor.submit(evaluate_model_worker, payload) for payload in payloads]
			for future in futures:
				result_key, result_value = future.result()
				all_results["models"][result_key] = result_value

	base_accuracy = all_results["models"]["base"]["accuracy"]
	sft_accuracy = all_results["models"]["sft"]["accuracy"]
	rl_accuracy = all_results["models"]["rl"]["accuracy"]

	all_results["comparisons"] = {
		"sft_vs_base_relative": (sft_accuracy - base_accuracy) / (base_accuracy + 1e-8),
		"rl_vs_sft_relative": (rl_accuracy - sft_accuracy) / (sft_accuracy + 1e-8),
		"rl_vs_base_relative": (rl_accuracy - base_accuracy) / (base_accuracy + 1e-8),
	}

	export_results(args.output_file, all_results)

	print("\n" + "=" * 80)
	print("MMLU 结果汇总")
	print("=" * 80)
	print(f"Base Model 准确率: {base_accuracy:.2%}")
	print(f"SFT Model 准确率:  {sft_accuracy:.2%}")
	print(f"RL Model 准确率:   {rl_accuracy:.2%}")
	print(f"SFT 相对变化 (vs Base): {all_results['comparisons']['sft_vs_base_relative']:+.2%}")
	print(f"RL 相对变化 (vs SFT):   {all_results['comparisons']['rl_vs_sft_relative']:+.2%}")
	print(f"RL 相对变化 (vs Base):  {all_results['comparisons']['rl_vs_base_relative']:+.2%}")
	print(f"结果已保存至: {args.output_file}")
	print("=" * 80 + "\n")


if __name__ == "__main__":
	main()
