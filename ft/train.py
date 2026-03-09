import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # 指定使用物理 GPU

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed
)
from trl import SFTTrainer

# 1. 全局设置
set_seed(42)
MODEL_PATH = "Qwen/Qwen3-0.6B"  
DATA_PATH = "/root/workspace/data/data_process/cold_start_sft_english_1000.jsonl"  # 完整路径
OUTPUT_DIR = "./qwen3-0.6b-cold-start-sft"

def main():
    print("加载 Tokenizer 和 模型 (使用 BF16 半精度加速)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # Qwen 默认可能没有 pad_token，使用 eos_token 代替
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 直接全参数加载模型到显卡
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cuda:0",  # CUDA_VISIBLE_DEVICES=7 后，这会对应 GPU 7
        trust_remote_code=True
    )

    # 全参数 FT 开启梯度检查点，降低显存峰值
    model.gradient_checkpointing_enable()

    print("加载并切分数据集...")
    # 加载刚才生成的 JSONL 数据
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    
    # 按照 Data Card 的设定：留出 5% (50条) 作为验证集
    dataset_split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    print(f"训练集大小: {len(train_dataset)} | 验证集大小: {len(eval_dataset)}")

    # 2. 训练超参数配置
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,                  # 跑 3 轮，足够让模型记住格式
        per_device_train_batch_size=2,       # 全参数 FT，保守设置避免峰值 OOM
        gradient_accumulation_steps=4,       # 等效 batch=8
        per_device_eval_batch_size=2,
        learning_rate=2e-5,
        warmup_ratio=0.1,                    # 预热步数
        weight_decay=0.01,
        logging_steps=5,                     # 每 5 步打印一次 Loss
        eval_strategy="steps",               # 改用 eval_strategy 替代 evaluation_strategy
        eval_steps=20,                       # 每 20 步在验证集上测一次 Loss
        save_strategy="epoch",               # 每跑完一轮保存一次权重
        bf16=True,                           # 开启 Bfloat16
        report_to="wandb",
        run_name="qwen3-0.6b-cold-start-sft",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
    )

    # 3. 初始化 SFT Trainer
    # trl 会自动读取 "messages" 列，并使用 tokenizer.apply_chat_template 将其转为模型能看懂的 token 序列
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,          # 用 processing_class 替代 tokenizer
    )

    # 4. 开始训练
    trainer.train()

    # 5. 保存最终模型
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()