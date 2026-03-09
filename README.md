# CSC6052NLP-ASS2

Reinforcement Learning Fine-tuning Pipeline for Math Reasoning Tasks

This repository contains scripts for training language models on mathematical reasoning tasks using:
- **Supervised Fine-Tuning (SFT)** with mixed datasets (MATH + ECQA)
- **Reinforcement Learning (RL)** using GRPO algorithm via [verl](https://github.com/volcengine/verl)
- **Evaluation** on GSM8K benchmark

## 📁 Project Structure

```
.
├── data/data_process/sft_data_proc.py  # SFT data preprocessing
├── ft/train.py                          # SFT training script
├── download.py                          # Model download helper
├── eval/evaluation.py                   # GSM8K evaluation script
└── verl/run_qwen3_0.6b_grpo_gsm8k.sh   # GRPO RL training launcher
```

## 🚀 Quick Start

### 1. Environment Setup

#### Pull verl Docker Image

```bash
docker pull verlai/verl:vllm015.dev
docker run -it --gpus all --name verl_workspace verlai/verl:vllm015.dev bash
```

#### Configure verl Environment

Follow the official [verl documentation](https://github.com/volcengine/verl) to complete the environment setup.

#### Install Additional Dependencies

```bash
pip install datasets trl wandb
```

#### Configure wandb (Optional)

For online logging:
```bash
wandb login
# Enter your API key when prompted
```

For offline usage:
```bash
export WANDB_MODE=offline
```

### 2. Download Base Model

Use `download.py` to download the base model (e.g., Qwen/Qwen3-0.6B):

```bash
python download.py
```

Models will be cached in `~/.cache/huggingface/hub/`.

### 3. Generate SFT Training Data

Run the data preprocessing script to create cold-start SFT data:

```bash
cd data/data_process
python sft_data_proc.py
```

This generates:
- Mixed dataset from MATH competitions and ECQA reasoning tasks
- Output: `cold_start_sft_english_1000.jsonl` (default: 1000 samples)

### 4. Supervised Fine-Tuning (SFT)

Train the base model with supervised learning:

```bash
cd ft
python train.py
```

**Configuration:**
- Training runs on GPU 7 by default (modify `CUDA_VISIBLE_DEVICES` if needed)
- Checkpoints saved to `./qwen3-0.6b-cold-start-sft/`
- Wandb project: automatically created
- Default: 3 epochs, bf16 precision, gradient checkpointing enabled

**Expected Results:**
- Training loss: ~0.91
- Eval loss: ~0.80
- Token accuracy: ~80%

### 5. Clone verl Repository

Before running RL training, clone the official verl repository:

```bash
git clone https://github.com/verl-project/verl.git
cd verl
```

### 6. Prepare GSM8K Dataset

Process GSM8K dataset using verl's preprocessing script:

```bash
# Make sure you're in the verl/ directory
python3 examples/data_preprocess/gsm8k.py
```

This will generate:
- Training set: `data/gsm8k/train.parquet`
- Test set: `data/gsm8k/test.parquet`

**Note:** Update the data paths in your training script if the output location differs.

### 7. Setup RL Training Script

Copy the GRPO training script to the verl repository root:

```bash
# Assuming you're still in verl/ directory
cp /path/to/this/repo/verl/run_qwen3_0.6b_grpo_gsm8k.sh .
```

**Update the following paths in `run_qwen3_0.6b_grpo_gsm8k.sh`:**
- `SFT_MODEL_PATH`: Path to your SFT checkpoint from step 4
- `TRAIN_DATA`: Path to GSM8K train.parquet (from step 6)
- `TEST_DATA`: Path to GSM8K test.parquet (from step 6)

### 8. Reinforcement Learning with GRPO

Launch GRPO training using the SFT checkpoint:

```bash
# In verl/ directory
bash run_qwen3_0.6b_grpo_gsm8k.sh
```

**Configuration:**
- Algorithm: GRPO (Group Relative Policy Optimization)
- Base model: SFT checkpoint from step 4
- Training data: GSM8K training set
- Validation data: GSM8K test set
- GPU memory utilization: 0.75 (adjustable)
- Batch size: 8
- Rollout samples per prompt: 4
- Total epochs: 10

**Monitoring:**
```bash

# Check wandb dashboard
# Project: verl_grpo_qwen3_0.6b_gsm8k
```

### 9. Model Evaluation

Evaluate models on GSM8K test set:

```bash
cd eval
python evaluation.py
```

**Evaluation targets:**
- Base model (before SFT)
- SFT model (after supervised training)
- RL model (after GRPO training, optional)

**Metrics:**
- Accuracy on GSM8K math problems
- Answer extraction accuracy
- Inference speed (tokens/sec)

Results saved to `eval_results.json`.

## 📊 Expected Performance

| Model Stage | GSM8K Accuracy | Relative Improvement |
|-------------|----------------|----------------------|
| Base (Qwen3-0.6B) | ~18% | Baseline |
| After SFT | ~29% | +61% |
| After RL (GRPO) | TBD | TBD |

## 🛠️ Troubleshooting

### GPU Memory Issues

If you encounter OOM errors during RL training, reduce `gpu_memory_utilization`:

```bash
# In run_qwen3_0.6b_grpo_gsm8k.sh
actor_rollout_ref.rollout.gpu_memory_utilization=0.5  # Try 0.4 or 0.3
```

### Slow Training

Adjust micro batch sizes:

```bash
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4  # Increase if memory allows
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
```

### wandb Authentication

Ensure your API key is at least 40 characters. Get it from: https://wandb.ai/authorize



## 📄 License

See [LICENSE](LICENSE) file for details.

## 🔄 Future Updates

- [ ] Add more evaluation metrics

