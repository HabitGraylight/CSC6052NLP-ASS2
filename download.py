import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

def download_model(model_id, cache_dir=None):

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    else:
        cache_dir = Path(cache_dir)
    
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    print(f"开始下载模型: {model_id}")
    print(f"目标路径: {cache_dir}")
    
    try:
        repo_path = snapshot_download(
            model_id,
            cache_dir=str(cache_dir),
            resume_download=True,
            local_files_only=False,
        )
        print(f"下载完成！模型路径: {repo_path}")
        return repo_path
    except Exception as e:
        print(f"下载失败: {e}")
        return None

if __name__ == "__main__":
    # 用命令行传入模型名
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
    cache_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    download_model(model_name, cache_dir)