import os

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

HF_MODEL_DIR=os.getenv("HF_MODEL_DIR")

snapshot_download(repo_id='Qwen/Qwen3-4B', local_dir=HF_MODEL_DIR)

print('模型下载完成!')