# download_models.py
from huggingface_hub import snapshot_download
import os

# 设置模型保存路径
os.environ["HF_HOME"] = "./models/huggingface"

# 下载模型
model_path = snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2")
print(f"模型已下载到: {model_path}")