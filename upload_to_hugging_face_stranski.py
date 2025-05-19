import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="runs/detect/train/weights",
    repo_id="ParkVerc/model_stranski",
    repo_type="model",
)
