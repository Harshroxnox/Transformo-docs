from huggingface_hub import login
from dotenv import load_dotenv
import wandb
import os

# Load environment variables from the .env file (if present)
load_dotenv()

Hugging_Face_Token = os.getenv("Hugging_Face_Token")
Wandb_key = os.getenv("Wandb_key")

# Login to hugging face using your token
login(token=Hugging_Face_Token)

# Login to wandb using your API key
wandb.login(key=Wandb_key)
