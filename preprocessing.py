from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import json

# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

# we are going to be dividing this dataset into 3 different parts and training on cloud for production
# run the below code for only the first time
"""
# saving a small sample of the dataset for development and preprocessing
ds = load_dataset("teknium/OpenHermes-2.5", split='train[:1%]')
ds.save_to_disk("dev-dataset")
"""

ds = load_from_disk("dev-dataset")
ds = ds["conversations"]
print("\n\nBefore preprocessing:-")
print(ds[0])

"""
    Open-Hermes dataset format 
    {
        "conversations": [
            {
                "from": "system",
                "value": ""
            },
            {
                "from": "human",
                "value": "Every day, a tree drops 7 leaves. How many leaves would it drop in a month of February in a non-leap year? Include your logic."
            },
            {
                "from": "gpt",
                "value": "Here's the logic behind this:\n\n1. We know that February has 28 days in a non-leap year.\n2. If the tree drops 7 leaves every day, then over the course of February, it would drop:\n   Leaves dropped in February = Leaves per day * Days in February\n   = 7 leaves * 28 days\n   = 196 leaves\n\nSo, the tree would drop 196 leaves in February in a non-leap year."
            }
        ],
        "source": "airoboros2.2",
        "category": "orca"
    },
"""


def preprocess_conversation(conversation):
    for msg in conversation:
        msg["role"] = msg.pop("from")
        msg["content"] = msg.pop("value")
        msg.pop("weight")
        if msg["role"] == "human":
            msg["role"] = "user"
        elif msg["role"] == "gpt":
            msg["role"] = "assistant"


for conv in ds:
    preprocess_conversation(conv)
print("\n\nAfter preprocessing:-")
print(ds[0])

single_example = tokenizer.apply_chat_template(ds[0], tokenize=False)
print("\n\nChat Template Applied to new dataset:")
print(single_example)

# Exporting the dataset. Run this code only once
"""
with open("processed-dev-ds.json", "w") as json_file:
    json.dump(ds, json_file, indent=4)
"""
