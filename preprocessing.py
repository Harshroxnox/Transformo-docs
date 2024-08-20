from transformers import AutoTokenizer
from datasets import load_from_disk, load_dataset
import json

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

# can be either "dev"(for development) or "prod"(for production)
env = "dev"
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

# run the below code for only the first time
# saving a small sample of the dataset for development and preprocessing
if env == "dev":
    ds = load_dataset("teknium/OpenHermes-2.5", split='train[:1%]')
    ds.save_to_disk("dev-dataset")
elif env == "prod":
    ds = load_dataset("teknium/OpenHermes-2.5")

    # Load the test, train and validation splits
    ds_test = ds['test']
    ds_val = ds['validation']
    ds_train = ds['train']

    # Shuffle first
    # Further split the train dataset into 3 parts
    total_len = len(ds_train)
    split_1 = total_len/3
    split_2 = (2*total_len)/3
    ds_train_1 = ds_train[:split_1]
    ds_train_2 = ds_train[split_1:split_2]
    ds_train_3 = ds_train[split_2:]

    # Exporting all the datasets for further use
    ds_train_1.save_to_disk("ds_train_1")
    ds_train_2.save_to_disk("ds_train_2")
    ds_train_3.save_to_disk("ds_train_3")
    ds_test.save_to_disk("ds_test")
    ds_val.save_to_disk("ds_val")


datasets_list = []
if env == "dev":
    datasets_list = ["dev-dataset"]
elif env == "prod":
    datasets_list = ["ds_train_1", "ds_train_1", "ds_train_1", "ds_test", "ds_val"]


def preprocess_conversation(conversation):
    for msg in conversation:
        msg["role"] = msg.pop("from")
        msg["content"] = msg.pop("value")
        msg.pop("weight")
        if msg["role"] == "human":
            msg["role"] = "user"
        elif msg["role"] == "gpt":
            msg["role"] = "assistant"


def load_process_export_ds(ds_name):
    pre_ds = load_from_disk(ds_name)
    pre_ds = pre_ds["conversations"]

    print("\n\nBefore preprocessing:-")
    print(pre_ds[0])

    for conv in pre_ds:
        preprocess_conversation(conv)

    print("\n\nAfter preprocessing:-")
    print(pre_ds[0])

    single_example = tokenizer.apply_chat_template(pre_ds[0], tokenize=False)
    print("\n\nChat Template Applied to new dataset:")
    print(single_example)

    # Exporting the dataset
    file_name = ds_name + ".json"
    with open(file_name, "w") as json_file:
        json.dump(pre_ds, json_file, indent=4)


for name in datasets_list:
    load_process_export_ds(name)
