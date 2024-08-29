"""
Converting the Open Hermes 2.5 dataset into a format that can be easily processed by the default mistral
tokenizer.apply_chat_template function after that shuffling the dataset and storing the train and test
sets separately. Then we further shuffle and divide the training set into 5 equal parts for effective training.
Since the dataset is very large (1M samples) this step is crucial.

After that we are exporting each dataset.pyarrow into a json file that can be loaded back into dataset.pyarrow
format using Dataset.from_json() function.

Size of Dataset --> 2GB(approx)

Dev: The dev version of this does all the preprocessing steps with a very small sample(1%) of dataset
NOTE: We are not tokenizing here that will be done in the fine-tuning.py file
"""
from transformers import AutoTokenizer
from datasets import load_from_disk, load_dataset, Dataset

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
    Input Format that can be easily formatted by Mistral-7b-Instruct-v0.3
    [
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
        {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]
"""

# can be either "dev"(for development) or "prod"(for production)
env = "prod"
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

# run the below code for only the first time
# saving a small sample of the dataset for development and preprocessing
if env == "dev":
    ds = load_dataset("teknium/OpenHermes-2.5", split='train[:1%]')
    ds = ds.shuffle(seed=42)
    ds.save_to_disk("dev-dataset")
elif env == "prod":
    ds = load_dataset("teknium/OpenHermes-2.5")

    # full dataset shuffle
    ds = ds.shuffle(seed=42)
    # load 125,000 samples of training dataset
    ds = ds["train"]
    ds = ds.select(range(125000))

    # performing train, test split
    ds = ds.train_test_split(test_size=0.05)
    ds_val = ds["test"]
    ds_train = ds["train"]

    # shuffle first
    ds_train = ds_train.shuffle(seed=42)

    # Divide the ds_train into 5 equal parts
    ds_train_1 = ds_train.select(range(23750))
    ds_train_2 = ds_train.select(range(23750, 47500))
    ds_train_3 = ds_train.select(range(47500, 71250))
    ds_train_4 = ds_train.select(range(71250, 95000))
    ds_train_5 = ds_train.select(range(95000, 118750))

    # Export all the datasets
    ds_train_1.save_to_disk("ds_train_1")
    ds_train_2.save_to_disk("ds_train_2")
    ds_train_3.save_to_disk("ds_train_3")
    ds_train_4.save_to_disk("ds_train_4")
    ds_train_5.save_to_disk("ds_train_5")
    ds_val.save_to_disk("ds_val")


# List of datasets to load, preprocess and export this depends on the env whether dev or prod
datasets_list = []
if env == "dev":
    datasets_list = ["dev-dataset"]
elif env == "prod":
    datasets_list = ["ds_train_1", "ds_train_2", "ds_train_3", "ds_train_4", "ds_train_5", "ds_val"]


def preprocess_conversation(conversation):
    # Changing the OpenHermes-2.5 format into Mistral-7b-instruct-v0.3 input format
    for msg in conversation:
        msg["role"] = msg.pop("from")
        msg["content"] = msg.pop("value")
        msg.pop("weight")
        if msg["role"] == "human":
            msg["role"] = "user"
        elif msg["role"] == "gpt":
            msg["role"] = "assistant"


def load_process_export_ds(ds_name):
    # Load the dataset
    pre_ds = load_from_disk(ds_name)
    pre_ds = pre_ds["conversations"]

    print("\n\nBefore preprocessing:-")
    print(pre_ds[0])

    # Apply the preprocessing steps
    for conv in pre_ds:
        preprocess_conversation(conv)

    print("\n\nAfter preprocessing:-")
    print(pre_ds[0])

    single_example = tokenizer.apply_chat_template(pre_ds[0], tokenize=False)
    print("\n\nChat Template Applied to new dataset:")
    print(single_example)

    # Converting the list of lists to hugging face dataset
    conversations = []
    id = 1
    for dialogue in pre_ds:
        dict_to_append = {
            "conversation_id": id,
            "dialogue": dialogue
        }
        conversations.append(dict_to_append)
        id = id + 1

    # Converting from dict to hugging face dataset format
    export_ds = Dataset.from_dict({
        "conversation_id": [conv["conversation_id"] for conv in conversations],
        "dialogue": [conv["dialogue"] for conv in conversations]
    })

    # Exporting the Dataset
    print(export_ds[0])
    export_ds.to_json("pre_"+ds_name+".json")


for name in datasets_list:
    load_process_export_ds(name)
