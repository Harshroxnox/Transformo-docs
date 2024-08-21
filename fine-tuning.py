from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model

env = "dev"
turn = 1

if env == "dev":
    # Development - importing the preprocessed dataset
    ds = Dataset.from_json("pre_dev-dataset.json")
    split_ds = ds.train_test_split(test_size=0.2)
    ds_train = split_ds["train"]
    ds_val = split_ds["test"]
else:
    ds_name = "pre_ds_train_" + f"{turn}" + ".json"
    ds_train = Dataset.from_json(ds_name)
    ds_val = Dataset.from_json("pre_ds_val.json")

lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=32,
    lora_dropout=0.1
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", return_tensors="pt")
tokenizer.pad_token = tokenizer.unk_token
eos_token_id = tokenizer.eos_token_id
context_length = 32768


def tokenization(example):
    formatted_text = tokenizer.apply_chat_template(example["dialogue"], tokenize=False)
    outputs = tokenizer(
        formatted_text,
        truncation=False,
    )
    # Concatenate all tokenized samples in a batch with an eos_token_id in between
    concatenated = []
    for input_ids in outputs["input_ids"]:
        concatenated.extend(input_ids + [eos_token_id])

    # Remove the last eos_token_id
    concatenated = concatenated[:-1]

    # Chunk the concatenated sequence into context_length-sized chunks
    input_batch = []
    for i in range(0, len(concatenated), context_length):
        chunk = concatenated[i:i + context_length]
        if len(chunk) == context_length:
            input_batch.append(chunk)

    return {"input_ids": input_batch}


# Apply the tokenization
ds_train = ds_train.map(tokenization, batched=True, remove_columns=ds_train.column_names)
ds_val = ds_val.map(tokenization, batched=True, remove_columns=ds_val.column_names)

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", device_map="auto")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="model-part"+f"{turn}",
    learning_rate=1e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    tokenizer=tokenizer
)

trainer.train()

# merging the model with adapter conv it from AutoPeftModelForCausalLM to AutoModelForCausalLM
model = model.merge_and_unload()
model.save_pretrained("dev_model")

