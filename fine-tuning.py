"""
This is the fine-tuning file where we train the trainable parameters on the Open-Hermes-2.5 dataset
Size of model --> 14GB(approx.)
Here we use low rank adapter of lora-matrix rank 16
Trainable lora parameters --> 6 Million params. (out of 7 Billion params.)
NOTE:
    For production this script needs to be run 3 times by incrementing the value of turn each time
    for turn = 1
        turn = 2
    and turn = 3
    After that the final model export can be quantized using llama-cpp
Dev:
    For dev we are just using a small sample dataset for checking if everything is working properly.
    No need to run this multiple times for dev obviously
    turn variable has no significance when in Dev mode
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model

# env and turn values initialized
env = "dev"
turn = 1

# Importing the datasets given by preprocessing.py file taking into consideration the env
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

# Define the quantization type for qlora
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Lora configuration settings
lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=32,
    lora_dropout=0.1
)

# Downloading and importing the tokenizer for the model
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", return_tensors="pt")

# Using the Out Of Vocabulary token for the padding token so model learns to predict EOS tokens
tokenizer.pad_token = tokenizer.unk_token

# Getting the EOS token and context length for model
eos_token_id = tokenizer.eos_token_id
context_length = 32768

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


def tokenization(example):
    # This function takes all the dialogues and formats it into a list of formatted_text
    formatted_text = tokenizer.apply_chat_template(example["dialogue"], tokenize=False)
    # Tokenizing the formatted texts with no truncation
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

# Download and import the model and get the adapter
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    quantization_config=quantization_config,
    device_map="auto"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Defining training arguments
training_args = TrainingArguments(
    output_dir="model-part"+f"{turn}",
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Start training
trainer.train()

# Save the adapter
model.save_pretrained("dev_model")
