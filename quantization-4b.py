"""
4 Bit Quantization (For development only):
This file is going to load the mistral-instruct fine-tuned model in 16 bit precision and going to quantize
the model in 4 bits (development) using bitsandbytes.

Quantization:
Original model has 7 Billion parameters each of 32 bits so total model size comes out to be around
(7 * 10^9* 32)/8 Bytes which is approx 28GB!!!
But after quantizing each parameter basically converting each 32 bit into 4 bits(or 6bits) total size
becomes only 3.5 GB (approx) this makes the model extremely efficient and doesn't affect the output
of the model that much.

On production if we are using 6bits quantization (5.25 GB approx) we might be able to use 32GB ram
system without GPU for inference.

NOTE:
    You need a cuda compatible GPU to run the following code. You must also install all the required
    libraries such as torch, transformers, accelerate and bitsandbytes(only the latest version).
    You also need to authenticate to hugging face in order to use the gated mistral model. You can do
    that using the "huggingface-cli login" command and providing your token.

In Cloud Environments (such as Colab, Kaggle etc.) Do the following before running the code:
# Install latest version of bitsandbytes
!pip install --upgrade bitsandbytes
from huggingface_hub import login

# Log in using the token
login("Your-token")
"""

# Importing necessary libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Setting quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Loading the model with the quantization configuration
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    quantization_config=quantization_config,
    device_map="auto"
)

# Downloading the tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

# Example messages
messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It"
                                     " adds just the right amount of zesty flavour to whatever I'm cooking up in "
                                     "the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

# Tokenizing our input string
model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

# Getting output vectors
generated_ids = model.generate(model_inputs, max_new_tokens=500, do_sample=True)

# Decoding and printing the output
output = tokenizer.batch_decode(generated_ids)[0]
print(output)

# Exporting the quantized model
model.save_pretrained("q4_model")
