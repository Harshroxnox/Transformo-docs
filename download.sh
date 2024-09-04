#!/bin/bash

cd rag
mkdir model
cd model

wget https://huggingface.co/harshroxnox/Mistral-inst-v0.3-finetune-quant/resolve/main/Mistral-7B-dev-Q5_K_M.gguf
