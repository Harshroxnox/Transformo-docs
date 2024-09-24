# Transformo-docs
- This is for preprocessing and fine-tuning an open source llm. Currently work is in progress...
- For the Retrieval Augmented Generation (RAG) built using an open source quantized model go to flask-rag.
- NOTE: For flask-rag and fine-tuning a model you need to create separate virtual environments with their 
  respective requirements.txt
- All the preprocessing, quantization and rag scripts written do work but there is large scope for 
  improvements such as performance for rag system and adjusting hyperparameters for fine-tuning.

## Technologies Used
- HuggingFace transformers, peft, datasets library is used for fine-tuning.
- Torch is mainly utilized.
- Weights and biases for tracking fine-tuning.
- PEFT library for qlora fine-tuning.
- llama.cpp for quantizing the model to Q5_K_M after fine-tuning.
- PyMuPDF for pdf to text in rag.
- llama.cpp for inference in rag.
- Qdrant for the vector database in rag.
- Flask is used for the web server.

NOTE: Since development is in progress everything is subject to change in the future. 