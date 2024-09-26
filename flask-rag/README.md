# Flask-RAG
## Important Points
- Make sure to have python3.10, docker and python3.10-venv installed 
- Create virtual env and activate it 
 ```bash
 python3 -m venv venv-rag
 ```
```bash
source venv-rag/bin/activate
```
- Install from requirements.txt
```bash
pip install -r requirements.txt
```
- Run `model-setup.sh`. This file downloads embedding model inside models folder then clones the 
 llama.cpp repo and downloads the `Phi-3.5-mini-Instruct` model inside llama.cpp/models then builds
 llama.cpp locally which is required to serve the model as an http web server.
```bash
chmod +x model-setup.sh
```
```bash
./model-setup.sh
```
- Serve the llm using llama-server. Go inside llama.cpp then run the following command:
```bash
./llama-server -c 512 -a "Phi-3.5-Instruct" -m models/Phi-3.5-mini-instruct-Q5_K_M.gguf --api-key abcd
```
Here the context length is 512 tokens and API-key is `abcd`.
- Run `Qdrant` Vector database for vector search. Run these commands inside the flask-rag folder  
```bash
docker pull qdrant/qdrant
```
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```
- run rag.py
```bash
python3 rag.py
```
