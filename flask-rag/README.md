# Flask-RAG
## Important Points
- Make sure to have python3.10, docker and venv installed 
- create virtual env and activate it 
- Install from requirements.txt
- run download.sh
- run qdrant 
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
