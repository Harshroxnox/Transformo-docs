from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
import llama_cpp
import pymupdf
import time
import uuid

text = ""
# Open a document
doc = pymupdf.open("./pdf/2310.06825v1.pdf")

# Extract all the text from the pdf document
for page in doc:
    result = page.get_text()
    text += result

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False
)

documents = text_splitter.create_documents([text])

print("no of documents: ")
print(len(documents))

print("single document: ")
print(documents[0])

embedding_model = llama_cpp.Llama(
    model_path="./models/bge-small-en-v1.5-f16.gguf",
    embedding=True,
    verbose=False
)

start = time.time()
document_embeddings = []
# Generate Embeddings for every single document in documents and append it into document_embeddings
for document in documents:
    embeddings = embedding_model.create_embedding([document.page_content])
    document_embeddings.extend([
        (document, embeddings["data"][0]["embedding"])
    ])

end = time.time()
all_text = [item.page_content for item in documents]
char_per_sec = len(''.join(all_text))/ (end-start)
print(f"TIME: {end-start:.2f} seconds / {char_per_sec:,.2f} chars/second")

print("\n")
print("single doc embedding: ")
print(document_embeddings[0])

client = QdrantClient(host="localhost", port=6333)

# If collection VectorDB exists then delete
if client.collection_exists(collection_name="VectorDB"):
    client.delete_collection(collection_name="VectorDB")

client.create_collection(
    collection_name="VectorDB",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

points = [
    PointStruct(
        id=str(uuid.uuid4()),
        vector=embeddings,
        payload={
            "text": document.page_content
        }
    )
    for document, embeddings in document_embeddings
]

operation_info = client.upsert(
    collection_name="VectorDB",
    wait=True,
    points=points
)

print("\n")
print("operation_info: ")
print(operation_info)

search_query = "What is this document all about?"
query_vector = embedding_model.create_embedding(search_query)['data'][0]['embedding']
search_result = client.search(
    collection_name="VectorDB",
    query_vector=query_vector,
    limit=3
)

print("\n")
print("search_result: ")
print(search_result)

llm = llama_cpp.Llama(
    model_path="./models/Main-Model-7.2B-Q5_K_M.gguf",
    verbose=False
)

template = """
You are a helpful assistant who answers questions using the provided context. If you don't know the answer, 
simply state that you don't know.

{context}

Question: {question}"""

stream = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": template.format(
            context="\n\n".join([row.payload['text'] for row in search_result]),
            question=search_query
        )}
    ],
    stream=True
)

for chunk in stream:
    print(chunk['choices'][0]['delta'].get('content', ''), end='')
