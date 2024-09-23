from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
from flask import Flask, request, jsonify
from qdrant_client import QdrantClient
import pymupdf
import llama_cpp
import uuid

# app = Flask(__name__)

client = QdrantClient(host="localhost", port=6333)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False
)

embedding_model = llama_cpp.Llama(
    model_path="./models/mxbai-embed-large-v1-f16.gguf",
    embedding=True,
    verbose=False,
)

llm = llama_cpp.Llama(
    model_path="./models/Phi-3.5-mini-instruct-Q5_K_M.gguf",
    n_ctx=512,
    n_threads=4,
    n_threads_batch=4,
    use_mlock=False,
    use_mmap=True,
    verbose=True,
)

template = """
You are a helpful assistant who answers questions using the provided context. If you don't know the answer, 
simply state that you don't know.

{context}

Question: {question}"""


def pdf_to_documents(arr_docs):
    text = ""
    for doc in arr_docs:
        # Extract all the text from the pdf document
        for page in doc:
            result = page.get_text()
            text += result

    return text_splitter.create_documents([text])


def generate_doc_embeddings(_documents):
    local_document_embeddings = []
    # Generate Embeddings for every single document in documents and append it into document_embeddings
    for document in _documents:
        embeddings = embedding_model.create_embedding([document.page_content])
        local_document_embeddings.extend([
            (document, embeddings["data"][0]["embedding"])
        ])

    return local_document_embeddings


def insert_in_db(_document_embeddings):
    # If collection VectorDB exists then delete
    if client.collection_exists(collection_name="VectorDB"):
        client.delete_collection(collection_name="VectorDB")

    client.create_collection(
        collection_name="VectorDB",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings,
            payload={
                "text": document.page_content
            }
        )
        for document, embeddings in _document_embeddings
    ]

    operation_info = client.upsert(
        collection_name="VectorDB",
        wait=True,
        points=points
    )

    print("\n")
    print("operation_info: ")
    print(operation_info)


def query(_search_query):
    query_vector = embedding_model.create_embedding(_search_query)['data'][0]['embedding']
    search_result = client.search(
        collection_name="VectorDB",
        query_vector=query_vector,
        limit=5
    )

    print("\n")
    print("search_result: ")
    print(search_result)

    ans = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": template.format(
                context="\n\n".join([row.payload['text'] for row in search_result]),
                question=_search_query
            )}
        ],
        # stream=True
    )
    ans = ans['choices'][0]['message']['content']
    print(ans)
    return ans
    # for chunk in stream:
    #     ans = chunk['choices'][0]['delta'].get('content', '')
    #     print(ans, end='')
    #     yield ans


def insert_pdf_vectordb(_arr_docs):
    documents = pdf_to_documents(_arr_docs)

    document_embeddings = generate_doc_embeddings(documents)

    insert_in_db(document_embeddings)


pdf_file = pymupdf.open("./pdf/Cross-Validators-Idea.pdf")
insert_pdf_vectordb([pdf_file])
query("what is the main idea?")

# @app.route('/question', methods=['POST'])
# def question():
#     data = request.get_json()
#     search_query = data.get("question")
#     ans = query(search_query)
#     return jsonify({"answer": ans})
#
#
# if __name__ == "__main__":
#     app.run(debug=True)
