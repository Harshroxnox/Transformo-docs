from rag import pdf_to_documents, generate_doc_embeddings, insert_in_db, query
from flask import Flask
import pymupdf

#app = Flask(__name__)

pdf_file = pymupdf.open("./pdf/2310.06825v1.pdf")
documents = pdf_to_documents(pdf_file)

print("no of documents: ")
print(len(documents))

print("single document: ")
print(documents[0])

document_embeddings = generate_doc_embeddings(documents)
print("\n")
print("single doc embedding: ")
print(document_embeddings[0])

insert_in_db(document_embeddings)

search_query = "What is this document all about?"
query(search_query)
