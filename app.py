from rag import pdf_to_documents, generate_doc_embeddings, insert_in_db, query
from flask import Flask
import pymupdf

#app = Flask(__name__)

pdf_file = pymupdf.open("./pdf/2310.06825v1.pdf")
documents = pdf_to_documents(pdf_file)

document_embeddings = generate_doc_embeddings(documents)

insert_in_db(document_embeddings)

search_query = "What is this document all about?"
for response in query(search_query):
    print(response, end='')
