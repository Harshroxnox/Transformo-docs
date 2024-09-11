import pymupdf
import llama_cpp
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = ""
# Open a document
doc = pymupdf.open("./pdf/2310.06825v1.pdf")

# Extract all the text from the pdf document
for page in doc:
    result = page.get_text()
    text += result

# Delete document
del doc

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False
)

documents = text_splitter.create_documents([text])

print(len(documents))

print(documents[0])

embedding_model = llama_cpp.Llama(
    model_path="./models/bge-small-en-v1.5-f16.gguf",
    embedding=True,
    verbose=False
)


