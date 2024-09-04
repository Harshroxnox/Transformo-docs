from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, set_global_tokenizer
from llama_index.llms.llama_cpp import LlamaCPP
from transformers import AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

reader = SimpleDirectoryReader(input_dir="pdf/")
documents = reader.load_data()
print(documents)

llm = LlamaCPP(
    # you can set the path to a pre-downloaded model instead of model_url
    model_path="./model/Mistral-7B-dev-Q5_K_M.gguf",
    temperature=0.1,
    max_new_tokens=256,
    # we will change this context window later according to mistral-instruct
    context_window=3900,
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

response_iter = llm.stream_complete("Can you write me a poem about fast cars?")
for response in response_iter:
    print(response.delta, end="", flush=True)

set_global_tokenizer(
    AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3").encode
)

# import your embeddings model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# create vector store index
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# set up query engine
query_engine = index.as_query_engine(llm=llm)

response = query_engine.query("Can you explain model architecture of mistral?")
print(response)
