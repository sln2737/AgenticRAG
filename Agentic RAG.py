#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install faiss-cpu chromadb llama-cpp-python sentence-transformers


# In[2]:


import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

# Load the embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load text documents (replace with your proprietary data source)
documents = [
    "Jamsetji Tata's vision laid the foundation for India's industrial revolution.",
    "The Tata group has pioneered industries like steel, aviation, and IT.",
    "Jamsetji's philosophy was about excellence, nation-building, and philanthropy.",
    "The Tata Trusts have contributed significantly to education and healthcare.",
]

# Generate embeddings
embeddings = np.array(embedding_model.encode(documents), dtype=np.float32)

# Create a FAISS index and add embeddings
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index and document mapping
faiss.write_index(index, "vector_store.index")

# Save document mapping
with open("doc_map.json", "w") as f:
    json.dump(documents, f)


# In[3]:


def retrieve_relevant_documents(query, k=3):
    """Retrieve top-k relevant documents for a query using FAISS"""
    query_embedding = np.array(embedding_model.encode([query]), dtype=np.float32)
    distances, indices = index.search(query_embedding, k)

    with open("doc_map.json", "r") as f:
        document_list = json.load(f)
    
    return [document_list[i] for i in indices[0]]


# In[6]:


from huggingface_hub import hf_hub_download,HfApi
import os

# Security note: Never hardcode tokens! Use environment variables instead
hf_token = os.getenv("HF_TOKEN", "[ACCESS_TOKEN]")  # Replace with your actual token

api = HfApi()
files = api.list_repo_files(
    repo_id="TheBloke/deepseek-llm-7B-base-GGUF",
    token=hf_token
)

for filename in files:
    print(filename)

model_path = hf_hub_download(
    repo_id="TheBloke/deepseek-llm-7B-base-GGUF",
    filename="deepseek-llm-7b-base.Q8_0.gguf",
    token=hf_token,
    local_dir="C:/models"
)


# In[9]:


from llama_cpp import Llama

# Load the on-prem model (adjust the path)
llm = Llama(model_path="C:/models/deepseek-llm-7b-base.Q8_0.gguf")

def generate_response(query):
    """Generate a response using retrieved context and the LLM"""
    retrieved_docs = retrieve_relevant_documents(query)
    context = "\n".join(retrieved_docs)
    
    prompt = f"""You are an AI agent using Retrieval-Augmented Generation (RAG). 
    Answer the query using the following retrieved documents:

    {context}

    Query: {query}
    Answer:
    """

    response = llm(prompt, max_tokens=300)
    return response["choices"][0]["text"]

# Example query
query = "What was Jamsetji Tata's industrial impact?"
response = generate_response(query)
print(response)


# In[10]:


class Agent:
    """Custom agent to decide whether to retrieve, generate, or refine responses"""

    def __init__(self, llm):
        self.llm = llm

    def decide_action(self, query):
        """Decide if retrieval is necessary or if LLM alone can answer"""
        prompt = f"""Determine if the query requires external retrieval. 
        Respond with 'retrieve' if knowledge from documents is needed, otherwise 'generate':

        Query: {query}
        Answer:
        """
        response = self.llm(prompt, max_tokens=10)["choices"][0]["text"].strip().lower()
        return response

    def execute(self, query):
        """Execute the best approach based on decision"""
        action = self.decide_action(query)

        if "retrieve" in action:
            return generate_response(query)
        else:
            return self.llm(query, max_tokens=300)["choices"][0]["text"]

# Initialize agent
agent = Agent(llm)

# Example agent decision
query = "Who founded Tata Steel?"
response = agent.execute(query)
print(response)


# In[12]:


# Example agent decision
query = "Who is the Owner of Tata Steel?"
response = agent.execute(query)
print(response)


# In[ ]:




