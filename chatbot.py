from transformers import AutoTokenizer, AutoModelForCausalLM
from pinecone import Pinecone
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from groq import Groq
import os

client = Groq(
    api_key=st.secrets.groq_api_key,
)

pc = Pinecone(api_key=st.secrets.pinecone_key) 

from sentence_transformers import SentenceTransformer

# Load a pre-trained Sentence-BERT model that outputs 384-dimensional embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def query_to_embedding(query):
    embedding = embedder.encode(query)
    return embedding

# Function to query Pinecone
def query_pinecone(query, index_name,top_k=5):
    embedding = query_to_embedding(query)
    index = pc.Index(index_name)
    result = index.query(vector=embedding.tolist(), top_k=top_k, include_metadata=True)
    documents = [match['metadata']['text'] for match in result['matches']]
    return documents

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

def generate_response(query, index_name):
    documents = query_pinecone(query, index_name)
    context = " ".join(documents)
    
    # Prepare the input for OpenAI
    input_text = f"Context: {context}"

    chat_completion = client.chat.completions.create(
           messages=[
            {"role": "system", "content": f"You are a helpful assistant.Answer the query asked by user based on this context. \n\n{input_text}"},
            {"role": "user", "content": query}
        ],
            model="llama3-8b-8192",
        )

    # To get the response text:
    answer = chat_completion.choices[0].message.content
    # Extract the generated response
    return answer