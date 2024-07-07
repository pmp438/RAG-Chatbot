import fitz  # PyMuPDF
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import streamlit as st

# index_name = "pdf-index-2"
# Initialize Pinecone
pc = Pinecone(api_key=st.secrets.pinecone_key) 


def extract_text_from_pdf(pdf):
    text = ""
    for page_num in range(len(pdf)):
        page = pdf.load_page(page_num)
        text += page.get_text()
    return text

def split_text(text, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_text(text)
    return chunks

def index_text(id, text, index_name):
    if index_name not in pc.list_indexes().names():
        print(f"Index {index_name} not found. Creating it")
        pc.create_index(index_name, dimension=384,
                        spec=ServerlessSpec(
                                cloud="aws",
                                region="us-east-1"
                            )) 
        print(f"Pinecone index: {index_name} created")
    index = pc.Index(index_name)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text).tolist()
    index.upsert(vectors=[{
      "id": id, 
      "values": embeddings, 
      "metadata": {"text":text}
    }])

def process_pdfs(directory_path, index_name):
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            print(f"Starting indexing file: {filename}")
            file_path = os.path.join(directory_path, filename)
            pdf = fitz.open(file_path)
            print("Extracting text")
            text = extract_text_from_pdf(pdf)
            print("Creating chunks of extracted text")
            text_chunks = split_text(text)
            for i, chunk in enumerate(text_chunks):
                print(f"Indexing chunk {i}")
                index_text(str(i), chunk, index_name)
                print(f"Chunk {i} indexed")
            print(f"Indexed {filename}")

# process_pdfs("PDFs")