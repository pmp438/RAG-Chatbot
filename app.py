import pinecone
from transformers import AutoTokenizer, AutoModelForCausalLM
from pinecone import Pinecone
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from groq import Groq
import os
import tempfile
import pandas as pd
from index import process_pdfs
from chatbot import generate_response

CSV_FILE = "indexes.csv"
pc = Pinecone(api_key=st.secrets.pinecone_key) 

def save_index_info(index_name, chatbot_name, chatbot_description):
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        if index_name not in df['index_name'].values:
            new_row = {"index_name": index_name, "chatbot_name": chatbot_name, "chatbot_description": chatbot_description}
            new_df = pd.DataFrame(new_row, index=[0])
            df = pd.concat([df, new_df], ignore_index=True)
            df.to_csv(CSV_FILE, index=False)
    else:
        df = pd.DataFrame([{"index_name": index_name, "chatbot_name": chatbot_name, "chatbot_description": chatbot_description}])
        df.to_csv(CSV_FILE, index=False)


st.title("RAG-based Chatbot and PDF Processor")

# Sidebar for navigation
page = st.sidebar.selectbox("Select a page", ["Chatbot", "PDF Processor"])

if page == "Chatbot":
    st.header("Chatbot")

    with st.sidebar:
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
            st.header("Existing Indexes")
            st.dataframe(df)
        else:
            st.warning("No index information available yet.")
    
    if "history" not in st.session_state:
        st.session_state.history = []

    available_indexes = pc.list_indexes().names()
    index_name = st.selectbox("Index", available_indexes)
    user_input = st.text_input("You:", "")

    if user_input:
        st.session_state.history.append({"user": user_input, "bot": generate_response(user_input, index_name)})
        chat = st.session_state.history[-1]
        st.write(f"User: {chat['user']}")
        st.write(f"Bot: {chat['bot']}")

elif page == "PDF Processor":
    st.header("PDF Processor")

    index_name = st.text_input("Enter the index name")
    chatbot_name = st.text_input("Enter the chatbot name")
    chatbot_description = st.text_input("Enter the chatbot description")

    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    # Read and display indexes.csv in a tabular format
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        st.header("Existing Indexes")
        st.dataframe(df)
    else:
        st.warning("No index information available yet.")

    if st.button("Process PDFs"):
        with st.spinner("Processing PDFS..."):
            if uploaded_files and index_name and chatbot_name and chatbot_description:
                save_index_info(index_name, chatbot_name, chatbot_description)
                with tempfile.TemporaryDirectory() as temp_dir:
                    for uploaded_file in uploaded_files:
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    process_pdfs(temp_dir, index_name)
                st.success("PDFs processed successfully")
            else:
                st.error("Please upload PDFs and provide all required information")