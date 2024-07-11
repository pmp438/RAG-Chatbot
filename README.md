[<img src="https://github.com/Sakalya100/AutoTabML/blob/main/Sample%20Data/AutoTabML%20Automated%20Code%20generation%20Using%20ML.png" width="5000px;"/>](https://github.com/pmp438)
# Med-Buddy - Personalised RAG-Chatbot for Medical Research Papers

## Overview
Med-Buddy is an implementation of a Retrieval-Augmented Generation (RAG) based chatbot. This chatbot leverages retrieval mechanisms to fetch relevant documents from an indexed database and uses a generative model to provide detailed and contextually relevant responses.

## Try it out here
The Med-Buddy Application is hosted on HuggingFace Spaces for anyone to go and test this amazing functionality.
[<img src="https://github.com/Sakalya100/AutoTabML/blob/main/Sample%20Data/5229488.png" width="200px;"/>](https://huggingface.co/spaces/pmp438/med-buddy)

## Features
- **Document Indexing:** Efficiently index documents for retrieval. Create as many chatbots with your different categories of documents and efficiently chat with them anytime you want without the need to re upload the documents. 
- **Contextual Responses:** Generate responses based on retrieved documents. Utilising Llama-7B Model, and Pinecone as the Vector Database, get contextual and proper answers to your queries.
- **Modular Design:** Easy to extend and integrate with various data sources. Components are divided and coded to easily integrate and expand use-cases in future.

## Features
- Pinecone
- Streamlit
- HuggingFace
- Python
- Llama-7B (Groq API)
- Sentence-Transformers(Embedding)

  
## Installation
Clone the repository:
```bash
git clone https://github.com/pmp438/RAG-Chatbot.git
cd RAG-Chatbot
```
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. **Index Documents:** Use `index.py` to index your documents. Or you can also directly index your documents through the client interface
2. **Start Chatbot:** Run `app.py` to start the chatbot server.'
```bash
streamlit run app.py
```
3. **Interact:** Use a client to interact with the chatbot.

## Files
- `README.md`: This file.
- `app.py`: Entry point for running the chatbot server.
- `chatbot.py`: Core logic for the chatbot.
- `index.py`: Script for indexing documents.
- `indexes.csv`: Indexed documents and the index csv having your chatbot wise index details on Pinecone.
- `requirements.txt`: Dependencies for the project.

## License
This project is licensed under the MIT License.

## Contact
For any queries, please open an issue on the repository.
