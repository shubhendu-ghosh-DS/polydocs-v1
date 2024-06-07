---
title: Polydocs
emoji: 🏃
colorFrom: purple
colorTo: pink
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


PolyDocs
PolyDocs is a Streamlit application that enables users to interactively chat with the contents of their PDF files using advanced AI models. This app leverages Google Generative AI for embedding and querying, allowing detailed and context-aware question-answering from PDF documents.

Features
Upload Multiple PDFs: Easily upload multiple PDF files for processing.
Intelligent Text Extraction: Extracts text from uploaded PDFs efficiently.
Context-Aware Question Answering: Ask questions about the content of your PDFs and receive detailed answers.
Interactive User Interface: Simple and intuitive interface for seamless user experience.
Persistent Vector Store: Stores embeddings of text chunks locally using FAISS for fast similarity search.
Demo
A demo of the app can be found here: PolyDocs Demo

Installation
To run this application locally, follow these steps:

Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/polydocs.git
cd polydocs
Set up a virtual environment (optional but recommended):

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Set up your Google API key:
Create a .env file in the root directory of your project and add your Google API key:

env
Copy code
GOOGLE_API_KEY=your_google_api_key_here
Usage
To run the application, execute:

bash
Copy code
streamlit run app.py
How to Use
Upload PDFs: Use the sidebar menu to upload your PDF files.
Process PDFs: Click the "Submit" button to process the uploaded PDFs.
Ask Questions: Enter your questions in the text input field and receive answers based on the content of the PDFs.
Technologies Used
Streamlit: For building the web application.
Google Generative AI: For embedding and querying text.
FAISS: For creating and querying the vector store.
PyPDF2: For extracting text from PDF files.
LangChain: For managing the conversational chain and text processing.
dotenv: For loading environment variables.
Code Overview
app.py
Imports and Configuration: Sets up necessary imports and configures the Google Generative AI API key.
Helper Functions:
get_pdf_text(pdf_docs): Extracts text from the uploaded PDF files.
get_text_chunks(text): Splits the extracted text into manageable chunks.
get_vector_store(text_chunks): Creates and saves a FAISS vector store from the text chunks.
get_conversational_chain(): Sets up the conversational chain using a custom prompt template.
user_input(user_question): Processes the user's question by querying the FAISS index and generating a response.
Main Function: Manages the user interface, handles file uploads, and processes user input.
requirements.txt
Lists all the dependencies required to run the application:

Copy code
streamlit
google-generativeai
python-dotenv
langchain
PyPDF2
chromadb
faiss-cpu
langchain_google_genai
langchain-community
Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

License
This project is licensed under the MIT License.
