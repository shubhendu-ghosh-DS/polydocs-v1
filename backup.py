import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# API key configuration
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Footer design and message
footer = """<style>
a:link, a:visited {
    color: blue;
    background-color: transparent;
    text-decoration: underline;
}
a:hover, a:active {
    color: red;
    background-color: transparent;
    text-decoration: underline;
}
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    color: black;
    text-align: center;
}
</style>
<div class="footer">
<p>Powered by Google Gemini.<a style='display: block; text-align: center;' href="https://www.linkedin.com/in/shubhendu-ghosh-423092205/" target="_blank">Developer</a></p>
</div>
"""





# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split extracted text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to load saved FAISS vector store
def load_vector_store():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        return vector_store
    except Exception as e:
        st.error("Vector store could not be loaded. Please upload a PDF and try again.")
        return None

# Function to get the QA chain for conversational AI
def get_conversational_chain():
    prompt_template = """
    Respond in as much detail as possible within the provided context. Provide full details. If the answer does not exist within the context, simply state, "answer not available in context". Do not provide incorrect answers.
Context:
{context}
Question:
{question}
Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user queries
def user_input(user_question, vector_store, chain):
    if vector_store is None or chain is None:
        return

    try:
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.markdown(f"""<p style="color: #000000;font-size: 15px;font-family: sans-serif; text-align:left;margin-bottom: 0px; height: 5px">{response["output_text"]}</p>""", unsafe_allow_html=True)
    except Exception as e:
        st.error("Could not generate an answer. Please ensure you have uploaded a relevant document and try again.")

# Main function to setup and run the app
def main():
    st.set_page_config("Chat with PDF")
    st.markdown("""<p style="color: #0352ff;font-size: 70px;font-family: arial; text-align:center; margin-bottom: 0px;" ><b>POLY</b><span style="color: #ec11f7;font-size: 70px;font-family: arial;"><b>DOCS</b></span></p>""", unsafe_allow_html=True)
    st.markdown("""<p style="color: #0352ff;font-size: 30px;font-family: sans-serif; text-align:center; margin-bottom: 50px;">Chat with your PDFs.</p>""", unsafe_allow_html=True)

    st.markdown("""<p style="color: #0352ff;font-size: 15px;font-family: sans-serif; text-align:left;margin-bottom: 0px; height: 5px">Ask a question based on uploaded PDF files.</p>""", unsafe_allow_html=True)
    user_question = st.text_input("")

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF files and click 'Submit' to process.", accept_multiple_files=True)

        if st.button("Submit"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Uploaded and processed")

    # Load the vector store and conversational chain
    vector_store = load_vector_store()
    chain = get_conversational_chain()

    # Handle user query if vector store and chain are loaded
    if user_question and vector_store is not None and chain is not None:
        user_input(user_question, vector_store, chain)

    st.markdown(footer, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
