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

load_dotenv()
os.getenv("GOOGLE_API_KEY")
#genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))




footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
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
<p>powered by google gemini<a style='display: block; text-align: center;' href="https://www.linkedin.com/in/shubhendu-ghosh-423092205/" target="_blank">Developer</a></p>
</div>
"""

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    print("embed is good")
     

    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)

    st.markdown(f"""<p style="color: #000000;font-size: 15px;font-family: sans-serif; text-align:left;margin-bottom: 0px; height: 5px">{response["output_text"]}</p>""", unsafe_allow_html=True)



def main():


    # Set page config and header
    st.set_page_config("Chat PDF")
    st.markdown("""<p style="color: #0352ff;font-size: 70px;font-family: arial; text-align:center; margin-bottom: 0px;" ><b>POLY</b><span style="color: #ec11f7;font-size: 70px;font-family: arial;"><b>DOCS</b></span></p>""", unsafe_allow_html=True)
    st.markdown("""<p style="color: #0352ff;font-size: 30px;font-family: sans-serif; text-align:center; margin-bottom: 50px;">Chat with your PDF</p>""", unsafe_allow_html=True)
    # Text input for user question
    st.markdown("""<p style="color: #0352ff;font-size: 15px;font-family: sans-serif; text-align:left;margin-bottom: 0px; height: 5px">Ask a Question from the PDF Files </p>""", unsafe_allow_html=True)
    user_question = st.text_input("")

    # If user inputs a question, process it
    if user_question:
        user_input(user_question)

    # Sidebar menu
    with st.sidebar:
        st.title("Menu")
        # File uploader for PDF files
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit Button", accept_multiple_files=True)
        # Button to submit and process PDF files
        if st.button("Submit"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Uploaded")
    
    st.markdown(footer,unsafe_allow_html=True)


if __name__ == "__main__":
    main()
