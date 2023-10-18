import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI as LangOpenAI
import openai

try:
    # Streamlit Config
    st.set_page_config(page_title="StudyAid", page_icon="📚", layout="wide")

    # Sidebar
    options = {
        "home": "Home",
        "chat": "Chat",
        "document": "Document Interactions"
    }

    with st.sidebar:
        openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
        if not openai_api_key:
            st.warning("API key is required to use the app.")
        option = st.selectbox("Select Option", list(options.values()))

    if not openai_api_key:
        st.warning("API key is missing. Please enter it in the sidebar.")
    else:
        # Home Page
        if option == options["home"]:
            st.header("Welcome to StudyAid 📚")
            st.info("Select an option from the sidebar to get started.")

        # Document Interactions
        elif option == options["document"]:
            st.header("Analyze Your Documents")
            pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])

            if pdf_file is not None:
                reader = PdfReader(pdf_file)
                raw_text = ''.join([page.extract_text() for page in reader.pages if page.extract_text()])
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = text_splitter.split_text(raw_text)
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                docsearch = FAISS.from_texts(texts, embeddings)
                query = st.text_input("What do you want to know from this document?", "Provide a brief summary of this document.")
                if st.button("Search"):
                    docs = docsearch.similarity_search(query)
                    chain = load_qa_chain(LangOpenAI(openai_api_key=openai_api_key))
                    answer = chain.run(input_documents=docs, question=query)
                    st.code(answer, language=None)
except Exception as e:
    st.error(f"An error occurred: {e}")
