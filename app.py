import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI as LangOpenAI
import openai

# Streamlit Config
st.set_page_config(page_title="StudyAid", page_icon="📚", layout="wide")

# API Key
openai_api_key = st.secrets["api_key"]

# Sidebar
options = {
    "home": "Home",
    "chat": "Chat",
    "document": "Document Interactions"
}

with st.sidebar:
    option = st.selectbox("Select Option", list(options.values()))

# Home Page
if option == options["home"]:
    st.header("Welcome to StudyAid 📚")
    st.info("Select an option from the sidebar to get started.")

# Document Interactions
elif option == options["document"]:
    try:
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
    except:
        st.error("An error occurred. Please try again.")

elif option == options["chat"]:
    try:
        st.header("Get answers, information, and solutions to your academic queries.")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        openai.api_key = openai_api_key

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question"):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                messages_list = [
                    {"role": "system", "content": "Your name is StudyAid. You're an AI assistant specialized in helping students with academic tasks, providing guidance in English or any other relevant language."}
                ]
                messages_list += [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                
                for response in openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=messages_list, stream=True):
                    full_response += response.choices[0].delta.get("content", "")
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    except:
        st.error("An unexpected error occurred. Please try again later.")

