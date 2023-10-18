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
        st.header("Obtén respuestas, información y solución a cualquier duda.")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        openai.api_key = openai_api_key
        for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if prompt := st.chat_input("haz una pregunta"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                messages_list = [
                    {"role": "system", "content": "Your name is InfoGPT. You're an assistant that helps journalists in daily tasks, writing in Spanish in the style of the New York Times.\
                                                You can only help with tasks related to writing articles, and you can't help with other tasks.\
                                                Before or after giving respones, don't include any other information that is not related to the task, for example explanations."}
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
        st.error("""
            **Ha ocurrido un error inesperado.**\n
            Por favor, intenta de nuevo más tarde.""",
        )

