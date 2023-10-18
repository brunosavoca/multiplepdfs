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
# try:
#         elif option == options["chat"]:
#                 st.header("Get answers, information, and solutions to your academic queries.")

#                 # Initialize session state variables if they don't exist
#                 if 'generated' not in st.session_state:
#                     st.session_state['generated'] = []
            
#                 if 'past' not in st.session_state:
#                     st.session_state['past'] = []
            
#                 # Function to send a query to the Hugging Face API
#                 def query(payload):
#                     response = requests.post(API_URL, headers=headers, json=payload)
#                     return response.json()
            
#                 # Text input for the user
#                 user_input = st.text_input("You: ", "Hello, how are you?", key="input")
            
#                 # Sending user input to the chat model and receiving a response
#                 if user_input:
#                     output = query({
#                         "inputs": {
#                             "past_user_inputs": st.session_state.past,
#                             "generated_responses": st.session_state.generated,
#                             "text": user_input,
#                         },
#                         "parameters": {"repetition_penalty": 1.33},
#                     })
            
#                     # Save chat history to session state
#                     st.session_state.past.append(user_input)
#                     st.session_state.generated.append(output["generated_text"])
            
#                 # Display the chat history
#                 if st.session_state['generated']:
#                     for i in range(len(st.session_state['generated']) - 1, -1, -1):            
#                         message(st.session_state["generated"][i], key=str(i))
#                         message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

