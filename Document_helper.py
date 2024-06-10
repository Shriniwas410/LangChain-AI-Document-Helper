import os
import tempfile
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Set the page configuration for the Streamlit app
st.set_page_config(page_title="AI Document Helper")
st.title("Chat with Documents")

# Cache the retriever configuration to avoid reloading documents multiple times
@st.cache_resource(ttl=1800)
def configure_retriever(uploaded_files):
    # Reading the uploaded documents
    document_list = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        filepath = os.path.join(temp_dir.name, file.name)
        with open(filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(filepath)
        document_list.extend(loader.load())

    # Spliting the documents using Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(document_list)

    # Create embeddings and store in vectordb in memory
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = FAISS.from_documents(splits, embeddings)
    
    # Define retriever
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 4})

    return retriever

# Custom callback handler to stream LLM responses in real-time
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # Append new tokens to the text unless they are part of the ignored run
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


# Custom callback handler to display retrieval process
class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        # Display the query being processed
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # Display the retrieved documents
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")

# Dropdown to select the model
model_choice = st.sidebar.selectbox("Select Model", ["Gemini", "OpenAI"])

# Input for the API key based on the selected model
def get_api_key(model_choice):
    if model_choice == "Gemini":
        api_key = st.sidebar.text_input("Gemini API Key", type="password")
        if not api_key:
            st.info("Please add your Gemini API key to continue.")
            st.stop()
        os.environ['GOOGLE_API_KEY'] = api_key
    else:
        api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        if not api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
    return api_key

# File uploader for PDF documents
def upload_files():
    uploaded_files = st.sidebar.file_uploader(
        label="Upload PDF files", type=["pdf"], accept_multiple_files=True
    )
    if not uploaded_files:
        st.info("Please upload PDF documents to continue.")
        st.stop()
    return uploaded_files

# Setup memory for contextual conversation
def setup_memory():
    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)
    return msgs, memory

# Function to generate response using the selected model
def generate_response(user_query, model_choice, api_key, retriever, memory):
    if model_choice == "Gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=api_key, temperature=0.5, streaming=True)
    else:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key, temperature=0.5, streaming=True)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True
    )
    
    retrieval_handler = PrintRetrievalHandler(st.container())
    stream_handler = StreamHandler(st.empty())
    response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
    return response

# Initialize chat history or clear it if requested
def initialize_chat_history(msgs):
    if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
        msgs.clear()
        msgs.add_ai_message("How can I help you?")

# Display chat history
def display_chat_history(msgs):
    avatars = {"human": "user", "ai": "assistant"}
    for msg in msgs.messages:
        st.chat_message(avatars[msg.type]).write(msg.content)

# Main function to run the app
def main():
    api_key = get_api_key(model_choice)
    uploaded_files = upload_files()
    retriever = configure_retriever(uploaded_files)
    msgs, memory = setup_memory()
    initialize_chat_history(msgs)
    display_chat_history(msgs)

    # Handle user input and generate response
    if user_query := st.chat_input(placeholder="Ask me anything!"):
        st.chat_message("user").write(user_query)
        response = generate_response(user_query, model_choice, api_key, retriever, memory)
        st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()