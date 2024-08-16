import streamlit as st
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
import os
import tempfile
import shutil
import time

# Create chroma_db directory if it doesn't exist
if not os.path.exists("./chroma_db"):
    os.makedirs("./chroma_db")

# Initialize embeddings
embeddings = GPT4AllEmbeddings()

# Function to create vector store from uploaded PDFs
def create_vector_store(pdf_docs, store_name):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = []
    for pdf in pdf_docs:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(pdf.read())
        temp_file.close()
        loader = PyPDFLoader(temp_file.name)
        pages = loader.load_and_split(text_splitter)
        texts.extend(pages)
        os.unlink(temp_file.name)
    
    vector_store = Chroma.from_documents(texts, embeddings, persist_directory=f"./chroma_db/{store_name}")
    return f"Vector store '{store_name}' created successfully!"

# Function to load existing vector store
def load_vector_store(store_name):
    return Chroma(persist_directory=f"./chroma_db/{store_name}", embedding_function=embeddings)

# Function to delete vector store
def delete_vector_store(store_name):
    store_path = f"./chroma_db/{store_name}"
    if os.path.exists(store_path):
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                shutil.rmtree(store_path)
                return f"Vector store '{store_name}' deleted successfully!"
            except PermissionError:
                if attempt < max_attempts - 1:
                    time.sleep(2)  # Wait for 2 seconds before retrying
                else:
                    return f"Unable to delete '{store_name}'. The file is in use. Please try again later."
    else:
        return f"Vector store '{store_name}' not found."

# Function to get response (with or without RAG)
def get_response(prompt, history, vector_store=None, system_prompt="You are a helpful AI assistant.", 
                 temperature=1.0, max_tokens=256, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, k=3):
    try:
        chat_model = ChatOllama(model="llama3", temperature=temperature, max_tokens=max_tokens,
                                top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
        
        if vector_store:
            qa_chain = ConversationalRetrievalChain.from_llm(
                chat_model,
                vector_store.as_retriever(search_kwargs={"k": k}),
                return_source_documents=True
            )
            converted_history = [(msg["content"], msg["content"]) for msg in history[:-1]]
            result = qa_chain({"question": prompt, "chat_history": converted_history})
            return result['answer'], result['source_documents']
        else:
            messages = [SystemMessage(content=system_prompt)]
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
            messages.append(HumanMessage(content=prompt))
            
            response = chat_model.invoke(messages)
            return response.content, None
    except Exception as e:
        return f"An error occurred: {str(e)}", None

# Streamlit UI
st.title("RAG-enabled Ollama Chatbot")

# System prompt at the top
system_prompt = st.text_area("SYSTEM", "Enter system instructions", height=100)

# Sidebar for vector store management and settings
with st.sidebar:
    st.header("Vector Store Management")
    
    # File uploader
    uploaded_files = st.file_uploader("Upload PDF documents", accept_multiple_files=True, type="pdf")
    store_name = st.text_input("Enter a name for the vector store:")
    if st.button("Create Vector Store") and uploaded_files and store_name:
        result = create_vector_store(uploaded_files, store_name)
        st.success(result)
    
    vector_stores = [d for d in os.listdir("./chroma_db") if os.path.isdir(os.path.join("./chroma_db", d))]
    selected_store = st.selectbox("Select a vector store:", ["None"] + vector_stores)

    # Vector store deletion
    st.subheader("Delete Vector Store")
    delete_store = st.selectbox("Select a vector store to delete:", ["None"] + vector_stores)
    if st.button("Delete Selected Store") and delete_store != "None":
        result = delete_vector_store(delete_store)
        st.info(result)
        if "deleted successfully" in result:
            time.sleep(1)  # Give a moment for the file system to update
            st.rerun()

    st.header("Chat Settings")
    
    # Hyperparameters
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.01)
    max_tokens = st.slider("Maximum Tokens", min_value=1, max_value=2048, value=256, step=1)
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
    frequency_penalty = st.slider("Frequency penalty", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
    presence_penalty = st.slider("Presence penalty", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
    k = st.slider("Number of k", min_value=1, max_value=20, value=3, step=1)

# Main chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is your question?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if selected_store != "None":
                vector_store = load_vector_store(selected_store)
                response, sources = get_response(prompt, st.session_state.messages, vector_store, system_prompt, 
                                                 temperature, max_tokens, top_p, frequency_penalty, presence_penalty, k)
            else:
                response, sources = get_response(prompt, st.session_state.messages, system_prompt=system_prompt, 
                                                 temperature=temperature, max_tokens=max_tokens, top_p=top_p, 
                                                 frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
        
        st.markdown(response)
        if sources:
            with st.expander("Sources"):
                for i, doc in enumerate(sources):
                    st.write(f"Source {i+1}:")
                    st.write(doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": response})

if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()