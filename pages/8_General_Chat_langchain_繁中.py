import streamlit as st
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Initialize ChatOllama
chat_model = ChatOllama(model="llama3")

def get_ollama_response(prompt, history):
    try:
        messages = [SystemMessage(content="你係我條頭馬，你問我答。")]
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=prompt))
        
        response = chat_model.invoke(messages)
        return response.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

st.title("Ollama Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("你有咩問題？"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Create a placeholder for the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("幫緊你"):
            response = get_ollama_response(prompt, st.session_state.messages)
        st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add a button to clear chat history
if st.button("清除對話紀錄"):
    st.session_state.messages = []
    st.rerun()