import streamlit as st
import ollama

def get_ollama_response(prompt, history):
    try:
        messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
        messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        
        response = ollama.chat(model='llama3', messages=messages)
        return response['message']['content']
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
if prompt := st.chat_input("What is your question?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_ollama_response(prompt, st.session_state.messages)
            st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()