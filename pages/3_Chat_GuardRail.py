import streamlit as st
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import json

# Initialize ChatOllama
chat_model = ChatOllama(model="llama3")

def is_sex_related(prompt):
    try:
        check_message = [
            SystemMessage(content="You are a content filter. Your task is to determine if the given text is related to sex or sexual topics. Respond with a JSON object containing a single key 'sex_related' with a boolean value. Do not provide any other commentary."),
            HumanMessage(content=f"Is the following text related to sex or sexual topics? Text: '{prompt}'")
        ]
        response = chat_model.invoke(check_message)
        result = json.loads(response.content)
        return result.get('sex_related', False)
    except Exception as e:
        st.error(f"Error in content filtering: {str(e)}")
        return False

def get_ollama_response(prompt, history):
    try:
        if is_sex_related(prompt):
            return "I'm sorry, but I can't respond to questions related to sex or sexual topics."

        messages = [SystemMessage(content="You are a helpful AI assistant. Provide accurate and ethical information. Do not share personal data or inappropriate content.")]
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

st.title("Two-Stage Guardrailed Ollama Chatbot")

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

    # Create a placeholder for the assistant's response
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

# Display information about guardrails
st.sidebar.title("About Guardrails")
st.sidebar.info(
    "This chatbot includes a two-stage content filtering system to ensure safe and appropriate interactions. "
    "It uses AI to detect and filter out sex-related topics from both user inputs and AI responses. "
    "The chatbot will not engage in or respond to any questions or discussions "
    "related to sexual content."
)