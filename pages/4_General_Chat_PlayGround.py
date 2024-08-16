import streamlit as st
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Function to get response
def get_response(prompt, history, system_prompt="You are a helpful AI assistant.", 
                 temperature=1.0, max_tokens=256, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0):
    try:
        chat_model = ChatOllama(model="llama3", temperature=temperature, max_tokens=max_tokens,
                                top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
        
        messages = [SystemMessage(content=system_prompt)]
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

# Streamlit UI
st.title("Ollama Chatbot")

# System prompt at the top
system_prompt = st.text_area("SYSTEM", "Enter system instructions", height=100)

# Sidebar for settings
with st.sidebar:
    st.header("Chat Settings")
    
    # Hyperparameters
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.01)
    max_tokens = st.slider("Maximum Tokens", min_value=1, max_value=2048, value=256, step=1)
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
    frequency_penalty = st.slider("Frequency penalty", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
    presence_penalty = st.slider("Presence penalty", min_value=0.0, max_value=2.0, value=0.0, step=0.01)

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
            response = get_response(prompt, st.session_state.messages, system_prompt, 
                                    temperature, max_tokens, top_p, frequency_penalty, presence_penalty)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()