import streamlit as st
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# 獲取回應的函數
def get_response(prompt, history, system_prompt="你是一個有幫助的AI助手。", 
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
        return f"發生錯誤：{str(e)}"

# Streamlit UI
st.title("Ollama 聊天機械人")

# 頂部的系統提示
system_prompt = st.text_area("系統(Prompt Engineering)", "輸入系統指令", height=100)

# 側邊欄設置
with st.sidebar:
    st.header("聊天設置")
    
    # 超參數
    temperature = st.slider("溫度", min_value=0.0, max_value=2.0, value=1.0, step=0.01)
    max_tokens = st.slider("最大字元數", min_value=1, max_value=2048, value=256, step=1)
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
    frequency_penalty = st.slider("頻率懲罰", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
    presence_penalty = st.slider("存在懲罰", min_value=0.0, max_value=2.0, value=0.0, step=0.01)

# 主聊天界面
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("你有什麼問題想問？"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response = get_response(prompt, st.session_state.messages, system_prompt, 
                                    temperature, max_tokens, top_p, frequency_penalty, presence_penalty)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

if st.button("清除聊天歷史記錄"):
    st.session_state.messages = []
    st.rerun()
