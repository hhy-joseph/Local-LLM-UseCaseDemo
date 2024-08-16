import streamlit as st
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import json

# 初始化 ChatOllama
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
            return "對不起，我不能回應與性或性相關的問題。"

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
        return f"發生錯誤：{str(e)}"

st.title("雙階段防護 Ollama 聊天機械人")

# 初始化聊天歷史記錄
if "messages" not in st.session_state:
    st.session_state.messages = []

# 在應用程式重新運行時顯示聊天歷史中的訊息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 處理用戶輸入
if prompt := st.chat_input("有什麼問題想問？"):
    # 在聊天訊息容器中顯示用戶訊息
    st.chat_message("user").markdown(prompt)
    # 將用戶訊息添加到聊天歷史記錄中
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 創建一個占位符，用於顯示助手的回應
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response = get_ollama_response(prompt, st.session_state.messages)
        st.markdown(response)

    # 將助手的回應添加到聊天歷史記錄中
    st.session_state.messages.append({"role": "assistant", "content": response})

# 添加一個按鈕以清除聊天歷史記錄
if st.button("清除聊天歷史記錄"):
    st.session_state.messages = []
    st.rerun()

# 顯示有關防護措施的信息
st.sidebar.title("關於防護措施")
st.sidebar.info(
    "這個聊天機械人包括一個雙階段內容過濾系統，以確保安全和適當的互動。"
    "它使用AI來檢測並過濾用戶輸入和AI回應中的性相關話題。"
    "機械人不會參與或回應任何與性內容相關的問題或討論。"
)
