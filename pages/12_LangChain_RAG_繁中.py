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

# 創建 chroma_db 目錄（如果不存在）
if not os.path.exists("./chroma_db"):
    os.makedirs("./chroma_db")

# 初始化嵌入
embeddings = GPT4AllEmbeddings()

# 從上傳的 PDF 文件創建向量存儲的函數
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
    return f"向量存儲 '{store_name}' 成功創建！"

# 加載現有的向量存儲的函數
def load_vector_store(store_name):
    return Chroma(persist_directory=f"./chroma_db/{store_name}", embedding_function=embeddings)

# 刪除向量存儲的函數
def delete_vector_store(store_name):
    store_path = f"./chroma_db/{store_name}"
    if os.path.exists(store_path):
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                shutil.rmtree(store_path)
                return f"向量存儲 '{store_name}' 成功刪除！"
            except PermissionError:
                if attempt < max_attempts - 1:
                    time.sleep(2)  # 等待2秒再重試
                else:
                    return f"無法刪除 '{store_name}'。文件正在使用中。請稍後再試。"
    else:
        return f"未找到向量存儲 '{store_name}'。"

# 獲取回應的函數（有或無 RAG）
def get_response(prompt, history, vector_store=None, system_prompt="你是一個有幫助的AI助手。", 
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
        return f"發生錯誤：{str(e)}", None

# Streamlit UI
st.title("支持 RAG 的 Ollama 聊天機械人")

# 頂部的系統提示
system_prompt = st.text_area("系統提示", "輸入系統指令", height=100)

# 側邊欄管理向量存儲和設置
with st.sidebar:
    st.header("向量存儲管理")
    
    # 文件上傳
    uploaded_files = st.file_uploader("上傳 PDF 文件", accept_multiple_files=True, type="pdf")
    store_name = st.text_input("為向量存儲輸入一個名稱：")
    if st.button("創建向量存儲") and uploaded_files and store_name:
        result = create_vector_store(uploaded_files, store_name)
        st.success(result)
    
    vector_stores = [d for d in os.listdir("./chroma_db") if os.path.isdir(os.path.join("./chroma_db", d))]
    selected_store = st.selectbox("選擇一個向量存儲：", ["無"] + vector_stores)

    # 刪除向量存儲
    st.subheader("刪除向量存儲")
    delete_store = st.selectbox("選擇一個要刪除的向量存儲：", ["無"] + vector_stores)
    if st.button("刪除選定的存儲") and delete_store != "無":
        result = delete_vector_store(delete_store)
        st.info(result)
        if "成功刪除" in result:
            time.sleep(1)  # 給文件系統一點時間更新
            st.rerun()

    st.header("聊天設置")
    
    # 超參數
    temperature = st.slider("溫度", min_value=0.0, max_value=2.0, value=1.0, step=0.01)
    max_tokens = st.slider("最大字元數", min_value=1, max_value=2048, value=256, step=1)
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
    frequency_penalty = st.slider("頻率懲罰", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
    presence_penalty = st.slider("存在懲罰", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
    k = st.slider("k 的數量", min_value=1, max_value=20, value=3, step=1)

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
            if selected_store != "無":
                vector_store = load_vector_store(selected_store)
                response, sources = get_response(prompt, st.session_state.messages, vector_store, system_prompt, 
                                                 temperature, max_tokens, top_p, frequency_penalty, presence_penalty, k)
            else:
                response, sources = get_response(prompt, st.session_state.messages, system_prompt=system_prompt, 
                                                 temperature=temperature, max_tokens=max_tokens, top_p=top_p, 
                                                 frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
        
        st.markdown(response)
        if sources:
            with st.expander("來源"):
                for i, doc in enumerate(sources):
                    st.write(f"來源 {i+1}:")
                    st.write(doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": response})

if st.button("清除聊天歷史"):
    st.session_state.messages = []
    st.rerun()
