import streamlit as st
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import GPT4AllEmbeddings
import networkx as nx

# 初始化 LLM
@st.cache_resource
def get_llm():
    return ChatOllama(model="llama3", temperature=0)

# 從文本創建一個簡單的圖
def create_simple_graph(text):
    sentences = text.split('.')
    G = nx.Graph()
    for i, sentence in enumerate(sentences):
        G.add_node(i, content=sentence.strip())
        if i > 0:
            G.add_edge(i-1, i)
    return G

# 從文本創建向量存儲
def create_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    embeddings = GPT4AllEmbeddings()
    return FAISS.from_texts(texts, embeddings)

# 獲取相關的子圖
def get_relevant_subgraph(G, query, vector_store):
    relevant_docs = vector_store.similarity_search(query, k=3)
    relevant_sentences = [doc.page_content for doc in relevant_docs]
    subgraph_nodes = [n for n, d in G.nodes(data=True) if any(sentence in d['content'] for sentence in relevant_sentences)]
    return G.subgraph(subgraph_nodes)

# Streamlit UI
st.title("簡單 GraphRAG with Ollama")
st.write('這個應用程式從您的文本中創建一個簡單的圖結構，並使用它進行問答。')

# 文本輸入
ph = """
時間是1906年，香港的街道充滿了活力。在熙熙攘攘的市場和殖民建築之中，一群不太可能成為英雄的人聚集在一起，準備拯救這一天。

在德輔道上的一家小茶館裡，餐館老闆娘梁太正在為她的每日特餐煩惱。她的招牌菜「虎爪」在剛開張的西式咖啡館的激烈競爭中難以吸引顧客。就在她即將失去希望時，門突然打開，一位英俊的年輕叛軍黃飛鴻走了進來。

黃飛鴻是紅燈籠秘密會的成員，他剛剛收到情報，無情的三合會領袖大刀吳計劃綁架英國總督的女兒維多利亞小姐。黃飛鴻提議與梁太結盟：作為交換，他將為茶館提供稀有的香草和香料，保證虎爪的成功。

梁太對拯救她的生意的前景很感興趣，於是同意合作。就在他們策劃計劃的時候，另一位不太可能的盟友也出現在門口——那就是臭名昭著的「打鬥愚蠢」劉力威。這位古怪的武術家以其怪異的戰鬥技術和華麗的服裝而聞名。

三人一起策劃了一個大膽的計劃：黃飛鴻將潛入大刀吳的幫派，而梁太則通過在街對面開一家競爭對手的茶館來創造分散注意力的效果，並推出一款無法抗拒的虎爪仿製品。劉力威則穿上他最誇張的服裝，在街上製造混亂，以分散三合會的注意力。

當夜幕降臨香港時，黃飛鴻潛入大刀吳的藏身之處，偷走了綁架計劃。同時，梁太和她的新「競爭對手」茶館以他們的虎爪仿製品吸引了大量顧客。劉力威則如常拿出他的標誌性雨傘，開始了一場瘋狂的舞蹈表演，讓三合會的打手們驚慌失措。

隨著大刀吳的計劃被挫敗，黃飛鴻與這位三合會領袖展開了一場史詩般的戰鬥，最終紅燈籠旗幟勝利飄揚。維多利亞小姐得救了，總督府為梁太、黃飛鴻和劉力威的英勇行為舉行了慶祝活動。

當三人在蒸騰的虎爪茶杯旁大笑時，香港的街道開始傳頌這三位非凡英雄的傳說——「茶館三傑」。梁太的謙遜茶館成為了抵抗和團結的象徵，黃飛鴻和劉力威經常光顧，分享他們的冒險故事，伴隨著那著名的虎爪茶。
"""
text = st.text_area("在此輸入您的文本：", height=200, placeholder=ph)

# 處理文本按鈕
if st.button("處理文本"):
    if text:
        with st.spinner("正在處理文本..."):
            graph = create_simple_graph(text)
            vector_store = create_vector_store(text)
            st.session_state['graph'] = graph
            st.session_state['vector_store'] = vector_store
            st.success("文本處理成功！")
    else:
        st.warning("請先輸入一些文本。")

# 查詢輸入
query = st.text_input("輸入您的問題：", placeholder='梁太是誰？')

# 獲取答案按鈕
if st.button("獲取答案"):
    if 'vector_store' in st.session_state and 'graph' in st.session_state and query:
        with st.spinner("思考中..."):
            llm = get_llm()
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm,
                retriever=st.session_state['vector_store'].as_retriever(),
                memory=memory
            )
            
            # 獲取相關子圖
            subgraph = get_relevant_subgraph(st.session_state['graph'], query, st.session_state['vector_store'])
            
            # 將子圖信息添加到查詢中
            subgraph_info = "\n".join([f"{d['content']}" for n, d in subgraph.nodes(data=True)])
            enhanced_query = f"基於以下上下文，請回答問題：{subgraph_info}\n\n問題：{query}"
            
            result = qa_chain({"question": enhanced_query})
            st.write("答案：", result['answer'])
    elif 'vector_store' not in st.session_state or 'graph' not in st.session_state:
        st.warning("請先處理文本。")
    else:
        st.warning("請輸入一個問題。")
