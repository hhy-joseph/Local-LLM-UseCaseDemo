import streamlit as st
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import GPT4AllEmbeddings
import networkx as nx

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatOllama(model="llama3", temperature=0)

# Create a simple graph from text
def create_simple_graph(text):
    sentences = text.split('.')
    G = nx.Graph()
    for i, sentence in enumerate(sentences):
        G.add_node(i, content=sentence.strip())
        if i > 0:
            G.add_edge(i-1, i)
    return G

# Create vector store from text
def create_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    embeddings = GPT4AllEmbeddings()
    return FAISS.from_texts(texts, embeddings)

# Get relevant subgraph
def get_relevant_subgraph(G, query, vector_store):
    relevant_docs = vector_store.similarity_search(query, k=3)
    relevant_sentences = [doc.page_content for doc in relevant_docs]
    subgraph_nodes = [n for n, d in G.nodes(data=True) if any(sentence in d['content'] for sentence in relevant_sentences)]
    return G.subgraph(subgraph_nodes)

# Streamlit UI
st.title("Simple GraphRAG with Ollama")
st.write('This application creates a simple graph structure from your text and uses it for question answering.')

# Text input
ph = """
The year was 1906, and the streets of Hong Kong were buzzing with energy. Amidst the bustling markets and colonial architecture, a group of unlikely heroes converged to save the day.

In a small tea shop on Des Voeux Road, restaurateur Madam Leung was fretting over her daily specials. Her prized dish, "Tiger's Claw," was struggling to attract customers amidst stiff competition from the newly opened Western-style cafes. Just as she was about to give up hope, the door swung open and a dashing young rebel, Wong Fei-hung, strode in.

Wong, a member of the Red Lantern Secret Society, had just received intel that the ruthless Triad leader, Big Sword Wu, planned to kidnap the British Governor's daughter, Lady Victoria. Wong proposed an alliance with Madam Leung: in exchange for her help, he would provide the tea shop with exclusive access to rare herbs and spices, guaranteeing the Tiger's Claw's success.

Madam Leung, intrigued by the prospect of saving her business, agreed to join forces. As they hatched a plan, another unlikely ally appeared at the doorstep – none other than the infamous "Fighting Fool" himself, Lui Lik-wei. This eccentric martial artist was known for his bizarre fighting techniques and flamboyant costumes.

Together, the trio devised a daring scheme: Wong would infiltrate Big Sword Wu's gang, while Madam Leung created a diversion by opening a rival tea shop across the street, complete with an irresistible Tiger's Claw clone. Lui Lik-wei, donning his most outrageous attire, would create chaos in the streets to distract the Triads.

As night fell on Hong Kong, Wong infiltrated Big Sword Wu's hideout and stole the kidnapping plans. Meanwhile, Madam Leung and her new "rival" tea shop drew a massive crowd with their Tiger's Claw imitations. Lui Lik-wei, true to form, brandished his signature umbrella and launched a wild dance routine, causing Triad henchmen to scatter in confusion.

With Big Sword Wu's plans foiled, Wong Fei-hung confronted the Triad leader, engaging him in an epic battle that culminated with the Red Lantern flag waving triumphantly. Lady Victoria was rescued, and Governor's House celebrated the heroics of Madam Leung, Wong Fei-hung, and Lui Lik-wei.

As the trio shared a hearty laugh over steaming cups of Tiger's Claw tea, the streets of Hong Kong whispered about the legendary "Tea Shop Trio" – a group of unconventional heroes who had brought honor to their city. And so, Madam Leung's humble tea shop became a symbol of resistance and unity, with Wong Fei-hung and Lui Lik-wei visiting often, sharing tales of their adventures over cups of that famous Tiger's Claw brew.
"""
text = st.text_area("Enter your text here:", height=200, placeholder=ph)

# Process text button
if st.button("Process Text"):
    if text:
        with st.spinner("Processing text..."):
            graph = create_simple_graph(text)
            vector_store = create_vector_store(text)
            st.session_state['graph'] = graph
            st.session_state['vector_store'] = vector_store
            st.success("Text processed successfully!")
    else:
        st.warning("Please enter some text first.")

# Query input
query = st.text_input("Enter your question:", placeholder='Who is Madam Leung?')

# Answer button
if st.button("Get Answer"):
    if 'vector_store' in st.session_state and 'graph' in st.session_state and query:
        with st.spinner("Thinking..."):
            llm = get_llm()
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm,
                retriever=st.session_state['vector_store'].as_retriever(),
                memory=memory
            )
            
            # Get relevant subgraph
            subgraph = get_relevant_subgraph(st.session_state['graph'], query, st.session_state['vector_store'])
            
            # Add subgraph information to the query
            subgraph_info = "\n".join([f"{d['content']}" for n, d in subgraph.nodes(data=True)])
            enhanced_query = f"Based on the following context, please answer the question: {subgraph_info}\n\nQuestion: {query}"
            
            result = qa_chain({"question": enhanced_query})
            st.write("Answer:", result['answer'])
    elif 'vector_store' not in st.session_state or 'graph' not in st.session_state:
        st.warning("Please process the text first.")
    else:
        st.warning("Please enter a question.")