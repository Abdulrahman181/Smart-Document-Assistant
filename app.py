import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain_classic.chains import RetrievalQA  
from langchain_core.prompts import PromptTemplate 

st.set_page_config(page_title="Smart Document Assistant", layout="wide")
st.title("Smart Document Assistant")

@st.cache_resource
def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGUF", 
        model_file="llama-2-7b-chat.Q4_K_M.gguf",
        model_type="llama",
        config={
            "temperature": 0.0,
            "max_new_tokens": 512,
            "context_length": 2048
        }
    )
    return llm

@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vector_store = FAISS.load_local("faiss_index_arxiv", embeddings, allow_dangerous_deserialization=True)
    return vector_store

llm = load_llm()
vector_store = load_vector_store()

template = """
You are a strict document-based assistant. 
Answer only based on the provided context. If not found, say "Not mentioned in the documents."

Context: {context}
Question: {question}

Answer:
"""
qa_prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_prompt}
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything about the documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("I'm still thinking and working out the answer..."):
            response = qa_chain.invoke({"query": prompt})
            answer = response["result"]
            sources = response["source_documents"]
            
            if "Not mentioned" in answer:
                full_response = answer
            else:
                full_response = f"{answer}\n\n**Sources:**\n"
                source_names = set()
                for doc in sources:
                    source_names.add(f"- {doc.metadata['source']} (Page {doc.metadata.get('page', 'N/A')})")
                
                full_response += "\n".join(source_names)

            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
