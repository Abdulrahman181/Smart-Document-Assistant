import os
from pathlib import Path

import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


DEFAULT_INDEX_PATH = Path(__file__).resolve().parent / "faiss_index_arxiv"
INDEX_PATH = Path(os.environ.get("FAISS_INDEX_PATH", str(DEFAULT_INDEX_PATH))).expanduser()

st.set_page_config(page_title="Smart Document Assistant", layout="wide")
st.title("Smart Document Assistant")


@st.cache_resource
def load_llm():
    return CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGUF",
        model_file="llama-2-7b-chat.Q4_K_M.gguf",
        model_type="llama",
        config={
            "temperature": 0.0,
            "max_new_tokens": 512,
            "context_length": 2048,
        },
    )


@st.cache_resource
def load_vector_store(index_path: str):
    if not Path(index_path).is_dir():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. Build it by following the README instructions."
        )
    if os.environ.get("TRUSTED_FAISS_INDEX") != "1":
        raise RuntimeError(
            "FAISS indexes loaded by this application contain a pickle docstore. "
            "Only load an index you created or otherwise fully trust. After verifying "
            "the index, set TRUSTED_FAISS_INDEX=1 to opt in."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


try:
    vector_store = load_vector_store(str(INDEX_PATH))
    llm = load_llm()
except (FileNotFoundError, RuntimeError) as error:
    st.error(str(error))
    st.stop()
except Exception:
    st.error(
        "The document assistant could not load its model or index. Check the "
        "installation and README setup steps, then try again."
    )
    st.stop()


template = """
You are a document-based assistant. Answer using the provided context. If the answer is
not supported by the context, say: "Not mentioned in the documents."

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
    chain_type_kwargs={"prompt": qa_prompt},
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
                source_names = {
                    f"- {doc.metadata.get('source', 'Unknown source')} "
                    f"(Page {doc.metadata.get('page', 'N/A')})"
                    for doc in sources
                }
                full_response = answer
                if source_names:
                    full_response += "\n\n**Sources:**\n" + "\n".join(sorted(source_names))

            st.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
