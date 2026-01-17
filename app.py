import streamlit as st
import asyncio
import nest_asyncio

from utils import read_pdf, chunk
from embeddings import embed, build_index
from rag import retrieve
from realtime_ws import stream_response

st.set_page_config(page_title="Realtime RAG Chat", layout="centered")
st.title("⚡ Realtime Chat + FAISS RAG")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.docs = None

# Upload PDF
pdf = st.file_uploader("Upload knowledge PDF", type=["pdf"])

if pdf:
    with st.spinner("Indexing document..."):
        text = read_pdf(pdf)
        chunks = chunk(text)
        vectors = embed(chunks)
        st.session_state.index = build_index(vectors)
        st.session_state.docs = chunks
    st.success("Document indexed!")

# Display chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask something...")

if prompt and st.session_state.index:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        # RAG retrieval
        chunks = retrieve(
            prompt,
            st.session_state.index,
            st.session_state.docs
        )
        context = "\n\n".join(chunks)

        answer = {"text": ""}

        async def run():
            async for token in stream_response(prompt, context):
                answer["text"] += token
                placeholder.markdown(answer["text"] + "▌")

        nest_asyncio.apply()
        asyncio.run(run())
        placeholder.markdown(answer["text"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer["text"]
    })
