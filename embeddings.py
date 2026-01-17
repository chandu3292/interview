import streamlit as st
import faiss
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
EMBED_MODEL = "text-embedding-3-small"

def embed(texts):
    res = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    return np.array([e.embedding for e in res.data]).astype("float32")

def build_index(vectors):
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index
