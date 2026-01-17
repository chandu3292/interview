from embeddings import embed

def retrieve(query, index, docs, k=3):
    qvec = embed([query])
    _, ids = index.search(qvec, k)
    return [docs[i] for i in ids[0]]
