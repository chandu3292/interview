from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def read_pdf(file):
    reader = PdfReader(file)
    return "".join(p.extract_text() or "" for p in reader.pages)

def chunk(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(text)
