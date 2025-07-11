import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_yourkey"

@st.cache_resource
def load_docs_to_qdrant(file_path):
    pages = PyPDFLoader(file_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    dim = len(embeddings.embed_query("test"))

    client = QdrantClient(url="http://localhost:6333")

    if not client.collection_exists("pdf_docs"):
        client.create_collection(
            collection_name="pdf_docs",
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    vs = QdrantVectorStore(client=client, collection_name="pdf_docs", embedding=embeddings)
    vs.add_documents(chunks)
    return vs

st.title("📚 Chat with PDF (Qdrant + HuggingFaceEndpoint)")

uploaded = st.file_uploader("Upload a PDF", type="pdf")
if uploaded:
    path = "temp.pdf"
    with open(path, "wb") as f:
        f.write(uploaded.read())
    st.success("Uploaded!")

    vectorstore = load_docs_to_qdrant(path)

    llm = HuggingFaceEndpoint(
        model="microsoft/Phi-3-mini-4k-instruct",       
        temperature=0.8,                    
        top_k=50,
        max_new_tokens=256,
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    query = st.text_input("Ask a question about your PDF:")
    if query:
        with st.spinner("Thinking..."):
            resp = qa.invoke({"query": query})
            st.write("🧠 Answer:", resp["result"])

