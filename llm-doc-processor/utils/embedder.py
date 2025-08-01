from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

def create_vectorstore(chunks, save_path="vectorstore/faiss_index"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # latest model
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    print("✅ Vectorstore saved to:", save_path)

def load_vectorstore(save_path="vectorstore/faiss_index"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(save_path, embeddings)
