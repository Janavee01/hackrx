from utils.parser import load_and_split_pdf
from utils.embedder import create_vectorstore
import os

  # Put your real OpenAI key here

if __name__ == "__main__":
    chunks = load_and_split_pdf("documents/HDFHLIP23024V072223.pdf")
    create_vectorstore(chunks)
