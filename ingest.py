import os
from dotenv import load_dotenv

# from langchain_community.document_loaders import DirectoryLoader,PDFMinerLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.embeddings import OpenAIEmbeddings

from langchain_community.vectorstores import Chroma

import chromadb

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
persist_directory = "./db"


def main():
    for root, dirs, files in os.walk("docs/"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PyPDFLoader(os.path.join(root, file))

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create Embeddings
    # embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create vector store
    # FIX: Module 'langchain_community.vectorstores.Chroma' has no 'from_documents'
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=persist_directory,
    )
    db.persist()
    db = None


# The if __name__ == '__main__': block ensures the main() function runs only when the script is executed directly, not when imported as a module.
if __name__ == "__main__":
    main()
