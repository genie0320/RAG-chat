# from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    PDFMinerLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain.embeddings import sentenceTransformerEmbeddings
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings

# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma

import os
from dotenv import load_dotenv
from constants import CHROMA_SETTINGS

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
persist_directory = "db"


def main():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create Embeddings
    # embeddings = sentenceTransformerEmbeddings()
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    # Create vector store
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=persist_directory,
        client_settings=CHROMA_SETTINGS,
    )
    db.persist()
    db = None

    # The if __name__ == '__main__': block ensures the main() function runs only when the script is executed directly, not when imported as a module.
    if __name__ == "__main__":
        main()
