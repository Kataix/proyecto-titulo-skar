import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Usamos os.path.join para que funcione en cualquier sistema operativo
DATA_PATH = os.path.join(os.getcwd(), "data")
DB_PATH = os.path.join(os.getcwd(), "vectorstores", "chroma")

def main():
    print("Iniciando el proceso de ingesta de datos...")

    # 1. Cargar los documentos
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    print(f"Se cargaron {len(documents)} documentos.")

    # 2. Dividir los documentos en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Se dividieron los documentos en {len(chunks)} chunks.")

    # 3. Generar embeddings y almacenar en ChromaDB
    embedding_model = HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v2-base-es",
        model_kwargs={'trust_remote_code': True}
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )

    print("¡Ingesta de datos completada y base de datos vectorial creada!")

if __name__ == "__main__":
    main()