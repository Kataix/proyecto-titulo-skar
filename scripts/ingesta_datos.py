import os
import argparse
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, ConfluenceLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Carga las variables de entorno desde el archivo .env al inicio
load_dotenv()

# --- CONSTANTES DE CONFIGURACIÓN ---

# Rutas del proyecto (se calculan dinámicamente para ser flexibles)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(PROJECT_ROOT, "vectorstores", "chroma")

# Modelo de embeddings a utilizar
EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v2-base-es"

# Lee las credenciales de Confluence de forma segura desde las variables de entorno
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONfluence_api_key = os.getenv("CONFLUENCE_API_KEY")
CONFLUENCE_SPACE_KEY = "SKAR" # O la clave del espacio que hayas configurado

# --- FUNCIONES DE CARGA DE DATOS ---

def cargar_documentos_locales():
    """
    Carga documentos desde la carpeta local /data.
    Busca archivos .md, .txt y .xlsx.
    """
    print(f"Cargando documentos desde la ruta local: {DATA_PATH}...")
    loader = DirectoryLoader(DATA_PATH, glob="**/*[.md|.txt|.xlsx]", use_multithreading=True, show_progress=True)
    return loader.load()

def cargar_documentos_confluence():
    """
    Carga documentos desde un espacio de Confluence.
    """
    # Verificación para asegurar que las variables de entorno fueron cargadas
    if not all([CONFLUENCE_URL, CONFLUENCE_USERNAME, CONfluence_api_key]):
        raise ValueError("Error: Las credenciales de Confluence (URL, USERNAME, API_KEY) no están configuradas en el archivo .env")

    print(f"Cargando documentos desde Confluence (Espacio: {CONFLUENCE_SPACE_KEY})...")
    loader = ConfluenceLoader(
        url=CONFLUENCE_URL,
        username=CONFLUENCE_USERNAME,
        api_key=CONfluence_api_key
    )
    # Carga hasta 50 páginas del espacio, sin incluir adjuntos
    return loader.load(space_key=CONFLUENCE_SPACE_KEY, include_attachments=False, limit=50)

# --- FUNCIÓN PRINCIPAL ---

def main():
    """
    Función principal para orquestar la ingesta de datos.
    Permite seleccionar la fuente de datos (local o confluence) mediante un argumento.
    """
    # 1. Configurar argumentos de línea de comandos para elegir la fuente
    parser = argparse.ArgumentParser(description="Proceso de ingesta de datos para SKAR.")
    parser.add_argument(
        "--fuente",
        type=str,
        required=True,
        choices=['local', 'confluence'],
        help="La fuente de los datos a ingestar: 'local' o 'confluence'."
    )
    args = parser.parse_args()

    print("Iniciando el proceso de ingesta de datos...")

    # 2. Cargar documentos según la fuente seleccionada
    if args.fuente == "local":
        documents = cargar_documentos_locales()
    elif args.fuente == "confluence":
        documents = cargar_documentos_confluence()
    else:
        # Esta comprobación es redundante gracias a 'choices', pero es una buena práctica
        print(f"Error: Fuente '{args.fuente}' no válida. Saliendo.")
        return

    if not documents:
        print("No se encontraron documentos para procesar. Saliendo.")
        return

    print(f"Se cargaron {len(documents)} documentos de la fuente '{args.fuente}'.")

    # 3. Dividir los documentos en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Se dividieron los documentos en {len(chunks)} chunks.")

    # 4. Generar embeddings y almacenar en ChromaDB
    print(f"Creando embeddings con el modelo: {EMBEDDING_MODEL_NAME}...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'trust_remote_code': True}
    )

    print(f"Guardando vectores en la base de datos: {DB_PATH}...")
    # Usamos Chroma.from_documents para crear/sobrescribir la BD y persistirla
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )

    print("\n¡Ingesta de datos completada exitosamente!")
    print(f"La base de datos vectorial ha sido creada/actualizada en: {DB_PATH}")

if __name__ == "__main__":
    main()