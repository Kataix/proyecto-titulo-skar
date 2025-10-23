# scripts/ingesta_datos.py (Versión Final con Pandas para Excel)

import os
import argparse
import re # Importa regex
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, ConfluenceLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from datetime import datetime
import pandas as pd # Importar pandas
from langchain_core.documents import Document # Para crear documentos manualmente

# Carga las variables de entorno desde el archivo .env al inicio
load_dotenv()

# --- CONSTANTES DE CONFIGURACIÓN ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(PROJECT_ROOT, "vectorstores", "chroma")
EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v2-base-es"

# Lee las credenciales de Confluence de forma segura
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_API_KEY = os.getenv("CONFLUENCE_API_KEY")
CONFLUENCE_SPACE_KEY = "SKAR" # O la clave que uses

# --- FUNCIÓN: Extracción de Metadatos para Incidentes ---
def extraer_metadatos_incidente(contenido, metadata_original):
    """Intenta extraer metadatos clave de documentos de INCIDENTE."""
    metadata = {}
    # Solo aplica si NO es del Excel
    source = metadata_original.get('source', '')
    if 'InventarioServidores.xlsx' in source:
        return metadata # No extrae de Excel aquí

    # Buscar Ticket ID
    match_ticket = re.search(r"Ticket:\s*(\d+)", contenido, re.IGNORECASE)
    if match_ticket: metadata["ticket_id"] = match_ticket.group(1).strip()
    # Buscar Sistemas Afectados
    match_sistemas = re.search(r"Sistemas afectados:\s*(.*)", contenido, re.IGNORECASE)
    if match_sistemas: metadata["sistemas_afectados"] = match_sistemas.group(1).strip()
    # Buscar Impacto
    match_impacto = re.search(r"Impacto:\s*(Alto|Medio|Bajo)", contenido, re.IGNORECASE)
    if match_impacto: metadata["impacto"] = match_impacto.group(1).strip()
    # Buscar Estado
    match_estado = re.search(r"Estado:\s*(Resuelto|Activo|Pendiente)", contenido, re.IGNORECASE)
    if match_estado: metadata["estado"] = match_estado.group(1).strip()
    return metadata
# --- FIN FUNCIÓN ---

# --- FUNCIONES DE CARGA DE DATOS ---
def cargar_documentos_locales():
    """
    Carga documentos desde /data. Trata .xlsx con pandas, el resto con DirectoryLoader.
    """
    print(f"Cargando documentos desde la ruta local: {DATA_PATH}...")

    documentos_cargados = []
    archivos_procesados = set()

    # 1. Procesar archivos Excel con Pandas
    archivos_excel = [f for f in os.listdir(DATA_PATH) if f.endswith('.xlsx')]
    for archivo_xlsx in archivos_excel:
        ruta_completa = os.path.join(DATA_PATH, archivo_xlsx)
        print(f"  Procesando Excel: {archivo_xlsx} con Pandas...")
        try:
            # Lee todas las hojas del Excel
            excel_data = pd.read_excel(ruta_completa, sheet_name=None)
            for nombre_hoja, df in excel_data.items():
                print(f"    Leyendo hoja: '{nombre_hoja}'...")
                # Convierte cada fila del DataFrame en un objeto Document
                for index, row in df.iterrows():
                    # Contenido: Combina todas las celdas de la fila en un string legible
                    contenido_fila = "\n".join([f"- {columna}: {valor}" for columna, valor in row.items() if pd.notna(valor)])
                    # Metadatos: Usa las columnas como metadatos, limpiando valores NaN y convirtiendo a string
                    metadatos_fila = {k: str(v) for k, v in row.to_dict().items() if pd.notna(v)}
                    # Añade fuente y estandariza nombre_servidor
                    metadatos_fila['source'] = ruta_completa
                    if 'NombreServidor' in metadatos_fila: # Usa el nombre exacto de tu columna
                         metadatos_fila['nombre_servidor'] = metadatos_fila['NombreServidor'].strip().upper()
                    elif 'Nombre Servidor' in metadatos_fila: # Añade posibles variaciones
                         metadatos_fila['nombre_servidor'] = metadatos_fila['Nombre Servidor'].strip().upper()

                    documentos_cargados.append(Document(page_content=contenido_fila, metadata=metadatos_fila))
            archivos_procesados.add(archivo_xlsx)
            print(f"    > {len(df) * len(excel_data)} filas procesadas de {archivo_xlsx}")
        except Exception as e_pandas:
            print(f"[ERROR] No se pudo procesar el archivo Excel '{archivo_xlsx}' con Pandas: {e_pandas}")

    # 2. Procesar el resto de archivos (.md, .txt) con DirectoryLoader
    print("  Procesando otros archivos (.md, .txt) con DirectoryLoader...")
    # Usamos un lambda en `glob` para excluir los .xlsx ya procesados
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*[.md|.txt]", # Solo .md y .txt
        use_multithreading=True,
        show_progress=True,
        # Excluir archivos ya procesados (si es necesario, aunque glob ya lo hace)
        # loader_kwargs={"exclude": list(archivos_procesados)}
    )
    try:
        documentos_resto = loader.load()
        documentos_cargados.extend(documentos_resto)
        print(f"    > {len(documentos_resto)} archivos .md/.txt cargados.")
    except Exception as e_dirloader:
         print(f"[ERROR] No se pudo cargar archivos .md/.txt con DirectoryLoader: {e_dirloader}")


    return documentos_cargados
# --- FIN FUNCIÓN MODIFICADA ---

def cargar_documentos_confluence():
    # ... (código igual) ...
    if not all([CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_KEY]): raise ValueError("Credenciales Confluence no en .env")
    print(f"Cargando documents desde Confluence (Espacio: {CONFLUENCE_SPACE_KEY})...")
    loader = ConfluenceLoader(url=CONFLUENCE_URL, username=CONFLUENCE_USERNAME, api_key=CONFLUENCE_API_KEY)
    return loader.load(space_key=CONFLUENCE_SPACE_KEY, include_attachments=False, limit=50)

# --- FUNCIÓN PRINCIPAL ---
def main():
    parser = argparse.ArgumentParser(description="Ingesta SKAR.")
    parser.add_argument("--fuente", type=str, required=True, choices=['local', 'confluence'], help="Fuente de datos.")
    args = parser.parse_args()

    print("Iniciando ingesta...")

    if args.fuente == "local": documents = cargar_documentos_locales() # Llama a la nueva función
    elif args.fuente == "confluence": documents = cargar_documentos_confluence()
    else: print(f"Error: Fuente '{args.fuente}' inválida."); return

    if not documents: print("No se encontraron documentos."); return
    print(f"Total documentos cargados de '{args.fuente}': {len(documents)}.")

    # --- Extraer metadatos de documentos NO Excel ---
    print("Extrayendo metadatos de documentos NO Excel...")
    doc_count_with_metadata = 0
    for doc in documents:
        source = doc.metadata.get('source', '')
        # Solo aplica si NO viene del Excel (asumiendo que source contiene el nombre)
        is_excel = False
        if source and isinstance(source, str):
            is_excel = source.endswith('.xlsx')

        if not is_excel:
             nuevos_metadatos = extraer_metadatos_incidente(doc.page_content, doc.metadata)
             if nuevos_metadatos:
                doc.metadata.update(nuevos_metadatos)
                doc_count_with_metadata += 1
                print(f"  > Metadatos Incidente: {source[-40:] if source else 'N/A'}, {nuevos_metadatos}")
        # Contar los que ya tenían metadatos del Excel (ej. nombre_servidor)
        elif 'nombre_servidor' in doc.metadata:
             doc_count_with_metadata += 1
             print(f"  > Metadatos Servidor: {source[-40:] if source else 'N/A'}, {{'nombre_servidor': {doc.metadata['nombre_servidor']}}}")


    print(f"Metadatos relevantes asignados/encontrados en {doc_count_with_metadata}/{len(documents)} documentos base.")

    # --- Chunking, Embedding, Guardado ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_documents(documents)
    print(f"Divididos en {len(chunks)} chunks.")
    if chunks:
        # Imprime metadatos de un chunk de Excel y uno de texto si es posible
        chunk_excel_example = next((c for c in chunks if c.metadata.get('source', '').endswith('.xlsx')), None)
        chunk_text_example = next((c for c in chunks if not c.metadata.get('source', '').endswith('.xlsx')), None)
        if chunk_excel_example: print(f"Ejemplo metadatos 1er chunk Excel: {chunk_excel_example.metadata}")
        if chunk_text_example: print(f"Ejemplo metadatos 1er chunk Texto: {chunk_text_example.metadata}")


    print(f"Creando embeddings con: {EMBEDDING_MODEL_NAME}...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'trust_remote_code': True})

    print(f"Guardando en BD: {DB_PATH}...")
    #  BD se sobrescribe o actualiza correctamente
    # Chroma.from_documents por defecto añade, si quieres sobrescribir, borra el directorio antes
    # O usa collection.add() con IDs si quieres actualizar
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=DB_PATH)

    print("\n¡Ingesta completada!")
    print(f"BD vectorial actualizada en: {DB_PATH}")
    print("NOTA: Revisa metadatos extraídos.")

    # --- Opcional: Ejecutar evaluación ---
    # (Código omitido)

if __name__ == "__main__":
    main()