# scripts/ingesta_datos.py (Versión con Extracción de Fechas)

import os
import argparse
import re
from datetime import datetime # Importar datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, ConfluenceLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import pandas as pd
from langchain_core.documents import Document

load_dotenv()

# --- CONSTANTES ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(PROJECT_ROOT, "vectorstores", "chroma")
EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v2-base-es"
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_API_KEY = os.getenv("CONFLUENCE_API_KEY")
CONFLUENCE_SPACE_KEY = "SKAR"

# --- FUNCIÓN: Extracción de Metadatos (Incluye Fechas) ---
def extraer_metadatos_incidente(contenido, metadata_original):
    """Extrae metadatos clave, incluyendo fechas inicio/fin."""
    metadata = {}
    source = metadata_original.get('source', '')
    if 'InventarioServidores.xlsx' in source: return metadata # No aplica a Excel aquí

    # Extraer Ticket ID, Sistemas, Impacto, Estado (como antes)
    # ... (código regex omitido por brevedad, igual que antes) ...
    match_ticket = re.search(r"Ticket:\s*(\d+)", contenido, re.IGNORECASE);
    if match_ticket: metadata["ticket_id"] = match_ticket.group(1).strip()
    match_sistemas = re.search(r"Sistemas afectados:\s*(.*)", contenido, re.IGNORECASE);
    if match_sistemas: metadata["sistemas_afectados"] = match_sistemas.group(1).strip()
    match_impacto = re.search(r"Impacto:\s*(Alto|Medio|Bajo)", contenido, re.IGNORECASE);
    if match_impacto: metadata["impacto"] = match_impacto.group(1).strip()
    match_estado = re.search(r"Estado:\s*(Resuelto|Activo|Pendiente)", contenido, re.IGNORECASE);
    if match_estado: metadata["estado"] = match_estado.group(1).strip()


    # --- NUEVO: Extraer Fechas ---
    fecha_inicio_int = None
    fecha_fin_int = None

    # Buscar Fecha Inicio (ej. Inicio: dd/mm/yyyy HH:MM)
    match_inicio = re.search(r"Inicio:\s*(\d{2}/\d{2}/\d{4})", contenido, re.IGNORECASE)
    if match_inicio:
        try:
            fecha_dt = datetime.strptime(match_inicio.group(1), "%d/%m/%Y")
            fecha_inicio_int = int(fecha_dt.strftime("%Y%m%d")) # Convertir a YYYYMMDD
            metadata["fecha_inicio_int"] = fecha_inicio_int
        except ValueError:
            print(f"Advertencia: Formato de fecha de inicio inválido en {source}: {match_inicio.group(1)}")

    # Buscar Fecha Fin (ej. Fin: dd/mm/yyyy HH:MM)
    match_fin = re.search(r"Fin:\s*(\d{2}/\d{2}/\d{4})", contenido, re.IGNORECASE)
    if match_fin:
        try:
            fecha_dt = datetime.strptime(match_fin.group(1), "%d/%m/%Y")
            fecha_fin_int = int(fecha_dt.strftime("%Y%m%d")) # Convertir a YYYYMMDD
            metadata["fecha_fin_int"] = fecha_fin_int
        except ValueError:
            print(f"Advertencia: Formato de fecha de fin inválido en {source}: {match_fin.group(1)}")
    # --- FIN NUEVO ---

    return metadata
# --- FIN FUNCIÓN ---

# --- FUNCIONES DE CARGA DE DATOS ---
def cargar_documentos_locales():
    """Carga documentos, procesa Excel con Pandas."""
    print(f"Cargando docs desde: {DATA_PATH}...")
    documentos_cargados = []
    # 1. Procesar Excel con Pandas (crea Document por fila, asigna metadatos)
    archivos_excel = [f for f in os.listdir(DATA_PATH) if f.endswith('.xlsx')]
    for archivo_xlsx in archivos_excel:
        ruta_completa = os.path.join(DATA_PATH, archivo_xlsx); print(f"  Procesando Excel: {archivo_xlsx}...")
        try:
            excel_data = pd.read_excel(ruta_completa, sheet_name=None)
            total_filas = 0
            for nombre_hoja, df in excel_data.items():
                print(f"    Leyendo hoja: '{nombre_hoja}'...")
                for index, row in df.iterrows():
                    contenido_fila = "\n".join([f"- {col}: {val}" for col, val in row.items() if pd.notna(val)])
                    metadatos_fila = {k: str(v) for k, v in row.to_dict().items() if pd.notna(v)}
                    metadatos_fila['source'] = ruta_completa
                    if 'NombreServidor' in metadatos_fila: metadatos_fila['nombre_servidor'] = metadatos_fila['NombreServidor'].strip().upper()
                    elif 'Nombre Servidor' in metadatos_fila: metadatos_fila['nombre_servidor'] = metadatos_fila['Nombre Servidor'].strip().upper()
                    documentos_cargados.append(Document(page_content=contenido_fila, metadata=metadatos_fila))
                total_filas += len(df)
            print(f"    > {total_filas} filas procesadas de {archivo_xlsx}")
        except Exception as e_pandas: print(f"[ERROR] Pandas Excel '{archivo_xlsx}': {e_pandas}")
    # 2. Procesar resto (.md, .txt)
    print("  Procesando otros archivos (.md, .txt)...")
    loader = DirectoryLoader(DATA_PATH, glob="**/*[.md|.txt]", use_multithreading=True, show_progress=True)
    try:
        documentos_resto = loader.load(); documentos_cargados.extend(documentos_resto)
        print(f"    > {len(documentos_resto)} archivos .md/.txt cargados.")
    except Exception as e_dirloader: print(f"[ERROR] DirectoryLoader: {e_dirloader}")
    return documentos_cargados

def cargar_documentos_confluence():
    # ... (código igual) ...
    if not all([CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_KEY]): raise ValueError("Credenciales Confluence no en .env")
    print(f"Cargando docs desde Confluence (Espacio: {CONFLUENCE_SPACE_KEY})..."); loader = ConfluenceLoader(url=CONFLUENCE_URL, username=CONFLUENCE_USERNAME, api_key=CONFLUENCE_API_KEY)
    return loader.load(space_key=CONFLUENCE_SPACE_KEY, include_attachments=False, limit=50)

# --- FUNCIÓN PRINCIPAL ---
def main():
    parser = argparse.ArgumentParser(description="Ingesta SKAR."); parser.add_argument("--fuente", type=str, required=True, choices=['local', 'confluence'], help="Fuente."); args = parser.parse_args()
    print("Iniciando ingesta...")
    if args.fuente == "local": documents = cargar_documentos_locales()
    elif args.fuente == "confluence": documents = cargar_documentos_confluence()
    else: print(f"Error: Fuente '{args.fuente}' inválida."); return
    if not documents: print("No se encontraron documentos."); return
    print(f"Total docs cargados de '{args.fuente}': {len(documents)}.")

    # --- Extraer metadatos de documentos NO Excel ---
    print("Extrayendo metadatos (Incidentes)..."); doc_count_with_metadata = 0
    for doc in documents:
        source = doc.metadata.get('source', ''); is_excel = source.endswith('.xlsx') if source else False
        if not is_excel:
             nuevos_metadatos = extraer_metadatos_incidente(doc.page_content, doc.metadata)
             if nuevos_metadatos:
                doc.metadata.update(nuevos_metadatos); doc_count_with_metadata += 1
                print(f"  > Metadatos Incidente: {source[-40:] if source else 'N/A'}, {nuevos_metadatos}")
        elif 'nombre_servidor' in doc.metadata: doc_count_with_metadata += 1 # Contar los de Excel
    print(f"Metadatos relevantes asignados/encontrados en {doc_count_with_metadata}/{len(documents)} docs base.")

    # --- Chunking, Embedding, Guardado ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_documents(documents)
    print(f"Divididos en {len(chunks)} chunks.");
    # (Impresión de ejemplo de metadatos omitida por brevedad)

    print(f"Creando embeddings..."); embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'trust_remote_code': True})
    print(f"Guardando en BD: {DB_PATH}..."); vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=DB_PATH)
    print("\n¡Ingesta completada!"); print(f"BD vectorial actualizada en: {DB_PATH}")

if __name__ == "__main__":
    main()