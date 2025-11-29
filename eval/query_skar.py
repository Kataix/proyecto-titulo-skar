# query_skar.py - Versión para Depuración de Traceback

import argparse
from dotenv import load_dotenv
import sys
import os
import traceback # <<<<< Asegúrate de que esta línea esté aquí

# Añade la ruta raíz del proyecto al sys.path para poder importar desde skar_api
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Alternativa si query_skar.py no está en la raíz

# Ajuste si query_skar.py está en la raíz del proyecto
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # Si query_skar.py está en la raíz
sys.path.insert(0, PROJECT_ROOT) # Añade la raíz al path para que pueda encontrar skar_api

# Importa la cadena RAG DESPUÉS de modificar el sys.path
try:
    from skar_api.pipeline import rag_chain
except ImportError as ie:
    print(f"\n[ERROR] No se pudo importar 'rag_chain' desde 'skar_api.pipeline'. Error: {ie}")
    print("Asegúrate de que la estructura de carpetas sea 'tu_proyecto/skar_api/pipeline.py'")
    print(f"sys.path actual: {sys.path}")
    exit()
except Exception as e:
    print(f"\n[ERROR] Ocurrió un error general al importar el pipeline: {e}")
    print(traceback.format_exc())
    exit()

def main():
    """
    Función principal para ejecutar consultas individuales a SKAR.
    """
    load_dotenv() # Carga el .env (aunque no lo usemos mucho con Ollama local)

    # Configurar argparse para recibir la pregunta
    parser = argparse.ArgumentParser(description="Consulta al Agente SKAR (Versión Simplificada).")
    parser.add_argument("-p", "--pregunta",
                        type=str,
                        required=True,
                        help="La pregunta que quieres hacerle al agente SKAR.")

    args = parser.parse_args()

    print("Pipeline RAG importado. Procesando pregunta...")

    try:
        # --- Invoca la cadena (entrada es string, salida es dict) ---
        # La cadena simplificada espera un string como entrada
        resultado = rag_chain.invoke(args.pregunta)

        # Extraer la respuesta y los contextos del diccionario resultante
        # Añadimos verificación por si 'resultado' no es un diccionario
        if isinstance(resultado, dict):
            respuesta = resultado.get('answer', '[ERROR: No se encontró la clave "answer" en la salida del pipeline]')
            contextos_docs = resultado.get('contexts', []) # Lista de objetos Document
        else:
            # Si 'resultado' NO es un diccionario, es probablemente un error
            print(f"[ERROR INTERNO] La cadena RAG devolvió un tipo inesperado: {type(resultado)}")
            print(f"Valor devuelto: {resultado}")
            respuesta = "[ERROR: La cadena RAG no devolvió un diccionario]"
            contextos_docs = []


        # --- Fin Invocación ---

        print("\n" + "="*50)
        print(f"PREGUNTA: {args.pregunta}")
        print("="*50)

        print("\nRESPUESTA GENERADA:")
        print(respuesta)

        print("\n" + "-"*50)
        # Mostramos los contextos que llegaron al LLM (post-reranking)
        print(f"CONTEXTO FINAL UTILIZADO ({len(contextos_docs)} fragmentos):")
        print("-" * 50)

        if not contextos_docs:
            print(" (No se recuperó o seleccionó ningún contexto)")
        else:
            # Iteramos sobre la lista de objetos Document
            for i, doc in enumerate(contextos_docs):
                print(f"\n--- Contexto {i+1} ---")
                # Intentamos obtener la fuente del metadata
                source = getattr(doc, 'metadata', {}).get('source', 'N/A')
                source_name = os.path.basename(source) if source != 'N/A' else 'N/A'
                print(f"Fuente: {source_name}")
                # Mostramos el contenido del documento
                print(f"Contenido: {getattr(doc, 'page_content', '')}")

    except Exception as e:
        # <<<<< ESTAS SON LAS LÍNEAS CLAVE PARA EL TRACEBACK
        print(f"\n[ERROR CRÍTICO] Ocurrió un error inesperado durante la invocación del pipeline: {e}")
        print("--- TRACEBACK COMPLETO ---")
        traceback.print_exc() # Imprime el traceback completo
        print("--------------------------")
        # >>>>> FIN LÍNEAS CLAVE

if __name__ == "__main__":
    main()