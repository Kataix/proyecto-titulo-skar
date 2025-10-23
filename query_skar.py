# query_skar.py
import argparse
from dotenv import load_dotenv

# Importa la cadena RAG completa 
try:
    from skar_api.pipeline import rag_chain 
except ImportError:
    print("\n[ERROR] No se pudo importar 'rag_chain' desde 'skar_api.pipeline'.")
    print("Asegúrate de que el archivo exista y no tenga errores de sintaxis.")
    exit()
except Exception as e:
    print(f"\n[ERROR] Ocurrió un error al importar el pipeline: {e}")
    exit()

def main():
    """
    Función principal para ejecutar consultas individuales a SKAR.
    """
    load_dotenv() # Carga el .env (aunque no se use mucho con Ollama local)

    # Configurar argparse para recibir la pregunta
    parser = argparse.ArgumentParser(description="Consulta al Agente SKAR.")
    parser.add_argument("-p", "--pregunta", 
                        type=str, 
                        required=True, 
                        help="La pregunta que quieres hacerle al agente SKAR.")

    args = parser.parse_args()

    print("Pipeline RAG importado. Procesando pregunta...")

    try:
        # Invocar el pipeline RAG (que ya incluye el Re-Ranker)
        resultado = rag_chain.invoke(args.pregunta)

        # Extraer la respuesta y los contextos del diccionario resultante
        respuesta = resultado.get('answer', '[ERROR: No se encontró la clave "answer"]')
        contextos = resultado.get('contexts', []) # Lista de strings

        print("\n" + "="*50)
        print(f"PREGUNTA: {args.pregunta}")
        print("="*50)

        print("\nRESPUESTA GENERADA:")
        print(respuesta)

        print("\n" + "-"*50)
        print(f"CONTEXTO UTILIZADO ({len(contextos)} fragmentos post-Reranking):")
        print("-" * 50)

        if not contextos:
            print(" (No se recuperó o seleccionó ningún contexto)")
        else:
            for i, ctx_str in enumerate(contextos):
                print(f"\n--- Contexto {i+1} ---")
                # No tenemos metadata fácil aquí, solo el contenido post-reranking
                print(ctx_str)

    except Exception as e:
        print(f"\n[ERROR] Ocurrió un error inesperado durante la invocación del pipeline: {e}")
        # Considerar añadir más detalles del error si es necesario para depurar
        # import traceback
        # print(traceback.format_exc())

if __name__ == "__main__":
    main()