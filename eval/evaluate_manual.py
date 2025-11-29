# eval/evaluacion_manual.py
import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import sys
import time
import traceback
import argparse # Importación necesaria

# --- Configuración de Ruta ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# --- Importación del Pipeline RAG ---
try:
    # Importa la cadena RAG (asegúrate que sea la v15 con filtro híbrido)
    from skar_api.pipeline import rag_chain
    print("Pipeline RAG importado exitosamente.")
except ImportError as ie:
    print(f"\n[ERROR] Importación fallida: {ie}. Verifica la ruta y 'pipeline.py'.")
    exit()
except Exception as e_import:
    print(f"\n[ERROR] al importar pipeline: {e_import}")
    traceback.print_exc()
    exit()

# --- Función Principal de Generación de Datos ---
def generar_datos_evaluacion(test_bench_file, output_csv_file):
    """
    Ejecuta el banco de pruebas completo y guarda preguntas, respuestas
    (ideal y generada), contextos y fuentes en un CSV para análisis manual.
    """
    print(f"\nCargando banco de pruebas desde: {test_bench_file}...")
    try:
        with open(test_bench_file, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        # Usamos TODOS los casos, incluso los de "Fuera de Alcance"
        valid_test_cases = test_cases
        print(f"Banco de pruebas cargado. {len(valid_test_cases)} casos a procesar.")
    except FileNotFoundError: print(f"[ERROR] No se encontró: {test_bench_file}"); return
    except json.JSONDecodeError: print(f"[ERROR] JSON inválido: '{test_bench_file}'."); return

    # --- Preparación de Datos ---
    resultados_lista = []
    print(f"\nGenerando respuestas y contextos para {len(valid_test_cases)} preguntas...")
    total_start_time = time.time()

    for i, test in enumerate(valid_test_cases):
        pregunta = test['pregunta']
        respuesta_ideal = test.get('respuesta_ideal', 'N/A')
        print(f"  Procesando pregunta {i+1}/{len(valid_test_cases)}: {pregunta[:60]}...")
        start_time = time.time()

        try:
            # --- Invocar Pipeline ---
            input_data = {"question": pregunta, "chat_history": []}
            resultado_dict = rag_chain.invoke(input_data)
            
            respuesta_generada = resultado_dict.get('answer', None)
            contextos_docs = resultado_dict.get('contexts', []) # Lista de objetos Document
            end_time = time.time(); latency = end_time - start_time
            
            if respuesta_generada is None or contextos_docs is None: 
                raise ValueError("Salida inválida de rag_chain (falta 'answer' o 'contexts')")

            # Extraer contextos y fuentes
            contextos_str_list = [getattr(doc, 'page_content', '') for doc in contextos_docs]
            
            sources_set = set()
            for doc in contextos_docs:
                if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                    source = doc.metadata.get('source')
                    if source:
                        sources_set.add(os.path.basename(source)) # Añade solo el nombre del archivo
            fuentes_str = ", ".join(sorted(list(sources_set))) if sources_set else "N/A"

            # Añadir a la lista de resultados
            resultados_lista.append({
                "categoria": test.get('categoria', 'N/A'),
                "pregunta": pregunta,
                "respuesta_ideal": respuesta_ideal,
                "respuesta_generada": respuesta_generada,
                "contextos_recuperados": json.dumps(contextos_str_list, ensure_ascii=False), # Contextos como JSON string
                "fuentes_recuperadas": fuentes_str, # Fuentes como string separado por comas
                "latencia_seg": round(latency, 2)
            })
            print(f"    Respuesta generada en {latency:.2f} seg.")

        except Exception as e_invoke:
            end_time = time.time(); latency = end_time - start_time
            print(f"[ERROR] Invocación fallida para la pregunta: '{pregunta}'. Error: {e_invoke}")
            print(f"    Tiempo hasta error: {latency:.2f} seg.")
            # Añadir placeholders para el CSV
            resultados_lista.append({
                "categoria": test.get('categoria', 'N/A'),
                "pregunta": pregunta,
                "respuesta_ideal": respuesta_ideal,
                "respuesta_generada": f"[ERROR_INVOKE: {e_invoke}]",
                "contextos_recuperados": "[]",
                "fuentes_recuperadas": "N/A",
                "latencia_seg": round(latency, 2)
            })

    total_end_time = time.time()
    print(f"\nGeneración completa en {total_end_time - total_start_time:.2f} seg.")

    # --- Guardar Resultados en CSV ---
    try:
        df = pd.DataFrame(resultados_lista)
        df.to_csv(output_csv_file, index=False, encoding='utf-8-sig')
        print(f"\n¡Resultados de evaluación manual guardados exitosamente en: {output_csv_file}")
        print("\n--- Vista Previa de Resultados ---")
        print(df.head())
        print("---------------------------------")
    except Exception as e_save:
        print(f"\n[ERROR] No se pudieron guardar los resultados en CSV: {e_save}")
        traceback.print_exc()

# --- Función Main del Script ---
def main():
    """Punto de entrada principal del script."""
    load_dotenv()
    parser = argparse.ArgumentParser(description="Genera datos de evaluación (para análisis manual) del pipeline RAG.")
    parser.add_argument("--suffix", type=str, default="", help="Sufijo para CSV (ej. '_con_reranker').")
    args = parser.parse_args()

    input_file = os.path.join(PROJECT_ROOT, "eval", "test_bench.json")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_name = f"evaluacion_manual{args.suffix}_{timestamp}.csv"
    output_file = os.path.join(PROJECT_ROOT, "eval", output_file_name)

    generar_datos_evaluacion(input_file, output_file)

if __name__ == "__main__":
    main()