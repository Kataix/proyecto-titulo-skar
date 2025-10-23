# eval/run_tests_batch.py
import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import sys

# Añade la ruta raíz del proyecto al sys.path para poder importar desde skar_api
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Importa la cadena RAG DESPUÉS de modificar el sys.path
try:
    from skar_api.pipeline import rag_chain
except ImportError:
    print("\n[ERROR] No se pudo importar 'rag_chain' desde 'skar_api.pipeline'.")
    print("Asegúrate de que el archivo exista y no tenga errores.")
    exit()
except Exception as e:
    print(f"\n[ERROR] Ocurrió un error al importar el pipeline: {e}")
    exit()

def run_test_batch(test_bench_file, output_csv_file):
    """
    Ejecuta un lote de pruebas desde un archivo JSON usando el pipeline importado
    y guarda los resultados en un CSV.
    """
    print(f"Cargando pipeline RAG...")
    # El pipeline ya se cargó al importar

    print(f"Cargando banco de pruebas desde {test_bench_file}...")
    try:
        with open(test_bench_file, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] No se encontró el archivo de pruebas: {test_bench_file}")
        return

    print(f"Se ejecutarán {len(test_cases)} pruebas...")

    resultados = []

    for i, test in enumerate(test_cases):
        pregunta = test['pregunta']
        print(f"Ejecutando prueba {i+1}/{len(test_cases)}: {pregunta[:50]}...")

        try:
            # Invocar el pipeline RAG completo
            resultado_dict = rag_chain.invoke(pregunta)

            # Extraer datos del diccionario resultante
            respuesta = resultado_dict.get('answer', '[ERROR: Sin respuesta]')
            contextos_list = resultado_dict.get('contexts', []) # Lista de strings

            # Añadir a la lista de resultados
            resultados.append({
                "categoria": test.get('categoria', 'N/A'),
                "pregunta": pregunta,
                "respuesta_ideal": test.get('respuesta_ideal', 'N/A'), # Añadido
                "respuesta_generada": respuesta,
                "contextos_recuperados": json.dumps(contextos_list, ensure_ascii=False), # Guardar como JSON
                # No podemos obtener las fuentes fácilmente aquí porque el reranker pierde la metadata original
                # Dejaremos las fuentes en blanco por ahora, nos enfocamos en el contenido
                "fuentes_contexto": "N/A (Post-Reranking)" 
            })

        except Exception as e:
            print(f"[ERROR] Falló la prueba para la pregunta: '{pregunta}'. Error: {e}")
            # import traceback # Descomenta para más detalle si necesitas depurar
            # print(traceback.format_exc())
            resultados.append({
                "categoria": test.get('categoria', 'N/A'),
                "pregunta": pregunta,
                "respuesta_ideal": test.get('respuesta_ideal', 'N/A'),
                "respuesta_generada": f"[ERROR: {e}]",
                "contextos_recuperados": "N/A",
                "fuentes_contexto": "N/A"
            })

    # Guardar resultados en un CSV
    try:
        df = pd.DataFrame(resultados)
        df.to_csv(output_csv_file, index=False, encoding='utf-8-sig') # utf-8-sig para compatibilidad Excel
        print(f"\n¡Pruebas completadas!")
        print(f"Resultados guardados en: {output_csv_file}")

    except Exception as e:
        print(f"\n[ERROR] No se pudieron guardar los resultados en CSV: {e}")

def main():
    load_dotenv()

    input_file = os.path.join(PROJECT_ROOT, "eval", "test_bench.json") # Ruta correcta

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Guarda los resultados dentro de la carpeta eval
    output_file = os.path.join(PROJECT_ROOT, "eval", f"resultados_pruebas_iniciales_{timestamp}.csv") 

    run_test_batch(input_file, output_file)

if __name__ == "__main__":
    main()