# eval/evaluacion_tiempos.py
import pandas as pd
import time
import json
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import traceback

# --- Configuración de Ruta ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# --- Importar COMPONENTES del Pipeline ---
# ¡Importante! Importamos los bloques de construcción, no la cadena 'rag_chain'
try:
    from skar_api.pipeline import (
        determine_retriever_question,
        hybrid_retrieval,
        rerank_and_format_with_memory,
        final_llm_chain,
        format_docs  # Para la versión SIN reranker
    )
    print("Componentes del pipeline importados exitosamente.")
except ImportError as e_import:
    print(f"[ERROR CRÍTICO] No se pudieron importar componentes desde 'skar_api.pipeline'. Asegúrate de que las funciones estén definidas globalmente.")
    print(f"Error: {e_import}")
    print(traceback.format_exc())
    exit()
except Exception as e:
    print(f"[ERROR] Error inesperado al importar: {e}")
    print(traceback.format_exc())
    exit()

# --- Configuración de las Pruebas ---
QUESTIONS_FILE = os.path.join(PROJECT_ROOT, "eval", "test_bench.json") # Ruta al banco de 50 preguntas
NUM_RUNS_PER_QUESTION = 3 # Número de veces que se ejecuta CADA pregunta por configuración
OUTPUT_TIMES_CSV = os.path.join(PROJECT_ROOT, "eval", "performance_timing_results.csv")
OUTPUT_PLOT_FILE_TOTAL = os.path.join(PROJECT_ROOT, "eval", "grafico_tiempos_totales.png")
OUTPUT_PLOT_FILE_COMPONENTS = os.path.join(PROJECT_ROOT, "eval", "grafico_tiempos_componentes.png")

def load_questions(filepath: str) -> List[str]:
    """Carga las 50 preguntas de tu archivo test_bench.json."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        questions = [item['pregunta'] for item in questions_data if 'pregunta' in item]
        if len(questions) < 50:
            print(f"Advertencia: Se cargaron {len(questions)} preguntas, se esperaban 50.")
        return questions
    except FileNotFoundError:
        print(f"Error: Archivo de preguntas no encontrado en {filepath}")
        exit()
    except KeyError:
        print(f"Error: La columna 'pregunta' no se encontró en el archivo {filepath}")
        exit()
    except Exception as e:
        print(f"Error al cargar las preguntas del archivo: {e}")
        exit()

def run_single_test(question: str, use_reranker: bool = True) -> Dict[str, Any]:
    """
    Ejecuta un solo pase del pipeline, midiendo cada etapa.
    """
    times = {
        "rewrite": 0.0,
        "retrieval": 0.0,
        "rerank": 0.0,
        "generation": 0.0,
        "total": 0.0
    }
    
    total_start_time = time.time()
    
    try:
        # --- 1. Reescritura de Pregunta ---
        # (Para prueba de latencia, usamos historial vacío)
        input_data = {"question": question, "chat_history": []}
        start_rewrite = time.time()
        retriever_q = determine_retriever_question(input_data)
        times['rewrite'] = time.time() - start_rewrite

        # --- 2. Recuperación Híbrida ---
        retrieval_inputs = {'retriever_q': retriever_q, 'original_q': question}
        start_retrieval = time.time()
        retrieved_data = hybrid_retrieval(retrieval_inputs) # Devuelve {'documents', 'retriever_q', 'original_q'}
        times['retrieval'] = time.time() - start_retrieval
        
        # --- 3. Re-Ranking (Condicional) ---
        if use_reranker:
            start_rerank = time.time()
            reranked_data = rerank_and_format_with_memory(retrieved_data) # Devuelve {'context_str', 'contexts_for_eval', 'question'}
            times['rerank'] = time.time() - start_rerank
            llm_input_data = {"context": reranked_data['context_str'], "question": reranked_data['question']}
            contexts_for_output = reranked_data['contexts_for_eval']
        else:
            # Si no hay reranker, formatea todos los documentos recuperados
            llm_input_data = {
                "context": format_docs(retrieved_data['documents']),
                "question": retrieved_data['original_q']
            }
            contexts_for_output = retrieved_data['documents'] # Pasa los docs originales
            times['rerank'] = 0.0 # El Rerank no tomó tiempo

        # --- 4. Generación con LLM ---
        start_generation = time.time()
        final_answer = final_llm_chain.invoke(llm_input_data)
        times['generation'] = time.time() - start_generation
        
        times['total'] = time.time() - total_start_time

        return {
            "answer": final_answer,
            "contexts": [getattr(doc, 'page_content', '') for doc in contexts_for_output], # Solo el texto
            "sources": list(set([os.path.basename(getattr(doc, 'metadata', {}).get('source', 'N/A')) for doc in contexts_for_output])),
            "times": times,
            "error": None
        }

    except Exception as e:
        print(f"[ERROR] Fallo en la ejecución del pipeline para la pregunta '{question}': {e}")
        traceback.print_exc()
        times['total'] = time.time() - total_start_time
        return {
            "answer": f"[ERROR: {e}]",
            "contexts": [],
            "sources": [],
            "times": times,
            "error": str(e)
        }

def run_tests_and_collect_times(questions: List[str]):
    """Ejecuta las pruebas para ambas configuraciones."""
    all_results = []
    
    # Iterar sobre ambas configuraciones
    for config_name, use_reranker_flag in [("SIN Re-Ranker", False), ("CON Re-Ranker", True)]:
        print(f"\n--- Iniciando pruebas {config_name} ---")
        
        for i, q in enumerate(questions):
            print(f"  Procesando pregunta {i+1}/{len(questions)} ({config_name}): {q[:70]}...")
            
            # Ejecutar N veces para promediar
            for run in range(NUM_RUNS_PER_QUESTION):
                result = run_single_test(q, use_reranker=use_reranker_flag)
                
                times = result['times']
                all_results.append({
                    "pregunta": q,
                    "configuracion": config_name,
                    "run": run + 1,
                    "tiempo_total": times['total'],
                    "tiempo_reescritura": times['rewrite'],
                    "tiempo_recuperacion": times['retrieval'],
                    "tiempo_rerank": times['rerank'],
                    "tiempo_generacion": times['generation'],
                    "respuesta_generada": result['answer'],
                    "error": result['error']
                })
                # Pequeña pausa para no saturar Ollama (opcional)
                # time.sleep(0.5) 
            
    return pd.DataFrame(all_results)

def analyze_and_plot(df_results: pd.DataFrame):
    """Calcula promedios y genera los gráficos."""
    
    # Calcular promedios por configuración y componente
    # Agrupamos por 'configuracion' y calculamos la media de las columnas de tiempo
    avg_times = df_results.groupby('configuracion')[['tiempo_total', 'tiempo_reescritura', 'tiempo_recuperacion', 'tiempo_rerank', 'tiempo_generacion']].mean()
    std_dev_times = df_results.groupby('configuracion')[['tiempo_total']].std()

    print("\n--- Tiempos Promedio (segundos) ---")
    print(avg_times.to_markdown())
    print("\n--- Desviación Estándar (Tiempo Total) ---")
    print(std_dev_times.to_markdown())

    # --- Generar Gráfico 1: Tiempos Totales ---
    plt.figure(figsize=(10, 6))
    # Usamos el DataFrame original 'df_results' para que seaborn calcule el promedio y la barra de error
    sns.barplot(
        x='configuracion', 
        y='tiempo_total', 
        data=df_results, 
        errorbar='sd', # Muestra la desviación estándar como barra de error
        palette='viridis',
        capsize=.1
    )
    plt.title('Tiempo Total Promedio de Consulta (con Desv. Est.)\n(50 Preguntas, 3 Ejecuciones c/u)', fontsize=16)
    plt.ylabel('Tiempo Promedio (segundos)', fontsize=12)
    plt.xlabel('Configuración del Pipeline', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Añadir valores promedio en las barras
    for i, config in enumerate(avg_times.index):
        avg_val = avg_times.loc[config, 'tiempo_total']
        plt.text(i, avg_val / 2, f'{avg_val:.2f} s', ha='center', va='center', color='white', weight='bold', fontsize=12)

    plt.savefig(OUTPUT_PLOT_FILE_TOTAL)
    print(f"\nGráfico de Tiempos Totales guardado como '{OUTPUT_PLOT_FILE_TOTAL}'")
    
    # --- Generar Gráfico 2: Desglose por Componente ---
    # Preparamos los datos para un gráfico de barras apiladas
    avg_times_stacked = avg_times[['tiempo_reescritura', 'tiempo_recuperacion', 'tiempo_rerank', 'tiempo_generacion']]
    
    avg_times_stacked.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='viridis_r') # 'viridis_r' para colores invertidos
    
    plt.title('Desglose de Tiempos Promedio por Componente del Pipeline', fontsize=16)
    plt.ylabel('Tiempo Promedio (segundos)', fontsize=12)
    plt.xlabel('Configuración del Pipeline', fontsize=12)
    plt.xticks(rotation=0) # Mantiene las etiquetas 'CON' y 'SIN' horizontales
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(title='Componente')
    plt.tight_layout() # Ajusta para que todo quepa
    plt.savefig(OUTPUT_PLOT_FILE_COMPONENTS)
    print(f"Gráfico de Componentes guardado como '{OUTPUT_PLOT_FILE_COMPONENTS}'")
    
    # No mostramos los gráficos (plt.show()) para que el script pueda correr en un entorno sin GUI
    print("\nAnálisis y generación de gráficos completados.")

    return avg_times

if __name__ == "__main__":
    # 1. Cargar las 50 preguntas
    questions = load_questions(QUESTIONS_FILE)
    print(f"Cargadas {len(questions)} preguntas para la evaluación de tiempos.")
    
    # 2. Ejecutar las pruebas y recolectar datos
    df_results = run_tests_and_collect_times(questions)
    
    # 3. Guardar los resultados crudos
    df_results.to_csv(OUTPUT_TIMES_CSV, index=False, encoding='utf-8-sig')
    print(f"\nResultados detallados de tiempos (datos crudos) guardados en '{OUTPUT_TIMES_CSV}'")
    
    # 4. Analizar los datos y generar gráficos/tablas
    avg_times = analyze_and_plot(df_results)

    print("\n--- ¡Análisis de Tiempos Completado! ---")
    print("Recuerda usar la tabla de Tiempos Promedio en tu informe LaTeX.")
    print(f"Los archivos de imagen '{OUTPUT_PLOT_FILE_TOTAL}' y '{OUTPUT_PLOT_FILE_COMPONENTS}' están en tu carpeta 'eval/'.")