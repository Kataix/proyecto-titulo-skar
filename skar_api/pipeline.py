# skar_api/pipeline.py (Versión v10 - Prompts Restaurados)

import os
import re
from operator import itemgetter
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from langchain_core.messages import HumanMessage, AIMessage
try:
    from langchain_experimental.llms.ollama_functions import OllamaFunctions
    OLLAMA_FUNCTIONS_AVAILABLE = True
except ImportError:
    OLLAMA_FUNCTIONS_AVAILABLE = False
    print("[ADVERTENCIA] langchain_experimental no instalado o OllamaFunctions no disponible.")
# Quitamos SelfQuery por ahora, usaremos filtro manual
# from langchain.retrievers.self_query.base import SelfQueryRetriever
# from langchain.chains.query_constructor.base import AttributeInfo


# --- CONSTANTES Y CONFIGURACIÓN ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "vectorstores", "chroma")
EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v2-base-es"
LLM_MODEL_NAME = "llama3"
# SELF_QUERY_LLM_MODEL_NAME = "llama3" # No se usa
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Patrón para nombres de servidor (ajústalo si es necesario)
SERVER_NAME_PATTERN = r"([A-Z]{2,}-[A-Z]{2,}-[A-Z0-9]{2,})"

# --- INICIALIZACIÓN DE COMPONENTES ---

print("Inicializando componentes del pipeline RAG...")

# 1. Modelo de Embeddings
print(f"Cargando embeddings: {EMBEDDING_MODEL_NAME}...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'trust_remote_code': True})
print("Embeddings cargados.")

# 2. Carga de la BD Vectorial
print(f"Cargando BD vectorial desde: {DB_PATH}...")
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
print("BD vectorial cargada.")

# --- RETRIEVER VECTORIAL SIMPLE ---
print("Configurando retriever vectorial base (k=20)...")
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# 4. Prompt Final (Reforzado) - CONTENIDO RESTAURADO
template = """
Responde la pregunta basándote únicamente en el siguiente contexto.
Si el contexto contiene información sobre múltiples incidentes o temas diferentes,
enfócate SÓLO en la información MÁS RELEVANTE para la pregunta específica del usuario.
Si no sabes la respuesta o el contexto no es relevante para la pregunta, simplemente di
que no tienes suficiente información. No intentes inventar una respuesta.
Cita siempre las fuentes de los documentos al final (ej. [Fuente: archivo.md]).

Contexto:
{context}

Pregunta:
{question}

Respuesta Concisa y Enfocada:
"""
prompt = ChatPromptTemplate.from_template(template)
print("Plantilla prompt final definida.")

# 5. Prompt para Reescribir Pregunta (Mejorado) - CONTENIDO RESTAURADO
CONDENSE_QUESTION_PROMPT_TEMPLATE = """Dada la siguiente conversación y una pregunta de seguimiento,
reescribe la pregunta de seguimiento para que sea una pregunta independiente y específica,
en su idioma original.
**Asegúrate de incluir entidades clave (como nombres de aplicaciones, números de ticket, códigos de error, etc.) mencionadas en el último intercambio del historial del chat para dar contexto a la pregunta.**

Historial del Chat (último intercambio):
{chat_history}

Pregunta de Seguimiento: {question}

Pregunta Independiente y Específica:"""
condense_question_prompt = ChatPromptTemplate.from_messages([
    ("system", CONDENSE_QUESTION_PROMPT_TEMPLATE),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])
print("Plantilla reescritura definida.")

# 6. LLM Principal
print(f"Inicializando LLM principal: {LLM_MODEL_NAME}..."); llm = ChatOllama(model=LLM_MODEL_NAME); print("LLM principal inicializado.")

# 7. Re-Ranker
print(f"Cargando Re-Ranker: {CROSS_ENCODER_MODEL}..."); cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL); print("Re-Ranker cargado.")

# --- FUNCIONES AUXILIARES ---

def format_docs(docs):
    # ... (código igual) ...
    formatted_docs = []
    for i, doc in enumerate(docs):
        if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
            source = doc.metadata.get('source', f'Doc {i+1}'); source_name = os.path.basename(source) if source else f'Doc {i+1}'
        else: source_name = f'Doc {i+1} (sin meta)'
        page_content = getattr(doc, 'page_content', ''); formatted_docs.append(f"[Fuente: {source_name}]\n{page_content}")
    return "\n\n---\n\n".join(formatted_docs)

def format_chat_history(chat_history_list_of_dicts):
    # ... (código igual, devuelve messages[-2:]) ...
    messages = []
    for message_dict in chat_history_list_of_dicts:
        if isinstance(message_dict, dict):
            role = message_dict.get("role"); content = message_dict.get("content", "")
            if role == "user": messages.append(HumanMessage(content=content))
            elif role == "assistant": messages.append(AIMessage(content=content))
        else: print(f"Warn: Item inesperado en history: {type(message_dict)}")
    return messages[-2:]

def rerank_and_format_with_memory(inputs):
    # ... (código igual, con corrección KeyError y usando retriever_q) ...
    question_for_rerank = inputs['retriever_q'] # Usa la clave correcta
    docs = inputs.get('documents', []); original_question = inputs.get('original_q', '')
    if not docs: print("DEBUG: ReRanker sin docs"); return {"context_str": "No info.", "contexts_for_eval": [], "question": original_question}
    if not isinstance(docs, list): print(f"DEBUG ERROR: docs no es lista: {type(docs)}"); return {"context_str": "Error.", "contexts_for_eval": [], "question": original_question}
    pairs = [[question_for_rerank, getattr(doc, 'page_content', '')] for doc in docs]; print(f"Re-ranking {len(pairs)} docs for: '{question_for_rerank[:50]}...'")
    try: scores = cross_encoder.predict(pairs)
    except Exception as e_predict: print(f"ERROR predict: {e_predict}. Saltando."); reranked_docs = docs[:3]; scores = [0.0] * len(reranked_docs); doc_scores = list(zip(reranked_docs, scores)); print("Fallback: Usando top 3.")
    else: doc_scores = list(zip(docs, scores)); doc_scores.sort(key=lambda x: x[1], reverse=True); reranked_docs = [doc for doc, score in doc_scores[:3]]; print("Top 3 docs por ReRanker.")
    print("--- DEBUG: Contenido Contexto Post-ReRanker ---")
    if not reranked_docs: print(" (Ninguno)")
    else:
        for i, (doc, _) in enumerate(doc_scores[:3]):
             try: numeric_score = float(doc_scores[i][1])
             except (TypeError, ValueError, IndexError): numeric_score = 0.0
             print(f"Doc {i+1} Score: {numeric_score:.4f} Meta: {getattr(doc, 'metadata', {})} Content: {getattr(doc, 'page_content', '')[:100]}...")
    print("---------------------------------------------")
    return {"context_str": format_docs(reranked_docs), "contexts_for_eval": reranked_docs, "question": original_question}


# --- CONSTRUCCIÓN DEL PIPELINE RAG ---

print("Construyendo pipeline con Búsqueda Híbrida Manual, Memoria y Re-Ranker...")

# 1. Cadena para reescribir la pregunta
question_rewriter_chain = condense_question_prompt | llm | StrOutputParser()

# 2. Función para determinar la pregunta del retriever (prioriza original específica)
def determine_retriever_question(input_dict):
    # ... (código igual, con validación de robustez) ...
    chat_history_list = input_dict.get("chat_history", []); question = input_dict.get("question", "")
    print(f"DEBUG: Recibido en determine -> Pregunta: '{question}', Historial: {chat_history_list}")
    if re.search(r"(ticket|incidente)\s*:?\s*(\d{5,}|[a-zA-Z]{2,}-\d+)", question, re.IGNORECASE): print("DEBUG: Pregunta específica, usando original."); return question
    if chat_history_list:
        print("DEBUG: Pregunta no específica Y hay historial, intentando reescribir.")
        rewriter_input = {"question": question, "chat_history": format_chat_history(chat_history_list)}
        if not rewriter_input["chat_history"]: print("DEBUG: Historial formateado vacío, usando original."); return question
        try:
            rewritten_q = question_rewriter_chain.invoke(rewriter_input); print(f"DEBUG: Pregunta Reescrita: {rewritten_q}")
            fallo_reescritura_keywords = ["no puedo reescribir", "no sé", "información insuficiente", "no hay suficiente información", "no está relacionado", "here is the rewritten question"]
            rewritten_q_lower = rewritten_q.lower()
            if len(rewritten_q) < 15 or any(keyword in rewritten_q_lower for keyword in fallo_reescritura_keywords): print(f"DEBUG: Pregunta reescrita inválida/negativa, usando original."); return question
            else:
                 # Limpia prefijo si existe (AJUSTADO PARA EL FORMATO REAL)
                 prefijo = "pregunta independiente y específica:"
                 if rewritten_q.lower().startswith(prefijo):
                     cleaned_q = rewritten_q[len(prefijo):].strip()
                     print(f"DEBUG: Limpiando prefijo -> '{cleaned_q}'"); return cleaned_q
                 return rewritten_q # Devuelve la reescrita aunque no empiece con el prefijo exacto
        except Exception as e: print(f"ERROR reescritura: {e}. Usando original."); return question
    else: print("DEBUG: No hay historial Y pregunta no específica, usando original."); return question

# 3. Función para Búsqueda Híbrida Manual
def hybrid_retrieval(inputs):
    # ... (código igual, con corrección NameError) ...
    retriever_q = inputs['retriever_q']; original_q = inputs['original_q']
    print(f"DEBUG Híbrido: Iniciando para: '{retriever_q}'")
    match_ticket = re.search(r"(ticket|incidente)\s*:?\s*(\d{5,})", retriever_q, re.IGNORECASE)
    match_server = None
    if not match_ticket: match_server = re.search(SERVER_NAME_PATTERN, retriever_q, re.IGNORECASE)

    retrieved_docs = []; filter_used = False; metadata_filter = None; filter_type = ""

    if match_ticket:
        ticket_id_to_find = match_ticket.group(2).strip()
        print(f"DEBUG Híbrido: Ticket ID detectado: '{ticket_id_to_find}'.")
        metadata_filter = {'ticket_id': ticket_id_to_find}; filter_type = f"ticket '{ticket_id_to_find}'"
    elif match_server and len(match_server.group(1)) > 5:
        server_name_to_find = match_server.group(1).strip().upper()
        print(f"DEBUG Híbrido: Server detectado: '{server_name_to_find}'.")
        metadata_filter = {'nombre_servidor': server_name_to_find}; filter_type = f"servidor '{server_name_to_find}'"

    if metadata_filter:
        print(f"DEBUG Híbrido: Intentando filtrar por {filter_type}...")
        try:
            filtered_retriever = vectorstore.as_retriever(search_kwargs={'k': 5, 'filter': metadata_filter})
            retrieved_docs = filtered_retriever.invoke(retriever_q)
            if retrieved_docs: print(f"DEBUG Híbrido: Docs encontrados vía filtro {filter_type}."); filter_used = True
            else: print(f"DEBUG Híbrido: Filtro {filter_type} sin resultados.")
        except Exception as e_filter: print(f"ERROR filtrado: {e_filter}. Usando vectorial."); retrieved_docs = []
    if not filter_used:
        print("DEBUG Híbrido: Búsqueda vectorial normal (k=20)...")
        retrieved_docs = base_retriever.invoke(retriever_q) # Usa base_retriever (k=20)
    print(f"DEBUG Híbrido: Docs recuperados ANTES de ReRanker ({len(retrieved_docs)} docs) ---")
    if retrieved_docs: print("\n".join([f"  - Meta: {getattr(doc, 'metadata', {})} Content: {getattr(doc, 'page_content', '')[:100]}..." for doc in retrieved_docs]))
    else: print("  (Ninguno)")
    print("-------------------------------------------------------")
    return {"documents": retrieved_docs, "retriever_q": retriever_q, "original_q": original_q}

# 4. Cadena Core RAG (ReRanker -> LLM) - Simplificada
core_rag_chain_simplified = (
    RunnableParallel({
        "documents": itemgetter("documents"),
        "retriever_q": itemgetter("retriever_q"), # Pasa la clave correcta
        "original_q": itemgetter("original_q")
    })
    | RunnableLambda(rerank_and_format_with_memory).with_config(run_name="Rerank")
    | RunnableParallel({
          "answer":
              RunnableParallel({ # Prepara entrada para LLM final
                  "context": itemgetter("context_str"),
                  "question": itemgetter("question") # Usa la pregunta ORIGINAL
              })
              | prompt # Aplica la plantilla de prompt
              # --- DEBUG PRINT: VER EL PROMPT FINAL ---
              | RunnableLambda(
                  lambda x: print(f"\n--- DEBUG: Prompt FINAL ENVIADO AL LLM ---\n{x}\n----------------------------------\n") or x
                ).with_config(run_name="DebugFinalPrompt")
              # --- FIN DEBUG ---
              | llm # Llama al LLM
              | StrOutputParser(), # Parsea la salida del LLM
          "contexts": itemgetter("contexts_for_eval") # Devuelve objetos Document
      })
).with_config(run_name="CoreRagChainSimplified")

# 5. Pipeline Final Completo (con Búsqueda Híbrida)
rag_chain_with_memory = (
    RunnableParallel({
        'retriever_q': RunnableLambda(determine_retriever_question),
        'original_q': itemgetter('question'),
        'chat_history': itemgetter('chat_history')
    }).with_config(run_name="DetermineRetrieverQuestion")
    | RunnableLambda(hybrid_retrieval).with_config(run_name="HybridRetrieval")
    | core_rag_chain_simplified
).with_config(run_name="FinalRagChainHybrid")

# Sobrescribimos la variable rag_chain que usan los otros scripts
rag_chain = rag_chain_with_memory

print("Pipeline RAG con Búsqueda Híbrida, Memoria y Re-Ranker construido exitosamente.")

# --- Opcional: Para probar el pipeline directamente ---
# if __name__ == '__main__':
#      (Código de prueba omitido)