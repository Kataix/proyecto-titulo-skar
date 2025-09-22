import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough

DB_PATH = os.path.join(os.getcwd(), "vectorstores", "chroma")
MODEL_NAME = "llama3"

# 1. Definir la plantilla del prompt (con la corrección)
PROMPT_TEMPLATE = """
### ROL DEL SISTEMA ###
Eres SKAR, un agente de IA experto en operaciones de TI. Tu propósito es responder preguntas basándote únicamente en el conocimiento proporcionado. Eres preciso, técnico y siempre verificable.

### INSTRUCCIONES ###
1. Analiza el CONOCIMIENTO RECUPERADO para responder la PREGUNTA DEL USUARIO.
2. Basa tu respuesta EXCLUSIVAMENTE en la información del conocimiento recuperado. No utilices ningún conocimiento externo o pre-entrenado.
3. Si el conocimiento recuperado no contiene la información necesaria para responder, debes indicar claramente: "No he encontrado información relevante en la base de conocimiento para responder a esa pregunta."
4. Al final de tu respuesta, es OBLIGATORIO citar las fuentes utilizadas en una sección titulada "Fuentes:", listando los identificadores de los documentos (ej. nombre de archivo) de donde extrajiste la información.

### CONOCIMIENTO RECUPERADO ###
{context}

### PREGUNTA DEL USUARIO ###
{question}

### RESPUESTA ###
"""

# 2. Cargar la base de datos vectorial y el modelo de embedding
embedding_model = HuggingFaceEmbeddings(
    model_name="jinaai/jina-embeddings-v2-base-es",
    model_kwargs={'trust_remote_code': True}
)

vectorstore = Chroma(
    persist_directory=DB_PATH, 
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever()

# 3. Crear la cadena de RAG con LangChain Expression Language (LCEL)
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
llm = ChatOllama(model=MODEL_NAME)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}

| prompt
| llm
| StrOutputParser()
)