# app.py (Modificado para mostrar fuentes si están disponibles)
import streamlit as st
import os # Necesario para os.path.basename
# Importa la cadena RAG (que ahora espera 'question' y 'chat_history')
from skar_api.pipeline import rag_chain

# --- Configuración de la Página ---
st.set_page_config(
    page_title="SKAR - Agente de Conocimiento",
    page_icon="🐶",
    layout="centered"
)

# --- Título y Descripción ---
st.title("🐶 SKAR: Tu Agente de Conocimiento Operativo")
st.caption("Desarrollado por Samuel Sáez para el Trabajo de Título de Ingeniería Civil Informática.")

# --- Inicialización del Historial de Chat ---
if "messages" not in st.session_state:
    # El primer mensaje es el saludo del bot
    st.session_state.messages = [
        {"role": "assistant", "content": "¿En qué puedo ayudarte hoy?"}
    ]

# --- Mostrar mensajes del historial ---
# Iteramos sobre una copia para evitar problemas si se modifica mientras se itera (Streamlit puede re-ejecutar)
for message in st.session_state.messages[:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Intentar mostrar fuentes si existen en mensajes del asistente
        sources_in_history = message.get("sources", [])
        if message["role"] == "assistant" and sources_in_history:
             st.caption("Fuentes: " + ", ".join(sources_in_history))

# --- Input del Usuario ---
if prompt := st.chat_input("Escribe tu pregunta a SKAR..."):
    # 1. Añadir mensaje del usuario al historial y mostrarlo
    #    Asegurarse de añadir como diccionario
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generar y mostrar la respuesta del asistente
    with st.chat_message("assistant"):
        with st.spinner("Procesando..."):

            # Preparamos la entrada para la cadena RAG con memoria
            # Pasamos todo el historial EXCEPTO el saludo inicial del bot
            # y EXCEPTO la pregunta actual del usuario
            # Usamos una copia para evitar modificar la lista original mientras se itera
            current_history = st.session_state.messages[1:-1]

            input_data = {
                "question": prompt,
                "chat_history": current_history # Pasa la copia
            }

            # --- DEBUG PRINT en app.py ---
            print(f"--- DEBUG App.py: Enviando a rag_chain -> Pregunta: '{prompt}', Historial: {current_history}")
            # --- FIN DEBUG ---

            # Invocamos la cadena con el diccionario de entrada
            response = "Lo siento, ocurrió un error inesperado." # Valor por defecto
            sources = [] # Valor por defecto
            try:
                response_dict = rag_chain.invoke(input_data)
                response = response_dict.get('answer', "Lo siento, ocurrió un error al generar la respuesta.")
                # --- EXTRAER FUENTES ---
                contexts_docs = response_dict.get('contexts', []) # Ahora son objetos Document
                if contexts_docs and isinstance(contexts_docs, list):
                    temp_sources = set() # Usar set para evitar duplicados
                    for doc in contexts_docs:
                        if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                           source = doc.metadata.get('source')
                           if source:
                               temp_sources.add(os.path.basename(source))
                    sources = sorted(list(temp_sources)) # Convertir a lista ordenada
                # --- FIN EXTRAER FUENTES ---

            except Exception as e:
                print(f"[ERROR CRÍTICO] Excepción al invocar rag_chain en app.py: {e}")
                import traceback
                print(traceback.format_exc()) # Imprime el traceback completo en la consola
                response = f"Lo siento, ocurrió un error interno al procesar tu pregunta. Por favor, revisa la consola del servidor para más detalles."
                sources = []

            st.markdown(response)
            # Mostrar fuentes si se encontraron
            if sources:
                st.caption("Fuentes: " + ", ".join(sources))


    # 3. Añadir la respuesta del asistente AL HISTORIAL (incluyendo fuentes)
    assistant_message = {"role": "assistant", "content": response}
    if sources:
        assistant_message["sources"] = sources # Añade fuentes al diccionario del historial
    st.session_state.messages.append(assistant_message)