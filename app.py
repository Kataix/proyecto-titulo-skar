# app.py
import streamlit as st
from skar_api.pipeline import rag_chain # Importamos la cadena RAG 

# --- Configuración de la Página ---
st.set_page_config(
    page_title="SKAR - Agente de Conocimiento",
    page_icon="🤖",
    layout="centered"
)

# --- Título y Descripción ---
st.title("🤖 SKAR: Tu Agente de Conocimiento Operativo")
st.caption("Desarrollado por Samuel Sáez para el Trabajo de Título de Ingeniería Civil Informática.")

# --- Inicialización del Historial de Chat ---
# Usamos st.session_state para que la conversación no se pierda
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¿En qué puedo ayudarte hoy?"}
    ]

# --- Mostrar mensajes del historial ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Input del Usuario ---
if prompt := st.chat_input("Escribe tu pregunta a SKAR..."):
    # 1. Añadir mensaje del usuario al historial y mostrarlo
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generar y mostrar la respuesta del asistente
    with st.chat_message("assistant"):
        with st.spinner("Procesando..."):
            # Invocamos la cadena RAG
            response = rag_chain.invoke(prompt)
            st.markdown(response)

    # 3. Añadir respuesta del asistente al historial
    st.session_state.messages.append({"role": "assistant", "content": response})