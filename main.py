import streamlit as st
from rag_engine import query_rag
import os
# ===============================
# Interfaz principal en Streamlit
# ===============================

st.set_page_config(
    page_title="RAG con PDFs (Gemini + ChromaDB)",
    page_icon="📚",
    layout="centered"
)

st.title("Chat con tus PDFs - Claro Insurance RAG")
st.markdown("Pregunta sobre tus documentos cargados en la carpeta `data/pdfs/`")

# Listar PDFs disponibles

pdf_dir = "data/pdfs"
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

if pdf_files:
    st.info(f"Se detectaron {len(pdf_files)} documentos PDF:")
    for f in pdf_files:
        st.write(f"- {f}")
else:
    st.warning("No hay PDFs en la carpeta `data/pdfs/`")

# Campo de entrada
question = st.text_area("Escribe tu pregunta:", height=100)

# Botón de búsqueda
if st.button("Consultar"):
    if not question.strip():
        st.warning("Por favor escribe una pregunta antes de continuar.")
    else:
        with st.spinner("Buscando respuesta entre tus documentos..."):
            try:
                answer = query_rag(question)
                st.markdown("### Respuesta:")
                st.write(answer)
            except Exception as e:
                st.error(f"Ocurrió un error: {e}")

# ===============================
# Pie de página
# ===============================
st.markdown("---")
st.caption("Powered by Google Gemini + ChromaDB + LangChain | © 2025 Claro Insurance RAG")
