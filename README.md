En muchos entornos técnicos, la información crítica suele estar distribuida en documentos extensos: manuales, políticas, requisitos regulatorios o reportes difíciles de navegar. Para resolver este problema se desarrolló una aplicación RAG (Retrieval-Augmented Generation) que permite interactuar con estos PDFs a través de un chat inteligente capaz de buscar, interpretar y responder preguntas con precisión contextual.

La solución combina varias tecnologías modernas:

* Google Gemini como modelo de lenguaje (LLM)
* ChromaDB + FAISS como vector store para búsquedas semánticas eficientes
* LangChain como framework para orquestar el flujo RAG
* Streamlit como interfaz web
* SentenceTransformers para generar embeddings
* Carga local de PDFs, organizada de forma simple y directa

Este repositorio explica el rol de cada componente, los motivos detrás de las decisiones técnicas y el código completo necesario para desplegar la aplicación en pocos minutos.
