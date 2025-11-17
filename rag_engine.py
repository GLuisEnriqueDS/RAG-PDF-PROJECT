import os
from dotenv import load_dotenv
import google.generativeai as genai

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from PyPDF2 import PdfReader 

# ==========================================
# CONFIGURACIÓN INICIAL
# ==========================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Falta GEMINI_API_KEY en el archivo .env")
genai.configure(api_key=GEMINI_API_KEY)

PDF_DIR = "data/pdfs"
CHROMA_DIR = "embeddings/chroma_db"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# Define la ruta para el índice de persistencia de FAISS
FAISS_INDEX_PATH = os.path.join(CHROMA_DIR, "faiss_index")

# ==========================================
# CONFIGURACIÓN DE EMBEDDINGS (Local)
# ==========================================
# Esto reemplaza a LocalEmbeddingFunction
print("Inicializando modelo de embedding local...")
EMBEDDINGS = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
print("Modelo de embedding local cargado.")

# ==========================================
# BASE VECTORIAL FAISS (Todo-en-uno)
# ==========================================

def get_vectorstore():
    # 1. Intentar cargar el índice existente (Persistencia)
    if os.path.exists(FAISS_INDEX_PATH):
        print("Cargando índice FAISS existente...")
        # Nota: 'allow_dangerous_deserialization=True' es necesario para LangChain 0.2+
        return FAISS.load_local(FAISS_INDEX_PATH, EMBEDDINGS, allow_dangerous_deserialization=True)
        
    # 2. Si no existe, crear el índice desde cero
    print("Creando FAISS Vector Store por primera vez...")
    
    docs = []
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    
    # Cargar y dividir en una sola pasada usando LangChain Loaders
    for f in pdf_files:
        path = os.path.join(PDF_DIR, f)
        try:
            # PyPDFLoader maneja la carga de forma más robusta
            loader = PyPDFLoader(path)
            docs.extend(loader.load()) 
        except Exception as e:
            print(f"❌ Error al cargar el PDF {f}: {e}")
            continue

    if not docs:
        print("No se encontraron documentos válidos. Devuelve None.")
        return None

    # 3. Dividir el texto (LangChain Text Splitter)
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        separator="\n",
        length_function=len,
        is_separator_regex=False
    )
    split_docs = text_splitter.split_documents(docs)
    
    print(f"Total de chunks a procesar: {len(split_docs)}")

    # 4. Crear el índice FAISS (aquí se generan los embeddings)
    print("Generando embeddings y creando índice FAISS...")
    # FAISS.from_documents genera los embeddings y crea el índice
    vectorstore = FAISS.from_documents(split_docs, EMBEDDINGS)
    
    # 5. Guardar el índice
    vectorstore.save_local(CHROMA_DIR, index_name="faiss_index")
    print("Índice FAISS creado y guardado correctamente.")
    
    return vectorstore

# ==========================================
# QUERY RAG
# ==========================================
def query_rag(question, top_k=3):
    vectorstore = get_vectorstore()
    if not vectorstore:
        return "No hay documentos para consultar."

    # 1. Obtener los documentos de contexto relevantes
    # similarity_search devuelve Documentos de LangChain
    results = vectorstore.similarity_search(question, k=top_k)
    
    # 2. Compilar el contexto (tomando el contenido de la página)
    context_text = "\n\n".join([doc.page_content for doc in results])
    
    # Verificar si la consulta de FAISS tiene al menos 1 resultado
    if not context_text.strip():
        return "No se encontró información relevante en los documentos para responder a la pregunta."


    prompt = f"""
Eres un asistente que responde preguntas usando únicamente la información de los documentos a continuación. Si la información no está presente, simplemente indica que no puedes responder basándote en el contexto.

--- CONTEXTO EXTRAÍDO DE DOCUMENTOS ---
{context_text}
--- FIN DEL CONTEXTO ---

Pregunta: {question}

Respuesta clara y concisa:
"""
    try:
        modelo = genai.GenerativeModel("gemini-2.0-flash")
        respuesta = modelo.generate_content(prompt)
        return respuesta.text
    except Exception as e:
        if "429" in str(e):
            return " El servicio de Gemini está temporalmente saturado. Intenta de nuevo en unos minutos."
        else:

            raise e
