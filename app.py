import os
import platform
import traceback
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# Configuración general
st.set_page_config(page_title="🧠 RAG con PDF", layout="wide")
st.markdown("""
    <style>
    .main-title { font-size: 40px; color: #2C3E50; text-align: center; margin-bottom: 10px; }
    .subtitle { font-size: 24px; color: #16A085; margin-top: 30px; }
    .footer { font-size: 14px; text-align: center; margin-top: 50px; color: grey; }
    </style>
""", unsafe_allow_html=True)

# Título Principal
st.markdown('<div class="main-title">📚 Sistema de Preguntas con RAG sobre PDFs</div>', unsafe_allow_html=True)
st.caption(f"🛠️ Python {platform.python_version()} - con LangChain, FAISS y GPT-4o")

# Mostrar imagen
with st.expander("📷 Ver imagen del sistema"):
    try:
        st.image(Image.open("Chat_pdf.png"), width=400)
    except Exception as e:
        st.warning(f"❌ Imagen no cargada: {e}")

# Sidebar para info y clave
with st.sidebar:
    st.markdown("## 🧾 Instrucciones")
    st.info("Este asistente analiza el contenido de un PDF y responde preguntas sobre él usando IA.")
    api_key = st.text_input("🔑 Clave API de OpenAI", type="password")

# Carga del PDF
st.markdown('<div class="subtitle">📄 Cargar documento PDF</div>', unsafe_allow_html=True)
pdf = st.file_uploader("Arrastra o selecciona un archivo PDF", type="pdf")

# Validación de claves
if not api_key:
    st.warning("🔐 Ingresa tu clave de API de OpenAI para continuar")
else:
    os.environ['OPENAI_API_KEY'] = api_key

# Procesamiento si hay PDF y clave
if pdf is not None and api_key:
    try:
        # Extracción de texto
        st.info("🧠 Extrayendo texto del PDF...")
        reader = PdfReader(pdf)
        full_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        st.success(f"✅ Texto extraído: {len(full_text)} caracteres")

        # Fragmentación del texto
        splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=20, length_function=len)
        chunks = splitter.split_text(full_text)
        st.success(f"📚 Fragmentos creados: {len(chunks)}")

        # Embeddings
        st.info("🔍 Generando embeddings y base de conocimiento...")
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Entrada de pregunta
        st.markdown('<div class="subtitle">❓ Pregunta sobre el documento</div>', unsafe_allow_html=True)
        user_question = st.text_area("Escribe tu pregunta 👇", placeholder="¿Qué dice el documento sobre...?")

        if user_question:
            # Búsqueda y respuesta
            with st.spinner("🧠 Pensando..."):
                docs = knowledge_base.similarity_search(user_question)
                llm = OpenAI(temperature=0, model_name="gpt-4o")
                chain = load_qa_chain(llm, chain_type="stuff")
                respuesta = chain.run(input_documents=docs, question=user_question)
            
            st.markdown("### 📬 Respuesta del modelo:")
            st.success(respuesta)

    except Exception as e:
        st.error("⚠️ Error procesando el PDF:")
        st.exception(e)
        st.code(traceback.format_exc(), language="python")
elif pdf is not None and not api_key:
    st.warning("🔑 Clave API requerida para procesar el PDF.")
else:
    st.info("📤 Esperando que cargues un PDF...")

