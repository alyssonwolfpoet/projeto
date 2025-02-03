from fastapi import FastAPI, HTTPException, UploadFile, Form, File
from typing import List
import ollama
import uvicorn
import os
import shutil
import logging
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import PGVector
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader, CSVLoader, UnstructuredODTLoader
import psycopg2

app = FastAPI()

# Configuração do Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = "nomic-embed-text"

# Configuração do PostgreSQL com LangChain PGVector
DB_CONNECTION_STRING = "postgresql://admin:admin@localhost:5432/llm_db"  # Atualizado

# Criar conexão com o banco de dados
try:
    conn = psycopg2.connect(DB_CONNECTION_STRING)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")  # Garantir que a extensão está ativada
    conn.commit()
    cur.close()
    conn.close()
except Exception as e:
    logging.error(f"Erro ao conectar ao banco de dados: {str(e)}")

vectorstore = PGVector(
    connection_string=DB_CONNECTION_STRING,
    embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL)
)

# Diretório para armazenar arquivos temporários
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Logger para erros e informações
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Função para salvar arquivos recebidos
def save_uploaded_files(files: List[UploadFile]) -> List[str]:
    file_paths = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_paths.append(file_path)
    return file_paths

# Função para processar arquivos e extrair embeddings
def process_and_store_embeddings(file_paths: List[str]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    
    documents = []
    
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".txt":
            loader = TextLoader(file_path)
        elif ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = UnstructuredWordDocumentLoader(file_path)
        elif ext == ".csv":
            loader = CSVLoader(file_path)
        elif ext == ".odt":
            loader = UnstructuredODTLoader(file_path)
        else:
            logger.warning(f"Formato de arquivo não suportado: {ext}")
            continue  

        docs = loader.load()
        chunks = text_splitter.split_documents(docs)
        documents.extend(chunks)

    if documents:
        vectorstore.add_documents(documents)

@app.post("/upload_documents/")
async def upload_documents(files: List[UploadFile] = File(...)):
    try:
        # Salvar arquivos recebidos
        file_paths = save_uploaded_files(files)
        
        # Processar arquivos e armazenar embeddings
        if file_paths:
            process_and_store_embeddings(file_paths)
        
        return {"message": "Arquivos carregados e processados com sucesso."}
    
    except Exception as e:
        logger.error(f"Erro ao processar arquivos: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro ao processar arquivos")

@app.post("/generate/")
async def generate_response(
    prompt: str = Form(...),
    temperature: float = Form(0.7),
    max_tokens: int = Form(200),
    top_p: float = Form(1.0)
):
    try:
        # Montar o prompt para enviar ao Ollama
        full_prompt = f"{prompt}"

        # Enviar para o Ollama
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": full_prompt}],
            options={"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p}
        )

        # Criar embedding do prompt e armazenar
        doc = Document(page_content=prompt)
        vectorstore.add_documents([doc])

        # Log para informação de processamento
        logger.info(f"Processed prompt: {prompt}")

        # Retornar a resposta do modelo
        return {"response": response.get("message", {}).get("content", "Erro ao gerar resposta.")}

    except Exception as e:
        logger.error(f"Erro inesperado: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno no servidor.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
