from fastapi import FastAPI, HTTPException, UploadFile, Form, File
from typing import List, Optional

# Importações do LangChain atualizadas
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Embeddings e Vetores
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import PGVector

# Modelos de linguagem
from langchain_community.chat_models import ChatOllama

# Processamento de documentos
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    UnstructuredWordDocumentLoader, 
    CSVLoader, 
    UnstructuredODTLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Outras bibliotecas
import ollama
import uvicorn
import os
import shutil
import logging
import psycopg2

# Configurações de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações do aplicativo
app = FastAPI(
    title="LLM Document Processing API",
    description="API avançada para processamento de documentos e geração de respostas com LangChain",
    version="0.3.0"
)

# Configurações de ambiente
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = "nomic-embed-text"
DB_CONNECTION_STRING = "postgresql://admin:admin@localhost:5432/llm_db"

# Diretórios e configurações
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Inicialização dos componentes
try:
    # Configuração de embeddings
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Configuração do vetor store
    vectorstore = PGVector(
        connection_string=DB_CONNECTION_STRING,
        embedding_function=embeddings,
        collection_name="document_collection"
    )

    # Modelo de linguagem
    llm = ChatOllama(model="llama3")

except Exception as e:
    logger.error(f"Erro na inicialização: {e}")
    raise

# Utilitários de processamento
def save_uploaded_files(files: List[UploadFile]) -> List[str]:
    file_paths = []
    for file in files:
        try:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_paths.append(file_path)
        except Exception as e:
            logger.error(f"Erro ao salvar arquivo {file.filename}: {e}")
    return file_paths

def process_documents(file_paths: List[str]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    
    all_documents = []
    
    for file_path in file_paths:
        try:
            ext = os.path.splitext(file_path)[[1]](https://api.python.langchain.com/en/latest/ibm_api_reference.html).lower()
            
            # Seleção do loader baseado na extensão
            loader_map = {
                ".txt": TextLoader,
                ".pdf": PyPDFLoader,
                ".docx": UnstructuredWordDocumentLoader,
                ".csv": CSVLoader,
                ".odt": UnstructuredODTLoader
            }
            
            loader_class = loader_map.get(ext)
            if not loader_class
            if not loader_class:
                logger.warning(f"Formato de arquivo não suportado: {ext}")
                continue

            # Carregar e processar documentos
            loader = loader_class(file_path)
            docs = loader.load()
            
            # Dividir documentos em chunks
            chunks = text_splitter.split_documents(docs)
            all_documents.extend(chunks)
        
        except Exception as e:
            logger.error(f"Erro ao processar arquivo {file_path}: {e}")
    
    return all_documents

# Cadeia de processamento RAG usando LCEL (LangChain Expression Language)
def create_rag_chain(retriever):
    # Template do prompt
    template = """Responda a pergunta baseando-se apenas no seguinte contexto:
    {context}
    
    Pergunta: {question}
    """
    
    # Criar prompt
    prompt = ChatPromptTemplate.from_template(template)
    
    # Definir cadeia RAG usando LCEL
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Rotas da API
@app.post("/upload_documents/")
async def upload_documents(files: List[UploadFile] = File(...)):
    try:
        # Salvar arquivos
        file_paths = save_uploaded_files(files)
        
        # Processar documentos
        documents = process_documents(file_paths)
        
        # Adicionar documentos ao vetor store
        if documents:
            vectorstore.add_documents(documents)
        
        return {
            "message": "Documentos carregados e processados com sucesso",
            "total_documents": len(documents)
        }
    
    except Exception as e:
        logger.error(f"Erro no upload de documentos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
async def query_documents(
    query: str = Form(...), 
    k: int = Form(3),
    temperature: float = Form(0.7)
):
    try:
        # Criar retriever
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
        
        # Criar cadeia RAG
        rag_chain = create_rag_chain(retriever)
        
        # Executar consulta
        response = rag_chain.invoke(query)
        
        # Buscar documentos relevantes
        relevant_docs = retriever.get_relevant_documents(query)
        
        return {
            "response": response,
            "relevant_documents": [
                {
                    "content": doc.page_content, 
                    "metadata": doc.metadata
                } for doc in relevant_docs
            ]
        }
    
    except Exception as e:
        logger.error(f"Erro na consulta: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Middleware de CORS
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint de saúde
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Configurações de inicialização
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )
