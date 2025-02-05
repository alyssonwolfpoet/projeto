# app/main.py
import os
import uuid
import logging
from typing import List, Optional, Union

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configurações
from config import Settings, get_settings

# Utilitários
from document_processor import DocumentProcessor

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicialização do app
app = FastAPI(title="RAG Multimodal Backend")

# Configuração de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos e Embeddings
class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embeddings = OllamaEmbeddings(model=settings.EMBEDDING_MODEL)
        self.llm = ChatOllama(model=settings.LLM_MODEL)
        
        self.vector_store = PGVector(
            connection_string=settings.DATABASE_URL,
            embedding_function=self.embeddings,
            collection_name=settings.COLLECTION_NAME
        )
        
        self.document_processor = DocumentProcessor(
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=500, 
                chunk_overlap=50
            )
        )
    
    def index_document(self, document: Document) -> List[Document]:
        """Processar e indexar um documento"""
        try:
            processed_docs = self.document_processor.process_document(document)
            self.vector_store.add_documents(processed_docs)
            return processed_docs
        except Exception as e:
            logger.error(f"Erro ao indexar documento: {e}")
            raise

    def query_documents(self, query: str, top_k: int = 3) -> str:
        """Realizar busca em documentos indexados"""
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
            
            template = """Answer the question based only on the following context:
            {context}
            
            Question: {question}
            """
            prompt = ChatPromptTemplate.from_template(template)
            
            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            return rag_chain.invoke(query)
        except Exception as e:
            logger.error(f"Erro na consulta: {e}")
            raise

# Modelos de Requisição
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: Optional[int] = Field(default=3, ge=1, le=10)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)

# Rotas
@app.post("/upload/")
async def upload_document(
    file: UploadFile = File(...), 
    settings: Settings = Depends(get_settings)
):
    try:
        # Inicializar serviço RAG
        rag_service = RAGService(settings)
        
        # Salvar arquivo temporariamente
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Processar documento
        document = rag_service.document_processor.load_document(temp_path)
        
        if document:
            # Indexar documento
            rag_service.index_document(document)
            
            # Remover arquivo temporário
            os.unlink(temp_path)
            
            return {
                "message": f"Documento {file.filename} indexado com sucesso",
                "document_id": document.metadata.get('id', str(uuid.uuid4()))
            }
        else:
            raise HTTPException(status_code=400, detail="Formato de documento não suportado")
    
    except Exception as e:
        logger.error(f"Erro no upload de documento: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index-url/")
async def index_url(
    url: str = Query(..., min_length=5, max_length=500),
    settings: Settings = Depends(get_settings)
):
    try:
        # Inicializar serviço RAG
        rag_service = RAGService(settings)
        
        # Carregar e indexar URL
        document = rag_service.document_processor.load_url(url)
        
        if document:
            rag_service.index_document(document)
            
            return {
                "message": f"URL {url} indexada com sucesso",
                "document_id": document.metadata.get('id', str(uuid.uuid4()))
            }
        else:
            raise HTTPException(status_code=400, detail="Não foi possível carregar a URL")
    
    except Exception as e:
        logger.error(f"Erro na indexação de URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
async def query_documents(
    request: QueryRequest, 
    settings: Settings = Depends(get_settings)
):
    try:
        # Inicializar serviço RAG
        rag_service = RAGService(settings)
        
        # Realizar consulta
        response = rag_service.query_documents(
            request.query, 
            request.top_k
        )
        
        return {"response": response}
    
    except Exception as e:
        logger.error(f"Erro na consulta de documentos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
async def chat(
    request: ChatRequest, 
    settings: Settings = Depends(get_settings)
):
    try:
        # Inicializar serviço RAG
        rag_service = RAGService(settings)
        
        # Template de conversação
        template = """You are a helpful AI assistant. 
        Respond to the following message:
        {message}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        # Cadeia de processamento
        chain = prompt | rag_service.llm | StrOutputParser()
        response = chain.invoke({"message": request.message})
        
        return {"response": response}
    
    except Exception as e:
        logger.error(f"Erro no chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/")
async def check_status():
    return {"status": "Backend está rodando"}

@app.delete("/clear-db/")
async def clear_database(settings: Settings = Depends(get_settings)):
    try:
        # Inicializar serviço RAG
        rag_service = RAGService(settings)
        
        # Limpar
        # Limpar coleção
        rag_service.vector_store.delete_collection()
        rag_service.vector_store.create_collection()
        
        return {"message": "Banco de dados vetorial limpo com sucesso"}
    except Exception as e:
        logger.error(f"Erro ao limpar banco de dados: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-files/")
async def list_indexed_files(settings: Settings = Depends(get_settings)):
    try:
        # Inicializar serviço RAG
        rag_service = RAGService(settings)
        
        # Recuperar documentos
        documents = rag_service.vector_store.similarity_search("", k=100)
        
        # Extrair metadados dos arquivos
        files = [
            {
                "source": doc.metadata.get('source', 'Unknown'),
                "id": doc.metadata.get('id', 'N/A')
            } 
            for doc in documents
        ]
        
        return {"files": list({v['source']:v for v in files}.values())}
    except Exception as e:
        logger.error(f"Erro ao listar arquivos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug-embeddings/{doc_id}")
async def get_document_embeddings(
    doc_id: str, 
    settings: Settings = Depends(get_settings)
):
    try:
        # Inicializar serviço RAG
        rag_service = RAGService(settings)
        
        # Buscar documento
        documents = rag_service.vector_store.similarity_search(f"id:{doc_id}", k=1)
        
        if not documents:
            raise HTTPException(status_code=404, detail="Documento não encontrado")
        
        # Gerar embedding
        embedding = rag_service.embeddings.embed_documents(
            [documents[[0]].page_content])[[0]]
        
        return {
            "doc_id": doc_id, 
            "embedding": embedding,
            "embedding_length": len(embedding)
        }
    except Exception as e:
        logger.error(f"Erro ao recuperar embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
