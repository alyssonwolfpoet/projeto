import os
import uuid
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

from typing import List

from config import Settings, get_settings
from services.rag_service import RAGService
from models.request_models import QueryRequest, DocumentMetadata

document_router = APIRouter(prefix="/documents", tags=["Documents"])

@document_router.post("/upload")
async def upload_document(
    file: UploadFile = File(...), 
    settings: Settings = Depends(get_settings)
):
    """Rota para upload e indexação de documentos"""
    try:
        # Validar tipo de arquivo
        allowed_extensions = [
            '.txt', '.pdf', '.csv', '.docx', '.doc', '.odt', 
            '.png', '.jpg', '.jpeg', '.bmp', '.gif'
        ]
        file_ext = os.path.splitext(file.filename)[[1]]("https://python.langchain.com/docs/langserve/").lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail="Tipo de arquivo não suportado"
            )
        
        # Criar serviço RAG
        rag_service = RAGService(settings)
        
        # Salvar arquivo temporariamente
        temp_dir = settings.UPLOAD_FOLDER
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}{file_ext}")
        
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Processar documento
        document = rag_service.document_processor.load_document(temp_path)
        
        if document:
            # Indexar documento
            indexed_docs = rag_service.index_document(document)
            
            # Remover arquivo temporário
            os.unlink(temp_path)
            
            return {
                "message": f"Documento {file.filename} indexado com sucesso",
                "document_id": document.metadata.get('id'),
                "chunks": len(indexed_docs)
            }
        else:
            # Remover arquivo temporário em caso de falha
            os.unlink(temp_path)
            raise HTTPException(
                status_code=400, 
                detail="Não foi possível processar o documento"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@document_router.post("/index-url")
async def index_url(
    url: str = Query(..., min_length=5, max_length=500),
    settings: Settings = Depends(get_settings)
):
    """Rota para indexação de URL"""
    try:
        # Criar serviço RAG
        rag_service = RAGService(settings)
        
        # Carregar e indexar URL
        document = rag_service.document_processor.load_url(url)
        
        if document:
            indexed_docs = rag_service.index_document(document)
            
            return {
                "message": f"URL {url} indexada com sucesso",
                "document_id": document.metadata.get('id'),
                "chunks": len(indexed_docs)
            }
        else:
            raise HTTPException(
                status_code=400, 
                detail="Não foi possível carregar a URL"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@document_router.post("/query")
async def query_documents(
    request: QueryRequest, 
    settings: Settings = Depends(get_settings)
):
    """Rota para consulta em documentos indexados"""
    try:
        # Criar serviço RAG
        rag_service = RAGService(settings)
        
        # Realizar consulta
        response = rag_service.query_documents(
            request.query, 
            request.top_k
        )
        
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@document_router.get("/list")
async def list_indexed_files(
    max_files: int = Query(default=100, ge=1, le=500),
    settings: Settings = Depends(get_settings)
):
    """Rota para listar arquivos indexados"""
    try:
        # Criar serviço RAG
        rag_service = RAGService(settings)
        
        # Recuperar lista de arquivos
        files = rag_service.list_indexed_files(max_files)
        
        return {
            "total_files": len(files),
            "files": files
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@document_router.get("/embeddings/{doc_id}")
async def get_document_embeddings(
    doc_id: str, 
    settings: Settings = Depends(get_settings)
):
    """Rota para recuperar embeddings de um documento"""
    try:
        # Criar serviço RAG
        rag_service = RAGService(settings)
        
        # Recuperar embeddings
        embeddings = rag_service.get_document_embeddings(doc_id)
        
        if not embeddings:
            raise HTTPException(status_code=404, detail="Documento não encontrado")
        
        return embeddings
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@document_router.delete("/clear")
async def clear_database(
    settings: Settings = Depends(get_settings)
):
    """Rota para limpar banco de dados"""
    try:
        # Criar serviço RAG
        rag_service = RAGService(settings)
        
        # Limpar banco de dados
        success = rag_service.clear_database()
        
        if success:
            return {"message": "Banco de dados vetorial limpo com sucesso"}
        else:
            raise HTTPException(
                status_code=500, 
                detail="Falha ao limpar o banco de dados"
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
