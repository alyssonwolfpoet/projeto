from fastapi import APIRouter, Depends, HTTPException
from config import Settings, get_settings
from services.rag_service import RAGService
from models.request_models import ChatRequest

chat_router = APIRouter(prefix="/chat", tags=["Chat"])

@chat_router.post("")
async def chat(
    request: ChatRequest, 
    settings: Settings = Depends(get_settings)
):
    """Rota para interação de chat"""
    try:
        # Criar serviço RAG
        rag_service = RAGService(settings)
        
        # Processar mensagem de chat
        response = rag_service.chat(request.message)
        
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
