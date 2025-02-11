from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Importar rotas
from routes.document_routes import document_router
from routes.chat_routes import chat_router

# Configuração do aplicativo
app = FastAPI(
    title="RAG Multimodal Backend",
    description="Backend para processamento de documentos e chat com IA",
    version="0.1.0"
)

#
# Configuração de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas as origens em ambiente de desenvolvimento
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir roteadores
app.include_router(document_router)
app.include_router(chat_router)

# Rota de status
@app.get("/status")
async def check_status():
    """Rota de verificação de status do serviço"""
    return {
        "status": "Serviço RAG está operacional",
        "version": "0.1.0"
    }

# Tratamento de exceções personalizadas
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Tratamento personalizado para erros de validação"""
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Erro de validação",
            "details": exc.errors()
        }
    )

# Configuração de inicialização
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )
