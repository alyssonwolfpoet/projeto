from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Union

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_ollama import ChatOllama

import base64
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalRequest(BaseModel):
    """Modelo de dados para requisição multimodal"""
    query: str
    image: Optional[str] = None  # Base64 encoded image

class MultimodalResponse(BaseModel):
    """Modelo de dados para resposta multimodal"""
    response: str
    model: str
    input_types: List[str]

class MultimodalService:
    """Serviço para processamento multimodal"""
    
    def __init__(self, model_name: str = "llava"):
        """
        Inicializa o serviço multimodal
        
        Args:
            model_name (str): Nome do modelo Ollama a ser usado
        """
        try:
            self.model = ChatOllama(model=model_name)
        except Exception as e:
            logger.error(f"Erro ao inicializar modelo {model_name}: {e}")
            raise

    def _prepare_messages(self, query: str, image: Optional[str] = None) -> List[BaseMessage]:
        """
        Prepara as mensagens para processamento multimodal
        
        Args:
            query (str): Texto de entrada
            image (Optional[str]): Imagem em base64 (opcional)
        
        Returns:
            List[BaseMessage]: Mensagens preparadas para o modelo
        """
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": query}
                ]
            )
        ]

        if image:
            messages[[0]](https://python.langchain.com/docs/integrations/document_transformers/markdownify/).content.append({
                "type": "image_url", 
                "image_url": {
                    "url": image
                }
            })

        return messages

    def generate_response(self, request: MultimodalRequest) -> MultimodalResponse:
        """
        Gera resposta multimodal
        
        Args:
            request (MultimodalRequest): Requisição multimodal
        
        Returns:
            MultimodalResponse: Resposta gerada
        """
        try:
            # Prepara as mensagens
            messages = self._prepare_messages(
                query=request.query, 
                image=request.image
            )

            # Gera resposta
            response = self.model.invoke(messages)

            # Prepara tipos de entrada
            input_types = ["text"] + (["image"] if request.image else [])

            return MultimodalResponse(
                response=response.content,
                model=self.model.model,
                input_types=input_types
            )

        except Exception as e:
            logger.error(f"Erro na geração multimodal: {e}")
            raise HTTPException(status_code=500, detail=str(e))

class MultimodalController:
    """Controlador para endpoints multimodais"""

    def __init__(self, service: MultimodalService):
        """
        Inicializa o controlador
        
        Args:
            service (MultimodalService): Serviço multimodal
        """
        self.service = service

    async def process_multimodal(
        self, 
        query: str, 
        image: Optional[UploadFile] = File(None)
    ) -> MultimodalResponse:
        """
        Processa requisição multimodal
        
        Args:
            query (str): Texto de entrada
            image (Optional[UploadFile]): Arquivo de imagem (opcional)
        
        Returns:
            MultimodalResponse: Resposta multimodal
        """
        try:
            # Converte imagem para base64, se fornecida
            base64
            base64_image = None
            if image:
                image_content = await image.read()
                base64_image = f"data:image/jpeg;base64,{base64.b64encode(image_content).decode('utf-8')}"

            # Cria requisição multimodal
            request = MultimodalRequest(
                query=query,
                image=base64_image
            )

            # Processa a requisição
            response = self.service.generate_response(request)
            return response

        except Exception as e:
            logger.error(f"Erro no processamento multimodal: {e}")
            raise HTTPException(status_code=500, detail=str(e))

def create_multimodal_app() -> FastAPI:
    """
    Cria e configura a aplicação FastAPI para multimodalidade
    
    Returns:
        FastAPI: Aplicação configurada
    """
    app = FastAPI(
        title="Multimodal AI Service",
        description="Serviço de geração de respostas multimodais"
    )

    # Inicializa serviços e controladores
    multimodal_service = MultimodalService(model_name="llava")
    multimodal_controller = MultimodalController(multimodal_service)

    @app.post("/multimodal-generate/", response_model=MultimodalResponse)
    async def multimodal_generate(
        query: str, 
        image: Optional[UploadFile] = File(None)
    ):
        """
        Endpoint para geração multimodal
        
        Args:
            query (str): Texto de consulta
            image (Optional[UploadFile]): Arquivo de imagem opcional
        
        Returns:
            MultimodalResponse: Resposta gerada
        """
        return await multimodal_controller.process_multimodal(query, image)

    return app

# Criação da aplicação
app = create_multimodal_app()

# Configurações de execução
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )
