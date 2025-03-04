# app/services/embedding_service.py

import requests
from app.config import OLLAMA_URL, LLAMA_MODEL

class EmbeddingService:
    def generate_embedding(self, text: str):
        """Gera o embedding para um texto fornecido usando o modelo Ollama"""
        response = requests.post(
            f"{OLLAMA_URL}/v1/embedding", 
            json={"model": LLAMA_MODEL, "input": text}
        )
        
        if response.status_code != 200:
            raise Exception("Erro ao obter embedding do Ollama")
        
        return response.json().get("embedding")
