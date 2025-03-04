# app/services/memory_manager.py

from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from app.config import OLLAMA_URL
import os

class MemoryManager:
    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.llama_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        self.conversation = ConversationChain(memory=self.memory, llm=self.llama_model)

    def add_message(self, user_message: str):
        """Adiciona a mensagem à memória e gera uma resposta"""
        response = self.conversation.predict(input=user_message)
        return response

    def get_memory(self):
        """Retorna o histórico de memória"""
        return self.memory.load_memory_variables({})
