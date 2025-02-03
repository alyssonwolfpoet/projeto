import os
import logging
from typing import List, Tuple, Dict, Optional

import streamlit as st
import requests
from requests.exceptions import RequestException, Timeout

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename='chat_app.log'
)

# Configurações da Aplicação
class AppConfig:
    ALLOWED_FILE_TYPES = ["txt", "pdf", "docx", "csv", "odt"]
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    MAX_UPLOAD_FILES = 5
    MAX_CHAT_HISTORY = 20

    # URLs de API
    UPLOAD_API_URL = os.getenv("UPLOAD_API_URL", "http://localhost:8000/upload-documents/")
    CHAT_API_URL = os.getenv("CHAT_API_URL", "http://localhost:8000/generate/")

# Classe de Serviços de Backend
class BackendService:
    @staticmethod
    def upload_documents(files: Dict[str, tuple]) -> str:
        """
        Envia documentos para o backend
        
        Args:
            files (Dict): Dicionário de arquivos para upload
        
        Returns:
            str: Mensagem de resultado do upload
        """
        try:
            response = requests.post(AppConfig.UPLOAD_API_URL, files=files)
            response.raise_for_status()
            return response.json().get("message", "Documentos enviados com sucesso.")
        except RequestException as e:
            logging.error(f"Erro no upload de documentos: {e}")
            return f"Erro ao enviar documentos: {e}"

    @staticmethod
    def generate_response(
        prompt: str, 
        temperature: float, 
        max_tokens: int, 
        top_p: float
    ) -> str:
        """
        Gera resposta do modelo de linguagem
        
        Args:
            prompt (str): Texto de entrada
            temperature (float): Parâmetro de criatividade
            max_tokens (int): Máximo de tokens na resposta
            top_p (float): Parâmetro de amostragem nuclear
        
        Returns:
            str: Resposta gerada
        """
        try:
            response = requests.post(
                AppConfig.CHAT_API_URL,
                data={
                    'prompt': prompt, 
                    'temperature': temperature, 
                    'max_tokens': max_tokens, 
                    'top_p': top_p
                }
            )
            response.raise_for_status()
            return response.json().get("response", "Erro na resposta do modelo.")
        except RequestException as e:
            logging.error(f"Erro na geração de resposta: {e}")
            return f"Erro ao obter resposta: {e}"

# Classe de Validação
class Validator:
    @staticmethod
    def validate_files(uploaded_files: List) -> bool:
        """
        Valida os arquivos enviados
        
        Args:
            uploaded_files (List): Lista de arquivos
        
        Returns:
            bool: Indica se os arquivos são válidos
        """
        for file in uploaded_files:
            # Verificar extensão
            if file.type.split('/')[-1] not in AppConfig.ALLOWED_FILE_TYPES:
                st.sidebar.error(f"Tipo de arquivo não permitido: {file.name}")
                return False
            
            # Verificar tamanho do arquivo
            if file.size > AppConfig.MAX_FILE_SIZE:
                st.sidebar.error(f"Arquivo muito grande: {file.name}. Limite de 10 MB.")
                return False
        
        # Verificar número máximo de arquivos
        if len(uploaded_files) > AppConfig.MAX_UPLOAD_FILES:
            st.sidebar.error(f"Número máximo de arquivos excedido. Limite: {AppConfig.MAX_UPLOAD_FILES}")
            return False
        
        return True

# Classe de Interface do Usuário
class UserInterface:
    @staticmethod
    def display_message(message: str, is_user: bool):
        """
        Exibe mensagens no chat com estilo personalizado
        
        Args:
            message (str): Texto da mensagem
            is_user (bool): Indica se a mensagem é do usuário
        """
        if is_user:
            st.markdown(
                f'<div style="text-align: right; color: #0000ff; padding: 10px; '
                f'border-radius: 10px; background-color: #f1f1f1; max-width: 70%; '
                f'margin-left: auto; margin-bottom: 5px;">{message}</div>', 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="text-align: left; color: #ffffff; padding: 10px; '
                f'border-radius: 10px; background-color: #007bff; max-width: 70%; '
                f'margin-right: auto; margin-bottom: 5px;">{message}</div>', 
                unsafe_allow_html=True
            )

    @staticmethod
    def render_sidebar():
        """Renderiza a barra lateral com configurações e upload"""
        st.sidebar.title("🤖 Configurações")
        
        # Seção de Upload de Documentos
        st.sidebar.header("📤 Carregar Documentos")
        uploaded_files = st.sidebar.file_uploader(
            "Envie seus arquivos", 
            type=AppConfig.ALLOWED_FILE_TYPES, 
            accept_multiple_files=True
        )
        
        # Botão de Upload
        if st.sidebar.button("Enviar Documentos"):
            UserInterface.handle_document_upload(uploaded_files)
        
        # Seção de Parâmetros
        st.sidebar.header("⚙️ Ajustes de Parâmetros")
        temperature = st.sidebar.slider("Temperatura", 0.0, 2.0, 0.7)
        max_tokens = st.sidebar.slider("Máximo de Tokens", 50, 1000, 200)
        top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 1.0)
        
        # Botão de Limpar Histórico
        if st.sidebar.button("🗑️ Limpar histórico de chat"):
            st.session_state.messages = []
            st.rerun()
        
        return temperature, max_tokens, top_p

    @staticmethod
    def handle_document_upload(uploaded_files):
        """
        Processa o upload de documentos
        
        Args:
            uploaded_files (List): Lista de arquivos enviados
        """
        if not uploaded_files:
            st.sidebar.warning("Por favor, selecione arquivos para upload.")
            return
        
        if not Validator.validate_files(uploaded_files):
            return
        
        # Preparar arquivos para upload
        files = {f"file_{i}": (file.name, file) for i, file in enumerate(uploaded_files)}
        
        with st.spinner("Enviando documentos..."):
            result = BackendService.upload_documents(files)
            
            # Exibir resultado do upload
            if "sucesso" in result.lower():
                st.sidebar.success(result)
            else:
                st.sidebar.error(result)

# Classe Principal da Aplicação
class ChatApplication:
    @staticmethod
    def initialize_session_state():
        """Inicializa o estado da sessão"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []

    @staticmethod
    def run():
        """
        Método principal para executar a aplicação Streamlit
        """
        # Configuração da página
        st.set_page_config(
            page_title="Chat LLM com LangChain + RAG", 
            layout="wide",
            page_icon="🤖"
        )
        
        # Título da aplicação
        st.title("🤖 Chat Inteligente com RAG")
        
        # Inicializar estado da sessão
        ChatApplication.initialize_session_state()
        
        # Renderizar sidebar e obter parâmetros
        temperature, max_tokens, top_p = UserInterface.render_sidebar()
        
        # Exibir mensagens anteriores
        for message, is_user in st.session_state.messages:
            UserInterface.display_message(message, is_user)
        
        # Caixa de entrada de texto
        user_input = st.text_input(
            "Digite sua pergunta ou comentário:", 
            key="user_input_field"
        )
        
        # Botão de envio
        if st.button("Enviar") or (user_input and st.session_state.get('submit_state', False)):
            ChatApplication.handle_user_input(
                user_input, 
                temperature, 
                max_tokens, 
                top_p
            )
    
    @staticmethod
    def handle_user_input(
        user_input: str, 
        temperature: float, 
        max_tokens: int, 
        top_p: float
    ):
        """
        Processa a entrada do usuário e gera resposta
        
        Args:
            user_input (str): Texto digitado pelo usuário
            temperature (float): Parâmetro de criatividade
            max_tokens (int): Máximo de tokens na resposta
            top_p (float): Parâmetro de amostragem nuclear
        """
        # Validar entrada
        if not user_input.strip():
            st.warning("Por favor, digite uma pergunta.")
            return
        
        # Adicionar mensagem do usuário ao histórico
        st.session_state.messages.append((user_input, True))
        
        # Limitar histórico de mensagens
        if len(st.session_state.messages) > AppConfig.MAX_CHAT_HISTORY:
            st.session_state.messages = st.session_state.messages[-AppConfig.MAX_CHAT_HISTORY:]
        
        # Gerar resposta
        with st.spinner("Gerando resposta..."):
            try:
                result = BackendService.generate_response(
                    user_input, 
                    temperature, 
                    max_tokens, 
                    top_p
                )
                
                # Adicionar resposta ao histórico
                st.session_state.messages.append((result, False))
                
                # Recarregar para mostrar nova mensagem
                st.rerun()
            
            except Exception as e:
                st.error(f"Erro ao processar sua solicitação: {e}")
                logging.error(f"Erro no processamento: {e}")

    @staticmethod
    def initialize_session_state():
        """
        Inicializa o estado da sessão com valores padrão
        """
        default_states = {
            'messages': [],
            'submit_state': False,
            'error_message': None
        }
        
        for key, value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = value

# Ponto de entrada da aplicação
def main():
    try:
        ChatApplication.run()
    except Exception as e:
        st.error(f"Erro crítico na aplicação: {e}")
        logging.critical(f"Erro crítico: {e}")

if __name__ == "__main__":
    main()
