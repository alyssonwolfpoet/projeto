import streamlit as st
import requests
import os

# URL das APIs backend
UPLOAD_API_URL = os.getenv("UPLOAD_API_URL", "http://localhost:8000/upload-documents/")
CHAT_API_URL = os.getenv("CHAT_API_URL", "http://localhost:8000/generate/")

# Função para enviar os documentos para o backend
def upload_documents_to_backend(files):
    try:
        # Envia os arquivos para o backend
        response = requests.post(UPLOAD_API_URL, files=files)
        if response.status_code == 200:
            return response.json().get("message", "Erro ao enviar documentos.")
        else:
            return f"Erro ao enviar documentos: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Erro ao comunicar com o backend: {e}"

# Função para enviar o prompt, arquivos e parâmetros para o backend e obter a resposta
def get_llm_response(prompt: str, temperature, max_tokens, top_p):
    try:
        # Envia o prompt e parâmetros para o backend para gerar a resposta
        response = requests.post(
            CHAT_API_URL,
            data={'prompt': prompt, 'temperature': temperature, 'max_tokens': max_tokens, 'top_p': top_p},
        )
        if response.status_code == 200:
            return response.json().get("response", "Erro na resposta do modelo.")
        else:
            return f"Erro ao obter resposta do modelo: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Erro ao comunicar com o backend: {e}"

# Definindo a interface no Streamlit
st.set_page_config(page_title="Chat LLM com LangChain + RAG", layout="wide")

# Função para exibir mensagens no chat
def display_message(message, is_user):
    if is_user:
        st.markdown(f'<div style="text-align: right; color: #0000ff; padding: 10px; border-radius: 10px; background-color: #f1f1f1; max-width: 70%; margin-left: auto; margin-bottom: 5px;">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="text-align: left; color: #ffffff; padding: 10px; border-radius: 10px; background-color: #007bff; max-width: 70%; margin-right: auto; margin-bottom: 5px;">{message}</div>', unsafe_allow_html=True)

# Título da página
st.title("🤖 Chat LLM com LangChain + RAG")

# Criação do painel lateral para enviar arquivos
st.sidebar.title("Carregar Documentos")
uploaded_files = st.sidebar.file_uploader("Envie seus arquivos", type=["txt", "pdf", "docx", "csv", "odt"], accept_multiple_files=True)

# Botão para enviar os documentos
if st.sidebar.button("Enviar Documentos"):
    if uploaded_files:
        files = {f"file_{i}": (file.name, file) for i, file in enumerate(uploaded_files)}
        with st.spinner("Enviando documentos..."):
            result = upload_documents_to_backend(files)
            st.sidebar.write(result)
    else:
        st.sidebar.write("Por favor, envie arquivos primeiro.")

# Ajustes personalizados de parâmetros
st.sidebar.header("Ajustes de Parâmetros")
temperature = st.sidebar.slider("Temperatura", 0.0, 2.0, 0.7)
max_tokens = st.sidebar.slider("Máximo de Tokens", 50, 1000, 200)
top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 1.0)

# Exibir uma área de chat onde o usuário pode digitar perguntas
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Mostrar mensagens anteriores no chat
for message, is_user in st.session_state.messages:
    display_message(message, is_user)

# Função para processar e enviar o prompt
def handle_user_input(user_input):
    if user_input:
        st.session_state.messages.append((user_input, True))  # Adicionar mensagem do usuário
        
        # Enviar para o backend e obter a resposta
        with st.spinner("Gerando resposta..."):
            result = get_llm_response(user_input, temperature, max_tokens, top_p)
        
        st.session_state.messages.append((result, False))  # Adicionar resposta do modelo
        st.rerun()  # Recarregar a página para exibir a nova resposta

# Caixa de entrada de texto do usuário
user_input = st.text_input("Digite sua pergunta ou comentário:", "")

# Botão para enviar a pergunta
if st.button("Enviar"):
    handle_user_input(user_input)

# Opção para limpar a tela
if st.sidebar.button("Limpar histórico de chat"):
    st.session_state.messages = []
    st.rerun()  # Recarregar a página após limpar o histórico
