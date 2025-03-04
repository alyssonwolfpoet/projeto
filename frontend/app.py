# frontend/app.py

import streamlit as st
import requests
from io import BytesIO

# Definindo a URL da API
API_URL = "http://localhost:8000"

# Função para upload de PDF
def upload_pdf():
    st.title("Envio de PDF para o Sistema")
    uploaded_file = st.file_uploader("Escolha um arquivo PDF", type="pdf")
    if uploaded_file is not None:
        st.write("Processando PDF...")
        pdf_content = uploaded_file.read()
        
        # Envia o PDF para o backend processar e armazenar
        response = requests.post(f"{API_URL}/documents/pdf/", files={"pdf_file": pdf_content})
        
        if response.status_code == 200:
            st.success("PDF enviado e processado com sucesso!")
            st.write(f"Documento ID: {response.json()['id']}")
            st.write(f"Conteúdo do Documento: {response.json()['content'][:500]}...")  # Exibe os primeiros 500 caracteres
        else:
            st.error("Erro ao processar o PDF.")

# Função para exibir todos os documentos armazenados
def show_documents():
    st.title("Documentos Armazenados")
    response = requests.get(f"{API_URL}/documents/")
    
    if response.status_code == 200:
        documents = response.json()
        for doc in documents:
            st.write(f"**ID:** {doc['id']}")
            st.write(f"**Conteúdo:** {doc['content'][:500]}...")  # Exibe os primeiros 500 caracteres
            st.write("---")
    else:
        st.error("Erro ao carregar os documentos.")

# Função para interagir com o modelo de linguagem
def chat_with_model():
    st.title("Interaja com o Modelo")
    user_input = st.text_input("Digite sua mensagem:")
    
    if user_input:
        response = requests.post(f"{API_URL}/chat/", json={"user_input": user_input})
        if response.status_code == 200:
            st.write("Resposta do Modelo:")
            st.write(response.json()["response"])
        else:
            st.error("Erro ao comunicar com o modelo.")

# Layout do Streamlit com opções para o usuário
st.sidebar.title("Opções")
option = st.sidebar.radio("Escolha uma opção:", ["Enviar PDF", "Visualizar Documentos", "Interagir com o Modelo"])

if option == "Enviar PDF":
    upload_pdf()
elif option == "Visualizar Documentos":
    show_documents()
elif option == "Interagir com o Modelo":
    chat_with_model()
