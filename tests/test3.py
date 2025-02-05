from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

# 1. Carregamento e Preparação dos Documentos
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"
]

# Carregar documentos
loader = WebBaseLoader(urls)
docs = loader.load()

# Dividir documentos em chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 2. Configuração de Embeddings e Vetorstore
embeddings = OllamaEmbeddings(model="llama3")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# 3. Configuração do Modelo LLM
llm = ChatOllama(model="llama3", temperature=0.7)

# 4. Prompt Template para RAG
prompt_template = ChatPromptTemplate.from_template("""
Você é um assistente de IA útil. Com base no contexto abaixo, responda à pergunta:

Contexto: {context}

Pergunta: {question}

Resposta detalhada:
""")

# 5. Construção do RAG Chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

# 6. Função para realizar consultas
def consultar_rag(pergunta):
    return rag_chain.invoke(pergunta)

# Exemplo de uso
resultado = consultar_rag("O que são agentes de IA?")
print(resultado)