# Importações necessárias
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Carregamento de Documentos
def load_documents(urls):
    """Carrega documentos de URLs específicas"""
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    return docs_list

# 2. Divisão de Texto
def split_documents(docs):
    """Divide os documentos em chunks menores"""
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, 
        chunk_overlap=50
    )
    return text_splitter.split_documents(docs)

# 3. Criação do Embedding e Vetorstore
def create_vectorstore(doc_splits):
    """Cria um vetorstore usando embeddings do Ollama"""
    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = Chroma.from_documents(
        documents=doc_splits, 
        embedding=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# 4. Configuração do Modelo RAG
def setup_rag_chain(retriever):
    """Configura o pipeline de RAG"""
    # Modelo local do Ollama
    llm = ChatOllama(model="llama3", temperature=0.7)
    
    # Template do prompt
    template = """Responda a pergunta baseando-se apenas no seguinte contexto:
    {context}
    
    Pergunta: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Construção do chain de RAG
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return rag_chain

# 5. Função principal de execução
def main():
    # URLs dos documentos para indexação
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    ]
    
    # Carrega e processa documentos
    docs = load_documents(urls)
    doc_splits = split_documents(docs)
    
    # Cria retriever
    retriever = create_vectorstore(doc_splits)
    
    # Configura chain de RAG
    rag_chain = setup_rag_chain(retriever)
    
    # Loop de interação
    while True:
        query = input("\nDigite sua pergunta (ou 'sair' para encerrar): ")
        
        if query.lower() == 'sair':
            break
        
        # Executa RAG
        response = rag_chain.invoke(query)
        print("\nResposta:", response)

# Executa o programa
if __name__ == "__main__":
    main()
