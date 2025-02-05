# Configuração e Uso do Sistema RAG

def main():
    # URLs ou caminhos de documentos para indexação
    sources = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"
    ]

    # Inicializa o sistema RAG
    rag_system = RAGSystem(
        model_name="llama3",
        embedding_model="nomic-embed-text",
        chunk_size=500,
        chunk_overlap=50
    )

    # 1. Carrega documentos
    print("Carregando documentos...")
    documents = rag_system.load_documents(sources)

    # 2. Prepara o vetorstore
    print("Preparando vetorstore...")
    retriever = rag_system.prepare_vectorstore(documents)

    # 3. Consultas de exemplo
    queries = [
        "O que são agentes de IA?",
        "Como funciona o prompt engineering?",
        "Quais são as principais técnicas de engenharia de prompt?"
    ]

    # 4. Realizando consultas com diferentes métodos
    print("\n--- Consultas Básicas ---")
    for query in queries:
        print(f"\nPergunta: {query}")
        
        # Consulta básica
        response = rag_system.advanced_query(query)
        print("Resposta:", response[:500] + "..." if len(response) > 500 else response)

    # 5. Demonstração de recuperação multi-query
    print("\n--- Multi-Query Retrieval ---")
    complex_query = "Explique as estratégias avançadas de prompt engineering"
    multi_docs = rag_system.multi_query_retrieval(complex_query)
    print(f"Documentos recuperados: {len(multi_docs)}")

    # 6. Avaliação de consulta
    print("\n--- Avaliação de Consulta ---")
    evaluation = rag_system.evaluate_query(
        "Quais são os desafios na criação de agentes de IA?"
    )
    print("Avaliação da Resposta:", evaluation)

    # 7. Adicionando documentos personalizados
    custom_docs = [
        Document(
            page_content="Agentes de IA são sistemas computacionais que podem perceber seu ambiente e tomar ações de forma autônoma.",
            metadata={"source": "custom_definition"}
        )
    ]
    rag_system.add_custom_documents(custom_docs)

    # 8. Consulta após adicionar documentos personalizados
    print("\n--- Consulta Após Adicionar Documentos Personalizados ---")
    custom_query = "Defina agentes de IA de forma simples"
    custom_response = rag_system.advanced_query(custom_query, return_sources=True)
    print("Resposta:", custom_response['answer'])
    print("\nFontes:")
    for source in custom_response['sources']:
        print("-", source['metadata'])

if _name_ == "_main_":
    main()


# pip install -U langchain langchain-core langchain-community langchain-ollama chromadb


# New chat
# New chat
# Crie uma estrutura de langchain mais atual v0.3 usando rag e pg ollama
# Done
# Research Plan & Sources
# Research the latest LangChain v0.3 RAG implementation

# langchain_ibm 0.1.9 — 🦜🔗 LangChain 0.2.7
# How to implement an integration package | 🦜️🔗 LangChain
# langchain_robocorp 0.0.10.post1 — 🦜🔗 LangChain 0.2.17
# LangChain v0.3 | 🦜️🔗 LangChain
# Docusaurus | 🦜️🔗 LangChain
# Evaluate a RAG application | 🦜️🛠️ LangSmith
# See 3 more documents
# Investigate Ollama integration for local language models

# ChatOllama | 🦜️🔗 LangChain
# Ollama | 🦜️🔗 LangChain
# OllamaEmbeddings | 🦜️🔗 LangChain
# OllamaLLM | 🦜️🔗 LangChain
# Run models locally | 🦜️🔗 LangChain
# OpenLLM | 🦜️🔗 LangChain
# See 2 more documents
# Create a sample code structure combining RAG with Ollama

# OllamaLLM | 🦜️🔗 LangChain
# ChatOllama | 🦜️🔗 LangChain
# Self-RAG using local LLMs | 🦜🕸️LangGraph
# Ollama | 🦜️🔗 LangChain
# arXiv | 🦜️🔗 LangChain
# arXiv | 🦜️🔗 LangChain
# See 8 more documents
# Selected Context
# langchain_ibm 0.1.9 — 🦜🔗 LangChain 0.2.7
# langchain_ibm 0.1.9 — 🦜🔗 LangChain 0.2.7 LangChain Core Community Experimental Text splitters ai21 airbyte anthropic astradb aws azure-dynamic-sessions chroma cohere couchbase elasticsearch exa fireworks google-community google-g
# How to implement an integration package | 🦜️🔗 LangChain
# - [Implementing LangChain components](#implementing-langchain-components)- [Properties](#properties) - [Run Methods](#run-methods) - [Implementation](#implementation) - [Attributes](#attributes) - [Methods](#methods) - [Implementation](#implemen
# langchain_robocorp 0.0.10.post1 — 🦜🔗 LangChain 0.2.17
# langchain_robocorp 0.0.10.post1 — 🦜🔗 LangChain 0.2.17 This is a legacy site. Please use the latest v0.2 and v0.3 API references instead. LangChain Core Community Experimental Text splitters ai21 airbyte anthropic astradb aws azure-dynam
# Answer
# Aqui está um exemplo de como criar uma estrutura de RAG (Retrieval-Augmented Generation) usando LangChain v0.3 com Ollama:
# Instalação das Dependências
# bash
# Copy
# pip install -U langchain langchain-core langchain-community langchain-ollama chromadb

# Código Completo de Implementação
# python
# Copy


