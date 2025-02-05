# Explicação da Estrutura
# Vamos detalhar cada parte do código:
# 1. Carregamento e Preparação dos Documentos
# Usa WebBaseLoader para carregar documentos de URLs
# RecursiveCharacterTextSplitter divide os documentos em chunks menores
# 2. Embeddings e Vetorstore
# OllamaEmbeddings gera embeddings usando o modelo Ollama
# Chroma cria um banco de vetores para recuperação de documentos
# 3. Modelo LLM
# ChatOllama usa o modelo Llama3 localmente
# 4. Prompt Template
# Template de prompt que inclui contexto e pergunta
# 5. RAG Chain
# Usa LCEL (LangChain Expression Language) para compor o pipeline
# Recupera contexto relevante
# Gera resposta baseada no contexto
# 6. Função de Consulta
# Método simples para realizar consultas no RAG
# Considerações Importantes
# Certifique-se de ter o Ollama instalado e o modelo Llama3 baixado
# A versão v0.3 usa Pydantic 2 e tem algumas mudanças de importação
# Ajuste os parâmetros como chunk_size, temperature, etc., conforme necessário


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