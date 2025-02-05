function List-DirWithContent {
    param (
        [string]$path = (Get-Location),   # Caminho padrão é o diretório atual
        [string]$outputFile = "output.txt" # Arquivo de saída padrão
    )

    # Criando ou limpando o arquivo de saída
    if (Test-Path $outputFile) {
        Remove-Item $outputFile
    }

    # Função para escrever no arquivo de saída
    function Write-ToFile {
        param ([string]$message)
        Add-Content -Path $outputFile -Value $message
    }

    # Função para exibir a estrutura de árvore com indentação
    function Show-Tree {
        param (
            [string]$currentPath,
            [int]$level = 0
        )

        # Obter todos os arquivos e subdiretórios no diretório atual
        $items = Get-ChildItem -Path $currentPath

        # Para cada item, seja pasta ou arquivo
        foreach ($item in $items) {
            # Adicionar a indentação conforme o nível de profundidade
            $indent = ' ' * ($level * 2)
            
            if ($item.PSIsContainer) {
                Write-ToFile "$indent└── Pasta: $($item.Name)"
                # Recursivamente chamar para subdiretórios
                Show-Tree -currentPath $item.FullName -level ($level + 1)
            } else {
                Write-ToFile "$indent└── Arquivo: $($item.Name)"
                try {
                    # Exibir o conteúdo do arquivo (primeiras 10 linhas)
                    Write-ToFile "$indent    Conteúdo do arquivo:"
                    Get-Content $item.FullName -First 10 | ForEach-Object { Write-ToFile "$indent    $_" }
                    Write-ToFile "$indent    ----------"
                } catch {
                    Write-ToFile "$indent    Não foi possível ler o conteúdo de $($item.FullName)"
                }
            }
        }
    }

    # Iniciando a exibição da árvore a partir do diretório raiz
    Show-Tree -currentPath $path

    Write-Host "Saída salva em $outputFile"
}

# Chama a função para o diretório especificado, salvando a saída em 'output.txt'
List-DirWithContent -path "C:\Users\Alysson\Desktop\myprojeto\rag_project" -outputFile "C:\Users\Alysson\Desktop\output.txt"
