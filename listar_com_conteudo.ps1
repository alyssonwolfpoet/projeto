# Função para listar e mostrar conteúdo dos arquivos
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

    # Listando a estrutura de diretórios e arquivos
    Get-ChildItem -Path $path -Recurse | ForEach-Object {
        if ($_ -is [System.IO.DirectoryInfo]) {
            Write-ToFile "Pasta: $($_.FullName)"
        } elseif ($_ -is [System.IO.FileInfo]) {
            Write-ToFile "Arquivo: $($_.FullName)"
            # Exibindo o conteúdo do arquivo
            try {
                Write-ToFile "Conteúdo do arquivo:"
                # Limita a exibição a 10 linhas para evitar grandes arquivos
                Get-Content $_.FullName -First 10 | ForEach-Object { Write-ToFile $_ }
                Write-ToFile "----------"
            } catch {
                Write-ToFile "Não foi possível ler o conteúdo de $($_.FullName)"
            }
        }
    }

    Write-Host "Saída salva em $outputFile"
}

# Chama a função para o diretório atual, salvando a saída em 'output.txt'
List-DirWithContent -path "C:\Users\Alysson\Desktop\myprojeto\rag_project" -outputFile "C:\Users\Alysson\Desktop\output.txt"
