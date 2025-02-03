docker system prune --all --force --volumes
# Parar todos os containers em execução
docker stop $(docker ps -aq)

# Remover todos os containers
docker rm $(docker ps -aq)

# Remover todas as imagens
docker rmi $(docker images -q) -f

# Remover todos os volumes
docker volume rm $(docker volume ls -q)

# Remover todas as redes não usadas (exceto a padrão)
docker network prune -f

Write-Host "Docker limpo! Todos os containers, imagens, volumes e redes foram removidos."
