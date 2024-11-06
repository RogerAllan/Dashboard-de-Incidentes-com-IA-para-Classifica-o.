# Dockerfile para o back-end com Flask
FROM python:3.9-slim

# Diretório de trabalho
WORKDIR /app

# Instala as dependências
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copia o código do back-end para o contêiner
COPY . .

# Porta para o Flask
EXPOSE 5000

# Comando para rodar a aplicação
CMD ["python", "app.py"]
