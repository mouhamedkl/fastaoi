# 🔹 Image de base Python 3.11
FROM python:3.11-slim

# 🔹 Installer ffmpeg (nécessaire pour diffusers)
RUN apt-get update && apt-get install -y ffmpeg git curl && rm -rf /var/lib/apt/lists/*

# 🔹 Copier le code
WORKDIR /app
COPY requirements.txt .
COPY main.py .

# 🔹 Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# 🔹 Exposer le port
EXPOSE 8000

# 🔹 Commande pour lancer FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
