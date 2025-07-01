# Dockerfile para Mnemosynth
FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de requirements
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY . .

# Exponer el puerto
EXPOSE 7860

# Establecer el directorio de trabajo para el script
WORKDIR /app/src/f5_tts/infer

# Comando por defecto
CMD ["python", "mnemosynth.py", "--host", "0.0.0.0", "--port", "7860"]
