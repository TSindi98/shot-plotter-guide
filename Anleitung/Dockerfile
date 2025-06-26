FROM python:3.9-slim

WORKDIR /app

# Labels hinzufügen
LABEL version="1.0"
LABEL description="Shot-Plotter Guide - Streamlit App"
LABEL maintainer="BVB Kooperationsprojekt"

# Installiere benötigte Pakete
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Kopiere requirements.txt zuerst, um Layer-Caching zu nutzen
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere den Rest der App
COPY . .

# Port, auf dem Streamlit läuft
EXPOSE 8501

# Healthcheck hinzufügen
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Streamlit starten mit der korrekten Datei
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"] 