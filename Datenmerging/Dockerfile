FROM python:3.9-slim

WORKDIR /app

# Kopiere alle Dateien aus dem aktuellen Verzeichnis
COPY . .

# Installiere Abhängigkeiten
RUN pip install --no-cache-dir streamlit pandas numpy plotly

# Exponiere Port 8501 (interner Port für Streamlit)
EXPOSE 8501

# Starte die Streamlit-Anwendung
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
