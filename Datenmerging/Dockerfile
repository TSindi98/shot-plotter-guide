FROM python:3.9-slim

WORKDIR /app

# Kopiere requirements.txt zuerst für besseres Caching
COPY requirements.txt .

# Installiere Abhängigkeiten aus requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere den Rest der Dateien
COPY . .

# Exponiere Port 8000 (interner Port für Streamlit)
EXPOSE 8401

# Starte die Hauptanwendung (integrierte Fußball-Datenanalyse Suite)
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8401", "--server.address=0.0.0.0"]
