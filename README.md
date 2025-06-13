# Shot-Plotter & Datenmerging Anleitung

Eine interaktive Anleitung für den Shot-Plotter und das Datenmerging-Tool, entwickelt im Rahmen des BVB Kooperationsprojekts.

## 🚀 Features

- **Interaktive Anleitung**: Schritt-für-Schritt Anleitung mit Screenshots
- **Docker Integration**: Einfache Installation über Docker Desktop
- **Mehrsprachige Dokumentation**: Deutsche Anleitung mit technischen Details
- **Beispielabläufe**: Praktische Beispiele zur Verwendung des Shot-Plotters
- **Widget-Erklärungen**: Detaillierte Beschreibung aller verfügbaren Optionen

## 📋 Inhaltsverzeichnis

1. [How to start](#how-to-start)
2. [Beispielablauf](#beispielablauf)
3. [Erklärung der Widgets](#erklärung-der-widgets)
4. [Extras](#extras)
5. [Download](#download)

## 🛠️ Installation

### Voraussetzungen

- Docker Desktop
- Mindestens 2 GB RAM
- 1 GB freier Speicherplatz
- Internetverbindung

### Schnellstart

1. Docker Desktop installieren
2. Shot-Plotter Image herunterladen:
   ```bash
   docker pull sindi98/shot-plotter-shot-plotter:v1.2
   ```
3. Container starten:
   ```bash
   docker run -d -p 8080:8080 --name shot-plotter sindi98/shot-plotter-shot-plotter:v1.2
   ```
4. Shot-Plotter öffnen: http://localhost:8080

## 📚 Anleitung

Die vollständige Anleitung ist als Streamlit-App verfügbar. Starte sie mit:

```bash
cd Anleitung
streamlit run streamlit_app.py
```

Oder nutze Docker Compose für alle Komponenten:

```bash
docker-compose up -d
```

## 🔧 Komponenten

- **Shot-Plotter**: Hauptanwendung für die Datenerfassung
- **Datenmerging**: Tool zur Verknüpfung von Daten
- **Streamlit Guide**: Interaktive Anleitung

## 🤝 Mitwirken

1. Fork das Repository
2. Erstelle einen Feature Branch
3. Committe deine Änderungen
4. Pushe zum Branch
5. Erstelle einen Pull Request

## 📝 Lizenz

© 2024 BVB Kooperationsprojekt

## 📞 Kontakt

Bei Fragen oder Problemen:
- Erstelle ein Issue in diesem Repository
- Kontaktiere das BVB Kooperationsprojekt Team
