# Docker-Setup für Shot-Plotter und Datenmerging

Dieses Projekt besteht aus zwei Docker-Containern:
1. **Shot-Plotter**: Eine Webanwendung zum Plotten von Ereignissen auf einem Sportfeld
2. **Datenmerging**: Eine Streamlit-Anwendung zur Verarbeitung und Integration von Daten

## Voraussetzungen

- Docker und Docker Compose müssen installiert sein
- Git für das Klonen des Repositories

## Installation und Start

1. Klone das Repository:
   ```
   git clone <repository-url>
   cd shot-plotter
   ```

2. Starte die Container mit Docker Compose:
   ```
   docker-compose up -d
   ```

3. Zugriff auf die Anwendungen:
   - Shot-Plotter: http://localhost:8080/html/soccer-ncaa.html?width=120&height=75
   - Datenmerging: http://localhost:8501

## Verwendung

### Shot-Plotter
Die Shot-Plotter-Anwendung ist unter http://localhost:8080/html/soccer-ncaa.html?width=120&height=75 erreichbar. Sie können durch einfaches Klicken auf das Spielfeld Ereignisse aufzeichnen.

### Datenmerging
Die Datenmerging-Anwendung läuft auf http://localhost:8501 und ermöglicht die Integration von:
1. Shot-Plotter CSV-Daten
2. Playermaker XML-Daten
3. Playermaker Possession Excel-Daten

## Container stoppen

Um die Container zu stoppen, führen Sie folgenden Befehl aus:
```
docker-compose down
```

## Fehlerbehebung

1. Wenn Port 8080 oder 8501 bereits verwendet wird, ändern Sie die Ports in der docker-compose.yml Datei.
2. Bei Problemen mit dem Shot-Plotter prüfen Sie, ob der Server korrekt läuft:
   ```
   docker logs shot-plotter
   ```
3. Bei Problemen mit der Datenmerging-Anwendung:
   ```
   docker logs data-merging
   ``` 