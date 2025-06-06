# Shot-Plotter & Datenmerging Anleitung

Diese Streamlit-App bietet eine umfassende Anleitung zur Verwendung des Shot-Plotters und des Datenmerging-Tools. Sie enthält Erklärungen, Anleitungen und Videos zur Demonstration der Funktionen.

## Funktionen

- **Interaktive Anleitungen** für den Shot-Plotter und das Datenmerging-Tool
- **Video-Tutorials** für komplexe Funktionen
- **FAQ** mit häufig gestellten Fragen und Antworten
- **Responsive Design** für die Nutzung auf verschiedenen Geräten

## Installation und Start

### Lokale Installation

1. Clone das Repository:
   ```bash
   git clone https://github.com/yourusername/shot-plotter.git
   cd shot-plotter
   ```

2. Installiere die Abhängigkeiten:
   ```bash
   pip install -r requirements.txt
   ```

3. Starte die App:
   ```bash
   streamlit run streamlit_app.py
   ```

4. Öffne die App im Browser unter `http://localhost:8501`

### Mit Docker

1. Baue das Docker-Image:
   ```bash
   docker build -t shot-plotter-guide:latest -f Dockerfile.streamlit .
   ```

2. Starte einen Container:
   ```bash
   docker run -d -p 8501:8501 --name shot-plotter-guide shot-plotter-guide:latest
   ```

3. Öffne die App im Browser unter `http://localhost:8501`

## Hinzufügen eigener Videos

Um eigene Videos hinzuzufügen:

1. Speichere die Videos im MP4-Format im Verzeichnis `videos/`
2. Füge in der `streamlit_app.py` den folgenden Code hinzu, um das Video einzubinden:
   ```python
   video_bytes = local_video("videos/dein_video.mp4")
   st.video(video_bytes)
   ```

## Multiplatform-Docker-Image erstellen

Um ein Docker-Image zu erstellen, das auf verschiedenen Plattformen (Intel/AMD und ARM) läuft:

```bash
docker buildx create --name multi-platform-builder --use
docker buildx inspect --bootstrap
docker buildx build --platform linux/amd64,linux/arm64 -t yourusername/shot-plotter-guide:latest -f Dockerfile.streamlit . --push
```

## Hinweise zur Videoproduktion

Für beste Ergebnisse bei der Erstellung von Anleitungsvideos:

1. Verwende einen Bildschirmrekorder wie OBS Studio oder QuickTime
2. Achte auf eine gute Auflösung (mind. 1080p)
3. Verwende Untertitel oder Texteinblendungen für wichtige Punkte
4. Halte die Videos kurz und fokussiert (max. 2-3 Minuten pro Thema)
5. Komprimiere die Videos auf ein geeignetes Format und eine angemessene Größe

