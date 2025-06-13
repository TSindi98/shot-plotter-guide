import streamlit as st
import base64
import os
from PIL import Image

# Seitenkonfiguration
st.set_page_config(
    page_title="Shot-Plotter & Datenmerging Anleitung",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Funktion zum Laden von Bildern
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Sidebar für Navigation
st.sidebar.title("Shot-Plotter Anleitung")
page = st.sidebar.radio(
    "Navigation",
    ["How to start", "Beispielablauf", "Erklärung der Widgets", "Extras", "Download"]
)

# Funktion zum Einbetten lokaler Videos
def local_video(file_path):
    with open(file_path, "rb") as video_file:
        video_bytes = video_file.read()
    return video_bytes

# How to start
if page == "How to start":
    st.title("How to start")
    st.write("""
    Hier erfährst du, wie du den Shot-Plotter mit Docker Desktop starten und einrichten kannst.
    """)
    
    # Funktion zum Anzeigen von Bildern mit Beschreibung
    def show_step(step_number, title, description_before, description_after):
        st.header(f"Schritt {step_number}: {title}")
        
        # Beschreibung vor dem Bild
        st.markdown(description_before)
        
        # Bild anzeigen
        image_path = f"images/{step_number}.png"
        if os.path.exists(image_path):
            st.image(image_path, use_column_width=True)
        else:
            st.error(f"Bild '{image_path}' nicht gefunden")
        
        # Beschreibung nach dem Bild
        st.markdown(description_after)
        st.markdown("---")  # Trennlinie
    
    # Schritt 1
    show_step(
        1, 
        "Docker Desktop öffnen", 
        """
        Falls du **Docker Desktop** noch nicht installiert hast, kannst du es hier herunterladen:
        
        [Docker Desktop herunterladen](https://www.docker.com/products/docker-desktop/)

        (Achte darauf, dass du die richtige Version für dein Betriebssystem herunterlädst)

        Erstelle dir einen Account bei Docker falls du noch keinen hast und melde dich damit bei Docker Desktop an.
        """,
        """
        Klicke links auf **DockerHub** und suche nach **sindi98**
        """
    )
    
    # Schritt 2
    show_step(
        2, 
        "Nach dem Shot-Plotter suchen", 
        """
        In der Suchergebnisliste findest du verschiedene Projekte:

        Klicke auf das Projekt **sindi98/shot-plotter-shot-plotter**. (Der Name ist nicht ganz zu sehen, aber man kann erkennen um welches Projekt es sich handelt)
        """,
        """
        
        """
    )
    
    # Schritt 3
    show_step(
        3, 
        "Das richtige Image auswählen", 
        """
        Auf der Projektseite findest du verschiedene **Versionen** des Programms:
        """,
        """
        Oben rechts findest du unter **Tag** die Versionen des Programms, wähle **v1.2** aus

        Klicke auf anschließend auf **Pull**
        
        *Hinweis: Der Download kann je nach Internetverbindung einige Minuten dauern.*
        """
    )
    
    # Schritt 4
    show_step(
        4, 
        "Zur Image-Seite wechseln", 
        """
        Nach dem erfolgreichen Download des Images:

        Klicke nun in der linken Seitenleiste auf **Images**
        """,
        """
        
        """
    )
    
    # Schritt 5
    show_step(
        5, 
        "Container konfigurieren", 
        """
        Du solltest nun ein Image mit dem Namen **sindi98/shot-plotter-shot-plotter** sehen. Außerdem sollte der Tag **v1.2** angezeigt werden.

        Klicke nun auf den blauen Button **Run** (Hier auf dem Bild in grün markiert)
        """,
        """
        *Hinweis: neben dem Namen shot-plotter-shot-plotter ist ein kleiner Kreis der noch nicht ausgefüllt ist. Im späteren Verlauf sollte dieser grün sein.*
        """
    )

    # Schritt 6
    show_step(
        6, 
        "Container Einstellungen", 
        """
        Öffne **"Optional Settings"** indem du auf den weißen Pfeil klickst.
        """,
        """
        """
    )

    # Schritt 7
    show_step(
        7, 
        "Container starten", 
        """
        Setze einen Namen für den Container. Vorzugsweise nenne ihn **shot-plotter**.

        Wähle **8080:8080** als Port aus, indem du in das Eingabefeld **8080** eingibst.

        Klicke auf **Run**
        """,
        """
        """
    )

    # Schritt 8
    show_step(
        8, 
        "Shot-Plotter im Browser öffnen", 
         """
        Der Container sollte nun gestartet worden sein. 

        Klicke nun auf den angezeigten Port. Im Bild ist dieser mit rot umrandet. 
        *Hinweis: Im Bild ist 7001 der Port, bei dir sollte aber 8080 angezeigt werden, sofern du die Standard-Einstellungen verwendet hast.*

        Du solltest nun die Seite des Shot-Plotters sehen.
        """,
        """
        Du kannst die Seite des Shot-Plotters immer wieder unter http://localhost:8080 in deinem Browser aufrufen.
        *Hinweis: Safari und Firefox sind als Browser zu empfehlen, bei Chrome kann es zu Problemen kommen.*

        Alternativ kannst du Docker Desktop öffnen und den Shot-Plotter unter **"Containers"** finden.
        """
    )




    # Alternative Installations-Methode über Kommandozeile
    st.header("Alternative: Installation über Kommandozeile")
    with st.expander("Für fortgeschrittene Benutzer"):
        st.markdown("""
        Der Shot-Plotter kann auch über die Kommandozeile gestartet werden:
        
        ```bash
        docker pull sindi98/shot-plotter-shot-plotter:v1.2
        docker run -d -p 8080:8080 --name shot-plotter sindi98/shot-plotter-shot-plotter:v1.2
        ```
        
        Nach der Installation kannst du den Shot-Plotter unter http://localhost:8080 aufrufen.
        """)

# Beispielablauf
elif page == "Beispielablauf":
    st.title("Beispielablauf")
    
    # Funktion zum Anzeigen von Bildern mit Beschreibung für den Beispielablauf
    def show_workflow_step(step_number, title, description):
        st.header(f"Schritt {step_number}: {title}")
        
        # Bild anzeigen
        image_path = f"images/workflow_{step_number}.png"
        if os.path.exists(image_path):
            st.image(image_path, use_column_width=True)
        else:
            st.error(f"Bild '{image_path}' nicht gefunden")
        
        # Beschreibung unter dem Bild
        st.markdown(description)
        st.markdown("---")  # Trennlinie
    
    # Schritt 1 des Workflows
    show_workflow_step(
        1, 
        "Aktion platzieren", 
        """
        1. Klicke auf das Spielfeld, um den Startpunkt einer Aktion zu markieren
        2. Halte die Shift-Taste gedrückt und klicke erneut, um den Endpunkt zu setzen (für Pass, Schuss, etc.)
        """
    )
    
    # Schritt 2 des Workflows
    show_workflow_step(
        2, 
        "Details eingeben", 
        """
        Im Seitenpanel kannst du folgende Informationen eintragen:
        
        1. **Halbzeit**: 1. oder 2. Halbzeit
        2. **Gegnerdruck**: Stärke des gegnerischen Drucks (0-4)
        3. **Outcome**: Erfolgreich oder Nicht Erfolgreich
        4. **Passhöhe**: Flach, Hoch oder Über Kniehöhe
        5. **Situation**: Aus dem Spiel, Anstoß, Abstoß, Ecke oder Freistoß
        6. **Zeit**: Spielzeit in Minuten:Sekunden
        7. **Aktionstyp**: Pass, Schuss, Dribbling, Kopfball oder Zweikampf
        """
    )
    
    # Schritt 3 des Workflows
    show_workflow_step(
        3, 
        "Aktion speichern", 
        """
        Klicke auf "Speichern", um die Aktion zu erfassen. Sie wird in der Tabelle unten angezeigt.
        """
    )
    
    # Komplettes Spiel analysieren
    st.header("Komplettes Spiel analysieren")
    show_workflow_step(
        4, 
        "Mehrere Aktionen verwalten", 
        """
        1. Erfasse nacheinander alle relevanten Aktionen
        2. Verwende die Tabelle unten, um bereits erfasste Aktionen zu überprüfen
        3. Nutze die Filter, um bestimmte Aktionstypen oder Situationen zu finden
        4. Exportiere die Daten am Ende für weitere Analysen
        """
    )

# Erklärung der Widgets
elif page == "Erklärung der Widgets":
    st.title("Erklärung der Widgets")
    
    st.markdown("""
    Die Widgets im Shot-Plotter ermöglichen die detaillierte Erfassung von Spielsituationen.
    Hier ist eine Erklärung aller verfügbaren Widgets:
    """)
    
    widgets = [
        {
            "name": "Halbzeit",
            "type": "Radio Button",
            "options": ["1", "2"],
            "description": "Gibt an, in welcher Halbzeit die Aktion stattgefunden hat."
        },
        {
            "name": "Team",
            "type": "Radio Button",
            "options": ["BVB", "Gegner"],
            "description": "Gibt an, welches Team die Aktion ausgeführt hat."
        },
        {
            "name": "Gegnerdruck",
            "type": "Radio Button",
            "options": ["0", "1", "2", "3", "4"],
            "description": "Gibt die Intensität des gegnerischen Drucks an. 0 = kein Druck, 4 = maximaler Druck."
        },
        {
            "name": "Outcome",
            "type": "Radio Button",
            "options": ["Erfolgreich", "Nicht Erfolgreich"],
            "description": "Gibt an, ob die Aktion erfolgreich war oder nicht."
        },
        {
            "name": "Passhöhe",
            "type": "Radio Button",
            "options": ["Flach", "Hoch", "Über Kniehöhe"],
            "description": "Gibt die Höhe des Passes an."
        },
        {
            "name": "Situation",
            "type": "Radio Button",
            "options": ["Aus dem Spiel", "Anstoß", "Abstoß", "Ecke", "Freistoß"],
            "description": "Gibt die Spielsituation an, aus der die Aktion entstanden ist."
        },
        {
            "name": "Zeit",
            "type": "Text",
            "description": "Ermöglicht die Eingabe der Spielzeit im Format mm:ss."
        },
        {
            "name": "Aktionstyp",
            "type": "Dropdown",
            "options": ["Pass", "Schuss", "Dribbling", "Kopfball", "Zweikampf"],
            "description": "Gibt die Art der Aktion an."
        }
    ]
    
    for widget in widgets:
        with st.expander(f"{widget['name']} ({widget['type']})"):
            st.markdown(f"**Beschreibung:** {widget['description']}")
            if "options" in widget:
                st.markdown("**Optionen:**")
                for option in widget["options"]:
                    st.markdown(f"- {option}")

# Extras
elif page == "Extras":
    st.title("Extras")
    
    st.header("Anpassen der Widgets")
    st.markdown("""
    Du kannst die verfügbaren Widgets und deren Optionen anpassen:
    
    1. Öffne die Einstellungen über das Zahnrad-Symbol
    2. Klicke auf "Create New Detail", um ein neues Widget hinzuzufügen
    3. Wähle den Widget-Typ (Radio Button, Dropdown, Text, etc.)
    4. Gib einen Titel und die verfügbaren Optionen ein
    5. Speichere die Änderungen
    
    Die Änderungen werden in der supported-sports.json-Datei gespeichert.
    """)
    
    st.header("Datenmerging")
    st.markdown("""
    Mit dem Datenmerging-Tool kannst du Daten aus dem Shot-Plotter mit anderen Datenquellen kombinieren:
    
    1. Exportiere die Daten aus dem Shot-Plotter als CSV
    2. Öffne das Datenmerging-Tool (verfügbar unter http://localhost:8000)
    3. Importiere die CSV-Datei und die zu verknüpfenden Daten
    4. Führe die Verknüpfung durch und exportiere das Ergebnis
    """)
    
    st.header("Tipps und Tricks")
    st.markdown("""
    ### Tastaturkürzel
    
    - **Shift + Klick**: Zwei-Punkt-Aktion (z.B. Pass von A nach B)
    - **ESC**: Aktuelle Aktion abbrechen
    
    ### Fehlerbehebung
    
    - **Browser-Cache leeren**: Bei Problemen mit der Anzeige
    - **Docker-Container neu starten**: Bei allgemeinen Problemen
    
    ```bash
    docker restart shot-plotter
    ```
    """)

# Download
elif page == "Download":
    st.title("Download")
    
    st.header("Shot-Plotter über Docker Desktop")
    st.markdown("""
    Der einfachste Weg, den Shot-Plotter zu installieren, ist über Docker Desktop:
    
    1. Öffne Docker Desktop
    2. Klicke auf den Tab "Images"
    3. Klicke auf "Pull" oder "Search on Docker Hub"
    4. Suche nach `sindi98/shot-plotter-shot-plotter`
    5. Wähle Tag `v1.2` aus und klicke auf "Pull"
    
    Eine detaillierte Anleitung mit Screenshots findest du im Bereich "How to start".
    """)
    
    if os.path.exists("images/docker_hub_pull.png"):
        st.image("images/docker_hub_pull.png", caption="Shot-Plotter über Docker Hub installieren", width=600)
    else:
        st.info("Bild 'docker_hub_pull.png' wird hier eingefügt")
    
    st.header("Systemanforderungen")
    st.markdown("""
    Für den Shot-Plotter benötigst du:
    
    - Docker Desktop installiert
    - Mind. 2 GB RAM
    - Mind. 1 GB freier Speicherplatz
    - Internetverbindung für den ersten Download
    """)
    
    st.header("Datenmerging-Tool herunterladen")
    st.markdown("""
    Das Datenmerging-Tool ist ebenfalls über Docker Hub verfügbar:
    
    1. Öffne Docker Desktop
    2. Klicke auf den Tab "Images"
    3. Klicke auf "Pull" oder "Search on Docker Hub"
    4. Suche nach `sindi98/shot-plotter-data-merging`
    5. Wähle das neueste Tag aus und klicke auf "Pull"
    6. Starte das Tool mit Port 8000
    """)
    
    st.header("Komplettes System mit Docker Compose")
    with st.expander("Für fortgeschrittene Benutzer"):
        st.markdown("""
        Du kannst auch alle Komponenten gleichzeitig mit Docker Compose starten.
        Erstelle dazu eine Datei namens `docker-compose.yml` mit folgendem Inhalt:
        
        ```yaml
        version: '3'
        
        services:
          shot-plotter:
            image: sindi98/shot-plotter-shot-plotter:v1.2
            container_name: shot-plotter
            ports:
              - "8080:8080"
            restart: unless-stopped
            networks:
              - shot-plotter-network
        
          data-merging:
            image: sindi98/shot-plotter-data-merging:latest
            container_name: data-merging
            ports:
              - "8000:8000"
            restart: unless-stopped
            networks:
              - shot-plotter-network
        
          streamlit-guide:
            image: sindi98/shot-plotter-guide:latest
            container_name: shot-plotter-guide
            ports:
              - "8501:8501"
            restart: unless-stopped
            networks:
              - shot-plotter-network
        
        networks:
          shot-plotter-network:
            driver: bridge
        ```
        
        Starte das System mit:
        
        ```bash
        docker-compose up -d
        ```
        """)
    
    st.header("Beispieldaten")
    st.markdown("""
    Hier kannst du Beispieldaten herunterladen, um den Shot-Plotter zu testen:
    
    - [Beispiel-CSV für Shot-Plotter](#) (Link wird später eingefügt)
    - [Beispiel-Datensatz für Datenmerging](#) (Link wird später eingefügt)
    """)

# Footer
st.markdown("---")
st.markdown("© 2024 BVB Kooperationsprojekt | Shot-Plotter v1.2 & Datenmerging") 