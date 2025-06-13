import streamlit as st
import base64
import os
from PIL import Image
import re

# Hilfsfunktion zum Laden von Bildern
def load_image(image_file):
    try:
        # Try to load the image from the Anleitung/images directory
        image_path = os.path.join("Anleitung", "images", image_file)
        if os.path.exists(image_path):
            return Image.open(image_path)
        else:
            st.warning(f"Image not found: {image_file}")
            return None
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

# Funktion zum Einbetten von Google Drive Videos
def embed_google_drive_video(video_url):
    """
    Konvertiert einen Google Drive Video-Link in einen einbettbaren iframe.
    
    Args:
        video_url (str): Der Google Drive Link zum Video
        
    Returns:
        str: HTML iframe Code zum Einbetten des Videos
    """
    # Extrahiere die Video-ID aus dem Google Drive Link
    file_id = None
    if 'drive.google.com/file/d/' in video_url:
        file_id = video_url.split('/file/d/')[1].split('/')[0]
    elif 'drive.google.com/open?id=' in video_url:
        file_id = video_url.split('id=')[1]
    
    if file_id:
        # Verwende die direkte Vorschau-URL mit Fokus auf Vollbildmodus
        embed_url = f'https://drive.google.com/file/d/{file_id}/preview?autoplay=0&fs=1'
        return f'<iframe src="{embed_url}" width="100%" height="720" allow="autoplay; fullscreen" style="border: none;"></iframe>'
    else:
        st.error("Invalid Google Drive Link")
        return None

# Funktion zum Einbetten von YouTube Videos
def embed_youtube_video(video_url):
    """
    Konvertiert einen YouTube Video-Link in einen einbettbaren iframe.
    
    Args:
        video_url (str): Der YouTube Link zum Video
        
    Returns:
        str: HTML iframe Code zum Einbetten des Videos
    """
    # Extrahiere die Video-ID aus dem YouTube Link
    video_id = None
    if 'youtube.com/watch?v=' in video_url:
        video_id = video_url.split('watch?v=')[1].split('&')[0]
    elif 'youtu.be/' in video_url:
        video_id = video_url.split('youtu.be/')[1].split('?')[0]
    
    if video_id:
        # Fokus auf Vollbildmodus und Qualität
        embed_url = f'https://www.youtube.com/embed/{video_id}?fs=1&hd=1'
        return f'<iframe src="{embed_url}" width="100%" height="720" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; fullscreen" allowfullscreen style="border: none;"></iframe>'
    else:
        st.error("Invalid YouTube Link")
        return None

# Funktion zum Anzeigen von Workflow-Schritten mit Videos
def show_workflow_step(step_number, title, description, video_url=None, video_type=None):
    """
    Zeigt einen Workflow-Schritt mit optionalem Video an.
    
    Args:
        step_number (int): Die Nummer des Schritts
        title (str): Der Titel des Schritts
        description (str): Die Beschreibung des Schritts
        video_url (str, optional): URL zum Video
        video_type (str, optional): Typ des Videos ('drive' oder 'youtube')
    """
    st.header(f"{step_number}. {title}")
    st.markdown(description)
    
    if video_url and video_type:
        if video_type == "drive":
            st.components.v1.html(embed_google_drive_video(video_url), height=720)
        elif video_type == "youtube":
            st.components.v1.html(embed_youtube_video(video_url), height=720)
    
    st.markdown("---")  # Trennlinie

# Konfiguration der Seite
st.set_page_config(
    page_title="Shot-Plotter Guide",
    page_icon="🎯",
    layout="wide"
)

# Sidebar für Navigation
st.sidebar.title("Shot-Plotter Guide")
page = st.sidebar.radio(
    "Navigation",
    ["How to start", "Example Workflow", "Widget Explanation", "Download"]
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
    Here you will learn how to start and set up the Shot-Plotter using Docker Desktop.
    """)
    
    # Funktion zum Anzeigen von Bildern mit Beschreibung
    def show_step(step_number, title, description_before, description_after):
        st.header(f"Step {step_number}: {title}")
        
        # Beschreibung vor dem Bild
        st.markdown(description_before)
        
        # Bild anzeigen
        image = load_image(f"{step_number}.png")
        if image:
            st.image(image, use_container_width=True)
        else:
            st.error(f"Bild für Schritt {step_number} konnte nicht geladen werden")
        
        # Beschreibung nach dem Bild
        st.markdown(description_after)
        st.markdown("---")  # Trennlinie
    
    # Schritt 1
    show_step(
        1, 
        "Open Docker Desktop", 
        """
        If you haven't installed **Docker Desktop** yet, you can download it here:
        
        [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)

        (Make sure to download the correct version for your operating system)

        Create a Docker account if you don't have one yet and sign in to Docker Desktop.
        """,
        """
        Click on **DockerHub** on the left and search for **sindi98**
        """
    )
    
    # Schritt 2
    show_step(
        2, 
        "Search for Shot-Plotter", 
        """
        In the search results list, you'll find various projects:

        Click on the project **sindi98/shot-plotter-shot-plotter**. (The name is not fully visible, but you can recognize which project it is)
        """,
        """
        
        """
    )
    
    # Schritt 3
    show_step(
        3, 
        "Select the correct image", 
        """
        On the project page, you'll find different **versions** of the program:

        In the top right, under **Tag**, you'll find the program versions, select **v1.2**.

        Click on **Pull** afterwards
        
        *Note: The download may take several minutes depending on your internet connection.*
        """,
        """
        """
    )
    
    # Schritt 4
    show_step(
        4, 
        "Switch to Images page", 
        """
        After successfully downloading the image:

        Click on **Images** in the left sidebar
        """,
        """
        
        """
    )
    
    # Schritt 5
    show_step(
        5, 
        "Configure container", 
        """
        You should now see an image named **sindi98/shot-plotter-shot-plotter**. The tag **v1.2** should also be displayed.

        Now click on the blue **Run** button (marked in green in this image)
        """,
        """
        *Note: Next to the name shot-plotter-shot-plotter is a small circle that is not filled yet. Later in the process, this should be green.*
        """
    )

    # Schritt 6
    show_step(
        6, 
        "Container Settings", 
        """
        Open **"Optional Settings"** by clicking on the white arrow.
        """,
        """
        """
    )

    # Schritt 7
    show_step(
        7, 
        "Start container", 
        """
        Set a name for the container. Preferably name it **shot-plotter**.

        Select **8080:8080** as the port by entering **8080** in the input field.

        Click on **Run**
        """,
        """
        """
    )

    # Schritt 8
    show_step(
        8, 
        "Open Shot-Plotter in browser", 
         """
        The container should now be started. 

        Now click on the displayed port. In the image, it's outlined in red. 

        *Note: In the image, the port is 7001, but you should see 8080 if you used the default settings.*

        You should now see the Shot-Plotter page.
        """,
        """
        """
    )

    # Schritt 9
    show_step(
        9,
        "Open Shot-Plotter in browser",
        """
        You can always access the Shot-Plotter page at http://localhost:8080 in your browser. (If you used the default settings)

        *Note: Safari and Firefox are recommended browsers, Chrome may cause issues.*

        Alternatively, you can open Docker Desktop and find the Shot-Plotter under **"Containers"**, as shown in the image.
        Go to **"Containers"** in Docker Desktop. The Shot-Plotter should be visible under **"shot-plotter"**. When you click on the port number, the Shot-Plotter will open in your browser.

        *Note: The dot next to the container name should be green when the container is running. If it's not filled and gray, you need to repeat the steps.*
        """,
        """
        """
    )

# Example Workflow
elif page == "Example Workflow":
    st.title("Example Workflow")
    
    # Allgemeines Setup
    show_workflow_step(
        1, 
        "General Setup", 
        """
        In this section, you'll learn how to set up and configure the Shot-Plotter for your analysis:

        1. Basic configuration settings
        2. Field setup and customization
        3. Initial settings for your analysis
        """,
        video_url="https://drive.google.com/file/d/1Sard0QX_EiRhO3mtN52QVg-MzfLNkoiO/view?usp=drive_link",
        video_type="drive"
    )
    
    # Location und Timing des Passes
    show_workflow_step(
        2, 
        "Location und Timing of passes", 
        """
        Learn how to accurately record pass locations and timing:

        1. How to mark pass starting and ending points
        2. Recording the exact timing of passes
        3. Tips for precise location marking
        """,
        video_url="https://drive.google.com/file/d/1qwz1p9AZaLtSpx7_jyE3gf14O_88tltE/view?usp=drive_link",
        video_type="drive"
    )

    # Spielrichtung und gespiegelte Koordinaten
    show_workflow_step(
        3,
        "Playing Direction and Mirrored Coordinates",
        """
        Understanding and handling different playing directions:

        1. Default playing direction is from left to right
        2. When the game is played from right to left, use the "Gespiegelte Koordinaten" button
        3. This ensures all data is consistently oriented for analysis
        """,
        video_url="https://drive.google.com/file/d/1M9q7B6rdaEzV-qR-WM3Q1D382NYd-Iru/view?usp=drive_link",
        video_type="drive"
    )

# Widget Explanation
elif page == "Widget Explanation":
    st.title("Widget Explanation")
    
    st.markdown("""
    The widgets in Shot-Plotter enable detailed recording of game situations.
    Here is an explanation of all available widgets with video tutorials:
    """)
    
    # Gegnerdruck
    show_workflow_step(
        1,
        "Gegnerdruck",
        """
        Learn how to assess and record opponent pressure:

        1. Understanding pressure levels (0-4)
        2. How to evaluate pressure situations
        3. Tips for consistent pressure assessment
        """,
        video_url="https://drive.google.com/file/d/1LoD-_ej1sxA3JzCmbZqudO5s7kar8Vqi/view?usp=drive_link",
        video_type="drive"
    )
    
    # Outcome
    show_workflow_step(
        2,
        "Outcome",
        """
        Understanding how to record action outcomes:

        1. Defining successful vs. unsuccessful actions
        2. Criteria for outcome assessment
        3. Examples of different outcomes
        """,
        video_url="https://drive.google.com/file/d/1UYiFCwREcc1iTMeGpAn5ydRKuyEnyDGH/view?usp=drive_link",
        video_type="drive"
    )
    
    # Aktionstyp
    show_workflow_step(
        3,
        "Action Type",
        """
        Explanation of different action types

        """,
        video_url="https://drive.google.com/file/d/14KoviY4fIZHljKfGwbB6grmx4p4b0GH_/view?usp=drive_link",
        video_type="drive"
    )
    
    # Passhöhe
    show_workflow_step(
        4,
        "Passhöhe",
        """
        How to record pass heights accurately:

        1. Define the pass height
        2. Explain the different pass heights
        """,
        video_url="https://drive.google.com/file/d/1YqhLkeUpoepsXrN75hPQv0_KpOQB_H5O/view?usp=drive_link",
        video_type="drive"
    )
    
    # Feldgröße
    show_workflow_step(
        5,
        "Feldgröße",
        """
        Understanding field size settings and their impact:

        1. How to set up field dimensions
        2. Impact on analysis
        3. Best practices for different field sizes
        """,
        video_url="https://drive.google.com/file/d/1xuEg--KBDVLgLU65oevoi6hNp5dP0I1n/view?usp=drive_link",
        video_type="drive"
    )
    
    # Rest of the widget explanations
    widgets = [
        {
            "name": "Half",
            "type": "Radio Button",
            "options": ["1", "2"],
            "description": "Indicates in which half the action took place."
        },
        {
            "name": "Team",
            "type": "Radio Button",
            "options": ["BVB", "Opponent"],
            "description": "Indicates which team executed the action. You only tag passes from BVB so you can only select BVB."
        },
        {
            "name": "Situation",
            "type": "Radio Button",
            "options": ["From Play", "Kick-off", "Goal Kick", "Corner", "Free Kick"],
            "description": "Indicates the game situation from which the action originated."
        },
        {
            "name": "Time",
            "type": "Text",
            "description": "Allows entering the game time in mm:ss format."
        }
    ]
    
    for widget in widgets:
        with st.expander(f"{widget['name']} ({widget['type']})"):
            st.markdown(f"**Description:** {widget['description']}")
            if "options" in widget:
                st.markdown("**Options:**")
                for option in widget["options"]:
                    st.markdown(f"- {option}")

# Extras
# elif page == "Extras":
#     st.title("Extras")
#     
#     st.header("Customizing Widgets")
#     st.markdown("""
#     You can customize the available widgets and their options:
#     
#     1. Open settings via the gear icon
#     2. Click on "Create New Detail" to add a new widget
#     3. Choose the widget type (Radio Button, Dropdown, Text, etc.)
#     4. Enter a title and available options
#     5. Save the changes
#     
#     The changes will be saved in the supported-sports.json file.
#     """)
#     
#     st.header("Data Merging")
#     st.markdown("""
#     With the data merging tool, you can combine data from Shot-Plotter with other data sources:
#     
#     1. Export the data from Shot-Plotter as CSV
#     2. Open the data merging tool (available at http://localhost:8000)
#     3. Import the CSV file and the data to be linked
#     4. Perform the linking and export the result
#     """)
#     
#     st.header("Tips and Tricks")
#     st.markdown("""
#     ### Keyboard Shortcuts
#     
#     - **Shift + Click**: Two-point action (e.g., pass from A to B)
#     - **ESC**: Cancel current action
#     
#     ### Troubleshooting
#     
#     - **Clear Browser Cache**: For display issues
#     - **Restart Docker Container**: For general issues
#     
#     ```bash
#     docker restart shot-plotter
#     ```
#     """)

# Download
elif page == "Download":
    st.title("Download")
    
    # Shot-Plotter Installation
    show_workflow_step(
        1,
        "Data Download",
        """
        Click on the download Button below the Video. Name it like the name of the current video.
        """,
        video_url="https://drive.google.com/file/d/1fzYCkfbXEC-RmbX8mO-F1JuAzLdn60nu/view?usp=drive_link",
        video_type="drive"
    )
    

# Footer
st.markdown("---")
st.markdown("© 2024 BVB Cooperation Project | Shot-Plotter v1.2 & Data Merging") 