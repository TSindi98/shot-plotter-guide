import streamlit as st
import base64
import os
from PIL import Image
import re

# Konfiguration der Seite
st.set_page_config(
    page_title="Shot-Plotter Guide",
    page_icon="🎯",
    layout="wide"
)

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
        # Verwende die direkte Vorschau-URL mit höherer Qualität
        embed_url = f'https://drive.google.com/file/d/{file_id}/preview?autoplay=0&hd=1'
        return f'<iframe src="{embed_url}" width="100%" height="720" allow="autoplay" style="border: none;"></iframe>'
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
        embed_url = f'https://www.youtube.com/embed/{video_id}'
        return f'<iframe src="{embed_url}" width="100%" height="720" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen style="border: none;"></iframe>'
    else:
        st.error("Invalid YouTube Link")
        return None

# Sidebar für Navigation
st.sidebar.title("Shot-Plotter Guide")
page = st.sidebar.radio(
    "Navigation",
    ["How to start", "Example Workflow", "Widget Explanation", "Extras", "Download"]
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
    
    # Funktion zum Anzeigen von Bildern mit Beschreibung für den Beispielablauf
    def show_workflow_step(step_number, title, description, video_url=None, video_type="drive"):
        st.header(f"{title}")
        
        # Video anzeigen, falls ein Link vorhanden ist
        if video_url:
            if video_type == "youtube":
                video_html = embed_youtube_video(video_url)
            else:  # default to drive
                video_html = embed_google_drive_video(video_url)
            if video_html:
                st.components.v1.html(video_html, height=480)
        else:
            # Bild anzeigen, wenn kein Video vorhanden ist
            image_path = f"images/workflow_{step_number}.png"
            if os.path.exists(image_path):
                st.image(image_path, use_container_width=True)
            else:
                st.error(f"Image '{image_path}' not found")
        
        # Beschreibung unter dem Video/Bild
        st.markdown(description)
        st.markdown("---")  # Trennlinie
    
    # Schritt 1 des Workflows
    show_workflow_step(
        1, 
        "Place Action", 
        """
        This video shows how to place an action on the field:

        1. Click on the field to mark the starting point of an action
        2. Hold the Shift key and click again to set the end point (for pass, shot, etc.)
        """,
        video_url="https://drive.google.com/file/d/1Sard0QX_EiRhO3mtN52QVg-MzfLNkoiO/view?usp=share_link"
    )
    
    # Schritt 2 des Workflows
    show_workflow_step(
        2, 
        "Enter Details", 
        """
        In the side panel, you can enter the following information:
        
        1. **Half**: 1st or 2nd half
        2. **Opponent Pressure**: Strength of opponent pressure (0-4)
        3. **Outcome**: Successful or Not Successful
        4. **Pass Height**: Flat, High, or Above Knee Height
        5. **Situation**: From Play, Kick-off, Goal Kick, Corner, or Free Kick
        6. **Time**: Game time in minutes:seconds
        7. **Action Type**: Pass, Shot, Dribbling, Header, or Duel
        """,
        # Here you can add the video link for step 2
        # video_url="YOUR_VIDEO_LINK_FOR_STEP_2"
    )

    # Schritt 3 des Workflows
    show_workflow_step(
        3,
        "Advanced Features",
        """
        This video demonstrates some advanced features of the Shot-Plotter:

        1. How to use filters to analyze specific situations
        2. How to export your data for further analysis
        3. Tips and tricks for efficient data collection
        """,
        video_url="https://youtu.be/boyOUWLbDII",
        video_type="youtube"
    )

# Widget Explanation
elif page == "Widget Explanation":
    st.title("Widget Explanation")
    
    st.markdown("""
    The widgets in Shot-Plotter enable detailed recording of game situations.
    Here is an explanation of all available widgets:
    """)
    
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
            "description": "Indicates which team executed the action."
        },
        {
            "name": "Opponent Pressure",
            "type": "Radio Button",
            "options": ["0", "1", "2", "3", "4"],
            "description": "Indicates the intensity of opponent pressure. 0 = no pressure, 4 = maximum pressure."
        },
        {
            "name": "Outcome",
            "type": "Radio Button",
            "options": ["Successful", "Not Successful"],
            "description": "Indicates whether the action was successful or not."
        },
        {
            "name": "Pass Height",
            "type": "Radio Button",
            "options": ["Flat", "High", "Above Knee Height"],
            "description": "Indicates the height of the pass."
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
        },
        {
            "name": "Action Type",
            "type": "Dropdown",
            "options": ["Pass", "Shot", "Dribbling", "Header", "Duel"],
            "description": "Indicates the type of action."
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
elif page == "Extras":
    st.title("Extras")
    
    st.header("Customizing Widgets")
    st.markdown("""
    You can customize the available widgets and their options:
    
    1. Open settings via the gear icon
    2. Click on "Create New Detail" to add a new widget
    3. Choose the widget type (Radio Button, Dropdown, Text, etc.)
    4. Enter a title and available options
    5. Save the changes
    
    The changes will be saved in the supported-sports.json file.
    """)
    
    st.header("Data Merging")
    st.markdown("""
    With the data merging tool, you can combine data from Shot-Plotter with other data sources:
    
    1. Export the data from Shot-Plotter as CSV
    2. Open the data merging tool (available at http://localhost:8000)
    3. Import the CSV file and the data to be linked
    4. Perform the linking and export the result
    """)
    
    st.header("Tips and Tricks")
    st.markdown("""
    ### Keyboard Shortcuts
    
    - **Shift + Click**: Two-point action (e.g., pass from A to B)
    - **ESC**: Cancel current action
    
    ### Troubleshooting
    
    - **Clear Browser Cache**: For display issues
    - **Restart Docker Container**: For general issues
    
    ```bash
    docker restart shot-plotter
    ```
    """)

# Download
elif page == "Download":
    st.title("Download")
    
    st.header("Shot-Plotter via Docker Desktop")
    st.markdown("""
    The easiest way to install Shot-Plotter is via Docker Desktop:
    
    1. Open Docker Desktop
    2. Click on the "Images" tab
    3. Click on "Pull" or "Search on Docker Hub"
    4. Search for `sindi98/shot-plotter-shot-plotter`
    5. Select tag `v1.2` and click on "Pull"
    
    You can find a detailed guide with screenshots in the "How to start" section.
    """)
    
    st.header("System Requirements")
    st.markdown("""
    For Shot-Plotter you need:
    
    - Docker Desktop installed
    - Min. 2 GB RAM
    - Min. 1 GB free storage space
    - Internet connection for initial download
    """)
    
    st.header("Download Data Merging Tool")
    st.markdown("""
    The data merging tool is also available via Docker Hub:
    
    1. Open Docker Desktop
    2. Click on the "Images" tab
    3. Click on "Pull" or "Search on Docker Hub"
    4. Search for `sindi98/shot-plotter-data-merging`
    5. Select the latest tag and click on "Pull"
    6. Start the tool with port 8000
    """)
    
    st.header("Complete System with Docker Compose")
    with st.expander("For advanced users"):
        st.markdown("""
        You can also start all components simultaneously with Docker Compose.
        Create a file named `docker-compose.yml` with the following content:
        
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
        
        Start the system with:
        
        ```bash
        docker-compose up -d
        ```
        """)
    
    st.header("Sample Data")
    st.markdown("""
    Here you can download sample data to test Shot-Plotter:
    
    - [Sample CSV for Shot-Plotter](#) (Link will be added later)
    - [Sample dataset for data merging](#) (Link will be added later)
    """)

# Footer
st.markdown("---")
st.markdown("© 2024 BVB Cooperation Project | Shot-Plotter v1.2 & Data Merging") 