import streamlit as st
import os
from PIL import Image

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_image(image_file):
    """Loads an image from the images folder."""
    try:
        image_path = os.path.join("Anleitung", "images", image_file)
        if os.path.exists(image_path):
            return Image.open(image_path)
        else:
            # Fallback: search directly in images folder
            image_path = os.path.join("images", image_file)
            if os.path.exists(image_path):
                return Image.open(image_path)
            else:
                st.warning(f"Image not found: {image_file}")
                return None
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def load_video(video_file):
    """Loads a video from the videos folder."""
    try:
        video_path = os.path.join("videos", video_file)
        if os.path.exists(video_path):
            return video_path
        else:
            st.warning(f"Video not found: {video_file}")
            return None
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
        return None

def embed_google_drive_video(video_url):
    """Converts a Google Drive video link into an embeddable iframe."""
    file_id = None
    if 'drive.google.com/file/d/' in video_url:
        file_id = video_url.split('/file/d/')[1].split('/')[0]
    elif 'drive.google.com/open?id=' in video_url:
        file_id = video_url.split('id=')[1]

    if file_id:
        embed_url = f'https://drive.google.com/file/d/{file_id}/preview?autoplay=0&fs=1'
        return f'<iframe src="{embed_url}" width="100%" height="480" allow="autoplay; fullscreen" style="border: none;"></iframe>'
    else:
        st.error("Invalid Google Drive Link")
        return None

def show_image_placeholder(name, description=""):
    """Shows a placeholder for a missing image."""
    st.info(f"📷 **Screenshot needed:** {name}")
    if description:
        st.caption(description)

def show_video_placeholder(name, description=""):
    """Shows a placeholder for a missing video."""
    st.warning(f"🎬 **Video needed:** {name}")
    if description:
        st.caption(description)

def show_step_with_image(step_number, title, description_before, image_file=None, description_after=""):
    """Displays a step with an optional image."""
    st.markdown("---")
    st.header(f"Step {step_number}: {title}")
    st.markdown(description_before)

    if image_file:
        image = load_image(image_file)
        if image:
            st.image(image, use_container_width=True)
        else:
            show_image_placeholder(image_file)

    if description_after:
        st.markdown(description_after)

def show_image(image_file, caption="", width=None, center=False):
    """Shows an image with a caption.

    Args:
        image_file: Name of the image file
        caption: Caption text
        width: Optional width in pixels. If None, uses full container width.
        center: If True and width is set, centers the image.
    """
    image = load_image(image_file)
    if image:
        if width and center:
            # Use columns to center the image
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, width=width, caption=caption)
        elif width:
            st.image(image, width=width, caption=caption)
        else:
            st.image(image, use_container_width=True, caption=caption)
    else:
        show_image_placeholder(image_file)

def show_video(video_file, caption=""):
    """Shows a video with a caption."""
    video = load_video(video_file)
    if video:
        st.video(video)
    else:
        show_video_placeholder(video_file)
    if caption:
        st.caption(caption)
# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Shot-Plotter Guide",
    page_icon="⚽",
    layout="wide"
)

# =============================================================================
# CUSTOM CSS - BVB STYLING
# =============================================================================

st.markdown("""
<style>
    /* BVB Yellow for main titles (h1 = st.title) and headers (h2 = st.header) */
    h1, h2 {
        color: #ffc400 !important;
    }

    /* Sidebar title also in BVB Yellow */
    [data-testid="stSidebar"] h1 {
        color: #ffc400 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

st.sidebar.title("Shot-Plotter Guide")
st.sidebar.markdown("---")

# Styled Navigation header
st.sidebar.markdown('<p style="font-size: 1.2em; font-weight: bold; color: #ffc400;">Navigation</p>', unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    [
        "1. Installation (Docker)",
        "2. User Interface",
        "3. Basic Functions",
        "4. Customize Widgets",
        "5. Keyboard Shortcuts",
        "6. Data Management",
        "7. Settings"
    ],
    label_visibility="collapsed"
)


# =============================================================================
# PAGE 1: INSTALLATION (DOCKER)
# =============================================================================

if page == "1. Installation (Docker)":
    st.title("Installation with Docker Desktop")

    st.markdown("""
    This guide shows you how to install and start the Shot-Plotter using Docker Desktop.
    Docker allows you to run the application on any computer without complicated setup.
    """)

    # Step 1
    show_step_with_image(
        1,
        "Install Docker Desktop",
        """
        If you haven't installed **Docker Desktop** yet, download it here:

        👉 [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)

        **Important:**
        - Choose the correct version for your operating system (Windows/Mac)
        - Create a Docker account and sign in
        - Verify your email address (otherwise you'll get errors later)

        After installation, open Docker Desktop and click on **DockerHub** on the left.
        """,
        "1.png",
        "Search for **sindi98** there"
    )

    # Step 2
    show_step_with_image(
        2,
        "Find Shot-Plotter",
        """
        In the search results, you'll find various projects.

        Click on **sindi98/shot-plotter-shot-plotter**.
        """,
        "2.png"
    )

    # Step 3
    show_step_with_image(
        3,
        "Select the correct version",
        """
        On the project page, you'll find different versions:

        1. Select version **v2.2** under **Tag** in the top right
        2. Click on **Pull**

        *The download may take several minutes depending on your internet connection.*
        """,
        "3.png"
    )

    # Step 4
    show_step_with_image(
        4,
        "Switch to Images",
        """
        After the successful download:

        Click on **Images** in the left sidebar
        """,
        "4.png"
    )

    # Step 5
    show_step_with_image(
        5,
        "Configure container",
        """
        You should now see an image named **sindi98/shot-plotter-shot-plotter** with the tag **v2.2**.

        Click on the blue **Run** button.
        """,
        "5.png",
        "*The small circle next to the name should turn green later when the container is running.*"
    )

    # Step 6
    show_step_with_image(
        6,
        "Open Optional Settings",
        """
        Click on **"Optional Settings"** (the white arrow) to open the settings.
        """,
        "6.png"
    )

    # Step 7
    show_step_with_image(
        7,
        "Start container",
        """
        Configure the container:

        1. **Name:** Give the container a name, e.g. `shot-plotter`
        2. **Port:** Enter `8080` in the port field (for 8080:8080)
        3. Click on **Run**
        """,
        "7.png"
    )

    # Step 8
    show_step_with_image(
        8,
        "Open Shot-Plotter in browser",
        """
        The container should now be running.

        Click on the displayed port (e.g. `8080:8080`) to open the Shot-Plotter in your browser.
        """,
        "8.png"
    )

    # Step 9
    show_step_with_image(
        9,
        "Future access",
        """
        You can access the Shot-Plotter anytime:

        **Option 1:** Directly in browser: `http://localhost:8080`

        **Option 2:** Via Docker Desktop:
        1. Go to **Containers**
        2. Find **shot-plotter**
        3. Click on the port link

        *The dot next to the container name should be green. If it's gray, restart the container.*

        **Browser recommendation:** Safari or Firefox (Chrome may cause issues with arrow keys or shortcuts)
        """,
        "9.png"
    )

# =============================================================================
# PAGE 2: USER INTERFACE
# =============================================================================

elif page == "2. User Interface":
    st.title("User Interface - Overview")

    st.markdown("""
    The Shot-Plotter consists of several areas that work together.
    Here you'll learn where to find what.
    """)

    # Overview image
    st.header("Complete Overview")
    show_image("ui-overview.png", "Screenshot of the entire user interface with numbered areas")

    st.markdown("""
    The interface is divided into the following areas:
    """)

    # Area 1: Header
    st.subheader("① Header Area")
    st.markdown("""
    - **Field selection:** Dropdown for "Entire Field" or "U13 (70x50m)"
    - **Reset Settings:** Button to reset all settings and go back to the default settings. It also deletes all recorded data!
    """)

    # Area 2: Settings
    st.subheader("② Settings Bar")
    st.markdown("""
    Here you can configure basic parameters:
    - **Width:** Field width in meters (90-120m)
    - **Height:** Field height in meters (60-80m)
    - **Skip Time:** Seconds to skip forward/backward in video with arrow keys (1-30s)
    - **Apply Settings:** Applies the changes
    """)

    # Area 3: Sidebar
    st.subheader("③ Details Sidebar (left)")
    st.markdown("""
    The left sidebar contains all **widgets** for data entry:
    - **Time:** Timestamp (linked to video)
    - **Team:** BVB or Opponent (or if any other team is playing you can change the options in the Customize Details)
    - **Half:** 1 or 2
    - **Default Options:** Widgets like Action Type, Outcome, etc. are just default options that can be changed in the Customize Details. You can also create your own widgets.
    - **Customize Details:** Button to customize widgets
    
    At the bottom of the sidebar you'll find:
    - **Customize Setup:** Button to customize widgets
    """)

    # Area 4: Playing Field
    st.subheader("④ Playing Field")
    st.markdown("""
    The interactive playing field:
    - **Click:** Records a position
    - **Shift + Click:** Two-point action (e.g., pass from A to B)
    - **Dots/Arrows:** Show recorded actions
    - **Colors:** Distinguish teams (Yellow = BVB, Grey = Opponent)
    """)

    # Area 5: Video
    st.subheader("⑤ Video Area")
    st.markdown("""
    Below the playing field:
    - **Video Upload:** Upload a game video
    - **Video Player:** Control the video
    - **"Open video in new window":** Opens the video in a separate window (useful for dual-monitor setups)
    """)

    # Area 6: Table
    st.subheader("⑥ Data Table")
    st.markdown("""
    Below the playing field:
    - Shows all recorded actions as a table
    - Each row = one action
    - Checkbox on the left for selecting/deleting
    """)

    # Area 7: Export
    st.subheader("⑦ Export Area")
    st.markdown("""
    At the very bottom:
    - **Download CSV:** Exports all data as CSV file
    - **Upload CSV:** Imports existing data
    - **Reset Settings:** Resets all settings (top right)
    """)


# =============================================================================
# PAGE 3: BASIC FUNCTIONS
# =============================================================================

elif page == "3. Basic Functions":
    st.title("Basic Functions")

    st.markdown("""
    Here you'll learn the most important functions of the Shot-Plotter.

    Use Safari or Firefox for the best experience. Chrome may cause issues with arrow keys or shortcuts.
    """)
    st.markdown("---")
    # Video Control
    st.header("Video Control")

    st.subheader("Upload Video")
    st.markdown("""
    1. Click on **"Choose file"** below the playing field
    2. Select your video file (MP4, MOV, etc.)
    3. The video appears in the player
    """)

    st.info("""
    **Tip:** You can also drag and drop the video file into the playing field.
    """)

    st.subheader("Video in Separate Window")
    st.markdown("""
    For dual-monitor setups:

    1. Upload the video
    2. Click on **"Open video in new window"**
    3. Drag the window to the second monitor
    4. Both windows are **synchronized** (Play/Pause, Position)
    """)


    st.subheader("Keyboard Control")
    st.markdown("""
    | Key | Function |
    |-----|----------|
    | **Spacebar** | Play / Pause |
    | **Right Arrow** | Skip forward (Skip Time) |
    | **Left Arrow** | Skip backward (Skip Time) |

    *You can adjust the Skip Time in the settings above (Default: 3 seconds)*
    """)

    st.markdown("---")

    # Timer
    st.header("Time Widget")

    st.markdown("""
    The Time widget shows the current game time:

    - **Format:** mm:ss (e.g., 12:34)
    - **Automatic:** When a video is playing, the video time is used
    - **Manual:** You can also enter the time manually

    For a two-point action, the time of the **first click** is saved.
    """)

    st.markdown("---")

    # Playing Direction
    st.header("Consider Playing Direction")

    st.markdown("""
    """)

    st.warning("**Important:** Before the first click, you must select the playing direction!")

    st.markdown("""
    """)
    st.markdown("""
    """)

    show_image("coords-mirrored.png", "Screenshot: Mirrored Coordinates selected", width=500, center=True)
    

    st.markdown("""
    """)
    st.markdown("""
    """)

    _, col1, col2, _ = st.columns([1, 2, 2, 1])

    with col1:
        st.subheader("Normal Coordinates")
        st.markdown("""
        Select **"Normal Coordinates"** when:¢
        - Your team plays from **left to right**
        """)

    with col2:
        st.subheader("Mirrored Coordinates")
        st.markdown("""
        Select **"Mirrored Coordinates"** when:
        - Your team plays from **right to left**
        """)

    st.markdown("""
    """)
    st.markdown("""
    """)
    st.markdown("""
    """)

    st.info("""
    **Why is this important?**

    The coordinates are saved so that all actions can be analyzed uniformly.
    Regardless of which direction is played - in the data, your team always plays "from left to right".
    """)

    st.markdown("---")

    # Single-Point vs Two-Point
    st.header("Recording Actions")

    st.warning("""
    **Important:** All information which are selected in the sidebar will be saved in the moment you click on the playing field. So make sure at first to select all the information you want to save and than confirm your selection by clicking on the playing field. If 2-Location actions are selected, the second point will confirm the action.
    """)
    st.markdown("---")

    show_video("two-point-action.mov", "Video: Demonstrate difference between single-point and two-point action")

    st.markdown("---")

    st.subheader("Single-Point Action")
    st.markdown("""
    For actions at a single position:

    1. Select the desired widget values (Team, Situation, etc.)
    2. Click on the position on the playing field
    3. The action is recorded and appears in the table

    """)


    st.subheader("Two-Location Action")
    st.markdown("""
    For actions with start and end points:

    1. Select the desired widget values
    2. Hold **Shift** and click on the **start point**
    3. A "ghost point" appears
    4. Hold **Shift** and click on the **end point**
    5. An arrow shows the action
    6. The second point will confirm the action and the action is recorded and appears in the table

    """)

    st.info("""
    You can also select the 2-Location action by clicking on the toggle above the playing field. Then you dont need to hold the Shift key anymore.
    """)




# =============================================================================
# PAGE 4: CUSTOMIZE WIDGETS
# =============================================================================

elif page == "4. Customize Widgets":
    st.title("Customize Widgets")

    st.markdown("""
    You can completely customize the widgets in the sidebar to your needs.
    This way you can create your own categories or modify existing ones.
    """)

    st.markdown("---")

    show_video("create-widget.mov", "Video: Create and change widgets")

    st.markdown("---")

    # Open Customize Details
    st.header("Open Customize Setup")

    st.markdown("""
    1. Scroll down in the left sidebar
    2. Click on **"Customize Setup"**
    3. A modal window opens
    """)

    st.markdown("---")

    # Widget Types
    st.header("Widget Types")

    st.markdown("There are different widget types for different inputs:")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Radio Buttons")
        st.markdown("""
        - Select one option from several
        - Good for few options (2-5)
        - Example: Half (1, 2)
        """)

        st.subheader("Dropdown")
        st.markdown("""
        - Select one option from a list
        - Good for many options
        - Example: Player selection
        """)

    with col2:
        st.subheader("Text Field")
        st.markdown("""
        - Free text input
        - For notes or names
        - Example: Comment
        """)

        st.subheader("Time Widget")
        st.markdown("""
        - Time input in mm:ss format
        - Linked to video
        - Only once per setup
        """)

    st.markdown("---")

    # Create New Widget
    st.header("Create New Widget")

    st.markdown("""
    1. Open **Customize Setup**
    2. Click on **"Create New Detail"**
    3. Select the **widget type**
    4. Enter a **title** (e.g., "Player Position")
    5. Add **options** (for Radio/Dropdown)
    6. Optionally add a **shortcut** (for Radio/Dropdown) - see **5. Keyboard Shortcuts** in the sidebar for more info
    7. Click on **"Save"**
    """)

    st.markdown("---")

    # Edit Widget
    st.header("Edit Widget")

    st.markdown("""
    1. Open **Customize Setup**
    2. Find the widget in the list
    3. Click on the **Edit icon** (pencil)
    4. Change title or options
    5. Click on **"Save"**
    """)

    st.markdown("---")

    # Delete Widget
    st.header("Delete Widget")

    st.markdown("""
    1. Open **Customize Setup**
    2. Find the widget in the list
    3. Click on the **Delete icon** (trash can)
    4. Confirm deletion

    **Caution:** Deleted widgets cannot be restored!
    """)


    st.markdown("---")

    # JSON Export/Import
    st.header("Share Setup (JSON Export/Import)")

    st.success("""
    **Tip:** You can export your widget setup as a JSON file and share it with others.
    This way everyone has the same setup!
    """)

    show_image("json-export-import.png", "Screenshot: JSON Export/Import Buttons", width=700, center=True)


    st.header("Export Setup")
    st.markdown("""
    1. Open **Customize Setup**
    2. Click on **"Download JSON"** (or similar)
    3. The file will be downloaded
    4. Share the file with your team
    """)

    st.header("Import Setup")
    st.markdown("""
    1. Open **Customize Setup**
    2. Click on **"Upload JSON"** (or similar)
    3. Select the JSON file
    4. The widgets will be imported

    """)

    st.warning("""
    **Caution:** Import overwrites your current widgets!
    """)

# =============================================================================
# PAGE 5: SETTINGS
# =============================================================================

elif page == "7. Settings":
    st.title("Settings")

    st.markdown("""
    Here you'll find all configuration options of the Shot-Plotter.
    """)

    # Field Size
    st.header("Field Size")

    st.markdown("""
    The field size affects the coordinate calculation.

    **Default settings:**
    - Width: 105m (standard field)
    - Height: 68m (standard field)

    **Adjust:**
    1. Change the values in the **Width** and **Height** fields
    2. Click on **Apply Settings**
    3. The page reloads with the new settings
    """)

    st.info("""
    **Allowed ranges:**
    - Width: 10 - 120 meters
    - Height: 10 - 80 meters
    """)

    st.markdown("---")

    # U13 Youth Field
    st.header("U13 Youth Field")

    st.markdown("""
    For youth games on smaller fields:

    1. Select **"U13 (70x50m)"** in the dropdown
    2. The playing field now shows an overlay:
       - Gray areas: Outside the youth field (not clickable)
       - Dashed lines: Penalty area and goal area of the youth field
    """)
    show_image("u13-field.png", "Screenshot: U13 Youth Field Overlay")

    st.markdown("---")

    # Skip Time
    st.header("Skip Time (Video)")

    st.markdown("""
    The Skip Time determines how many seconds the video skips forward/backward.

    **Adjust:**
    1. Change the value in the **Skip Time** field
    2. Click on **Apply Settings**

    **Recommendation:**
    - 3 seconds for detailed analysis
    - 5-10 seconds for quick browsing
    """)

    st.markdown("---")

    # Reset
    st.header("Reset Settings")

    st.markdown("""
    If something doesn't work or you want to start over:

    1. Click on **"Reset Settings"** (top right)
    2. All settings will be reset
    3. The page reloads

    **Caution:** This also deletes all recorded data!
    Export your data as CSV first if you want to keep it.
    """)

    st.warning("Reset deletes: Widgets, recorded actions, and all settings.")

# =============================================================================
# PAGE 6: DATA MANAGEMENT
# =============================================================================

elif page == "6. Data Management":
    st.title("Data Management")

    st.markdown("""
    Here you'll learn how to manage, export, and import your recorded data.
    """)

    # CSV Export
    st.header("Export Data (CSV Download)")

    st.markdown("""
    Export your recorded actions as a CSV file:

    1. Scroll to the bottom of the page
    2. Click on **"Download CSV"**
    3. Enter a filename 
    4. The file will be downloaded

    """)

    st.error("""
    **Important:** Structure your filenames logically, consistently or look at the guidelines of your project.
    """)

    show_image_placeholder("csv-download.png", "Screenshot: CSV Download area")

    st.markdown("---")

    # CSV Import
    st.header("Import Data (CSV Upload)")

    st.markdown("""
    Import previously exported data:

    1. Scroll to the bottom of the page
    2. Click on **"Upload CSV"** or **"Choose file"**
    3. Select the CSV file
    4. The data will be loaded and displayed on the playing field

    Import adds data to existing data.
    For a clean import, click "Reset Settings" first.
    """)

    st.markdown("---")

    # Table
    st.header("Using the Table")

    st.subheader("View Actions")
    st.markdown("""
    The table shows all recorded actions:
    - Each row = one action
    - Columns correspond to widgets
    - Coordinates (X, Y, X2, Y2) for position
    """)

    st.subheader("Select Actions")
    st.markdown("""
    - Click on the **checkbox** on the left to select an action
    - Multiple actions can be selected at once
    - Selected actions are highlighted on the playing field
    """)

    st.subheader("Delete Actions")
    st.markdown("""
    1. Select the actions to delete via checkbox
    2. Click on **"Delete Selected"** (or trash icon)
    3. Confirm deletion
    """)

    st.subheader("Delete All Actions")
    st.markdown("""
    1. Click on **"Delete All"**
    2. Confirm in the dialog

    """)

    st.error("""
    **Caution:** This cannot be undone!
    """)

    st.markdown("---")

    # Filter
    st.header("Filter Data")

    st.markdown("""
    You can filter the table by various criteria:

    1. Use the filter options above the table
    2. Select e.g., only "Successful" or only "1st Half"
    3. The table and playing field show only filtered actions

    **Tip:** Filters help with targeted analysis of specific game situations.
    """)

# =============================================================================
# PAGE 5: KEYBOARD SHORTCUTS
# =============================================================================

elif page == "5. Keyboard Shortcuts":
    st.title("Keyboard Shortcuts for Widgets")

    st.markdown("""
    Using keyboard shortcuts for widgets makes tagging **significantly faster and more efficient**.
    Instead of clicking through options with your mouse, you can simply press a key to select a value.
    """)

    st.success("""
    **Why use shortcuts?**

    When tagging a game, you might record hundreds of actions. Using shortcuts instead of mouse clicks
    can save you a lot of time and reduce fatigue. Your mouse can stay on the playing field to click on the positions.
    """)

    st.markdown("---")

    # How to assign shortcuts
    st.header("Assigning Shortcuts to Widgets")

    st.markdown("""
    You can assign keyboard shortcuts to **Radio Button** and **Dropdown** widgets. Each option
    within a widget can have its own shortcut key.

    **How to assign a shortcut:**

    1. Open **Customize Setup** in the sidebar
    2. Click the **Edit icon** (pencil) on a Radio Button or Dropdown widget
    3. For each option, you'll see a **Shortcut** field
    4. Enter a single key (e.g., `1`, `2`, `a`, `b`, `q`)
    5. Click **Save**

    Now when you press that key, the corresponding option will be selected automatically.
    """)

    st.error("""
    **Caution:** The shortcut keys must be unique. If you assign the same shortcut to multiple options, the last assigned option will be used.
    """)

    st.markdown("---")

    # Example setup
    st.header("Example: Efficient Shortcut Setup")

    st.markdown("""
    Here's an example of a shortcut setup for quick tagging. But its up to you to organize it the way you want.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Action Type")
        st.code("""
A → Pass
S → Shot
D → Dribble
F → Header
G → Tackle
        """)

        st.subheader("Outcome")
        st.code("""
Q → Successful
W → Unsuccessful
        """)

    with col2:
        st.subheader("Opponent Pressure")
        st.code("""
0 → 0 (no pressure)
1 → 1
2 → 2
3 → 3
4 → 4 (high pressure)
        """)

        st.subheader("Pass Height")
        st.code("""
Y → Low
X → High
        """)

    st.info("""
    **Tip:** Use keys that are easy to reach with your left hand (like `1-5`, `Q-W`, `A-S-D-F`, `Z-X`)
    so your right hand stays free for the mouse to click on the playing field.
    """)




# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    © 2025 BVB Cooperation Project | Shot-Plotter Guide
</div>
""", unsafe_allow_html=True)
