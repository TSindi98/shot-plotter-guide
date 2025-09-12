from ultralytics import YOLO
import cv2
import numpy as np

# --- 1. YOLO Prediction ---
model = YOLO("yolov8m.pt")
image_path = "Bildschirmfoto 2025-09-03 um 09.43.26 (3) Kopie.png"
results = model.predict(image_path, save=True, classes=[0])  # nur Personen

clicked_colors = {}

def pick_color(image_path, role_name):
    """Klick auf ein Bild, um HSV-Wert für ein Team/Rolle zu wählen"""
    img = cv2.imread(image_path)

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv_val = hsv_img[y, x].astype(int)
            clicked_colors[role_name] = hsv_val
            print(f"{role_name}: HSV={hsv_val}")
            cv2.destroyAllWindows()

    cv2.imshow(f"Klicke auf Trikotfarbe für {role_name}", img)
    cv2.setMouseCallback(f"Klicke auf Trikotfarbe für {role_name}", click_event)
    cv2.waitKey(0)

# Farben für Teams/Rollen auswählen
pick_color(image_path, "BVB")
pick_color(image_path, "Schalke")
pick_color(image_path, "Schiedsrichter")
pick_color(image_path, "Torwart")


def assign_role(crop, color_dict):
    """Weist Spieler anhand dominanter HSV-Farbe einer Rolle zu"""
    h, w, _ = crop.shape
    crop_top = crop[0:int(h*0.4), :]  # nur oberes Trikot
    hsv = cv2.cvtColor(crop_top, cv2.COLOR_BGR2HSV)
    avg_color = np.mean(hsv.reshape(-1, 3), axis=0)

    distances = {role: np.linalg.norm(avg_color - np.array(color)) 
                 for role, color in color_dict.items()}
    assigned_role = min(distances, key=distances.get)
    return assigned_role, avg_color

frame = cv2.imread(image_path)

for r in results:
    for box in r.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        role, avg = assign_role(crop, clicked_colors)

        # Farbcode fürs Zeichnen festlegen
        color_map = {
            "BVB": (0, 255, 0),
            "Schalke": (255, 0, 0),
            "Schiedsrichter": (0, 255, 255),
            "Torwart": (255, 0, 255)
        }
        color = color_map.get(role, (200, 200, 200))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, role, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imwrite("players_team_colors.png", frame)
print("Fertig! Ergebnis in players_team_colors.png gespeichert.")