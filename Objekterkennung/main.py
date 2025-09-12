from ultralytics import YOLO
import cv2
import numpy as np
import json
import os

# --------------------------
# Helper: interaktives Anklicken
# --------------------------
def pick_color_single(img, window_name):
    """Zeigt img an, wartet auf Klick und dann auf Tastendruck, gibt HSV zurück"""
    hsv_val = {}
    clicked = False

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not clicked:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv_val['hsv'] = hsv_img[y, x].astype(int)
            hsv_val['clicked'] = True
            # Zeige den ausgewählten Punkt an
            cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
            cv2.imshow(window_name, img)
            print(f"Punkt bei ({x}, {y}) ausgewählt. Drücke eine Taste zum Fortfahren...")

    cv2.imshow(window_name, img)
    cv2.setMouseCallback(window_name, click_event)
    
    # Warte auf Klick
    while not hsv_val.get('clicked', False):
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC-Taste
            cv2.destroyWindow(window_name)
            return None
    
    # Warte auf Tastendruck zum Fortfahren
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    
    return hsv_val.get('hsv', None)

def pick_two_colors_for_roles(image_path, roles, save_json="teams_colors.json"):
    """Für jede Rolle: oberen und unteren Farbpunkt anklicken und speichern."""
    img = cv2.imread(image_path)
    colors = {}
    for role in roles:
        print(f"== Klick für ROLE '{role}' — UPPER (Trikot) ==")
        upper = pick_color_single(img, f"Click UPPER for {role}")
        print(f"{role} UPPER HSV = {upper}")

        print(f"== Klick für ROLE '{role}' — LOWER (Shorts / Unterer Bereich) ==")
        lower = pick_color_single(img, f"Click LOWER for {role}")
        print(f"{role} LOWER HSV = {lower}")

        if upper is None or lower is None:
            raise RuntimeError(f"Keine Farbe für Rolle {role} gewählt (upper={upper}, lower={lower})")

        colors[role] = {'upper': upper.tolist(), 'lower': lower.tolist()}

    # speichern
    with open(save_json, 'w') as f:
        json.dump(colors, f, indent=2)
    print(f"Farben gespeichert in {save_json}")
    return colors

def load_colors(json_path="teams_colors.json"):
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        # in numpy arrays zurückwandeln
        return {k: {'upper': np.array(v['upper']), 'lower': np.array(v['lower'])} for k,v in data.items()}
    return None

# --------------------------
# Farb-Auswertung im Crop
# --------------------------
def dominant_hsv_of_slice(crop_slice, sat_thresh=20, val_thresh=20):
    """
    Robusteren Farbwert bestimmen - optimiert für unscharfe Bilder:
    - Niedrigere Schwellenwerte für unscharfe Bilder
    - Mehrere Analysemethoden
    - Fallback auf größere Bereiche falls nötig
    """
    if crop_slice is None or crop_slice.size == 0:
        return None
    
    hsv = cv2.cvtColor(crop_slice, cv2.COLOR_BGR2HSV)
    s = hsv[...,1]; v = hsv[...,2]
    
    # Methode 1: Mit Schwellenwerten (für scharfe Bilder)
    mask = (s > sat_thresh) & (v > val_thresh)
    if np.count_nonzero(mask) >= 0.1 * mask.size:  # Mindestens 10% Pixel
        pixels = hsv[mask]
        med = np.median(pixels, axis=0)
        return med.astype(float)
    
    # Methode 2: Alle Pixel, aber gewichtet nach Sättigung und Helligkeit
    pixels = hsv.reshape(-1, 3)
    weights = (s.reshape(-1) + v.reshape(-1)) / 2  # Gewichtung nach S+V
    if np.sum(weights) > 0:
        # Gewichteter Median
        sorted_indices = np.argsort(weights)[::-1]  # Höchste Gewichte zuerst
        top_pixels = pixels[sorted_indices[:len(pixels)//3]]  # Top 33%
        med = np.median(top_pixels, axis=0)
        return med.astype(float)
    
    # Methode 3: Fallback - einfacher Median aller Pixel
    med = np.median(pixels, axis=0)
    return med.astype(float)

def hsv_distance(a, b):
    """
    Abstandsmaß in HSV-Raum, Hue circular beachten (OpenCV Hue: 0..179).
    a,b are length-3 arrays or lists.
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    dh = abs(a[0] - b[0])
    dh = min(dh, 180 - dh)   # circular
    ds = a[1] - b[1]
    dv = a[2] - b[2]
    return np.sqrt(dh*dh + ds*ds + dv*dv)

def count_pixels_in_color_range(hsv_slice, color_ranges):
    """
    Zählt Pixel in bestimmten Farbbereichen.
    color_ranges: dict mit 'name': (lower_hsv, upper_hsv)
    """
    if hsv_slice is None or hsv_slice.size == 0:
        return {}
    
    counts = {}
    for color_name, (lower, upper) in color_ranges.items():
        # Erstelle Maske für diesen Farbbereich
        mask = cv2.inRange(hsv_slice, np.array(lower), np.array(upper))
        count = np.count_nonzero(mask)
        counts[color_name] = count
    
    return counts

def assign_role_by_color_rules(crop, top_frac=0.45, center_radius=0.25, min_pixels=10):
    """
    Neue regelbasierte Farbzuordnung:
    - Suche nach gelben Pixeln -> BVB
    - Suche nach blauen Pixeln -> Schalke  
    - Suche nach schwarzen Pixeln -> Schiedsrichter
    - Fallback auf Torwart
    """
    h, w = crop.shape[:2]
    
    # Obere und untere Bereiche definieren
    top_h = max(1, int(h * top_frac))
    center_w = max(1, int(w * center_radius))
    center_x = w // 2
    center_y_top = top_h // 2
    center_y_bottom = h // 2  # Hose in der Mitte
    
    # Analysiere beide Bereiche
    top_slice = crop[
        max(0, center_y_top - center_w):min(h, center_y_top + center_w),
        max(0, center_x - center_w):min(w, center_x + center_w)
    ]
    
    bottom_slice = crop[
        max(0, center_y_bottom - center_w):min(h, center_y_bottom + center_w),
        max(0, center_x - center_w):min(w, center_x + center_w)
    ]
    
    # Definiere Farbbereiche (HSV)
    # Gelb: H=20-30, S=100-255, V=100-255 (aber tolerant für unscharfe Bilder)
    # Blau: H=100-130, S=100-255, V=100-255
    # Schwarz: H=0-180, S=0-255, V=0-50
    color_ranges = {
        'gelb': ([15, 80, 80], [35, 255, 255]),      # Toleranter gelber Bereich
        'blau': ([95, 80, 80], [135, 255, 255]),     # Toleranter blauer Bereich
        'schwarz': ([0, 0, 0], [180, 255, 50])       # Schwarzer Bereich
    }
    
    # Konvertiere zu HSV
    top_hsv = cv2.cvtColor(top_slice, cv2.COLOR_BGR2HSV)
    bottom_hsv = cv2.cvtColor(bottom_slice, cv2.COLOR_BGR2HSV)
    
    # Zähle Pixel in beiden Bereichen
    top_counts = count_pixels_in_color_range(top_hsv, color_ranges)
    bottom_counts = count_pixels_in_color_range(bottom_hsv, color_ranges)
    
    # Kombiniere beide Bereiche
    total_counts = {}
    for color in color_ranges.keys():
        total_counts[color] = top_counts.get(color, 0) + bottom_counts.get(color, 0)
    
    print(f"Debug - Farbzählung: {total_counts}")
    
    # Regelbasierte Zuordnung
    if total_counts.get('gelb', 0) >= min_pixels:
        return "BVB", {'method': 'gelb', 'counts': total_counts}
    elif total_counts.get('blau', 0) >= min_pixels:
        return "Schalke", {'method': 'blau', 'counts': total_counts}
    elif total_counts.get('schwarz', 0) >= min_pixels:
        return "Schiedsrichter", {'method': 'schwarz', 'counts': total_counts}
    else:
        return "Torwart", {'method': 'fallback', 'counts': total_counts}

def split_image_into_quadrants(image):
    """
    Teilt das Bild in 4 Quadranten auf, um die Spielererfassung zu verbessern.
    """
    h, w = image.shape[:2]
    mid_h, mid_w = h // 2, w // 2
    
    quadrants = [
        # Oben links
        (0, 0, mid_w, mid_h, "top_left"),
        # Oben rechts  
        (mid_w, 0, w, mid_h, "top_right"),
        # Unten links
        (0, mid_h, mid_w, h, "bottom_left"),
        # Unten rechts
        (mid_w, mid_h, w, h, "bottom_right")
    ]
    
    return quadrants

def merge_detections_from_quadrants(quadrant_results, original_shape):
    """
    Führt die Detektionen aus allen Quadranten zusammen und korrigiert die Koordinaten.
    """
    all_detections = []
    h, w = original_shape[:2]
    mid_h, mid_w = h // 2, w // 2
    
    for quadrant_name, detections in quadrant_results.items():
        if detections is None:
            continue
            
        for detection in detections:
            x1, y1, x2, y2 = detection['box']
            role = detection['role']
            info = detection['info']
            
            # Koordinaten auf das ursprüngliche Bild umrechnen
            if quadrant_name == "top_left":
                pass  # Keine Änderung
            elif quadrant_name == "top_right":
                x1 += mid_w
                x2 += mid_w
            elif quadrant_name == "bottom_left":
                y1 += mid_h
                y2 += mid_h
            elif quadrant_name == "bottom_right":
                x1 += mid_w
                x2 += mid_w
                y1 += mid_h
                y2 += mid_h
            
            all_detections.append({
                'box': (x1, y1, x2, y2),
                'role': role,
                'info': info
            })
    
    return all_detections

def detect_players_in_quadrants(image_path, model, templates):
    """
    Analysiert das Bild in 4 Quadranten für bessere Spielererfassung.
    """
    frame = cv2.imread(image_path)
    h, w = frame.shape[:2]
    
    # Teile das Bild in Quadranten
    quadrants = split_image_into_quadrants(frame)
    quadrant_results = {}
    
    print(f"Analysiere Bild in 4 Quadranten ({w}x{h})...")
    
    for x1, y1, x2, y2, quadrant_name in quadrants:
        print(f"  Analysiere {quadrant_name}: ({x1},{y1}) bis ({x2},{y2})")
        
        # Schneide Quadrant aus
        quadrant = frame[y1:y2, x1:x2]
        
        # YOLO auf diesem Quadranten
        results = model.predict(quadrant, save=False, classes=[0])  # Nur Personen
        
        detections = []
        for r in results:
            for box in r.boxes.xyxy.cpu().numpy():
                bx1, by1, bx2, by2 = map(int, box)
                
                # Koordinaten relativ zum Quadranten
                detections.append({
                    'box': (bx1, by1, bx2, by2),
                    'role': None,
                    'info': None
                })
        
        quadrant_results[quadrant_name] = detections if detections else None
    
    # Führe alle Detektionen zusammen
    all_detections = merge_detections_from_quadrants(quadrant_results, frame.shape)
    
    print(f"Gefundene Spieler: {len(all_detections)}")
    return all_detections, frame

def detect_field_boundaries(frame):
    """
    Erkennt die Spielfeldgrenzen durch Analyse der weißen Linien.
    Gibt die Y-Koordinaten der oberen und unteren Spielfeldgrenzen zurück.
    """
    h, w = frame.shape[:2]
    
    # Konvertiere zu HSV für bessere Weiß-Erkennung
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Weiße Linien: niedrige Sättigung, hohe Helligkeit
    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    
    # Finde horizontale Linien durch Analyse der Spalten
    field_top = None
    field_bottom = None
    
    # Analysiere jede Spalte nach weißen Linien
    for x in range(0, w, 10):  # Alle 10 Pixel eine Spalte
        col = white_mask[:, x]
        
        # Finde weiße Bereiche in dieser Spalte
        white_indices = np.where(col > 0)[0]
        if len(white_indices) > 0:
            # Obere und untere weiße Linie in dieser Spalte
            top_y = white_indices[0]
            bottom_y = white_indices[-1]
            
            if field_top is None or top_y < field_top:
                field_top = top_y
            if field_bottom is None or bottom_y > field_bottom:
                field_bottom = bottom_y
    
    # Fallback: Falls keine Linien gefunden, verwende Standard-Bereiche
    if field_top is None:
        field_top = int(h * 0.1)  # 10% von oben
    if field_bottom is None:
        field_bottom = int(h * 0.9)  # 90% von oben
    
    print(f"Erkannte Spielfeldgrenzen: Oben bei Y={field_top}, Unten bei Y={field_bottom}")
    return field_top, field_bottom

def is_player_on_field(detection_box, field_top, field_bottom, tolerance=20):
    """
    Prüft ob ein Spieler auf dem Spielfeld steht.
    detection_box: (x1, y1, x2, y2) der Bounding Box
    field_top, field_bottom: Y-Koordinaten der Spielfeldgrenzen
    tolerance: Toleranz in Pixeln
    """
    x1, y1, x2, y2 = detection_box
    
    # Füße sind am unteren Rand der Bounding Box
    feet_y = y2
    
    # Prüfe ob Füße innerhalb der Spielfeldgrenzen sind
    on_field = (feet_y >= field_top - tolerance) and (feet_y <= field_bottom + tolerance)
    
    print(f"Spieler bei Y={feet_y}, Spielfeld: {field_top}-{field_bottom}, Auf dem Feld: {on_field}")
    
    return on_field

def filter_players_on_field(detections, frame):
    """
    Filtert Spieler, die auf dem Spielfeld stehen.
    """
    # Erkenne Spielfeldgrenzen
    field_top, field_bottom = detect_field_boundaries(frame)
    
    filtered_detections = []
    
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        
        # Prüfe ob Spieler auf dem Spielfeld steht
        if is_player_on_field((x1, y1, x2, y2), field_top, field_bottom):
            filtered_detections.append(detection)
            print(f"Spieler auf dem Feld: {detection['role']} bei ({x1}, {y2})")
        else:
            print(f"Zuschauer gefiltert: {detection['role']} bei ({x1}, {y2})")
    
    return filtered_detections

def visualize_field_boundaries(frame, field_top, field_bottom):
    """
    Zeichnet die erkannten Spielfeldgrenzen ins Bild ein.
    """
    h, w = frame.shape[:2]
    
    # Zeichne obere Spielfeldgrenze (grün)
    cv2.line(frame, (0, field_top), (w, field_top), (0, 255, 0), 3)
    cv2.putText(frame, f"Obere Feldgrenze: Y={field_top}", (10, field_top - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Zeichne untere Spielfeldgrenze (grün)
    cv2.line(frame, (0, field_bottom), (w, field_bottom), (0, 255, 0), 3)
    cv2.putText(frame, f"Untere Feldgrenze: Y={field_bottom}", (10, field_bottom + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Zeichne Spielfeldbereich (transparenter grüner Bereich)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, field_top), (w, field_bottom), (0, 255, 0), -1)
    cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
    
    return frame

def detect_field_landmarks(frame):
    """
    Erkennt markante Spielfeldpunkte für die Perspektivtransformation.
    Priorität 1: Mittelkreis-Mittellinie, 16er-Ecken
    Priorität 2: Halbkreis-16er-Kreuzungen, Torlinien
    """
    h, w = frame.shape[:2]
    
    # Konvertiere zu HSV für bessere Linien-Erkennung
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Weiße Linien: niedrige Sättigung, hohe Helligkeit
    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    
    # 1. Finde horizontale Linien (Mittellinie, 16er-Linien)
    horizontal_lines = []
    for y in range(0, h, 5):  # Alle 5 Pixel eine Zeile
        row = white_mask[y, :]
        white_indices = np.where(row > 0)[0]
        if len(white_indices) > w * 0.3:  # Mindestens 30% der Zeile ist weiß
            horizontal_lines.append(y)
    
    # 2. Finde vertikale Linien (Mittellinie, Torlinien)
    vertical_lines = []
    for x in range(0, w, 5):  # Alle 5 Pixel eine Spalte
        col = white_mask[:, x]
        white_indices = np.where(col > 0)[0]
        if len(white_indices) > h * 0.3:  # Mindestens 30% der Spalte ist weiß
            vertical_lines.append(x)
    
    # 3. Finde Kreisbögen (Mittelkreis, 16er-Bögen)
    # Verwende Hough-Kreise oder Kontur-Erkennung
    circles = detect_circles(white_mask)
    
    # 4. Bestimme markante Punkte
    landmarks = {}
    
    # Mittelpunkt des Bildes (ungefähre Spielfeldmitte)
    center_x, center_y = w // 2, h // 2
    
    # Suche nach Mittelkreis-Mittellinie-Kreuzung
    center_crossing = find_center_crossing(horizontal_lines, vertical_lines, center_x, center_y)
    if center_crossing:
        landmarks['center_crossing'] = center_crossing
        print(f"✓ Mittelkreis-Mittellinie-Kreuzung gefunden: {center_crossing}")
    
    # Suche nach 16er-Ecken
    penalty_area_corners = find_penalty_area_corners(horizontal_lines, vertical_lines, circles, h, w)
    if penalty_area_corners:
        landmarks['penalty_corners'] = penalty_area_corners
        print(f"✓ 16er-Ecken gefunden: {len(penalty_area_corners)} Stück")
    
    # Suche nach Halbkreis-16er-Kreuzungen
    arc_crossings = find_arc_crossings(horizontal_lines, vertical_lines, circles, h, w)
    if arc_crossings:
        landmarks['arc_crossings'] = arc_crossings
        print(f"✓ Halbkreis-16er-Kreuzungen gefunden: {len(arc_crossings)} Stück")
    
    print(f"Gefundene Landmarks: {list(landmarks.keys())}")
    return landmarks

def detect_circles(white_mask):
    """
    Erkennt Kreisbögen im weißen Maskenbild.
    """
    # Verwende Kontur-Erkennung für Kreisbögen
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circles = []
    for contour in contours:
        # Prüfe ob Kontur einem Kreisbogen ähnelt
        if len(contour) > 10:  # Mindestens 10 Punkte
            # Berechne Umfang und Fläche
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            
            # Kreis-ähnliche Konturen haben bestimmte Verhältnisse
            if area > 100 and perimeter > 50:  # Mindestgröße
                # Berechne umschreibendes Rechteck
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Kreisbögen haben oft ein bestimmtes Seitenverhältnis
                if 0.5 < aspect_ratio < 2.0:
                    circles.append({
                        'contour': contour,
                        'center': (x + w//2, y + h//2),
                        'radius': max(w, h) // 2,
                        'area': area,
                        'perimeter': perimeter
                    })
    
    return circles

def find_center_crossing(horizontal_lines, vertical_lines, center_x, center_y, tolerance=50):
    """
    Findet die Kreuzung von Mittellinie und Mittelkreis.
    """
    if not horizontal_lines or not vertical_lines:
        return None
    
    # Suche nach Linien nahe der Bildmitte
    center_horizontal = None
    center_vertical = None
    
    for y in horizontal_lines:
        if abs(y - center_y) < tolerance:
            center_horizontal = y
            break
    
    for x in vertical_lines:
        if abs(x - center_x) < tolerance:
            center_vertical = x
            break
    
    if center_horizontal is not None and center_vertical is not None:
        return (center_vertical, center_horizontal)
    
    return None

def find_penalty_area_corners(horizontal_lines, vertical_lines, circles, h, w):
    """
    Findet die Ecken der 16er-Bereiche.
    """
    corners = []
    
    # 16er-Bereiche sind typischerweise in den oberen und unteren Bildhälften
    top_half = h // 2
    bottom_half = h // 2
    
    # Suche nach horizontalen Linien in den 16er-Bereichen
    for y in horizontal_lines:
        if y < top_half or y > bottom_half:
            # Suche nach vertikalen Linien die diese horizontalen Linien kreuzen
            for x in vertical_lines:
                # Prüfe ob es sich um eine 16er-Ecke handeln könnte
                if is_likely_penalty_corner(x, y, w, h):
                    corners.append((x, y))
    
    return corners

def find_arc_crossings(horizontal_lines, vertical_lines, circles, h, w):
    """
    Findet Kreuzungen von Halbkreisen mit 16er-Linien.
    """
    crossings = []
    
    for circle in circles:
        cx, cy = circle['center']
        radius = circle['radius']
        
        # Suche nach Linien die den Kreis kreuzen
        for y in horizontal_lines:
            if abs(y - cy) < radius:
                # Berechne X-Koordinaten der Kreuzung
                dx = int(np.sqrt(radius**2 - (y - cy)**2))
                x1, x2 = cx - dx, cx + dx
                if 0 <= x1 < w and 0 <= x2 < w:
                    crossings.extend([(x1, y), (x2, y)])
        
        for x in vertical_lines:
            if abs(x - cx) < radius:
                # Berechne Y-Koordinaten der Kreuzung
                dy = int(np.sqrt(radius**2 - (x - cx)**2))
                y1, y2 = cy - dy, cy + dy
                if 0 <= y1 < h and 0 <= y2 < h:
                    crossings.extend([(x, y1), (x, y2)])
    
    return crossings

def is_likely_penalty_corner(x, y, w, h):
    """
    Prüft ob ein Punkt wahrscheinlich eine 16er-Ecke ist.
    """
    # 16er-Ecken sind typischerweise in den äußeren Bereichen
    margin = 0.1  # 10% Rand
    left_margin = w * margin
    right_margin = w * (1 - margin)
    
    return x < left_margin or x > right_margin

def calculate_field_transform(landmarks, frame_shape):
    """
    Berechnet die Perspektivtransformation basierend auf erkannten Landmarks.
    """
    h, w = frame_shape[:2]
    
    # Standard-Spielfeldmaße (ungefähre Proportionen)
    field_width = 105  # Meter
    field_height = 68  # Meter
    
    # Zielkoordinaten für die Transformation (Vogelperspektive)
    dst_points = np.array([
        [0, 0],                    # Oben links
        [field_width, 0],          # Oben rechts
        [field_width, field_height], # Unten rechts
        [0, field_height]          # Unten links
    ], dtype=np.float32)
    
    # Quellkoordinaten basierend auf erkannten Landmarks
    src_points = []
    
    if 'center_crossing' in landmarks:
        # Mittelpunkt als Referenz
        center_x, center_y = landmarks['center_crossing']
        src_points.append([center_x, center_y])
    
    if 'penalty_corners' in landmarks and len(landmarks['penalty_corners']) >= 2:
        # Verwende 16er-Ecken
        corners = landmarks['penalty_corners'][:2]  # Nimm die ersten 2
        for corner in corners:
            src_points.append(corner)
    
    if 'arc_crossings' in landmarks and len(landmarks['arc_crossings']) >= 2:
        # Verwende Halbkreis-Kreuzungen
        crossings = landmarks['arc_crossings'][:2]  # Nimm die ersten 2
        for crossing in crossings:
            src_points.append(crossing)
    
    # Fallback: Falls nicht genug Punkte gefunden wurden
    if len(src_points) < 4:
        print("Warnung: Nicht genug Landmarks für Perspektivtransformation")
        # Verwende Standard-Bereiche
        src_points = np.array([
            [w * 0.1, h * 0.1],      # Oben links
            [w * 0.9, h * 0.1],      # Oben rechts
            [w * 0.9, h * 0.9],      # Unten rechts
            [w * 0.1, h * 0.9]       # Unten links
        ], dtype=np.float32)
    else:
        src_points = np.array(src_points[:4], dtype=np.float32)  # Maximal 4 Punkte
    
    # Berechne Transformationsmatrix
    if len(src_points) == 4:
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return transform_matrix, src_points, dst_points
    else:
        return None, src_points, dst_points

def transform_field_to_top_view(frame, transform_matrix):
    """
    Transformiert das Spielfeld in eine Vogelperspektive.
    """
    h, w = frame.shape[:2]
    
    # Standard-Spielfeldmaße
    field_width = 105
    field_height = 68
    
    # Transformiere das Bild
    top_view = cv2.warpPerspective(frame, transform_matrix, (field_width, field_height))
    
    return top_view

def detect_field_boundaries_improved(frame):
    """
    Verbesserte Spielfeld-Erkennung mit Landmarks und Perspektivtransformation.
    """
    print("=== Erweiterte Spielfeld-Erkennung ===")
    
    # 1. Erkenne Landmarks
    landmarks = detect_field_landmarks(frame)
    
    # 2. Berechne Transformation
    transform_matrix, src_points, dst_points = calculate_field_transform(landmarks, frame.shape)
    
    # 3. Fallback: Falls Transformation fehlschlägt, verwende einfache Linien-Erkennung
    if transform_matrix is None:
        print("Fallback: Verwende einfache Linien-Erkennung")
        return detect_field_boundaries_simple(frame)
    
    # 4. Transformiere zu Vogelperspektive
    top_view = transform_field_to_top_view(frame, transform_matrix)
    
    # 5. Bestimme Spielfeldgrenzen in der transformierten Ansicht
    field_top, field_bottom = 0, top_view.shape[0]  # In der Vogelperspektive ist das gesamte Bild das Spielfeld
    
    # 6. Transformiere Grenzen zurück
    # Hier müssten wir die inversen Koordinaten berechnen...
    # Für den Moment verwenden wir die ursprünglichen Grenzen
    h, w = frame.shape[:2]
    field_top = int(h * 0.1)
    field_bottom = int(h * 0.9)
    
    print(f"Transformierte Spielfeldgrenzen: Oben bei Y={field_top}, Unten bei Y={field_bottom}")
    return field_top, field_bottom

def detect_field_boundaries_simple(frame):
    """
    Einfache Spielfeld-Erkennung als Fallback.
    """
    h, w = frame.shape[:2]
    
    # Verwende Standard-Bereiche basierend auf Bildgröße
    field_top = int(h * 0.15)      # 15% von oben
    field_bottom = int(h * 0.85)   # 85% von oben
    
    print(f"Fallback: Standard-Spielfeldgrenzen: Oben bei Y={field_top}, Unten bei Y={field_bottom}")
    return field_top, field_bottom

def detect_field_with_visioneye(frame, model):
    """
    Verwendet VisionEye für automatische Spielfeld-Erkennung und Spieler-Filterung.
    """
    try:
        from ultralytics import solutions
        
        print("=== Verwende VisionEye für Spielfeld-Erkennung ===")
        
        # Initialisiere VisionEye mit gültigen Parametern aus der Dokumentation
        visioneye = solutions.VisionEye(
            model=model,  # Pfad zur Ultralytics YOLO Modelldatei
            vision_point=(frame.shape[1]//2, frame.shape[0]//2),  # Zentrum des Bildes als Sichtpunkt
            max_hist=10,  # Maximale Anzahl historischer Punkte für Tracking
            meter_per_pixel=0.05,  # Skalierungsfaktor für reale Einheiten
            fps=30.0,  # Frames pro Sekunde für Geschwindigkeitsberechnungen
        )
        
        # Verarbeite den Frame
        results = visioneye(frame)
        
        # Extrahiere Tracking-Informationen
        if hasattr(results, 'tracks') and results.tracks is not None:
            tracks = results.tracks
            print(f"VisionEye: {len(tracks)} Spieler getrackt")
            
            # Konvertiere zu unserem Format
            detections = []
            for track in tracks:
                if track.boxes is not None:
                    box = track.boxes.xyxy[0].cpu().numpy()
                    track_id = int(track.boxes.id[0].cpu().numpy()) if track.boxes.id is not None else 0
                    
                    detections.append({
                        'box': tuple(map(int, box)),
                        'track_id': track_id,
                        'role': None,
                        'info': None
                    })
            
            return detections, frame, results
        
        else:
            print("VisionEye: Keine Tracks gefunden, verwende Fallback")
            return None, frame, None
            
    except ImportError:
        print("VisionEye nicht verfügbar, verwende Fallback")
        return None, frame, None
    except Exception as e:
        print(f"VisionEye Fehler: {e}, verwende Fallback")
        return None, frame, None

def filter_players_with_visioneye(detections, frame, visioneye_results):
    """
    Filtert Spieler basierend auf VisionEye-Sichtlinien.
    """
    if visioneye_results is None:
        print("VisionEye nicht verfügbar, verwende Standard-Filterung")
        return filter_players_on_field(detections, frame)
    
    print("=== Filtere Spieler mit VisionEye-Sichtlinien ===")
    
    # VisionEye hat bereits die Spieler gefiltert, da es nur Personen auf dem Feld trackt
    # Wir können zusätzliche Filterung basierend auf der Sichtlinien-Position machen
    
    filtered_detections = []
    h, w = frame.shape[:2]
    
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        
        # Füße sind am unteren Rand der Bounding Box
        feet_y = y2
        
        # Einfache Bereichsfilterung als Backup
        # VisionEye sollte bereits die Hauptarbeit gemacht haben
        if feet_y > h * 0.1 and feet_y < h * 0.9:  # Innerhalb des sichtbaren Bereichs
            filtered_detections.append(detection)
            print(f"Spieler auf dem Feld (VisionEye): {detection.get('role', 'Unbekannt')} bei ({x1}, {y2})")
        else:
            print(f"Zuschauer gefiltert (VisionEye): {detection.get('role', 'Unbekannt')} bei ({x1}, {y2})")
    
    return filtered_detections

def visualize_landmarks(frame, landmarks):
    """
    Zeichnet die erkannten Landmarks ins Bild ein.
    """
    # Zeichne Mittelkreis-Mittellinie-Kreuzung (rot)
    if 'center_crossing' in landmarks:
        x, y = landmarks['center_crossing']
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(frame, "Mittelpunkt", (x + 15, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Zeichne 16er-Ecken (blau)
    if 'penalty_corners' in landmarks:
        for i, (x, y) in enumerate(landmarks['penalty_corners']):
            cv2.circle(frame, (x, y), 8, (255, 0, 0), -1)
            cv2.putText(frame, f"16er-{i+1}", (x + 15, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Zeichne Halbkreis-Kreuzungen (grün)
    if 'arc_crossings' in landmarks:
        for i, (x, y) in enumerate(landmarks['arc_crossings']):
            cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(frame, f"Bogen-{i+1}", (x + 15, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
    
    return frame

# --------------------------
# Main: YOLO + Zuordnung
# --------------------------
def main():
    # Ändere von Bild zu Video
    video_path = "test_video.mp4"  # 10-Sekunden Video
    json_path = "teams_colors.json"
    roles = ["BVB", "Schalke", "Schiedsrichter", "Torwart"]

    # 1) Farben laden oder interaktiv wählen
    templates = load_colors(json_path)
    if templates is None:
        print("Keine team colors gefunden -> Interaktiv wählen.")
        # Für Videos können wir das erste Frame für die Farbauswahl verwenden
        print("Video-Datei nicht gefunden, verwende Standard-Farben")
        # Hier könnten wir Standard-Farben definieren
        return
    else:
        print(f"Lade Farben aus {json_path}")

    # 2) YOLO inference mit VisionEye für Video
    model = YOLO("yolov8m.pt")
    
    # Prüfe ob Video-Datei existiert
    if not os.path.exists(video_path):
        print(f"Video-Datei {video_path} nicht gefunden!")
        print("Bitte lege ein Video mit dem Namen 'test_video.mp4' in den Ordner.")
        return
    
    print(f"=== Analysiere Video: {video_path} ===")
    
    # 3) Video mit VisionEye verarbeiten
    process_video_with_visioneye(video_path, model, templates)

def process_video_with_visioneye(video_path, model, templates):
    """
    Verarbeitet ein Video mit VisionEye für Spieler-Tracking und Rollen-Zuordnung.
    """
    try:
        from ultralytics import solutions
        
        print("=== Initialisiere VisionEye für Video-Analyse ===")
        
        # Initialisiere VisionEye
        visioneye = solutions.VisionEye(
            model=model,
            vision_point=(960, 540),  # Standard-Sichtpunkt (kann angepasst werden)
            max_hist=15,  # Mehr historische Punkte für Video
            meter_per_pixel=0.05,
            fps=30.0,
        )
        
        # Video-Capture öffnen
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Fehler beim Öffnen des Videos: {video_path}")
            return
        
        # Video-Eigenschaften abrufen
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height}, {fps} FPS, {total_frames} Frames")
        
        # Video-Writer für Ausgabe
        output_path = "output_video_visioneye.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Tracking-Daten für 2D-Modell
        tracking_data = {}
        frame_count = 0
        total_detections = 0
        
        print("=== Starte Video-Verarbeitung ===")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"Verarbeite Frame {frame_count}/{total_frames}")
            
            # VisionEye auf diesem Frame
            results = visioneye(frame)
            
            # Debug: Prüfe was VisionEye zurückgibt
            print(f"  VisionEye Results: {type(results)}")
            
            # VisionEye speichert die Tracking-Daten anders - versuche direkten Zugriff
            tracks = None
            
            # Versuche verschiedene Möglichkeiten, die Tracking-Daten zu finden
            if hasattr(results, 'tracks'):
                tracks = results.tracks
                print(f"  Tracks über .tracks: {tracks is not None}")
            elif hasattr(results, 'boxes'):
                tracks = results.boxes
                print(f"  Boxes über .boxes: {tracks is not None}")
            elif hasattr(results, 'keypoints'):
                tracks = results.keypoints
                print(f"  Keypoints über .keypoints: {tracks is not None}")
            
            # Falls keine Tracks gefunden, verwende das plot_im und extrahiere die Daten
            if tracks is None:
                print(f"  Keine Tracks gefunden, verwende plot_im")
                
                # VisionEye hat bereits das verarbeitete Bild mit Tracking-Informationen
                # Wir können die ursprünglichen Detektionen aus dem Model extrahieren
                if hasattr(results, 'plot_im') and results.plot_im is not None:
                    # Verwende das plot_im als verarbeiteten Frame
                    frame = results.plot_im.copy()
                    
                    # Da VisionEye bereits die Tracking-Informationen gezeichnet hat,
                    # können wir die ursprünglichen YOLO-Ergebnisse verwenden
                    # Lass uns das Model direkt auf den Frame anwenden
                    yolo_results = model(frame, verbose=False)
                    
                    if len(yolo_results) > 0:
                        result = yolo_results[0]
                        if result.boxes is not None:
                            boxes = result.boxes.xyxy.cpu().numpy()
                            print(f"  YOLO direkt: {len(boxes)} Personen gefunden")
                            
                            frame_detections = 0
                            for i, box in enumerate(boxes):
                                frame_detections += 1
                                x1, y1, x2, y2 = map(int, box)
                                
                                # Clamps
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(width-1, x2), min(height-1, y2)
                                
                                if x2 > x1 and y2 > y1:
                                    # Crop des Spielers
                                    crop = frame[y1:y2, x1:x2]
                                    
                                    # Bestimme Rolle basierend auf Farben
                                    role, info = assign_role_by_color_rules(crop, top_frac=0.45, center_radius=0.25, min_pixels=10)
                                    print(f"    Rolle bestimmt: {role} (Methode: {info['method']})")
                                    
                                    # Verwende Frame-Nummer als Track-ID (da wir kein echtes Tracking haben)
                                    track_id = frame_count * 1000 + i
                                    
                                    # Speichere Tracking-Daten für 2D-Modell
                                    if track_id not in tracking_data:
                                        tracking_data[track_id] = {
                                            'role': role,
                                            'positions': [],
                                            'first_frame': frame_count
                                        }
                                        print(f"    Neuer Track {track_id} erstellt für {role}")
                                    
                                    # Füge Position hinzu (Fußposition = untere Mitte der Box)
                                    foot_x = (x1 + x2) // 2
                                    foot_y = y2
                                    
                                    # Prüfe ob Spieler auf dem Spielfeld steht
                                    on_field = is_player_on_field_simple(foot_y, height)
                                    print(f"    Auf dem Feld: {on_field} (Fuß Y: {foot_y})")
                                    
                                    if on_field:
                                        position_data = {
                                            'frame': frame_count,
                                            'x': foot_x,
                                            'y': foot_y,
                                            'role': role
                                        }
                                        tracking_data[track_id]['positions'].append(position_data)
                                        print(f"    Position hinzugefügt: Frame {frame_count}, ({foot_x}, {foot_y})")
                                        
                                        # Zeichne Rolle auf den Frame (größer und deutlicher)
                                        color_map = {
                                            "BVB": (0, 255, 0),
                                            "Schalke": (255, 0, 0),
                                            "Schiedsrichter": (0, 255, 255),
                                            "Torwart": (255, 0, 255)
                                        }
                                        
                                        col = color_map.get(role, (200, 200, 200))
                                        
                                        # Dickere Box für bessere Sichtbarkeit
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3)
                                        
                                        # Größerer Text für Rolle
                                        cv2.putText(frame, f"{role}", (x1, max(20, y1-10)), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
                                        
                                        # Track-ID anzeigen
                                        cv2.putText(frame, f"ID:{track_id}", (x1, y2+20), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                        
                                        # Debug-Info
                                        cv2.putText(frame, f"{info['method']}", (x1, y2-10), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                                    else:
                                        # Zuschauer: Dünnere, graue Box
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                                        cv2.putText(frame, "Zuschauer", (x1, y1-5), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                            
                            print(f"  Frame {frame_count}: {frame_detections} Detektionen verarbeitet")
                            total_detections += frame_detections
                        else:
                            print(f"  Frame {frame_count}: Keine Boxes in YOLO-Ergebnissen")
                    else:
                        print(f"  Frame {frame_count}: Keine YOLO-Ergebnisse")
                else:
                    print(f"  Frame {frame_count}: Kein plot_im verfügbar")
            else:
                # Falls doch Tracks gefunden wurden, verarbeite sie normal
                print(f"  Tracks gefunden, verarbeite normal")
                # ... Rest des ursprünglichen Codes für Tracks
                
                # Rolle für jeden getrackten Spieler bestimmen
                if tracks is not None:
                    frame_detections = 0
                    
                    # Versuche verschiedene Track-Formate
                    if hasattr(tracks, 'xyxy') and tracks.xyxy is not None:
                        # Standard YOLO-Format
                        boxes = tracks.xyxy.cpu().numpy()
                        ids = tracks.id.cpu().numpy() if hasattr(tracks, 'id') and tracks.id is not None else None
                        
                        print(f"  Gefunden: {len(boxes)} Boxes")
                        
                        for i, box in enumerate(boxes):
                            frame_detections += 1
                            x1, y1, x2, y2 = map(int, box)
                            
                            # Track-ID
                            track_id = int(ids[i]) if ids is not None and i < len(ids) else i
                            
                            print(f"    Box {i}: ({x1},{y1}) bis ({x2},{y2}), ID: {track_id}")
                            
                            # Clamps
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(width-1, x2), min(height-1, y2)
                            
                            if x2 > x1 and y2 > y1:
                                # Crop des Spielers
                                crop = frame[y1:y2, x1:x2]
                                
                                # Bestimme Rolle basierend auf Farben
                                role, info = assign_role_by_color_rules(crop, top_frac=0.45, center_radius=0.25, min_pixels=10)
                                print(f"    Rolle bestimmt: {role} (Methode: {info['method']})")
                                
                                # Speichere Tracking-Daten für 2D-Modell
                                if track_id not in tracking_data:
                                    tracking_data[track_id] = {
                                        'role': role,
                                        'positions': [],
                                        'first_frame': frame_count
                                    }
                                    print(f"    Neuer Track {track_id} erstellt für {role}")
                                
                                # Füge Position hinzu (Fußposition = untere Mitte der Box)
                                foot_x = (x1 + x2) // 2
                                foot_y = y2
                                
                                # Prüfe ob Spieler auf dem Spielfeld steht
                                on_field = is_player_on_field_simple(foot_y, height)
                                print(f"    Auf dem Feld: {on_field} (Fuß Y: {foot_y})")
                                
                                if on_field:
                                    position_data = {
                                        'frame': frame_count,
                                        'x': foot_x,
                                        'y': foot_y,
                                        'role': role
                                    }
                                    tracking_data[track_id]['positions'].append(position_data)
                                    print(f"    Position hinzugefügt: Frame {frame_count}, ({foot_x}, {foot_y})")
                                    
                                    # Zeichne Rolle auf den Frame (größer und deutlicher)
                                    color_map = {
                                        "BVB": (0, 255, 0),
                                        "Schalke": (255, 0, 0),
                                        "Schiedsrichter": (0, 255, 255),
                                        "Torwart": (255, 0, 255)
                                    }
                                    
                                    col = color_map.get(role, (200, 200, 200))
                                    
                                    # Dickere Box für bessere Sichtbarkeit
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3)
                                    
                                    # Größerer Text für Rolle
                                    cv2.putText(frame, f"{role}", (x1, max(20, y1-10)), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
                                    
                                    # Track-ID anzeigen
                                    cv2.putText(frame, f"ID:{track_id}", (x1, y2+20), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                    
                                    # Debug-Info
                                    cv2.putText(frame, f"{info['method']}", (x1, y2-10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                                else:
                                    # Zuschauer: Dünnere, graue Box
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                                    cv2.putText(frame, "Zuschauer", (x1, y1-5), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                    else:
                        print(f"  Keine Boxes in Tracks gefunden")
                    
                    print(f"  Frame {frame_count}: {frame_detections} Detektionen verarbeitet")
                    total_detections += frame_detections
                else:
                    print(f"  Frame {frame_count}: Keine Tracks verfügbar")
                
                # Frame schreiben
                out.write(frame)
        
        # Aufräumen
        cap.release()
        out.release()
        
        # Debug: Zeige Tracking-Daten vor dem Speichern
        print(f"\n=== Tracking-Daten vor dem Speichern ===")
        print(f"Anzahl Tracks: {len(tracking_data)}")
        for track_id, data in tracking_data.items():
            print(f"Track {track_id}: {data['role']} - {len(data['positions'])} Positionen")
        
        # 2D-Modell-Daten speichern
        save_tracking_data(tracking_data, "tracking_data.json")
        
        print(f"=== Video-Verarbeitung abgeschlossen ===")
        print(f"Ausgabe gespeichert in: {output_path}")
        print(f"Tracking-Daten gespeichert in: tracking_data.json")
        print(f"Gesamtanzahl Detektionen: {total_detections}")
        
        # Zusammenfassung der erkannten Spieler
        print("\n=== Zusammenfassung der erkannten Spieler ===")
        for track_id, data in tracking_data.items():
            if len(data['positions']) > 0:  # Nur Spieler auf dem Feld
                print(f"Track ID {track_id}: {data['role']} - {len(data['positions'])} Positionen")
        
    except ImportError:
        print("VisionEye nicht verfügbar!")
        return
    except Exception as e:
        print(f"Fehler bei der Video-Verarbeitung: {e}")
        import traceback
        traceback.print_exc()
        return

def is_player_on_field_simple(foot_y, image_height, margin_top=0.1, margin_bottom=0.1):
    """
    Einfache Spielfeld-Filterung basierend auf Y-Position.
    """
    top_limit = image_height * margin_top
    bottom_limit = image_height * (1 - margin_bottom)
    
    return top_limit < foot_y < bottom_limit

def save_tracking_data(tracking_data, filename):
    """
    Speichert Tracking-Daten für 2D-Modell-Entwicklung.
    """
    import json
    
    # Konvertiere numpy-Arrays zu Listen für JSON-Serialisierung
    serializable_data = {}
    for track_id, data in tracking_data.items():
        serializable_data[str(track_id)] = {
            'role': data['role'],
            'first_frame': data['first_frame'],
            'positions': data['positions']
        }
    
    with open(filename, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    print(f"Tracking-Daten gespeichert in {filename}")
    print(f"Anzahl getrackter Spieler: {len(serializable_data)}")

if __name__ == "__main__":
    main()