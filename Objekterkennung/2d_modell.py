import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from datetime import datetime
import os

def load_tracking_data(filename="tracking_data.json"):
    """
    Lädt die Tracking-Daten aus der JSON-Datei.
    """
    if not os.path.exists(filename):
        print(f"Tracking-Datei {filename} nicht gefunden!")
        return None
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    print(f"Tracking-Daten geladen: {len(data)} Spieler")
    return data

def convert_coordinates_to_field_system(tracking_data, image_width, image_height):
    """
    Konvertiert Bildkoordinaten in das standardisierte Fußballfeld-Koordinatensystem.
    
    Fußballfeld: Mittelpunkt (0,0), x: -52.5m bis +52.5m, y: -34m bis +34m
    
    Args:
        tracking_data: Dictionary mit Tracking-Daten
        image_width: Breite des Originalbildes
        image_height: Höhe des Originalbildes
    
    Returns:
        dict: Konvertierte Koordinaten im Feld-System
    """
    # Fußballfeld-Dimensionen (in Metern)
    field_width = 105  # -52.5m bis +52.5m
    field_height = 68  # -34m bis +34m
    
    # Annahme: Das Bild zeigt das gesamte Spielfeld
    # Wir müssen die Bildkoordinaten auf das Feld-System mappen
    
    converted_data = {}
    
    for track_id, player_data in tracking_data.items():
        if len(player_data['positions']) > 0:
            converted_positions = []
            
            for pos in player_data['positions']:
                # Bildkoordinaten (x, y)
                img_x = pos['x']
                img_y = pos['y']
                
                # Konvertiere zu Feld-Koordinaten
                # x: links (0) -> -52.5m, rechts (image_width) -> +52.5m
                field_x = (img_x / image_width) * field_width - field_width/2
                
                # y: oben (0) -> +34m, unten (image_height) -> -34m
                # (y-Achse ist invertiert im Bild)
                field_y = -(img_y / image_height) * field_height + field_height/2
                
                converted_positions.append({
                    'frame': pos['frame'],
                    'x': field_x,  # in Metern, -52.5 bis +52.5
                    'y': field_y,  # in Metern, -34 bis +34
                    'role': pos['role']
                })
            
            converted_data[track_id] = {
                'role': player_data['role'],
                'first_frame': player_data['first_frame'],
                'positions': converted_positions
            }
    
    return converted_data

def create_football_field_2d():
    """
    Erstellt ein standardisiertes Fußballfeld für das 2D-Modell.
    Mittelpunkt (0,0), x: -52.5m bis +52.5m, y: -34m bis +34m
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Spielfeldmaße
    field_width = 105
    field_height = 68
    
    # Zeichne das Spielfeld - BVB Gelb
    ax.add_patch(patches.Rectangle((-field_width/2, -field_height/2), field_width, field_height, 
                                 fill=True, color='#ffc400', alpha=0.8))
    ax.add_patch(patches.Rectangle((-field_width/2, -field_height/2), field_width, field_height, 
                                 fill=False, color='black', linewidth=2))
    
    # Mittellinie
    ax.plot([0, 0], [-field_height/2, field_height/2], 'k-', linewidth=2)
    
    # Mittelfeldkreis
    center_circle_radius = 9.975  # 105 * 0.095
    center_circle = patches.Circle((0, 0), center_circle_radius, fill=False, color='black', linewidth=2)
    ax.add_patch(center_circle)
    
    # Strafräume
    penalty_area_width = 16.485  # 105 * 0.157
    penalty_area_height = 40.324  # 68 * 0.593
    ax.add_patch(patches.Rectangle((-field_width/2, -penalty_area_height/2), penalty_area_width, penalty_area_height, 
                                 fill=False, color='black', linewidth=2))
    ax.add_patch(patches.Rectangle((field_width/2 - penalty_area_width, -penalty_area_height/2), 
                                 penalty_area_width, penalty_area_height, fill=False, color='black', linewidth=2))
    
    # Torräume
    goal_area_width = 5.985  # 105 * 0.057
    goal_area_height = 19.992  # 68 * 0.294
    ax.add_patch(patches.Rectangle((-field_width/2, -goal_area_height/2), goal_area_width, goal_area_height, 
                                 fill=False, color='black', linewidth=2))
    ax.add_patch(patches.Rectangle((field_width/2 - goal_area_width, -goal_area_height/2), 
                                 goal_area_width, goal_area_height, fill=False, color='black', linewidth=2))
    
    # Achsen konfigurieren
    field_margin = 5
    ax.set_xlim(-field_width/2 - field_margin, field_width/2 + field_margin)
    ax.set_ylim(-field_height/2 - field_margin, field_height/2 + field_margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X-Koordinate (Meter)')
    ax.set_ylabel('Y-Koordinate (Meter)')
    ax.set_title('Fußballfeld 2D-Modell (Mittelpunkt: 0,0)')
    
    return fig, ax

def plot_player_positions(fig, ax, converted_data, frame_number=None):
    """
    Zeichnet die Spieler-Positionen auf dem Fußballfeld.
    
    Args:
        fig: Matplotlib Figure
        ax: Matplotlib Axes
        converted_data: Konvertierte Tracking-Daten
        frame_number: Bestimmter Frame (None = alle Frames)
    """
    # Farben für verschiedene Rollen
    color_map = {
        "BVB": "green",
        "Schalke": "blue", 
        "Schiedsrichter": "yellow",
        "Torwart": "red"
    }
    
    # Zeichne Spieler-Positionen
    for track_id, player_data in converted_data.items():
        role = player_data['role']
        positions = player_data['positions']
        
        if frame_number is not None:
            # Nur bestimmten Frame anzeigen
            frame_positions = [pos for pos in positions if pos['frame'] == frame_number]
        else:
            # Alle Frames anzeigen
            frame_positions = positions
        
        if frame_positions:
            # Extrahiere Koordinaten
            x_coords = [pos['x'] for pos in frame_positions]
            y_coords = [pos['y'] for pos in frame_positions]
            
            # Zeichne Spieler als Punkte
            color = color_map.get(role, "gray")
            ax.scatter(x_coords, y_coords, c=color, s=100, alpha=0.7, 
                      label=f"{role} (ID: {track_id})")
            
            # Zeichne Bewegungslinie zwischen Frames
            if len(frame_positions) > 1:
                ax.plot(x_coords, y_coords, color=color, alpha=0.5, linewidth=2)
    
    # Legende
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    return fig, ax

def create_animation_frames(converted_data, output_dir="animation_frames"):
    """
    Erstellt Einzelbilder für eine Animation der Spieler-Bewegungen.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Finde alle Frame-Nummern
    all_frames = set()
    for player_data in converted_data.values():
        for pos in player_data['positions']:
            all_frames.add(pos['frame'])
    
    all_frames = sorted(list(all_frames))
    
    print(f"Erstelle {len(all_frames)} Animations-Frames...")
    
    for frame_num in all_frames:
        # Erstelle Fußballfeld
        fig, ax = create_football_field_2d()
        
        # Zeichne Spieler-Positionen für diesen Frame
        fig, ax = plot_player_positions(fig, ax, converted_data, frame_num)
        
        # Frame-Nummer anzeigen
        ax.text(0, 35, f"Frame: {frame_num}", fontsize=12, ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Speichere Frame
        output_path = os.path.join(output_dir, f"frame_{frame_num:03d}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if frame_num % 10 == 0:
            print(f"  Frame {frame_num}/{len(all_frames)} gespeichert")
    
    print(f"Alle Frames in {output_dir} gespeichert!")

def analyze_player_movements(converted_data):
    """
    Analysiert die Bewegungen der Spieler und erstellt Statistiken.
    """
    analysis = {}
    
    for track_id, player_data in converted_data.items():
        role = player_data['role']
        positions = player_data['positions']
        
        if len(positions) > 1:
            # Berechne Bewegungsstatistiken
            x_coords = [pos['x'] for pos in positions]
            y_coords = [pos['y'] for pos in positions]
            
            # Gesamtstrecke
            total_distance = 0
            for i in range(1, len(positions)):
                dx = x_coords[i] - x_coords[i-1]
                dy = y_coords[i] - y_coords[i-1]
                total_distance += np.sqrt(dx*dx + dy*dy)
            
            # Durchschnittsgeschwindigkeit (m/Frame)
            avg_speed = total_distance / (len(positions) - 1) if len(positions) > 1 else 0
            
            # Bewegungsbereich
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)
            
            analysis[track_id] = {
                'role': role,
                'total_frames': len(positions),
                'total_distance': total_distance,
                'avg_speed': avg_speed,
                'x_range': x_range,
                'y_range': y_range,
                'start_position': (x_coords[0], y_coords[0]),
                'end_position': (x_coords[-1], y_coords[-1])
            }
    
    return analysis

def main():
    """
    Hauptfunktion zum Erstellen des 2D-Modells.
    """
    print("=== 2D-Fußballfeld-Modell Generator ===")
    
    # 1. Tracking-Daten laden
    tracking_data = load_tracking_data()
    if tracking_data is None:
        return
    
    # 2. Bild-Dimensionen (aus dem Video)
    # Diese Werte müssen an dein Video angepasst werden
    image_width = 1080  # Anpassen an dein Video (aus Terminal: 1080x1920)
    image_height = 1920  # Anpassen an dein Video (aus Terminal: 1080x1920)
    
    print(f"Bild-Dimensionen: {image_width}x{image_height}")
    
    # 3. Koordinaten konvertieren
    print("Konvertiere Koordinaten...")
    converted_data = convert_coordinates_to_field_system(tracking_data, image_width, image_height)
    
    # 4. 2D-Modell erstellen
    print("Erstelle 2D-Modell...")
    fig, ax = create_football_field_2d()
    
    # 5. Spieler-Positionen zeichnen
    fig, ax = plot_player_positions(fig, ax, converted_data)
    
    # 6. Modell speichern
    output_path = "football_field_2d_model.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"2D-Modell gespeichert: {output_path}")
    
    # 7. Animation-Frames erstellen
    print("Erstelle Animations-Frames...")
    create_animation_frames(converted_data)
    
    # 8. Bewegungsanalyse
    print("Analysiere Spieler-Bewegungen...")
    analysis = analyze_player_movements(converted_data)
    
    # 9. Analyse speichern
    analysis_path = "player_movement_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Bewegungsanalyse gespeichert: {analysis_path}")
    
    # 10. Zusammenfassung anzeigen
    print("\n=== Zusammenfassung ===")
    for track_id, stats in analysis.items():
        print(f"Spieler {track_id} ({stats['role']}):")
        print(f"  Frames: {stats['total_frames']}")
        print(f"  Gesamtstrecke: {stats['total_distance']:.2f}m")
        print(f"  Durchschnittsgeschwindigkeit: {stats['avg_speed']:.2f}m/Frame")
        print(f"  Bewegungsbereich: {stats['x_range']:.2f}m x {stats['y_range']:.2f}m")
        print()
    
    print("=== 2D-Modell erfolgreich erstellt! ===")

if __name__ == "__main__":
    main() 