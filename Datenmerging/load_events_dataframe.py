import json
import pandas as pd
from pathlib import Path

def load_events_as_dataframe(json_file_path):
    """
    Lädt die event_database.json Datei und konvertiert alle Events in einen pandas DataFrame.
    
    Args:
        json_file_path (str): Pfad zur JSON-Datei
    
    Returns:
        pandas.DataFrame: DataFrame mit allen Events
    """
    
    # JSON-Datei laden
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Liste für alle Events
    all_events = []
    
    # Durch alle Matches iterieren
    for match_id, match_data in data['events_by_match'].items():
        match_info = match_data['match_info']
        events = match_data['events']
        
        # Durch alle Events des Matches iterieren
        for event in events:
            # Event-Daten kopieren
            event_row = event.copy()
            
            # Match-Informationen hinzufügen
            event_row['match_id'] = match_info['match_id']
            event_row['match_date'] = match_info['date']
            event_row['match_team'] = match_info['team']
            event_row['match_opponent'] = match_info['opponent']
            event_row['match_youth_level'] = match_info['youth_level']
            event_row['match_venue'] = match_info['venue']
            
            # Verschachtelte Daten flach machen
            if 'additional_data' in event_row:
                additional_data = event_row.pop('additional_data')
                for key, value in additional_data.items():
                    event_row[f'additional_{key}'] = value
            
            if 'passing_network' in event_row:
                passing_network = event_row.pop('passing_network')
                for key, value in passing_network.items():
                    event_row[f'passing_{key}'] = value
            
            all_events.append(event_row)
    
    # DataFrame erstellen
    df = pd.DataFrame(all_events)
    
    return df

def main():
    """
    Hauptfunktion zum Laden und Anzeigen der Events als DataFrame.
    """
    
    # Pfad zur JSON-Datei
    json_file_path = 'event_database.json'
    
    try:
        # DataFrame laden
        print("Lade event_database.json...")
        events_df = load_events_as_dataframe(json_file_path)
        
        # Informationen über den DataFrame ausgeben
        print(f"\nDataFrame erfolgreich erstellt!")
        print(f"Anzahl Events: {len(events_df)}")
        print(f"Anzahl Spalten: {len(events_df.columns)}")
        
        # Übersicht über die Spalten
        print(f"\nSpalten im DataFrame:")
        for i, col in enumerate(events_df.columns, 1):
            print(f"{i:2d}. {col}")
        
        # Erste 5 Zeilen anzeigen
        print(f"\nErste 5 Events:")
        print(events_df.head())
        
        # Grundlegende Statistiken
        print(f"\nÜbersicht über Action Types:")
        print(events_df['action_type'].value_counts())
        
        print(f"\nÜbersicht über Matches:")
        print(events_df['match_id'].value_counts())
        
        print(f"\nÜbersicht über Spieler (Top 10):")
        print(events_df['player'].value_counts().head(10))
        
        # DataFrame für weitere Analyse verfügbar machen
        globals()['events_df'] = events_df
        print(f"\n✅ DataFrame ist als 'events_df' Variable verfügbar!")
        print("Sie können jetzt mit dem DataFrame arbeiten, z.B.:")
        print("- events_df.head() für die ersten Zeilen")
        print("- events_df.info() für Informationen über die Datentypen")
        print("- events_df.describe() für statistische Kennzahlen")
        print("- events_df[events_df['action_type'] == 'PASS'] für alle Pässe")
        
        return events_df
        
    except FileNotFoundError:
        print(f"❌ Fehler: Datei '{json_file_path}' nicht gefunden!")
        print("Bitte überprüfen Sie den Pfad zur JSON-Datei.")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Fehler beim Lesen der JSON-Datei: {e}")
        return None
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        return None

if __name__ == "__main__":
    events_df = main() 