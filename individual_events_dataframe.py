# Individuelle Events DataFrame für DataSpell
# ==========================================

import json
import pandas as pd
import numpy as np

# Pandas Optionen für bessere Darstellung
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 100)

def load_individual_events_dataframe(json_file_path):
    """
    Lädt die event_database.json und erstellt einen DataFrame wo JEDES EVENT eine eigene Zeile ist.
    """
    
    # JSON-Datei laden
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Liste für alle individuellen Events
    all_individual_events = []
    
    print(f"🔄 Verarbeite Events aus {len(data['events_by_match'])} Matches...")
    
    # Durch alle Matches iterieren
    for match_id, match_data in data['events_by_match'].items():
        match_info = match_data['match_info']
        events = match_data.get('events', [])
        
        print(f"   📋 {match_id}: {len(events)} Events")
        
        # Durch JEDES EINZELNE Event iterieren
        for event_index, event in enumerate(events):
            # Flache Kopie des Events erstellen
            individual_event = {}
            
            # Grundlegende Event-Daten
            individual_event['event_number'] = event_index + 1
            individual_event['event_id'] = event.get('event_id', f"{match_id}_event_{event_index}")
            individual_event['timestamp'] = event.get('timestamp', None)
            individual_event['player'] = event.get('player', '')
            individual_event['position'] = event.get('position', '')
            individual_event['action_type'] = event.get('action_type', '')
            individual_event['outcome'] = event.get('outcome', '')
            individual_event['team'] = event.get('team', '')
            individual_event['half'] = event.get('half', None)
            
            # Positionsdaten
            individual_event['start_x'] = event.get('start_x', None)
            individual_event['start_y'] = event.get('start_y', None)
            individual_event['end_x'] = event.get('end_x', None)
            individual_event['end_y'] = event.get('end_y', None)
            
            # Match-Informationen hinzufügen
            individual_event['match_id'] = match_info.get('match_id', match_id)
            individual_event['match_date'] = match_info.get('date', '')
            individual_event['match_team'] = match_info.get('team', '')
            individual_event['match_opponent'] = match_info.get('opponent', '')
            individual_event['match_youth_level'] = match_info.get('youth_level', '')
            individual_event['match_venue'] = match_info.get('venue', '')
            
            # Additional Data flach machen
            additional_data = event.get('additional_data', {})
            for key, value in additional_data.items():
                # Spaces und spezielle Zeichen in Spaltennamen ersetzen
                clean_key = key.replace(' ', '_').replace('-', '_')
                individual_event[f'add_{clean_key}'] = value
            
            # Passing Network Daten
            passing_network = event.get('passing_network', {})
            for key, value in passing_network.items():
                clean_key = key.replace(' ', '_').replace('-', '_')
                individual_event[f'pass_{clean_key}'] = value
            
            # Event zur Liste hinzufügen
            all_individual_events.append(individual_event)
    
    # DataFrame erstellen
    df = pd.DataFrame(all_individual_events)
    
    print(f"✅ DataFrame mit {len(df)} individuellen Events erstellt!")
    return df

def analyze_individual_events(df):
    """
    Analysiert den DataFrame mit individuellen Events
    """
    
    print("\n" + "="*80)
    print("📊 INDIVIDUELLE EVENTS ANALYSE")
    print("="*80)
    
    # Grundlegende Statistiken
    print(f"\n📈 Gesamtstatistiken:")
    print(f"   • Anzahl Events: {len(df):,}")
    print(f"   • Anzahl Spalten: {len(df.columns)}")
    print(f"   • Anzahl Matches: {df['match_id'].nunique()}")
    print(f"   • Anzahl Spieler: {df['player'].nunique()}")
    print(f"   • Anzahl Action Types: {df['action_type'].nunique()}")
    
    # Erste Events anzeigen
    print(f"\n📋 Erste 5 Events:")
    print(df.head())
    
    # Action Types Verteilung
    print(f"\n⚽ Action Types Verteilung:")
    action_counts = df['action_type'].value_counts()
    print(action_counts)
    
    # Spieler Aktivität
    print(f"\n👤 Top 10 aktivste Spieler:")
    player_counts = df['player'].value_counts().head(10)
    print(player_counts)
    
    # Erfolgsrate
    print(f"\n🎯 Erfolgsrate gesamt:")
    success_rate = (df['outcome'] == 'Erfolgreich').mean() * 100
    print(f"   {success_rate:.1f}% aller Events waren erfolgreich")
    
    # Erfolgsrate pro Action Type
    print(f"\n🎯 Erfolgsrate pro Action Type:")
    success_by_action = df.groupby('action_type')['outcome'].apply(
        lambda x: (x == 'Erfolgreich').sum() / len(x) * 100 if len(x) > 0 else 0
    ).round(1).sort_values(ascending=False)
    print(success_by_action)
    
    # Events pro Match
    print(f"\n🏟️ Events pro Match:")
    events_per_match = df['match_id'].value_counts().sort_values(ascending=False)
    print(events_per_match)
    
    # Zeitliche Verteilung
    if 'timestamp' in df.columns:
        print(f"\n⏰ Zeitliche Verteilung der Events:")
        timestamp_stats = df['timestamp'].describe()
        print(timestamp_stats)
    
    # Halbzeit Verteilung
    if 'half' in df.columns:
        print(f"\n🕐 Events pro Halbzeit:")
        half_counts = df['half'].value_counts().sort_index()
        print(half_counts)

def main():
    """
    Hauptfunktion zum Laden und Analysieren der individuellen Events
    """
    
    json_file_path = 'Datenmerging/event_database.json'
    
    try:
        # DataFrame mit individuellen Events laden
        print("📊 Lade individuelle Events aus event_database.json...")
        events_df = load_individual_events_dataframe(json_file_path)
        
        # Analyse durchführen
        analyze_individual_events(events_df)
        
        # DataFrame für weitere Verwendung verfügbar machen
        globals()['events_df'] = events_df
        
        print("\n" + "="*80)
        print("🚀 DATAFRAME VERFÜGBAR FÜR WEITERE ANALYSEN")
        print("="*80)
        
        print(f"""
✅ Der DataFrame 'events_df' ist jetzt verfügbar!

🔍 NÜTZLICHE BEFEHLE:

   # Grundlegende Exploration
   events_df.head(10)                           # Erste 10 Events
   events_df.tail(10)                           # Letzte 10 Events
   events_df.info()                             # Spalten-Informationen
   events_df.describe()                         # Statistische Kennzahlen
   
   # Filtern nach Kriterien
   events_df[events_df['action_type'] == 'PASS']              # Nur Pässe
   events_df[events_df['player'] == 'Mika Rohmann']           # Events von Mika Rohmann
   events_df[events_df['outcome'] == 'Erfolgreich']           # Nur erfolgreiche Events
   events_df[events_df['match_id'] == 'U12_BVB-BMG_2023-11-04']  # Events eines Matches
   
   # Sortieren
   events_df.sort_values('timestamp')                         # Nach Zeit sortieren
   events_df.sort_values(['match_id', 'timestamp'])           # Nach Match und Zeit
   
   # Spieler-spezifische Analysen
   events_df[events_df['player'] == 'Spielername']['action_type'].value_counts()
   events_df.groupby('player')['outcome'].apply(lambda x: (x=='Erfolgreich').mean()*100)
   
   # Match-spezifische Analysen
   events_df[events_df['match_id'] == 'Match_ID']['action_type'].value_counts()
   
   # Exportieren für weitere Verwendung
   events_df.to_csv('alle_events.csv', index=False)          # Als CSV speichern
   events_df.to_excel('alle_events.xlsx', index=False)       # Als Excel speichern
""")
        
        return events_df
        
    except FileNotFoundError:
        print(f"❌ Fehler: Datei '{json_file_path}' nicht gefunden!")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Fehler beim Lesen der JSON-Datei: {e}")
        return None
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        return None

if __name__ == "__main__":
    # Hauptfunktion ausführen
    events_df = main()
    
    # Zusätzliche Information
    if events_df is not None:
        print(f"\n💡 TIPP: Jede Zeile im DataFrame repräsentiert EIN EINZELNES EVENT!")
        print(f"📊 Insgesamt {len(events_df):,} individuelle Events verfügbar.")
        
        # DataFrame auch in Console verfügbar machen für DataSpell
        print(f"\n🎯 DATAFRAME ZUGRIFF:")
        print(f"   Variable: events_df")
        print(f"   Dimensionen: {events_df.shape}")
        print(f"   Spalten: {list(events_df.columns[:10])}..." if len(events_df.columns) > 10 else f"   Spalten: {list(events_df.columns)}") 