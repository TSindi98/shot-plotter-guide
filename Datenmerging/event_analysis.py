# Event Database DataFrame Analyse für DataSpell
# ================================================

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Pandas Optionen für bessere Darstellung
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

def load_events_as_dataframe(json_file_path):
    """
    Lädt die event_database.json Datei und konvertiert alle Events in einen pandas DataFrame.
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

# =============================================================================
# HAUPTANALYSE
# =============================================================================

# DataFrame laden
print("📊 Lade event_database.json...")
json_file_path = 'Datenmerging/event_database.json'
events_df = load_events_as_dataframe(json_file_path)

print(f"✅ DataFrame erfolgreich erstellt!")
print(f"📈 Anzahl Events: {len(events_df):,}")
print(f"📊 Anzahl Spalten: {len(events_df.columns)}")
print(f"🗓️ Anzahl Matches: {events_df['match_id'].nunique()}")

# =============================================================================
# DATAFRAME ÜBERSICHT
# =============================================================================

print("\n" + "="*60)
print("📋 DATAFRAME ÜBERSICHT")
print("="*60)

# Spalten anzeigen
print("\n🏷️ Verfügbare Spalten:")
for i, col in enumerate(events_df.columns, 1):
    print(f"{i:2d}. {col}")

# Erste Zeilen
print(f"\n📄 Erste 3 Events:")
display(events_df.head(3))

# Grundlegende Statistiken
print(f"\n📊 Datentypen und fehlende Werte:")
display(events_df.info())

# =============================================================================
# STATISTISCHE ANALYSEN
# =============================================================================

print("\n" + "="*60)
print("📈 STATISTISCHE ANALYSEN")
print("="*60)

# Action Types
print("\n⚽ Action Types:")
action_counts = events_df['action_type'].value_counts()
display(action_counts)

# Erfolgsrate
print("\n🎯 Erfolgsrate nach Action Type:")
success_rate = events_df.groupby('action_type')['outcome'].apply(
    lambda x: (x == 'Erfolgreich').sum() / len(x) * 100
).round(1).sort_values(ascending=False)
display(success_rate)

# Matches
print("\n🏟️ Events pro Match:")
match_counts = events_df['match_id'].value_counts()
display(match_counts)

# Top Spieler
print("\n👤 Top 10 aktivste Spieler:")
top_players = events_df['player'].value_counts().head(10)
display(top_players)

# Youth Level Distribution
print("\n👶 Events nach Altersklasse:")
youth_level_counts = events_df['match_youth_level'].value_counts()
display(youth_level_counts)

# =============================================================================
# VISUALISIERUNGEN
# =============================================================================

print("\n" + "="*60)
print("📊 VISUALISIERUNGEN")
print("="*60)

# 1. Action Types
plt.figure(figsize=(12, 6))
action_counts.plot(kind='bar', color='steelblue')
plt.title('🎬 Verteilung der Action Types', fontsize=14, fontweight='bold')
plt.xlabel('Action Type')
plt.ylabel('Anzahl Events')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Erfolgsrate
plt.figure(figsize=(12, 6))
success_rate.plot(kind='bar', color='green')
plt.title('🎯 Erfolgsrate nach Action Type (%)', fontsize=14, fontweight='bold')
plt.xlabel('Action Type')
plt.ylabel('Erfolgsrate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Events pro Match
plt.figure(figsize=(12, 6))
match_counts.plot(kind='bar', color='orange')
plt.title('🏟️ Events pro Match', fontsize=14, fontweight='bold')
plt.xlabel('Match ID')
plt.ylabel('Anzahl Events')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =============================================================================
# SPEZIALANALYSEN
# =============================================================================

print("\n" + "="*60)
print("🔍 SPEZIALANALYSEN")
print("="*60)

# Nur Pässe
passes_df = events_df[events_df['action_type'] == 'PASS']
print(f"\n⚽ Pass-Analyse:")
print(f"   Anzahl Pässe: {len(passes_df):,}")
if len(passes_df) > 0:
    pass_success = (passes_df['outcome'] == 'Erfolgreich').mean() * 100
    print(f"   Passgenauigkeit: {pass_success:.1f}%")

# Schüsse
shots_df = events_df[events_df['action_type'] == 'SHOT']
print(f"\n🥅 Schuss-Analyse:")
print(f"   Anzahl Schüsse: {len(shots_df):,}")
if len(shots_df) > 0:
    shot_success = (shots_df['outcome'] == 'Erfolgreich').mean() * 100
    print(f"   Trefferquote: {shot_success:.1f}%")

# Halbzeit-Verteilung
if 'half' in events_df.columns:
    print(f"\n⏰ Events nach Halbzeit:")
    half_dist = events_df['half'].value_counts().sort_index()
    display(half_dist)

# =============================================================================
# ANWEISUNGEN FÜR WEITERE ANALYSEN
# =============================================================================

print("\n" + "="*60)
print("🚀 WEITERE ANALYSEMÖGLICHKEITEN")
print("="*60)

print("""
Der DataFrame 'events_df' ist jetzt verfügbar! Hier sind einige nützliche Befehle:

🔍 EXPLORATION:
   events_df.head()                    # Erste Zeilen anzeigen
   events_df.info()                    # Spalten-Informationen
   events_df.describe()                # Statistische Kennzahlen
   events_df.shape                     # Dimensionen des DataFrames

🎯 FILTERING:
   events_df[events_df['action_type'] == 'PASS']                    # Nur Pässe
   events_df[events_df['outcome'] == 'Erfolgreich']                # Nur erfolgreiche Events
   events_df[events_df['player'] == 'Spielername']                 # Events eines Spielers
   events_df[events_df['match_youth_level'] == 'U12']              # Nur U12 Events

📊 GRUPPIERUNGEN:
   events_df.groupby('player')['action_type'].value_counts()       # Actions pro Spieler
   events_df.groupby('match_id')['outcome'].value_counts()         # Outcomes pro Match
   events_df.groupby(['action_type', 'outcome']).size()           # Action-Outcome Kombinationen

📈 AGGREGATIONEN:
   events_df['timestamp'].describe()                               # Zeitstatistiken
   events_df.groupby('action_type')['timestamp'].mean()           # Durchschnittliche Zeiten
   events_df.pivot_table(values='timestamp', index='player', columns='action_type', aggfunc='count')

🎨 VISUALISIERUNGEN:
   events_df['action_type'].value_counts().plot(kind='pie')        # Kreisdiagramm
   events_df.boxplot(column='timestamp', by='action_type')         # Boxplot
   events_df.hist(figsize=(15, 10))                               # Histogramm aller numerischen Spalten
""")

print(f"\n✅ DataFrame 'events_df' ist bereit für Ihre Analysen in DataSpell!")
print(f"💡 Tipp: Verwenden Sie display(events_df) für eine schöne tabellarische Darstellung") 