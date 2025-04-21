import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import io
import base64
import json
from datetime import datetime, timedelta
import os
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import re
import plotly.graph_objects as go
import plotly.express as px

# Seitentitel und Beschreibung
st.set_page_config(page_title="Fußball-Passdaten Integration", layout="wide")
st.title("Fußball-Passdaten Integration")
st.markdown("""
Diese App verarbeitet und integriert Daten aus drei Quellen:
1. **Shot-Plotter CSV**: Enthält Positionsdaten
2. **Playermaker XML**: Enthält Passdaten
3. **Playermaker Possession Excel**: Enthält Zeitdaten
""")

# XML Parser aus XML_Parser.py importieren
@dataclass
class BallPossessionEvent:
    event_id: int
    start_time: float
    end_time: float
    player: str                  # aus code
    passed_from: Optional[str]   # aus "Passed by" label
    passed_to: Optional[str]     # aus "Passed to" label
    release_leg: Optional[str]   # aus Technical Balance
    release_velocity: Optional[str]  # aus Release Velocity
    in_team_possession: bool     # aus IBP in TBP
    action_type: str  # 'PASS', 'LOSS', oder 'POTENTIAL_SHOT'

    def to_dict(self):
        return {
            'event_id': self.event_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.end_time - self.start_time,
            'player': self.player,
            'passed_from': self.passed_from,
            'passed_to': self.passed_to,
            'release_leg': self.release_leg,
            'release_velocity': self.release_velocity,
            'in_team_possession': self.in_team_possession,
            'action_type': self.action_type
        }

def parse_playermaker_data(xml_file) -> List[BallPossessionEvent]:
    """Parst die Playermaker XML und gibt eine Liste von BallPossessionEvent-Objekten zurück"""
    content = xml_file.read()
    root = ET.fromstring(content)
    events = []
    
    # Nur Instanzen mit IBP verarbeiten
    for instance in root.findall('.//instance'):
        has_ibp = False
        for label in instance.findall('label'):
            group = label.find('group')
            if group is not None and group.text == 'IBP':
                has_ibp = True
                break
        
        if not has_ibp:
            continue
            
        # Basis-Informationen
        original_end_time = float(instance.find('end').text)
        
        # Zeitkorrektur: 4 Sekunden abziehen für die exakte Übereinstimmung mit der Possession Summary
        corrected_end_time = original_end_time - 4.0
        
        event = BallPossessionEvent(
            event_id=int(instance.find('ID').text),
            start_time=float(instance.find('start').text),
            end_time=corrected_end_time,  # Korrigierte Endzeit
            player=instance.find('code').text,
            passed_from=None,
            passed_to=None,
            release_leg=None,
            release_velocity=None,
            in_team_possession=False,
            action_type='LOSS'  # Standard-Wert, wird später aktualisiert
        )
        
        # Label-Informationen extrahieren
        for label in instance.findall('label'):
            group = label.find('group')
            text = label.find('text')
            
            if group is None or text is None:
                continue
                
            if group.text == 'Passing Network':
                if 'Passed by' in text.text:
                    event.passed_from = text.text.replace('Passed by ', '')
                elif 'Passed to' in text.text:
                    event.passed_to = text.text.replace('Passed to ', '')
                    event.action_type = 'PASS'
                    
            elif group.text == 'Technical Balance' and 'leg Release' in text.text:
                event.release_leg = text.text.split()[0]  # Extrahiert 'RIGHT' oder 'LEFT'
                
            elif group.text == 'Release Velocity':
                event.release_velocity = text.text
                
            elif group.text == 'IBP in TBP':
                event.in_team_possession = 'In team ball possession' in text.text
        
        events.append(event)
    
    # Nach Zeit sortieren
    sorted_events = sorted(events, key=lambda x: x.end_time)
    st.write(f"XML nach Zeit sortiert: {len(sorted_events)} Einträge")
    return sorted_events

def process_playermaker_possession(df):
    """Verarbeitet DataFrame aus Playermaker Possession Excel oder CSV und berechnet die Endzeit"""
    try:
        # Zuerst prüfen, ob wir die richtige Headerzeile finden und neu indizieren müssen
        # Suche nach der Zeile mit "Season" als erste Spalte
        header_row = -1
        for i, row in df.iterrows():
            first_col = row.iloc[0] if not row.empty else None
            if isinstance(first_col, str) and first_col.lower() == "season":
                header_row = i
                st.success(f"Header-Zeile in Position {i} gefunden")
                break
        
        # Wenn eine Header-Zeile gefunden wurde, neu indizieren
        if header_row >= 0:
            # Extrahiere die Header-Zeile
            header = df.iloc[header_row].values
            # Nehme die Daten ab der nächsten Zeile
            data_rows = df.iloc[header_row + 1:].values
            # Erstelle ein neues DataFrame mit den richtigen Überschriften
            df = pd.DataFrame(data_rows, columns=header)
            st.success("DataFrame mit korrekten Überschriften neu indiziert")
        
        # Wir suchen nach den relevanten Spalten
        time_col = None
        release_time_col = None
        player_name_col = None
        possession_type_col = None
        
        # Debug-Info
        st.write("Verfügbare Spalten nach Neuindizierung:", df.columns.tolist())
        
        # Versuche, die Spalten zu identifizieren
        for col in df.columns:
            if isinstance(col, str):
                col_lower = str(col).lower()
                if "possession time" in col_lower and "min" in col_lower:
                    time_col = col
                    st.success(f"Gefundene Zeitspalte: {col}")
                elif "time to release" in col_lower and "sec" in col_lower:
                    release_time_col = col
                    st.success(f"Gefundene Release-Zeit-Spalte: {col}")
                elif "player name" in col_lower or "playername" in col_lower:
                    player_name_col = col
                    st.success(f"Gefundene Player Name Spalte: {col}")
                elif "possession type" in col_lower or "possessiontype" in col_lower:
                    possession_type_col = col
                    st.success(f"Gefundene Possession Type Spalte: {col}")
        
        # Wenn die Spalten nicht gefunden wurden, suche nach Spaltenindizes in einer CSV
        if time_col is None or release_time_col is None:
            st.warning("Standardspalten nicht gefunden, versuche alternative Methode...")
            
            # Zeige die ersten 5 Zeilen, um zu sehen, was wir haben
            st.write("Erste 5 Zeilen der Daten:")
            st.write(df.head())
            
            # Potenzielle Spalten anzeigen
            potential_columns = []
            for i, col in enumerate(df.columns):
                val_sample = df.iloc[0, i] if not df.empty and i < df.shape[1] else "N/A"
                potential_columns.append(f"Spalte {i}: {col} (Beispielwert: {val_sample})")
            
            st.write("Potenzielle Spalten:", potential_columns)
            
            # Manuelle Auswahl für bestimmte bekannte Formate
            if len(df.columns) >= 20:  # Annahme für das Format der bereitgestellten CSV
                # Für das Format der bereitgestellten CSV
                try:
                    # Überprüfe mögliche Spalten
                    for i, col in enumerate(df.columns):
                        col_str = str(col).lower()
                        if "possession time" in col_str:
                            time_col = col
                            st.success(f"Gefunden: Possession Time in Spalte {i}: {col}")
                        elif "time to release" in col_str:
                            release_time_col = col
                            st.success(f"Gefunden: Time to Release in Spalte {i}: {col}")
                        elif "player name" in col_str or "playername" in col_str:
                            player_name_col = col
                            st.success(f"Gefunden: Player Name in Spalte {i}: {col}")
                        elif "possession type" in col_str or "possessiontype" in col_str:
                            possession_type_col = col
                            st.success(f"Gefunden: Possession Type in Spalte {i}: {col}")
                except Exception as e:
                    st.error(f"Fehler bei der manuellen Spaltenidentifikation: {str(e)}")
        
        # Wenn immer noch keine Spalten gefunden wurden
        if time_col is None or release_time_col is None:
            st.error("Konnte die erforderlichen Spalten nicht automatisch finden.")
            
            # Spalten basierend auf Indizes verwenden
            if len(df.columns) >= 18:
                # Wir verwenden direkt die Spaltenindizes
                # Normalerweise ist "Possession Time" an Position 15 und "Time to Release" an Position 17
                time_col = df.columns[15]  # Index für "Possession Time"
                release_time_col = df.columns[17]  # Index für "Time to Release"
                
                # Versuche auch Player Name und Possession Type zu finden
                if len(df.columns) > 5:
                    player_name_col = df.columns[5]  # Typischer Index für Player Name
                if len(df.columns) > 8:
                    possession_type_col = df.columns[8]  # Typischer Index für Possession Type
                
                st.warning(f"Verwende Spaltenindizes - Zeit: {time_col}, Release: {release_time_col}, Player: {player_name_col}, Type: {possession_type_col}")
            else:
                st.error("Nicht genügend Spalten im DataFrame. Prüfen Sie das Dateiformat.")
                return pd.DataFrame()
        
        # Konvertiere die Zeit-Spalten in numerische Werte falls nötig
        if df[time_col].dtype == 'object':
            # Versuche, nicht-numerische Zeilen zu entfernen
            st.write("Konvertiere Spalten in numerische Werte...")
            
            # Für beide Spalten: Ersetze Kommata durch Punkte (für europäisches Format)
            if isinstance(df[time_col].iloc[0], str):
                df[time_col] = df[time_col].str.replace(',', '.')
            if isinstance(df[release_time_col].iloc[0], str):
                df[release_time_col] = df[release_time_col].str.replace(',', '.')
            
            # Konvertiere zu float
            df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
            df[release_time_col] = pd.to_numeric(df[release_time_col], errors='coerce')
            
            # Entferne Zeilen mit NaN-Werten
            df = df.dropna(subset=[time_col, release_time_col])
        
        # Zeige die ersten Zeilen der bereinigten Daten
        st.write("Erste 3 Zeilen der bereinigten Daten:")
        display_cols = [col for col in [time_col, release_time_col, player_name_col, possession_type_col] if col is not None]
        st.write(df[display_cols].head(3))
        
        # Berechnung der Endzeit in Sekunden
        df['start_time_sec'] = df[time_col] * 60  # Minuten in Sekunden umrechnen
        df['end_time_sec'] = df['start_time_sec'] + df[release_time_col]
        
        # Stellt sicher, dass Player Name und Possession Type Spalten existieren
        if player_name_col is not None:
            df['Player Name'] = df[player_name_col]
        
        if possession_type_col is not None:
            df['Possession Type'] = df[possession_type_col]
        
        st.success(f"Endzeit erfolgreich berechnet für {len(df)} Einträge")
        return df
    
    except Exception as e:
        st.error(f"Fehler beim Verarbeiten der Possession-Daten: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        return pd.DataFrame()

def add_passed_to_and_from_column(possession_df, xml_events_df):
    """Fügt die passed_to und passed_from Spalte aus der XML zur Possession-Datei hinzu"""
    if possession_df.empty or xml_events_df.empty:
        return possession_df
    
    # Prüfe, ob die erforderlichen Spalten existieren
    required_columns = ['passed_to', 'passed_from', 'end_time', 'player']
    missing_columns = [col for col in required_columns if col not in xml_events_df.columns]
    
    if missing_columns:
        st.warning(f"XML-Daten enthalten folgende Spalten nicht: {', '.join(missing_columns)}")
    
    # Debug-Info: Zeige die Anzahl der Einträge und Zeitwerte nur im Debug-Modus
    debug_mode = st.session_state.get('debug_matching', False)
    if debug_mode:
        st.write(f"Possession-Einträge: {len(possession_df)}, XML-Einträge: {len(xml_events_df)}")
    
    # Stelle sicher, dass alle Zeitwerte numerisch sind
    if 'end_time_sec' in possession_df.columns:
        possession_df['end_time_sec'] = pd.to_numeric(possession_df['end_time_sec'], errors='coerce')
    
    if 'end_time' in xml_events_df.columns:
        xml_events_df['end_time'] = pd.to_numeric(xml_events_df['end_time'], errors='coerce')
    
    # Zeige Zeitvergleich nur im Debug-Modus
    if debug_mode and not possession_df.empty and not xml_events_df.empty:
        st.write("Zeitvergleich (erste 5 Einträge):")
        comparison_df = pd.DataFrame({
            'Possession-Endzeit': possession_df['end_time_sec'].head(5).values if 'end_time_sec' in possession_df.columns else ["N/A"] * 5,
            'XML-Endzeit': xml_events_df['end_time'].head(5).values if 'end_time' in xml_events_df.columns else ["N/A"] * 5
        })
        st.write(comparison_df)
    
    # Zeitsynchronisierte Zuordnung über end_time_sec
    if debug_mode:
        st.write("Verwende die Endzeit für die Zuordnung...")
    
    # Erstelle ein mapping von end_time zu passed_to/from Werten aus XML
    xml_time_to_passes = {}
    for _, row in xml_events_df.iterrows():
        if 'end_time' not in row or pd.isna(row['end_time']):
            continue
            
        end_time = float(row['end_time'])  # Explizite Konvertierung zu float
        passed_to = row['passed_to'] if 'passed_to' in row and pd.notna(row['passed_to']) else None
        passed_from = row['passed_from'] if 'passed_from' in row and pd.notna(row['passed_from']) else None
        player = row['player'] if 'player' in row and pd.notna(row['player']) else None
        
        if end_time not in xml_time_to_passes:
            xml_time_to_passes[end_time] = []
        
        xml_time_to_passes[end_time].append({
            'passed_to': passed_to,
            'passed_from': passed_from,
            'player': player
        })
    
    # Kopie des possession_df erstellen, um das Original nicht zu verändern
    updated_possession_df = possession_df.copy()
    
    # Füge die Spalten basierend auf dem nächsten Zeitwert hinzu
    for i, row in updated_possession_df.iterrows():
        if 'end_time_sec' not in row or pd.isna(row['end_time_sec']):
            continue
            
        end_time = float(row['end_time_sec'])  # Explizite Konvertierung zu float
        
        # Finde den nächsten Zeitwert in XML mit einer Toleranz von 0.5 Sekunden
        best_match = None
        min_diff = float('inf')
        
        for xml_time, passes_list in xml_time_to_passes.items():
            diff = abs(float(xml_time) - end_time)  # Stelle sicher, dass beide Werte float sind
            if diff < min_diff and diff <= 0.5:
                min_diff = diff
                best_match = passes_list[0] if passes_list else None
        
        # Füge die Werte hinzu, wenn ein Match gefunden wurde
        if best_match:
            if best_match['passed_to'] is not None:
                updated_possession_df.at[i, 'passed_to'] = best_match['passed_to']
            if best_match['passed_from'] is not None:
                updated_possession_df.at[i, 'passed_from'] = best_match['passed_from']
            if best_match['player'] is not None and ('Player Name' not in updated_possession_df.columns or pd.isna(updated_possession_df.at[i, 'Player Name'])):
                updated_possession_df.at[i, 'Player Name'] = best_match['player']
    
    # Anzahl der hinzugefügten Werte anzeigen
    passed_to_count = updated_possession_df['passed_to'].notna().sum() if 'passed_to' in updated_possession_df.columns else 0
    passed_from_count = updated_possession_df['passed_from'].notna().sum() if 'passed_from' in updated_possession_df.columns else 0
    
    st.success(f"passed_to-Werte: {passed_to_count}, passed_from-Werte: {passed_from_count} hinzugefügt")
    
    return updated_possession_df

def merge_data_by_time(shot_plotter_df, possession_df, time_window=3.0):
    """Führt Shot-Plotter-Daten und Possession-Daten anhand der Zeit zusammen
    mit einem flexiblen Zeitfenster für das bestmögliche Matching.
    
    Benötigte Spalten im finalen Ergebnis:
    - Name (Player Name aus Possession Summary)
    - Zeit (korrigierte Zeit aus XML/Possession Summary)
    - Passed from (aus XML)
    - Passed to (aus XML)
    - Possession type (aus Possession Summary)
    - X, Y Koordinaten (aus CSV/Shot-Plotter)
    """
    debug_mode = st.session_state.get('debug_matching', False)
    
    if shot_plotter_df.empty or possession_df.empty:
        return pd.DataFrame()
    
    # Stelle sicher, dass erforderliche Spalten vorhanden sind
    if 'Time' not in shot_plotter_df.columns:
        st.error("Die Shot-Plotter-Datei enthält keine 'Time'-Spalte.")
        return pd.DataFrame()
    
    if 'end_time_sec' not in possession_df.columns:
        st.error("Die verarbeiteten Possession-Daten enthalten keine 'end_time_sec'-Spalte.")
        return pd.DataFrame()
    
    # Konvertiere Zeitwerte zu float, um Typprobleme zu vermeiden
    shot_plotter_df['Time'] = pd.to_numeric(shot_plotter_df['Time'], errors='coerce')
    possession_df['end_time_sec'] = pd.to_numeric(possession_df['end_time_sec'], errors='coerce')
    
    # Entferne Zeilen mit ungültigen Zeitwerten
    shot_plotter_df = shot_plotter_df.dropna(subset=['Time'])
    possession_df = possession_df.dropna(subset=['end_time_sec'])
    
    # Finde die "Possession Type" Spalte
    possession_type_col = None
    for col in possession_df.columns:
        if isinstance(col, str) and "possession type" in str(col).lower():
            possession_type_col = col
            if debug_mode:
                st.success(f"Gefundene Possession Type Spalte: {col}")
            break
    
    # Finde die "Player Name" Spalte
    player_name_col = None
    for col in possession_df.columns:
        if isinstance(col, str) and "player name" in str(col).lower():
            player_name_col = col
            if debug_mode:
                st.success(f"Gefundene Player Name Spalte: {col}")
            break
    
    # Debug-Info über die zu vergleichenden Zeiten
    if 'debug_matching' in st.session_state and st.session_state.get('debug_matching', False):
        st.subheader("Debug: Zeitvergleich für Matching")
        st.write("Shot-Plotter Zeiten (die ersten 10):")
        shot_plotter_times = shot_plotter_df[['Time']].head(10)
        if 'Original_Time' in shot_plotter_df.columns:
            shot_plotter_times = shot_plotter_df[['Original_Time', 'Time']].head(10)
        st.write(shot_plotter_times)
        
        st.write("Possession End-Zeiten (die ersten 10):")
        st.write(possession_df[['end_time_sec']].head(10))
        
        st.info(f"Zeitfenster für Matching: {time_window} Sekunden")
        
        # Zeige die Zeitdifferenzen für die ersten 5 Einträge
        if not shot_plotter_df.empty and not possession_df.empty:
            st.write("Zeitdifferenzen für die ersten 3 Shot-Plotter-Einträge:")
            for i, shot_row in shot_plotter_df.head(3).iterrows():
                shot_time = float(shot_row['Time'])
                st.write(f"Shot-Plotter Zeit: {shot_time:.2f} s")
                
                # Berechne alle Zeitdifferenzen
                time_diffs = pd.DataFrame({
                    'Possession Zeit': possession_df['end_time_sec'],
                    'Differenz': (possession_df['end_time_sec'] - shot_time).abs()
                })
                
                # Zeige die 5 nächsten Zeiten
                closest_times = time_diffs.sort_values('Differenz').head(5)
                st.write(closest_times)
                
                # Zeige, ob ein Match gefunden wurde
                match_found = any(closest_times['Differenz'] <= time_window)
                if match_found:
                    st.success(f"✓ Match gefunden innerhalb von {time_window} Sekunden!")
                else:
                    st.error(f"✗ Kein Match innerhalb von {time_window} Sekunden!")
    
    # Merged-DataFrame initialisieren
    merged_data = []
    matches_found = 0
    total_entries = len(shot_plotter_df)
    
    # Im Debug-Modus mehr Informationen anzeigen
    if debug_mode:
        st.write(f"Versuche, {total_entries} Shot-Plotter-Einträge mit {len(possession_df)} Possession-Einträgen zu matchen")
        st.write(f"Verwendetes Zeitfenster: {time_window} Sekunden")
        
        # Zeige die ersten Zeilen beider Dataframes
        st.write("Erste Zeilen der Shot-Plotter-Daten:")
        st.write(shot_plotter_df.head(3))
        st.write("Erste Zeilen der Possession-Daten:")
        st.write(possession_df.head(3))
    
    # Tracking für nicht-gematchte Einträge
    matched_shot_indices = set()
    
    # Für jeden Eintrag in Shot-Plotter nach übereinstimmenden Zeiten suchen
    for idx, shot_row in shot_plotter_df.iterrows():
        shot_time = float(shot_row['Time'])  # Zeit in Sekunden, explizite Konvertierung zu float
        
        # Erweiterte Suche: Wir suchen nicht nur im exakten Zeitfenster,
        # sondern sortieren alle potenziellen Treffer nach zeitlicher Nähe
        try:
            time_diff_series = (possession_df['end_time_sec'] - shot_time).abs()
            
            # Sortiere nach zeitlichem Abstand und wähle die nächsten Einträge
            # (bis zum maximalen Zeitfenster)
            closest_matches = possession_df.loc[time_diff_series <= time_window].copy()
            
            if not closest_matches.empty:
                # Füge den Zeitabstand als Spalte hinzu, um danach zu sortieren
                closest_matches['time_diff'] = time_diff_series[time_diff_series <= time_window]
                closest_matches = closest_matches.sort_values('time_diff')
                
                # Wähle den besten Match (den mit der kleinsten Zeitdifferenz)
                best_match = closest_matches.iloc[0]
                
                # Erstelle einen kombinierten Eintrag mit den gewünschten Spalten
                merged_entry = {
                    # Koordinaten aus Shot-Plotter
                    'X': shot_row['X'],
                    'Y': shot_row['Y']
                }
                
                # Füge weitere Spalten aus Shot-Plotter hinzu, wenn vorhanden
                if 'X2' in shot_row and pd.notna(shot_row['X2']):
                    merged_entry['X2'] = shot_row['X2']
                if 'Y2' in shot_row and pd.notna(shot_row['Y2']):
                    merged_entry['Y2'] = shot_row['Y2']
                if 'Type' in shot_row and pd.notna(shot_row['Type']):
                    merged_entry['Type'] = shot_row['Type']
                if 'Distance' in shot_row and pd.notna(shot_row['Distance']):
                    merged_entry['Distance'] = shot_row['Distance']
                if 'Outcome' in shot_row and pd.notna(shot_row['Outcome']):
                    merged_entry['Outcome'] = shot_row['Outcome']
                
                # Spielername aus Possession Summary
                if player_name_col and player_name_col in best_match and pd.notna(best_match[player_name_col]):
                    merged_entry['Player Name'] = best_match[player_name_col]
                
                # Zeit (korrigierte Zeit aus Possession Summary)
                merged_entry['Zeit'] = float(best_match['end_time_sec'])  # Explizite float-Konvertierung
                
                # Füge passed_from und passed_to aus XML hinzu
                if 'passed_to' in best_match and pd.notna(best_match['passed_to']):
                    merged_entry['passed_to'] = best_match['passed_to']
                
                if 'passed_from' in best_match and pd.notna(best_match['passed_from']):
                    merged_entry['passed_from'] = best_match['passed_from']
                
                # Possession Type aus Possession Summary
                if possession_type_col and possession_type_col in best_match and pd.notna(best_match[possession_type_col]):
                    merged_entry['Possession Type'] = best_match[possession_type_col]
                
                merged_data.append(merged_entry)
                matched_shot_indices.add(idx)
                matches_found += 1
            else:
                # Kein Match gefunden - diesen Eintrag später als unmatched hinzufügen
                pass
        except Exception as e:
            st.error(f"Fehler beim Verarbeiten des Eintrags {idx}: {str(e)}")
            continue
    
    # Füge alle nicht-gematchten Shot-Plotter-Einträge hinzu
    unmatched_entries = 0
    for idx, shot_row in shot_plotter_df.iterrows():
        if idx not in matched_shot_indices:
            # Erstelle Eintrag mit verfügbaren Shot-Plotter-Daten und Platzhaltern für den Rest
            unmatched_entry = {
                'X': shot_row['X'], 
                'Y': shot_row['Y'],
                'Player Name': 'Unbekannter Spieler',
                'Zeit': float(shot_row['Time'])  # Verwende die Zeit aus Shot-Plotter
            }
            
            # Füge weitere Spalten aus Shot-Plotter hinzu, wenn vorhanden
            if 'X2' in shot_row and pd.notna(shot_row['X2']):
                unmatched_entry['X2'] = shot_row['X2']
            if 'Y2' in shot_row and pd.notna(shot_row['Y2']):
                unmatched_entry['Y2'] = shot_row['Y2']
            if 'Type' in shot_row and pd.notna(shot_row['Type']):
                unmatched_entry['Type'] = shot_row['Type']
            if 'Distance' in shot_row and pd.notna(shot_row['Distance']):
                unmatched_entry['Distance'] = shot_row['Distance']
            if 'Outcome' in shot_row and pd.notna(shot_row['Outcome']):
                unmatched_entry['Outcome'] = shot_row['Outcome']
            
            # Leere Werte für die anderen Felder
            unmatched_entry['passed_from'] = None
            unmatched_entry['passed_to'] = None
            unmatched_entry['Possession Type'] = None
            
            merged_data.append(unmatched_entry)
            unmatched_entries += 1
    
    result_df = pd.DataFrame(merged_data)
    
    # Info über Match-Qualität
    if not result_df.empty:
        st.info(f"Matches gefunden: {matches_found} von {total_entries} Einträgen ({matches_found/total_entries*100:.1f}%)")
        if unmatched_entries > 0:
            st.info(f"Ungematchte Einträge hinzugefügt: {unmatched_entries} (werden mit 'Unbekannter Spieler' angezeigt)")
        
        # Anzeigen der gefundenen Spieler-IDs
        if 'passed_from' in result_df.columns:
            unique_passed_from = result_df['passed_from'].dropna().unique()
            st.success(f"Gefundene passed_from Werte: {len(unique_passed_from)}")
            
        if 'passed_to' in result_df.columns:
            unique_passed_to = result_df['passed_to'].dropna().unique()
            st.success(f"Gefundene passed_to Werte: {len(unique_passed_to)}")
        
        # Prüfe, ob Player Name vorhanden ist
        if 'Player Name' not in result_df.columns or result_df['Player Name'].isnull().all():
            # Versuche, Player Name aus passed_from zu erstellen, wenn nicht vorhanden
            if 'passed_from' in result_df.columns:
                st.warning("Player Name nicht gefunden. Verwende passed_from als Player Name.")
                result_df['Player Name'] = result_df['passed_from']
    
    # Stelle sicher, dass alle gewünschten Spalten vorhanden sind
    for col in ['Player Name', 'passed_to', 'passed_from', 'Possession Type']:
        if col not in result_df.columns:
            result_df[col] = None
    
    # Definiere die Reihenfolge der Spalten entsprechend den Anforderungen
    column_order = ['Player Name', 'Zeit', 'passed_from', 'passed_to', 'Possession Type', 'X', 'Y']
    
    # Füge restliche Spalten hinzu
    for col in result_df.columns:
        if col not in column_order:
            column_order.append(col)
    
    # Reihenfolge der Spalten anpassen (nur vorhandene Spalten)
    available_columns = [col for col in column_order if col in result_df.columns]
    result_df = result_df[available_columns]
    
    return result_df

def create_sportscode_xml(merged_data, player_col=None, time_window=4.0):
    """Erzeugt eine Sportscode-kompatible XML-Datei aus den zusammengeführten Daten
    
    Der Player Name wird als code-Element verwendet (Hauptakteur)
    passed_from ist die Ballstation davor
    passed_to ist die Ballstation danach
    """
    
    # XML-Root-Element erstellen
    root = ET.Element("file")
    
    # SESSION_INFO hinzufügen (aktuelle Zeit verwenden)
    session_info = ET.SubElement(root, "SESSION_INFO")
    start_time = ET.SubElement(session_info, "start_time")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f%z")
    start_time.text = current_time
    
    # ALL_INSTANCES Element hinzufügen
    all_instances = ET.SubElement(root, "ALL_INSTANCES")
    
    # Instanz für Start Time hinzufügen (wie in der Beispiel-XML)
    start_instance = ET.SubElement(all_instances, "instance")
    
    ET.SubElement(start_instance, "ID").text = "1"
    ET.SubElement(start_instance, "start").text = "0"
    ET.SubElement(start_instance, "end").text = "2"
    ET.SubElement(start_instance, "code").text = "Start Time"
    
    label = ET.SubElement(start_instance, "label")
    ET.SubElement(label, "text").text = ""
    
    # Finde den Spielerspalten-Namen für den Hauptakteur (Player Name), wenn nicht angegeben
    if player_col is None or player_col not in merged_data.columns:
        # Priorisiere "Player Name" Spalte, dann "passed_from"
        if 'Player Name' in merged_data.columns:
            player_col = 'Player Name'
        elif 'passed_from' in merged_data.columns:
            player_col = 'passed_from'
        else:
            st.error("Keine Spielerspalte für Hauptakteur gefunden!")
            return "Error: Keine gültige Spielerspalte gefunden"
    
    # Zähler für IDs
    id_counter = 2  # Start bei 2, da 1 bereits für Start Time verwendet wurde
    
    # Für jeden Eintrag eine Instance erstellen
    for i, row in merged_data.iterrows():
        instance = ET.SubElement(all_instances, "instance")
        
        # ID, Start/End Zeiten setzen
        ET.SubElement(instance, "ID").text = str(id_counter)
        id_counter += 1
        
        # Startzeit = Zeit - time_window, Endzeit = Zeit + time_window
        # Verwende 'Zeit' statt 'Time'
        time_value = row['Zeit'] if 'Zeit' in row else 0
        ET.SubElement(instance, "start").text = str(time_value - time_window)
        ET.SubElement(instance, "end").text = str(time_value + time_window)
        
        # Der Hauptakteur wird als code-Element verwendet
        player_name = row[player_col] if pd.notna(row[player_col]) else "Unknown Player"
        ET.SubElement(instance, "code").text = str(player_name)
        
        # Wichtige Informationen als Labels hinzufügen
        
        # Vorherige Ballstation (passed_from) als Label
        if 'passed_from' in row and pd.notna(row['passed_from']):
            passed_from_label = ET.SubElement(instance, "label")
            ET.SubElement(passed_from_label, "group").text = "passed from"
            ET.SubElement(passed_from_label, "text").text = str(row['passed_from'])
        
        # Nächste Ballstation (passed_to) als Label
        if 'passed_to' in row and pd.notna(row['passed_to']):
            passed_to_label = ET.SubElement(instance, "label")
            ET.SubElement(passed_to_label, "group").text = "passed to"
            ET.SubElement(passed_to_label, "text").text = str(row['passed_to'])
        
        # Possession Type (Ballbesitzart) als Label
        if 'Possession Type' in row and pd.notna(row['Possession Type']):
            possession_type_label = ET.SubElement(instance, "label")
            ET.SubElement(possession_type_label, "group").text = "Possession Type"
            ET.SubElement(possession_type_label, "text").text = str(row['Possession Type'])
        
        # Weitere Informationen als Labels
        
        # Start X Position
        if 'X' in row and pd.notna(row['X']):
            start_x_label = ET.SubElement(instance, "label")
            ET.SubElement(start_x_label, "group").text = "Start X"
            ET.SubElement(start_x_label, "text").text = f"{row['X']:.2f}"
        
        # Start Y Position
        if 'Y' in row and pd.notna(row['Y']):
            start_y_label = ET.SubElement(instance, "label")
            ET.SubElement(start_y_label, "group").text = "Start Y"
            ET.SubElement(start_y_label, "text").text = f"{row['Y']:.2f}"
        
        # End X Position
        if 'X2' in row and pd.notna(row['X2']):
            end_x_label = ET.SubElement(instance, "label")
            ET.SubElement(end_x_label, "group").text = "End X"
            ET.SubElement(end_x_label, "text").text = f"{row['X2']:.2f}"
        
        # End Y Position
        if 'Y2' in row and pd.notna(row['Y2']):
            end_y_label = ET.SubElement(instance, "label")
            ET.SubElement(end_y_label, "group").text = "End Y"
            ET.SubElement(end_y_label, "text").text = f"{row['Y2']:.2f}"
        
        # Distanz
        if 'Distance' in row and pd.notna(row['Distance']):
            distance_label = ET.SubElement(instance, "label")
            ET.SubElement(distance_label, "group").text = "Distance"
            ET.SubElement(distance_label, "text").text = f"{row['Distance']:.2f}"
        
        # Typ
        if 'Type' in row and pd.notna(row['Type']):
            type_label = ET.SubElement(instance, "label")
            ET.SubElement(type_label, "group").text = "Type"
            ET.SubElement(type_label, "text").text = str(row['Type'])
        
        # Outcome/Ergebnis
        if 'Outcome' in row and pd.notna(row['Outcome']):
            outcome_label = ET.SubElement(instance, "label")
            ET.SubElement(outcome_label, "group").text = "Outcome"
            ET.SubElement(outcome_label, "text").text = str(row['Outcome'])
    
    # XML als String zurückgeben
    return ET.tostring(root, encoding='unicode')

def synchronize_by_specific_pass(playermaker_df, xml_time, video_time):
    """
    Synchronisiert die Zeiten zwischen dem Playermaker-System und der Video-Zeit
    basierend auf einem manuell angegebenen Pass.
    
    Parameters:
    -----------
    playermaker_df : DataFrame
        DataFrame mit den Playermaker-Daten, enthält 'end_time_sec' Spalte
    xml_time : float
        Die Zeit des Passes im Playermaker-System (nach -4 Sekunden Korrektur)
    video_time : float
        Die entsprechende Zeit des gleichen Passes im Video
        
    Returns:
    --------
    synchronized_df : DataFrame
        Kopie des playermaker_df mit synchronisierten Zeiten
    time_diff : float
        Berechneter Zeitunterschied zwischen den Systemen
    """
    if playermaker_df.empty:
        st.error("Keine Playermaker-Daten für die Zeitsynchronisation verfügbar.")
        return playermaker_df, 0.0
    
    # Stelle sicher, dass die erforderliche Spalte vorhanden ist
    if 'end_time_sec' not in playermaker_df.columns:
        st.error("Playermaker enthält keine 'end_time_sec'-Spalte.")
        return playermaker_df, 0.0
    
    # Stelle sicher, dass alle Zeitwerte numerisch sind
    playermaker_df['end_time_sec'] = pd.to_numeric(playermaker_df['end_time_sec'], errors='coerce')
    
    # Konvertiere Eingabewerte zu float, falls sie es noch nicht sind
    xml_time = float(xml_time)
    video_time = float(video_time)
    
    # Berechne den Zeitunterschied zwischen der Playermaker-Zeit und der Video-Zeit
    time_diff = video_time - xml_time
    
    # Debug-Info anzeigen
    st.info("Zeitsynchronisation basierend auf spezifischem Pass:")
    st.write(f"Pass im Playermaker-System: {xml_time:.2f} s")
    st.write(f"Gleicher Pass im Video: {video_time:.2f} s")
    st.write(f"Berechnete Zeitdifferenz: {time_diff:.2f} s")
    
    # Kopiere das DataFrame, um das Original nicht zu verändern
    synchronized_df = playermaker_df.copy()
    
    # Passe die Zeiten im Playermaker DataFrame an
    synchronized_df['original_end_time_sec'] = synchronized_df['end_time_sec']  # Original speichern
    synchronized_df['end_time_sec'] = synchronized_df['end_time_sec'] + time_diff
    
    if 'start_time_sec' in synchronized_df.columns:
        synchronized_df['start_time_sec'] = pd.to_numeric(synchronized_df['start_time_sec'], errors='coerce')
        synchronized_df['original_start_time_sec'] = synchronized_df['start_time_sec']  # Original speichern
        synchronized_df['start_time_sec'] = synchronized_df['start_time_sec'] + time_diff
    
    # Zeige die ersten 5 synchronisierten Einträge
    st.write("Synchronisierte Zeitwerte (erste 5 Einträge):")
    time_columns = ['original_end_time_sec', 'end_time_sec']
    if 'start_time_sec' in synchronized_df.columns:
        time_columns.extend(['original_start_time_sec', 'start_time_sec'])
    
    st.write(synchronized_df[time_columns].head())
    
    return synchronized_df, time_diff

def convert_time_to_seconds(time_str):
    """
    Konvertiert Zeit im Format MM:SS.HH oder MM:SS:HH zu Sekunden
    
    Parameters:
    -----------
    time_str : str
        Zeit im Format "MM:SS.HH" oder "MM:SS:HH"
        
    Returns:
    --------
    float
        Zeit in Sekunden
    """
    # Überprüfe, ob die Zeit bereits eine Zahl ist
    if isinstance(time_str, (int, float)):
        return float(time_str)
    
    # Wenn die Zeit eine Zeichenkette ist
    if isinstance(time_str, str):
        # Bereinigen und normalisieren
        time_str = time_str.strip()
        
        try:
            # Falls es bereits ein einfacher Float-String ist (z.B. "123.45")
            if time_str.replace('.', '', 1).isdigit():
                return float(time_str)
            
            # Format MM:SS.HH oder MM:SS:HH
            # Zuerst Prüfen, ob das Format MM:SS.HH ist
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 2:  # Format ist MM:SS.HH oder MM:SS,HH
                    minutes = int(parts[0])
                    # Behandle den Sekundenteil
                    if '.' in parts[1]:
                        seconds_parts = parts[1].split('.')
                        seconds = int(seconds_parts[0])
                        hundredths = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
                        return minutes * 60 + seconds + hundredths / 100
                    elif ',' in parts[1]:
                        seconds_parts = parts[1].split(',')
                        seconds = int(seconds_parts[0])
                        hundredths = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
                        return minutes * 60 + seconds + hundredths / 100
                    else:
                        return minutes * 60 + float(parts[1])
                elif len(parts) == 3:  # Format ist MM:SS:HH
                    minutes = int(parts[0])
                    seconds = int(parts[1])
                    hundredths = int(parts[2])
                    return minutes * 60 + seconds + hundredths / 100
            
            # Fallback: Versuche direkte Konvertierung (könnte fehlschlagen)
            return float(time_str)
        except (ValueError, IndexError) as e:
            st.error(f"Fehler bei der Zeitkonvertierung für '{time_str}': {str(e)}")
            return 0.0
    
    # Wenn kein gültiges Format
    st.error(f"Ungültiges Zeitformat: {time_str}")
    return 0.0

# Definiere Tabs für den Workflow
tabs = st.tabs(["Daten hochladen", "Daten verarbeiten", "Ergebnisse"])

# Bei App-Start den ersten Tab aktivieren
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

# Tab-Wechsel Funktion
def change_tab(tab_index):
    st.session_state.active_tab = tab_index

# Setze aktiven Tab
current_tab = st.session_state.active_tab
st.session_state.tabs = tabs
if current_tab < len(tabs):
    # Simuliere Klick auf den Tab
    tabs[current_tab].selectbox = True

# Tab 1: Daten hochladen
with tabs[0]:
    st.header("Daten hochladen")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Shot-Plotter CSV")
        shot_plotter_file = st.file_uploader("CSV-Datei mit Positionsdaten", type=['csv'], key="shot_plotter_uploader")
        if shot_plotter_file is not None:
            st.session_state['shot_plotter_file'] = shot_plotter_file
    
    with col2:
        st.subheader("Playermaker XML")
        xml_file = st.file_uploader("XML-Datei mit Passdaten", type=['xml'], key="xml_uploader")
        if xml_file is not None:
            st.session_state['xml_file'] = xml_file
    
    with col3:
        st.subheader("Playermaker Possession")
        possession_file = st.file_uploader("Excel-Datei mit Zeitdaten", type=['xlsx', 'csv'], key="possession_uploader")
        if possession_file is not None:
            st.session_state['possession_file'] = possession_file
    
    # Prüfe, ob alle Dateien hochgeladen wurden
    all_files_uploaded = ('shot_plotter_file' in st.session_state and 
                         'xml_file' in st.session_state and 
                         'possession_file' in st.session_state)
    
    if all_files_uploaded:
        st.success("Alle Dateien wurden hochgeladen!")
        if st.button("Weiter zum nächsten Schritt", key="continue_btn"):
            # Sofortiges Löschen von vorherigen Daten, um Neuverarbeitung zu erzwingen
            for key in ['shot_plotter_df', 'xml_events', 'possession_df', 'merged_data']:
                if key in st.session_state:
                    del st.session_state[key]
            change_tab(1)
            st.rerun()
    else:
        st.info("Bitte laden Sie alle drei Dateien hoch.")


# Tab 2: Daten verarbeiten
with tabs[1]:
    st.header("Daten verarbeiten")
    
    # Debug-Info
    st.write("Verfügbare Session-State-Schlüssel:", list(st.session_state.keys()))
    
    # Debug-Checkbox für Zeitvergleiche
    debug_matching = st.checkbox("Debug-Modus für Zeitvergleiche aktivieren", value=False, 
                              help="Zeigt detaillierte Informationen über die Zeitvergleiche beim Matching")
    st.session_state['debug_matching'] = debug_matching
    
    # Laden der Dateien aus dem Session State
    has_all_files = ('shot_plotter_file' in st.session_state and 
                     'xml_file' in st.session_state and 
                     'possession_file' in st.session_state)
    
    if not has_all_files:
        st.error("Bitte laden Sie zuerst alle Dateien im ersten Tab hoch.")
        if st.button("Zurück zum Upload", key="back_to_upload"):
            change_tab(0)
            st.rerun()
    else:
        # Verarbeite Shot-Plotter CSV
        if 'shot_plotter_df' not in st.session_state:
            try:
                # Zurücksetzen des Dateizeigers
                st.session_state.shot_plotter_file.seek(0)
                shot_plotter_df = pd.read_csv(st.session_state.shot_plotter_file)
                
                # Zeige Daten vor der Konvertierung
                st.write("Shot-Plotter Daten vor der Zeitkonvertierung:")
                st.write(shot_plotter_df.head())
                
                # Überprüfe, ob die Time-Spalte existiert
                if 'Time' in shot_plotter_df.columns:
                    # Überprüfe und konvertiere das Zeitformat
                    time_sample = str(shot_plotter_df['Time'].iloc[0]) if not shot_plotter_df.empty else ""
                    st.write(f"Erkanntes Zeitformat: {time_sample}")
                    
                    # Prüfe, ob die Zeit konvertiert werden muss
                    if ':' in time_sample or '.' in time_sample:
                        st.info(f"Konvertiere Zeitformat '{time_sample}' zu Sekunden...")
                        # Speichere das Originalformat
                        shot_plotter_df['Original_Time'] = shot_plotter_df['Time']
                        # Konvertiere zu Sekunden
                        shot_plotter_df['Time'] = shot_plotter_df['Time'].apply(convert_time_to_seconds)
                        
                        # Zeige Ergebnis der Konvertierung
                        conversion_sample = pd.DataFrame({
                            'Original': shot_plotter_df['Original_Time'].head(5),
                            'Sekunden': shot_plotter_df['Time'].head(5)
                        })
                        st.write("Zeitkonvertierung (Beispiel):")
                        st.write(conversion_sample)
                
                # Speichere das DataFrame im Session State
                st.session_state.shot_plotter_df = shot_plotter_df
                st.success(f"Shot-Plotter CSV geladen: {len(shot_plotter_df)} Einträge")
            except Exception as e:
                st.error(f"Fehler beim Laden der CSV: {str(e)}")
                import traceback
                st.text(traceback.format_exc())
                st.session_state.shot_plotter_df = pd.DataFrame()
        
        # Verarbeite XML
        if 'xml_events' not in st.session_state:
            try:
                # Zurücksetzen des Dateizeigers
                st.session_state.xml_file.seek(0)
                st.write("XML wird geparst...")
                events = parse_playermaker_data(st.session_state.xml_file)
                st.session_state.xml_events = pd.DataFrame([event.to_dict() for event in events])
                st.success(f"XML erfolgreich geparst: {len(st.session_state.xml_events)} Einträge")
            except Exception as e:
                st.error(f"Fehler beim Parsen der XML: {str(e)}")
                st.write("Fehlerdetails:", str(e))
                st.session_state.xml_events = pd.DataFrame()
        
        # Verarbeite Possession Excel/CSV
        if 'possession_df' not in st.session_state:
            try:
                # Zurücksetzen des Dateizeigers
                st.session_state.possession_file.seek(0)
                
                # Bestimme Dateityp (CSV oder Excel) anhand der Dateiendung
                file_type = st.session_state.possession_file.name.split('.')[-1].lower()
                
                st.write(f"Possession-Daten werden verarbeitet... (Dateityp: {file_type})")
                
                if file_type == 'csv':
                    # Versuche verschiedene Trennzeichen
                    for sep in [';', ',', '\t']:
                        try:
                            df = pd.read_csv(st.session_state.possession_file, sep=sep)
                            if len(df.columns) > 1:
                                st.success(f"CSV erfolgreich mit Trennzeichen '{sep}' gelesen")
                                break
                        except:
                            st.session_state.possession_file.seek(0)  # Zurücksetzen für den nächsten Versuch
                    else:
                        # Wenn keines der spezifischen Trennzeichen funktioniert hat
                        st.session_state.possession_file.seek(0)
                        df = pd.read_csv(st.session_state.possession_file, sep=None, engine='python')
                        st.success("CSV mit automatischer Trennzeichenerkennung gelesen")
                else:  # xlsx
                    df = pd.read_excel(st.session_state.possession_file)
                    st.success("Excel-Datei erfolgreich gelesen")
                
                # Zeige die ersten Zeilen zur Überprüfung
                st.write("Vorschau der gelesenen Daten:")
                st.write(df.head(3))
                
                # Verarbeite die Daten weiter
                st.session_state.possession_df = process_playermaker_possession(df)
                
                if not st.session_state.possession_df.empty:
                    st.success(f"Possession-Datei erfolgreich verarbeitet: {len(st.session_state.possession_df)} Einträge")
                else:
                    st.error("Keine Daten in der Possession-Datei gefunden oder Format nicht erkannt.")
            except Exception as e:
                st.error(f"Fehler bei der Verarbeitung der Possession-Datei: {str(e)}")
                st.write("Fehlerdetails:", str(e))
                import traceback
                st.text(traceback.format_exc())
                st.session_state.possession_df = pd.DataFrame()
        
        # Datenvorschau anzeigen
        data_valid = (not st.session_state.get('shot_plotter_df', pd.DataFrame()).empty and
                     not st.session_state.get('xml_events', pd.DataFrame()).empty and
                     not st.session_state.get('possession_df', pd.DataFrame()).empty)
        
        if data_valid:
            st.success("Alle Dateien wurden erfolgreich geladen und verarbeitet!")
            
            with st.expander("Shot-Plotter Daten (erste 5 Zeilen)", expanded=False):
                st.dataframe(st.session_state.shot_plotter_df.head())
            
            with st.expander("XML-Events (erste 5 Zeilen)", expanded=False):
                st.dataframe(st.session_state.xml_events.head())
            
            with st.expander("Possession-Daten (erste 5 Zeilen)", expanded=False):
                st.dataframe(st.session_state.possession_df.head())
            
            # Merge-Parameter
            st.subheader("Daten zusammenführen")
            
            # Manuelle Zeitsynchronisation
            st.subheader("Zeitsynchronisation")
            sync_enabled = st.checkbox("Zeitsynchronisation aktivieren", value=True,
                                      help="Aktivieren Sie diese Option, um die Zeiten zwischen dem Video und dem Playermaker-System zu synchronisieren.")
            
            if sync_enabled:
                # Zeige eine Tabelle mit verfügbaren Pässen an, wenn XML-Events geladen sind
                if 'xml_events' in st.session_state and not st.session_state.xml_events.empty:
                    with st.expander("Verfügbare Pässe aus dem Playermaker-System anzeigen"):
                        # Filtere Pässe (mit passed_to gesetzt)
                        passes_df = st.session_state.xml_events[
                            st.session_state.xml_events['passed_to'].notna()
                        ].sort_values('end_time')
                        
                        if not passes_df.empty:
                            # Zeige relevante Spalten für die Passidentifikation
                            display_columns = ['event_id', 'end_time', 'player', 'passed_from', 'passed_to']
                            available_columns = [col for col in display_columns if col in passes_df.columns]
                            
                            st.write("Verfügbare Pässe (sortiert nach Zeit):")
                            st.dataframe(passes_df[available_columns].head(10), height=300)
                            
                            st.info("""
                            Identifizieren Sie einen charakteristischen Pass und notieren Sie seine Zeit (end_time).
                            Finden Sie den gleichen Pass im Video und notieren Sie die Video-Zeit.
                            Geben Sie beide Zeiten unten ein, um die Synchronisation durchzuführen.
                            """)
                        else:
                            st.warning("Keine Pässe in den Playermaker-Daten gefunden.")
                
                # Zeige auch verfügbare Einträge aus Shot-Plotter an
                if 'shot_plotter_df' in st.session_state and not st.session_state.shot_plotter_df.empty:
                    with st.expander("Verfügbare Einträge aus dem Video/Shot-Plotter anzeigen"):
                        # Sortiere nach Zeit
                        shot_plotter_sorted = st.session_state.shot_plotter_df.sort_values('Time')
                        
                        # Zeige relevante Spalten für die Zeitidentifikation
                        display_columns = ['Time']
                        
                        # Füge Original_Time hinzu, falls vorhanden
                        if 'Original_Time' in shot_plotter_sorted.columns:
                            display_columns.insert(0, 'Original_Time')
                        
                        # Füge weitere Spalten hinzu
                        for col in ['Type', 'Outcome', 'X', 'Y']:
                            if col in shot_plotter_sorted.columns:
                                display_columns.append(col)
                        
                        st.write("Verfügbare Shot-Plotter-Einträge (sortiert nach Zeit):")
                        st.dataframe(shot_plotter_sorted[display_columns].head(15), height=300)
                        
                        st.info("""
                        Vergleichen Sie die Einträge mit den Pässen aus dem Playermaker-System.
                        Die konvertierten Zeiten ('Time' in Sekunden) werden für die Synchronisation benötigt.
                        Beispiel: Für den Eintrag mit Original_Time '0:42.46' ist die konvertierte Zeit 42.46 Sekunden.
                        """)
                
                # UI für die manuelle Passauswahl
                st.info("Geben Sie einen spezifischen Pass für die Zeitsynchronisation an:")
                
                # Spalten für die Eingabefelder
                col1, col2 = st.columns(2)
                
                with col1:
                    # Zeiteingabe für Playermaker-System
                    playermaker_time = st.number_input(
                        "Playermaker-Zeit (Sekunden)",
                        value=0.0,
                        min_value=0.0,
                        step=0.01,
                        help="Die Zeit des spezifischen Passes im Playermaker-System (korrigierte end_time)"
                    )
                    
                    # Beispiel anzeigen
                    st.caption("Beispiel: Wenn in der XML ein Pass mit korrigierter Endzeit von 7.99 Sekunden existiert")
                
                with col2:
                    # Zeiteingabe für Video
                    video_time = st.number_input(
                        "Video-Zeit (Sekunden)",
                        value=0.0,
                        min_value=0.0,
                        step=0.01,
                        help="Die Zeit des gleichen Passes im Video"
                    )
                    
                    # Beispiel anzeigen
                    st.caption("Beispiel: Wenn derselbe Pass im Video bei 42.71 Sekunden zu sehen ist")
                
                # Zeitdifferenz anzeigen
                time_diff = video_time - playermaker_time
                if time_diff != 0:
                    st.success(f"Die Playermaker-Zeiten werden um {time_diff:+.2f} Sekunden angepasst.")
            
            # Zeitfenster für Matching
            st.subheader("Matching-Parameter")
            time_window = st.slider("Zeitfenster für Matching (Sekunden)", 1.0, 30.0, 5.0, 0.5,
                                   help="Maximaler Zeitunterschied für das Matching zwischen Video und Playermaker-Daten")
            
            # Debug-Option für die Zeitvergleiche
            debug_matching = st.checkbox("Debug-Modus für Zeitvergleiche aktivieren", value=False,
                                        help="Zeigt detaillierte Informationen über die Zeitvergleiche beim Matching an")
            
            if st.button("Daten zusammenführen", key="merge_btn"):
                # Debug-Modus speichern, damit er in den Funktionen verfügbar ist
                st.session_state.debug_matching = st.session_state.get('debug_matching', False)
                
                with st.spinner("Daten werden zusammengeführt..."):
                    # 1. passed_to und passed_from Spalte hinzufügen
                    st.write("Füge passed_to und passed_from Spalte zu Possession-Daten hinzu...")
                    updated_possession = add_passed_to_and_from_column(
                        st.session_state.possession_df.copy(),
                        st.session_state.xml_events
                    )
                    
                    # 1.5 Zeitsynchronisation wenn aktiviert
                    if sync_enabled and playermaker_time > 0 and video_time > 0:
                        st.write("Synchronisiere Zeiten basierend auf dem angegebenen Pass...")
                        updated_possession, time_diff = synchronize_by_specific_pass(
                            updated_possession,
                            playermaker_time,
                            video_time
                        )
                        
                        # Zeige Hinweis zur Synchronisation
                        if abs(time_diff) > 0.1:  # Nur anzeigen, wenn eine signifikante Zeitdifferenz besteht
                            if time_diff > 0:
                                st.success(f"Zeitsynchronisation durchgeführt: Playermaker-Zeiten wurden um +{time_diff:.2f} Sekunden angepasst.")
                            else:
                                st.success(f"Zeitsynchronisation durchgeführt: Playermaker-Zeiten wurden um {time_diff:.2f} Sekunden angepasst.")
                        else:
                            st.info("Zeitsynchronisation durchgeführt: Keine signifikante Zeitdifferenz gefunden.")
                    elif sync_enabled:
                        st.warning("Bitte geben Sie gültige Zeiten für die Synchronisation an.")
                    else:
                        st.warning("Zeitsynchronisation deaktiviert. Die Zeiten wurden nicht angepasst.")
                    
                    # 2. Daten zusammenführen
                    st.write("Führe Daten anhand der Zeit zusammen...")
                    merged_data = merge_data_by_time(
                        st.session_state.shot_plotter_df,
                        updated_possession,
                        time_window
                    )
                    
                    if not merged_data.empty:
                        # Speichere die Zeitdifferenz im session_state für spätere Verwendung
                        if sync_enabled:
                            st.session_state.time_diff = time_diff
                        
                        st.session_state.merged_data = merged_data
                        st.success(f"Daten erfolgreich zusammengeführt: {len(merged_data)} Einträge")

                        # Statistiken anzeigen
                        st.subheader("Statistiken")
                        col1, col2 = st.columns(2)
                        with col1:
                            if 'Outcome' in merged_data.columns:
                                success_rate = merged_data[merged_data['Outcome'] == 'Erfolgreich'].shape[0] / len(merged_data) * 100
                                st.metric("Erfolgsrate", f"{success_rate:.1f}%")
                        with col2:
                            if 'Distance' in merged_data.columns:
                                avg_distance = merged_data['Distance'].mean()
                                st.metric("Durchschnittliche Passdistanz", f"{avg_distance:.1f} m")
                        
                        if st.button("Weiter zu den Ergebnissen", key="goto_results_btn"):
                            change_tab(2)
                            st.rerun()
                    else:
                        st.error("Keine übereinstimmenden Daten gefunden. Versuchen Sie ein größeres Zeitfenster oder passen Sie die Zeitsynchronisation an.")
        else:
            if not st.session_state.get('shot_plotter_df', pd.DataFrame()).empty:
                st.success("✓ Shot-Plotter CSV geladen")
            else:
                st.error("✗ Problem beim Laden der Shot-Plotter CSV")
                
            if not st.session_state.get('xml_events', pd.DataFrame()).empty:
                st.success("✓ XML erfolgreich geparst")
            else:
                st.error("✗ Problem beim Parsen der XML")
                
            if not st.session_state.get('possession_df', pd.DataFrame()).empty:
                st.success("✓ Possession-Datei geladen")
            else:
                st.error("✗ Problem beim Laden der Possession-Datei")

# Tab 3: Ergebnisse
with tabs[2]:
    st.header("Ergebnisse")
    
    if 'merged_data' in st.session_state and not st.session_state.merged_data.empty:
        merged_data = st.session_state.merged_data
        
        # Definiere die Hauptspalten, die wir hervorheben wollen
        main_columns = ['Player Name', 'Zeit', 'passed_from', 'passed_to', 'Possession Type', 'X', 'Y']
        
        # Zeige die Hauptspalten zuerst in der Übersicht
        available_main_columns = [col for col in main_columns if col in merged_data.columns]
        other_columns = [col for col in merged_data.columns if col not in main_columns]
        display_columns = available_main_columns + other_columns
        
        # Übersicht
        st.subheader("Zusammengeführte Daten")
        st.write("Hauptspalten in der finalen Tabelle:")
        st.markdown("""
        - **Player Name**: Spielername aus der Possession Summary
        - **Zeit**: Korrigierte Zeit aus XML/Possession Summary (synchronisiert mit Video)
        - **passed_from**: Ballabsender aus XML
        - **passed_to**: Ballempfänger aus XML
        - **Possession Type**: Art des Ballbesitzes aus der Possession Summary
        - **X, Y**: Koordinaten aus dem Shot-Plotter (manuelle CSV)
        """)
        
        # Anzeigen der zusammengeführten Daten
        st.dataframe(merged_data[display_columns])
        
        # Erweiterte Visualisierungen
        st.subheader("Daten-Visualisierung")
        viz_tabs = st.tabs(["Statistiken", "Feldansicht", "Zeitliche Verteilung", "Passnetzwerk"])
        
        with viz_tabs[0]:
            # Statistiken
            st.markdown("#### Passstatistiken")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'Outcome' in merged_data.columns:
                    success_count = merged_data[merged_data['Outcome'] == 'Erfolgreich'].shape[0]
                    fail_count = merged_data[merged_data['Outcome'] == 'Nicht_Erfolgreich'].shape[0]
                    success_rate = success_count / (success_count + fail_count) * 100 if (success_count + fail_count) > 0 else 0
                    st.metric("Erfolgsrate", f"{success_rate:.1f}%")
                    
                    # Kreisdiagramm
                    fig = px.pie(
                        names=['Erfolgreich', 'Nicht erfolgreich'],
                        values=[success_count, fail_count],
                        color_discrete_sequence=['#4CAF50', '#F44336'],
                        title='Passerfolg'
                    )
                    st.plotly_chart(fig)
            
            with col2:
                if 'Distance' in merged_data.columns:
                    avg_distance = merged_data['Distance'].mean()
                    max_distance = merged_data['Distance'].max()
                    min_distance = merged_data['Distance'].min()
                    st.metric("Durchschn. Passdistanz", f"{avg_distance:.1f} m")
                    st.metric("Längster Pass", f"{max_distance:.1f} m")
                    st.metric("Kürzester Pass", f"{min_distance:.1f} m")
            
            with col3:
                # Zeige relevante Spielerinformationen
                player_cols = ['Player Name', 'passed_from', 'passed_to']
                for col in player_cols:
                    if col in merged_data.columns:
                        unique_values = merged_data[col].dropna().unique()
                        st.metric(f"Anzahl {col}", len(unique_values))
                
                # Typ-Verteilung, wenn vorhanden
                if 'Type' in merged_data.columns:
                    type_counts = merged_data['Type'].value_counts()
                    st.write("Pass-Typen:")
                    for typ, count in type_counts.items():
                        st.text(f"{typ}: {count}")
            
            # Spielerinformationen-Tabelle hinzufügen
            st.subheader("Spielerstatistiken")
            
            # Erstelle eine Tabelle mit Spielerinformationen, wenn Player Name, passed_from oder passed_to vorhanden sind
            if any(col in merged_data.columns for col in ['Player Name', 'passed_from', 'passed_to']):
                # Alle Spieler sammeln (Player Name, passed_from, passed_to)
                all_players = set()
                
                for col in ['Player Name', 'passed_from', 'passed_to']:
                    if col in merged_data.columns:
                        all_players.update(merged_data[col].dropna().unique())
                
                # Statistiken für jeden Spieler sammeln
                player_stats = []
                
                for player in all_players:
                    stats = {"Spieler": player}
                    
                    # Als Hauptakteur
                    if 'Player Name' in merged_data.columns:
                        player_passes = merged_data[merged_data['Player Name'] == player]
                        stats["Pässe als Hauptakteur"] = len(player_passes)
                        
                        if 'Outcome' in merged_data.columns and len(player_passes) > 0:
                            success = player_passes[player_passes['Outcome'] == 'Erfolgreich'].shape[0]
                            stats["Erfolgsrate"] = f"{success / len(player_passes) * 100:.1f}%" if len(player_passes) > 0 else "N/A"
                    
                    # Als Absender
                    if 'passed_from' in merged_data.columns:
                        stats["Pässe als Absender"] = merged_data[merged_data['passed_from'] == player].shape[0]
                    
                    # Als Empfänger
                    if 'passed_to' in merged_data.columns:
                        stats["Pässe als Empfänger"] = merged_data[merged_data['passed_to'] == player].shape[0]
                    
                    player_stats.append(stats)
                
                # Sortiere nach der Gesamtzahl der Pässe
                player_stats.sort(key=lambda x: 
                    x.get("Pässe als Hauptakteur", 0) + 
                    x.get("Pässe als Absender", 0) + 
                    x.get("Pässe als Empfänger", 0), 
                    reverse=True)
                
                # Anzeigen als DataFrame
                st.dataframe(pd.DataFrame(player_stats))
        
        with viz_tabs[1]:
            st.markdown("#### Feldansicht")
            st.markdown("Passvisualisierung auf dem Spielfeld - Startpunkt (X,Y) mit Farbcodierung nach Erfolg.")
            
            # Prüfe, ob passed_from/to und Player Name vorhanden sind
            has_player_info = any(col in merged_data.columns for col in ['passed_from', 'passed_to', 'Player Name'])
            
            # Tooltip-Texte mit Spielerinformationen
            hover_texts = []
            for i, row in merged_data.iterrows():
                text = f"Zeit: {row['Zeit']:.1f}s<br>Distanz: {row['Distance']:.1f}m<br>Ergebnis: {row['Outcome']}"
                
                # Füge Spielerinformationen hinzu, wenn vorhanden
                if 'Player Name' in row and pd.notna(row['Player Name']):
                    text += f"<br>Spieler: {row['Player Name']}"
                if 'passed_from' in row and pd.notna(row['passed_from']):
                    text += f"<br>Absender: {row['passed_from']}"
                if 'passed_to' in row and pd.notna(row['passed_to']):
                    text += f"<br>Empfänger: {row['passed_to']}"
                    
                hover_texts.append(text)
            
            # Erzeuge den Feldplot mit aktualisierten Tooltips
            field_fig = go.Figure()
            
            # Füge Scatter-Plot für Startpositionen hinzu
            field_fig.add_trace(go.Scatter(
                x=merged_data['X'],
                y=merged_data['Y'],
                mode='markers',
                marker=dict(
                    color=['green' if o == 'Erfolgreich' else 'red' for o in merged_data['Outcome']],
                    size=8
                ),
                text=hover_texts,
                hoverinfo='text',
                name='Startpositionen'
            ))
            
            # Füge Linien für Pässe hinzu
            for i, row in merged_data.iterrows():
                field_fig.add_trace(go.Scatter(
                    x=[row['X'], row['X2']],
                    y=[row['Y'], row['Y2']],
                    mode='lines',
                    line=dict(
                        color='green' if row['Outcome'] == 'Erfolgreich' else 'red',
                        width=1
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Füge Spielernamen als Annotation hinzu, wenn vorhanden
            if has_player_info:
                st.subheader("Spieler-Informationen")
                player_cols = st.columns(3)
                
                with player_cols[0]:
                    if 'Player Name' in merged_data.columns:
                        unique_players = merged_data['Player Name'].dropna().unique()
                        st.write("Spieler:", ", ".join([str(p) for p in unique_players]))
                
                with player_cols[1]:
                    if 'passed_from' in merged_data.columns:
                        unique_senders = merged_data['passed_from'].dropna().unique()
                        st.write("Absender:", ", ".join([str(p) for p in unique_senders]))
                
                with player_cols[2]:
                    if 'passed_to' in merged_data.columns:
                        unique_receivers = merged_data['passed_to'].dropna().unique()
                        st.write("Empfänger:", ", ".join([str(p) for p in unique_receivers]))
            
            # Feldeinstellungen
            field_fig.update_layout(
                title='Passpositionen auf dem Spielfeld',
                xaxis=dict(title='X-Position', range=[-60, 60]),
                yaxis=dict(title='Y-Position', range=[-40, 40]),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            st.plotly_chart(field_fig, use_container_width=True)
        
        with viz_tabs[2]:
            st.markdown("#### Zeitliche Verteilung")
            # Zeitverteilung der Passes
            if 'Zeit' in merged_data.columns:
                time_hist = px.histogram(
                    merged_data,
                    x='Zeit',
                    title='Zeitliche Verteilung der Pässe',
                    labels={'Zeit': 'Zeit (s)', 'count': 'Anzahl'},
                    color_discrete_sequence=['#2196F3']
                )
                st.plotly_chart(time_hist)
        
        # Tab 4: Passnetzwerk
        with viz_tabs[3]:
            st.markdown("#### Passnetzwerk-Analyse")
            
            # Prüfe, ob die benötigten Spalten vorhanden sind
            required_cols = ['passed_from', 'passed_to']
            has_required_cols = all(col in merged_data.columns for col in required_cols)
            
            if has_required_cols:
                # Filtern der Daten, wo passed_from und passed_to nicht leer sind
                network_data = merged_data.dropna(subset=required_cols)
                
                if not network_data.empty:
                    # Zähle die Pässe zwischen Spieler-Paaren
                    pass_counts = network_data.groupby(['passed_from', 'passed_to']).size().reset_index(name='count')
                    
                    st.write("Passhäufigkeiten zwischen Spielern:")
                    st.dataframe(pass_counts.sort_values('count', ascending=False))
                    
                    # Einfaches Netzwerkdiagramm
                    network_fig = go.Figure()
                    
                    # Liste aller Spieler
                    all_players = list(set(pass_counts['passed_from'].tolist() + pass_counts['passed_to'].tolist()))
                    
                    # Erzeuge Kreiskoordinaten für die Spielerpositionen
                    n = len(all_players)
                    radius = 1
                    angles = [2 * np.pi * i / n for i in range(n)]
                    player_positions = {
                        player: (radius * np.cos(angle), radius * np.sin(angle)) 
                        for player, angle in zip(all_players, angles)
                    }
                    
                    # Füge Kanten für Pässe hinzu
                    for _, row in pass_counts.iterrows():
                        from_player = row['passed_from']
                        to_player = row['passed_to']
                        count = row['count']
                        
                        # Positionsdaten
                        from_pos = player_positions[from_player]
                        to_pos = player_positions[to_player]
                        
                        # Füge eine Linie hinzu, wobei die Breite die Anzahl der Pässe darstellt
                        network_fig.add_trace(go.Scatter(
                            x=[from_pos[0], to_pos[0]],
                            y=[from_pos[1], to_pos[1]],
                            mode='lines',
                            line=dict(width=1 + count, color='rgba(70, 130, 180, 0.8)'),
                            text=f"{from_player} → {to_player}: {count} Pässe",
                            hoverinfo='text',
                            showlegend=False
                        ))
                    
                    # Füge Spieler als Knoten hinzu
                    for player, pos in player_positions.items():
                        # Zähle Pässe von diesem Spieler
                        outgoing = pass_counts[pass_counts['passed_from'] == player]['count'].sum() if player in pass_counts['passed_from'].values else 0
                        # Zähle Pässe zu diesem Spieler
                        incoming = pass_counts[pass_counts['passed_to'] == player]['count'].sum() if player in pass_counts['passed_to'].values else 0
                        
                        # Größe basierend auf der Summe der ein- und ausgehenden Pässe
                        node_size = 10 + (outgoing + incoming) * 2
                        
                        network_fig.add_trace(go.Scatter(
                            x=[pos[0]],
                            y=[pos[1]],
                            mode='markers+text',
                            marker=dict(size=node_size, color='blue'),
                            text=player,
                            textposition="top center",
                            name=player,
                            hovertext=f"{player}<br>Ausgehende Pässe: {outgoing}<br>Eingehende Pässe: {incoming}",
                            hoverinfo='text'
                        ))
                    
                    # Layout
                    network_fig.update_layout(
                        title='Passnetzwerk',
                        showlegend=False,
                        xaxis=dict(
                            showgrid=False,
                            zeroline=False,
                            showticklabels=False,
                            range=[-1.2, 1.2]
                        ),
                        yaxis=dict(
                            showgrid=False,
                            zeroline=False,
                            showticklabels=False,
                            range=[-1.2, 1.2]
                        ),
                        width=700,
                        height=700
                    )
                    
                    st.plotly_chart(network_fig, use_container_width=True)
                else:
                    st.warning("Keine Daten für Passnetzwerk-Analyse verfügbar.")
            else:
                st.warning(f"Für die Passnetzwerk-Analyse werden die Spalten {', '.join(required_cols)} benötigt.")
        
        # Export-Optionen
        st.subheader("Daten exportieren")
        
        # Definiere die Hauptspalten für den Export und stelle sicher, dass sie in der richtigen Reihenfolge sind
        main_columns = ['Player Name', 'Zeit', 'passed_from', 'passed_to', 'Possession Type', 'X', 'Y']
        available_main_columns = [col for col in main_columns if col in merged_data.columns]
        other_columns = [col for col in merged_data.columns if col not in main_columns]
        export_columns = available_main_columns + other_columns
        
        # Vorbereiten der Export-Daten mit richtiger Spaltenreihenfolge
        export_data = merged_data[export_columns].copy()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # CSV Export
            csv = export_data.to_csv(index=False)
            st.download_button(
                label="Als CSV herunterladen",
                data=csv,
                file_name="merged_pass_data.csv",
                mime="text/csv"
            )
        
        with col2:
            # JSON Export
            json_str = export_data.to_json(orient="records")
            st.download_button(
                label="Als JSON herunterladen",
                data=json_str,
                file_name="merged_pass_data.json",
                mime="application/json"
            )
        
        with col3:
            # Excel Export
            buffer = io.BytesIO()
            export_data.to_excel(buffer, index=False)
            buffer.seek(0)
            st.download_button(
                label="Als Excel herunterladen",
                data=buffer,
                file_name="merged_pass_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col4:
            # XML-Exportoptionen
            with st.expander("Sportscode XML Exportoptionen"):
                st.markdown("""
                ### Sportscode XML-Export
                
                Diese Option ermöglicht den Export der zusammengeführten Daten in das Sportscode XML-Format. 
                Die Struktur entspricht dem angegebenen Format:
                
                - Jeder Pass wird als `<instance>` dargestellt
                - Der Hauptakteur (Player Name) wird als `<code>` Element gespeichert
                - Alle relevanten Informationen werden als `<label>` Elemente mit folgenden Gruppen gespeichert:
                    - `passed from`: vorherige Ballstation 
                    - `passed to`: nächste Ballstation
                    - `Start X`, `Start Y`: Startposition
                    - `End X`, `End Y`: Endposition
                    - `Distance`: Passdistanz
                    - `Type`: Passtyp
                    - `Outcome`: Ergebnis des Passes
                    - `Time to Release`: Zeit bis zum Abspiel
                    - `Possession Type`: Art des Ballbesitzes
                - Die `<start>` und `<end>` Zeiten definieren das Zeitfenster um den Zeitpunkt des Passes
                """)
                
                # Option für die Spielerspalte
                player_options = [col for col in merged_data.columns 
                                  if col in ['Player Name', 'passed_from', 'passed_to']]
                
                default_index = 0
                if 'Player Name' in player_options:
                    default_index = player_options.index('Player Name')
                
                xml_player_col = st.selectbox(
                    "Spielerspalte für 'code'-Element (Hauptakteur):",
                    options=player_options,
                    index=default_index,
                    help="Diese Spalte wird für den Spielernamen im 'code'-Element verwendet (Hauptakteur)."
                )
                
                xml_time_window = st.slider(
                    "Zeitfenster für Instanzen (Sekunden)",
                    min_value=1.0,
                    max_value=10.0,
                    value=4.0,
                    step=0.5,
                    help="Definiert das Zeitfenster vor und nach dem Event (start = Zeit - Fenster, end = Zeit + Fenster)"
                )
                
                # Funktion mit angepassten Optionen aufrufen
                sportscode_xml = create_sportscode_xml(
                    merged_data, 
                    player_col=xml_player_col, 
                    time_window=xml_time_window
                )
                
                # XML-Vorschau
                if st.checkbox("XML-Vorschau anzeigen"):
                    st.subheader("XML-Vorschau")
                    
                    # Eine einzelne Instanz zur Vorschau auswählen
                    sample_entry = None
                    for i, line in enumerate(sportscode_xml.split('\n')):
                        if '<instance>' in line:
                            sample_start = i
                        if '</instance>' in line and 'ID' not in line:  # Die erste Instance überspringen (Start Time)
                            sample_end = i
                            sample_entry = '\n'.join(sportscode_xml.split('\n')[sample_start:sample_end+1])
                            break
                    
                    if sample_entry:
                        st.code(sample_entry, language="xml")
                        st.info(f"Dies ist eine Vorschau einer einzelnen Instanz. Die vollständige XML enthält {merged_data.shape[0]} Instanzen.")
                    else:
                        st.error("Keine Instanz für die Vorschau gefunden.")
                
                # XML Export für Sportscode
                st.download_button(
                    label="Als Sportscode XML herunterladen",
                    data=sportscode_xml,
                    file_name="sportscode_data.xml",
                    mime="text/xml"
                )
        
        # Navigation
        st.markdown("---")
        if st.button("Zurück zur Datenverarbeitung", key="back_to_processing"):
            change_tab(1)
            st.rerun()
            
        if st.button("Neue Analyse starten", key="restart"):
            # Session-Variablen zurücksetzen
            for key in ['shot_plotter_df', 'xml_events', 'possession_df', 'merged_data']:
                if key in st.session_state:
                    del st.session_state[key]
            change_tab(0)
            st.rerun()
    else:
        st.info("Noch keine zusammengeführten Daten vorhanden. Bitte führen Sie zuerst Ihre Daten im vorherigen Tab zusammen.")
        if st.button("Zurück zur Datenverarbeitung", key="go_to_processing"):
            change_tab(1)
            st.rerun() 