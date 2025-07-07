# ================================================================================
# FUSSBALL-PASSDATEN INTEGRATION - STREAMLIT APP
# ================================================================================
# Diese App verarbeitet und integriert Daten aus drei Quellen:
# 1. Shot-Plotter CSV (Positionsdaten)
# 2. Playermaker XML (Passdaten) 
# 3. Playermaker Possession Excel (Zeitdaten)
# ================================================================================

# ================================================================================
# IMPORTS UND KONFIGURATION
# ================================================================================
import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import io
import base64
import json
import uuid
import math
from datetime import datetime, timedelta
import os
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import re
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO, StringIO
import zipfile

# Seitentitel und Beschreibung
st.set_page_config(page_title="Fußball-Passdaten Integration", layout="wide")
st.title("Fußball-Passdaten Integration")
st.markdown("""
Diese App verarbeitet und integriert Daten aus drei Quellen:
1. **Shot-Plotter CSV**: Enthält Positionsdaten
2. **Playermaker XML**: Enthält Passdaten
3. **Playermaker Possession Excel**: Enthält Zeitdaten
""")

# ================================================================================
# POSITION-MAPPING-FUNKTIONEN
# ================================================================================

def extract_unique_players(possession_df):
    """Extrahiert alle einzigartigen Spielernamen aus der Possession Summary
    
    Args:
        possession_df: DataFrame mit Possession-Daten
        
    Returns:
        list: Liste aller einzigartigen Spielernamen
    """
    unique_players = set()
    
    # Sammle Spieler aus verschiedenen Spalten
    player_columns = ['Player Name', 'passed_from', 'passed_to']
    
    for col in player_columns:
        if col in possession_df.columns:
            # Füge alle nicht-null Werte hinzu
            players = possession_df[col].dropna().unique()
            unique_players.update(players)
    
    # Entferne leere Strings und None-Werte
    unique_players = {player for player in unique_players if player and str(player).strip()}
    
    return sorted(list(unique_players))


def create_position_mapping_interface(unique_players):
    """Erstellt eine Streamlit-Oberfläche für das Mapping von Spielern zu Positionen
    
    Args:
        unique_players: Liste der einzigartigen Spielernamen
        
    Returns:
        dict: Dictionary mit Spielername -> Position Mapping
    """
    available_positions = ['TW', 'RV', 'LV', 'LIV', 'RIV', '6er', '8er', '10er', 'RM', 'LM', 'LF', 'RF', 'ST']
    
    st.subheader("Spieler-Position Mapping")
    st.write(f"Gefundene Spieler: {len(unique_players)}")
    st.write("Bitte ordnen Sie jedem Spieler eine Position zu:")
    
    position_mapping = {}
    
    # Erstelle eine Spalten-Layout für bessere Übersicht
    cols_per_row = 3
    for i in range(0, len(unique_players), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(unique_players):
                player = unique_players[i + j]
                with col:
                    # Verwende einen eindeutigen Key für jedes Selectbox
                    position = st.selectbox(
                        f"Position für {player}:",
                        options=['Auswählen...'] + available_positions,
                        key=f"position_{player}_{i}_{j}"
                    )
                    
                    if position != 'Auswählen...':
                        position_mapping[player] = position
    
    # Zeige eine Übersicht des aktuellen Mappings
    if position_mapping:
        st.subheader("Aktuelles Mapping")
        mapping_df = pd.DataFrame([
            {'Spieler': player, 'Position': position} 
            for player, position in position_mapping.items()
        ])
        st.dataframe(mapping_df)
        
        # Warnung wenn nicht alle Spieler gemappt sind
        unmapped_players = [p for p in unique_players if p not in position_mapping]
        if unmapped_players:
            st.warning(f"Noch nicht gemappte Spieler: {', '.join(unmapped_players)}")
        else:
            st.success("Alle Spieler wurden einer Position zugeordnet!")
    
    return position_mapping


def add_position_to_dataframe(df, position_mapping):
    """Fügt Position-Information zum DataFrame hinzu basierend auf dem Player Name
    
    Args:
        df: DataFrame mit Spielerdaten
        position_mapping: Dictionary mit Spielername -> Position Mapping
        
    Returns:
        DataFrame: DataFrame mit hinzugefügter Position-Spalte
    """
    # Erstelle eine Kopie des DataFrames
    result_df = df.copy()
    
    # Initialisiere Position-Spalten
    result_df['Position'] = None
    result_df['passed_from_Position'] = None
    result_df['passed_to_Position'] = None
    
    # Füge Positionen basierend auf Player Name hinzu
    if 'Player Name' in result_df.columns:
        for idx, row in result_df.iterrows():
            player_name = row['Player Name']
            if pd.notna(player_name) and player_name in position_mapping:
                result_df.at[idx, 'Position'] = position_mapping[player_name]
    
    # Füge Positionen basierend auf passed_from hinzu
    if 'passed_from' in result_df.columns:
        for idx, row in result_df.iterrows():
            passed_from_player = row['passed_from']
            if pd.notna(passed_from_player) and passed_from_player in position_mapping:
                result_df.at[idx, 'passed_from_Position'] = position_mapping[passed_from_player]
    
    # Füge Positionen basierend auf passed_to hinzu
    if 'passed_to' in result_df.columns:
        for idx, row in result_df.iterrows():
            passed_to_player = row['passed_to']
            if pd.notna(passed_to_player) and passed_to_player in position_mapping:
                result_df.at[idx, 'passed_to_Position'] = position_mapping[passed_to_player]
    
    return result_df

# ================================================================================
# XML DATA CLASSES UND PARSER
# ================================================================================

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

# ================================================================================
# DATENVERARBEITUNGSFUNKTIONEN
# ================================================================================

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
        receiving_leg_col = None
        release_foot_zone_col = None
        release_velocity_col = None
        releasing_leg_col = None
        
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
                elif "receiving leg" in col_lower or "receivingleg" in col_lower:
                    receiving_leg_col = col
                    st.success(f"Gefundene Receiving Leg Spalte: {col}")
                elif "release foot zone" in col_lower or "releasefootzone" in col_lower:
                    release_foot_zone_col = col
                    st.success(f"Gefundene Release Foot Zone Spalte: {col}")
                elif "release velocity" in col_lower and "m/sec" in col_lower:
                    release_velocity_col = col
                    st.success(f"Gefundene Release Velocity Spalte: {col}")
                elif "releasing leg" in col_lower or "releasingleg" in col_lower:
                    releasing_leg_col = col
                    st.success(f"Gefundene Releasing Leg Spalte: {col}")
        
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
                        elif "receiving leg" in col_str or "receivingleg" in col_str:
                            receiving_leg_col = col
                            st.success(f"Gefunden: Receiving Leg in Spalte {i}: {col}")
                        elif "release foot zone" in col_str or "releasefootzone" in col_str:
                            release_foot_zone_col = col
                            st.success(f"Gefunden: Release Foot Zone in Spalte {i}: {col}")
                        elif "release velocity" in col_str and "m/sec" in col_str:
                            release_velocity_col = col
                            st.success(f"Gefunden: Release Velocity in Spalte {i}: {col}")
                        elif "releasing leg" in col_str or "releasingleg" in col_str:
                            releasing_leg_col = col
                            st.success(f"Gefunden: Releasing Leg in Spalte {i}: {col}")
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
        display_cols = [col for col in [time_col, release_time_col, player_name_col, possession_type_col, receiving_leg_col, release_foot_zone_col, release_velocity_col, releasing_leg_col] if col is not None]
        st.write(df[display_cols].head(3))
        
        # Berechnung der Endzeit in Sekunden
        df['start_time_sec'] = df[time_col] * 60  # Minuten in Sekunden umrechnen
        df['end_time_sec'] = df['start_time_sec'] + df[release_time_col]
        
        # Stellt sicher, dass Player Name und Possession Type Spalten existieren
        if player_name_col is not None:
            df['Player Name'] = df[player_name_col]
        
        if possession_type_col is not None:
            df['Possession Type'] = df[possession_type_col]
        
        # Füge die neuen Spalten hinzu
        if receiving_leg_col is not None:
            df['Receiving Leg'] = df[receiving_leg_col]
        
        if release_foot_zone_col is not None:
            df['Release Foot Zone'] = df[release_foot_zone_col]
        
        if release_velocity_col is not None:
            df['Release Velocity'] = df[release_velocity_col]
        
        if releasing_leg_col is not None:
            df['Releasing Leg'] = df[releasing_leg_col]
        
        # Behalte auch die originale Time to Release Spalte
        if release_time_col is not None:
            df['Time to Release'] = df[release_time_col]
        
        # Bereinige leere Strings im finalen DataFrame
        df = clean_empty_strings(df)
        
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
        
        # Finde den nächsten Zeitwert in XML mit einer Toleranz von 0.1 Sekunden
        best_match = None
        min_diff = float('inf')
        
        for xml_time, passes_list in xml_time_to_passes.items():
            diff = abs(float(xml_time) - end_time)  # Stelle sicher, dass beide Werte float sind
            if diff < min_diff and diff <= 0.1:
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
    
    # Bereinige leere Strings im finalen DataFrame
    updated_possession_df = clean_empty_strings(updated_possession_df)
    
    return updated_possession_df

def merge_data_by_time(shot_plotter_df, possession_df, time_window=3.0):
    """Führt Shot-Plotter-Daten und Possession-Daten anhand der Zeit zusammen
    mit einem flexiblen Zeitfenster für das bestmögliche Matching.
    
    Benötigte Spalten im finalen Ergebnis:
    - Name (Player Name aus Possession Summary)
    - Position (wird durch separates Mapping hinzugefügt)
    - Zeit (korrigierte Zeit aus XML/Possession Summary)
    - Passed from (aus XML)
    - passed_from_Position (Position des Absenders, durch Mapping hinzugefügt)
    - Passed to (aus XML)
    - passed_to_Position (Position des Empfängers, durch Mapping hinzugefügt)
    - Possession type (aus Possession Summary)
    - Receiving Leg (aus Possession Summary)
    - Release Foot Zone (aus Possession Summary)
    - Release Velocity (aus Possession Summary)
    - Releasing Leg (aus Possession Summary)
    - Time to Release (aus Possession Summary)
    - X, Y Koordinaten (aus CSV/Shot-Plotter)
    - Neue Spalten: Team, Halbzeit, Gegnerdruck, Outcome, Passhöhe, Situation, Aktionstyp, X2, Y2
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
    
    # Debug: Zeige alle verfügbaren Spalten in Possession-Daten
    if debug_mode:
        st.subheader("Debug: Verfügbare Spalten in Possession-Daten")
        st.write("Alle Spalten:", possession_df.columns.tolist())
        
        if player_name_col:
            st.write(f"Player Name Spalte gefunden: '{player_name_col}'")
            st.write("Beispielwerte aus Player Name Spalte:")
            sample_values = possession_df[player_name_col].dropna().head(10).tolist()
            st.write(sample_values)
        else:
            st.warning("Keine Player Name Spalte gefunden!")
            st.write("Suche nach alternativen Spalten...")
            
            # Suche nach alternativen Spaltennamen
            alternative_cols = []
            for col in possession_df.columns:
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in ['player', 'name', 'spieler', 'code', 'id']):
                    alternative_cols.append(col)
            
            if alternative_cols:
                st.write("Gefundene alternative Spalten:", alternative_cols)
                for col in alternative_cols:
                    sample_values = possession_df[col].dropna().head(5).tolist()
                    st.write(f"'{col}': {sample_values}")
            else:
                st.error("Keine alternativen Spieler-Spalten gefunden!")
    
    # Wenn keine Player Name Spalte gefunden wurde, versuche Alternativen
    if player_name_col is None:
        # Suche nach alternativen Spaltennamen
        for col in possession_df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['player', 'name', 'spieler']):
                player_name_col = col
                if debug_mode:
                    st.success(f"Alternative Player Name Spalte gefunden: {col}")
                break
        
        # Falls immer noch nichts gefunden, verwende die erste Spalte, die nicht Zeit-bezogen ist
        if player_name_col is None:
            for col in possession_df.columns:
                col_lower = str(col).lower()
                if not any(keyword in col_lower for keyword in ['time', 'zeit', 'start', 'end', 'duration', 'dauer']):
                    player_name_col = col
                    if debug_mode:
                        st.warning(f"Verwende Fallback-Spalte als Player Name: {col}")
                    break
    
    
    # Merged-DataFrame initialisieren
    merged_data = []
    matches_found = 0
    total_entries = len(shot_plotter_df)
    
    # Tracking für bereits verwendete Possession-Einträge
    used_possession_indices = set()
    duplicate_attempts = 0
    
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
                
                # Suche nach dem besten verfügbaren Match (noch nicht verwendet)
                best_match = None
                best_match_index = None
                
                for match_idx, match_row in closest_matches.iterrows():
                    if match_idx not in used_possession_indices:
                        best_match = match_row
                        best_match_index = match_idx
                        break
                    else:
                        duplicate_attempts += 1
                        if debug_mode:
                            st.warning(f"Possession-Eintrag {match_idx} bereits verwendet - überspringe für Shot-Plotter-Zeit {shot_time:.2f}s")
                
                if best_match is not None and best_match_index is not None:
                    # Markiere diesen Possession-Eintrag als verwendet
                    used_possession_indices.add(best_match_index)
                    
                    # Erstelle einen kombinierten Eintrag mit den gewünschten Spalten
                    merged_entry = {
                        # Koordinaten aus Shot-Plotter
                        'X': shot_row['X'],
                        'Y': shot_row['Y']
                    }
                    
                    # Füge alle neuen Spalten aus Shot-Plotter hinzu
                    new_columns = ['Team', 'Halbzeit', 'Gegnerdruck', 'Outcome', 'Passhöhe', 'Situation', 'Aktionstyp', 'X2', 'Y2']
                    for col in new_columns:
                        if col in shot_row and pd.notna(shot_row[col]):
                            merged_entry[col] = shot_row[col]
                    
                    # Spielername aus Possession Summary
                    if player_name_col and player_name_col in best_match and pd.notna(best_match[player_name_col]):
                        merged_entry['Player Name'] = best_match[player_name_col]
                    else:
                        # Fallback: Verwende passed_from aus XML als Player Name
                        if 'passed_from' in best_match and pd.notna(best_match['passed_from']):
                            merged_entry['Player Name'] = best_match['passed_from']
                        else:
                            merged_entry['Player Name'] = 'Unbekannter Spieler'
                    
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
                    
                    # Füge die neuen Spalten aus der Possession Summary hinzu
                    if 'Receiving Leg' in best_match and pd.notna(best_match['Receiving Leg']):
                        merged_entry['Receiving Leg'] = best_match['Receiving Leg']
                    
                    if 'Release Foot Zone' in best_match and pd.notna(best_match['Release Foot Zone']):
                        merged_entry['Release Foot Zone'] = best_match['Release Foot Zone']
                    
                    if 'Release Velocity' in best_match and pd.notna(best_match['Release Velocity']):
                        merged_entry['Release Velocity'] = best_match['Release Velocity']
                    
                    if 'Releasing Leg' in best_match and pd.notna(best_match['Releasing Leg']):
                        merged_entry['Releasing Leg'] = best_match['Releasing Leg']
                    
                    if 'Time to Release' in best_match and pd.notna(best_match['Time to Release']):
                        merged_entry['Time to Release'] = best_match['Time to Release']
                    
                    merged_data.append(merged_entry)
                    matches_found += 1
                    
                    if debug_mode:
                        st.success(f"✅ Match: Shot-Plotter-Zeit {shot_time:.2f}s ↔ Possession-Zeit {best_match['end_time_sec']:.2f}s (Diff: {best_match['time_diff']:.3f}s, Index: {best_match_index})")
                else:
                    if debug_mode:
                        st.warning(f"❌ Kein verfügbarer Match für Shot-Plotter-Zeit {shot_time:.2f}s (alle Kandidaten bereits verwendet)")
            else:
                # Kein Match gefunden - diesen Eintrag später als unmatched hinzufügen
                if debug_mode:
                    st.info(f"🔍 Kein Match im Zeitfenster von {time_window}s für Shot-Plotter-Zeit {shot_time:.2f}s")
        except Exception as e:
            if debug_mode:
                st.error(f"Fehler beim Verarbeiten von Shot-Entry {idx}: {str(e)}")
            continue
    
    # Zusammenfassung
    if merged_data:
        merged_df = pd.DataFrame(merged_data)
        # Bereinige leere Strings im finalen DataFrame
        merged_df = clean_empty_strings(merged_df)
        
        success_rate = matches_found/total_entries*100
        st.success(f"✅ {matches_found} von {total_entries} Einträgen erfolgreich zusammengeführt ({success_rate:.1f}%)")
        
        if duplicate_attempts > 0:
            st.info(f"🛡️ {duplicate_attempts} Duplicate-Matches verhindert - jeder Possession-Eintrag wird nur einmal verwendet")
        
        return merged_df
    else:
        st.warning("Keine übereinstimmenden Zeiten gefunden. Überprüfen Sie das Zeitfenster oder die Zeitdaten.")
        return pd.DataFrame()

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
            
            # Format MM:SS.HH (wie in der CSV: "0:00.00", "1:03.64")
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 2:  # Format ist MM:SS.HH
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

def clean_empty_strings(df):
    """Konvertiert leere Strings zu NaN-Werten"""
    # Kopie des DataFrames erstellen
    df_cleaned = df.copy()
    
    # Für alle Spalten: leere Strings zu NaN konvertieren
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':  # Nur für String-Spalten
            # Ersetze leere Strings und Whitespace-Strings mit NaN
            df_cleaned[col] = df_cleaned[col].replace(['', ' ', '  ', '\t', '\n'], np.nan)
    
    return df_cleaned

def fix_column_names(df):
    """Korrigiert Spaltennamen, die durch Kodierungsprobleme beschädigt wurden"""
    column_mapping = {
        'Passhˆhe': 'Passhöhe',
        'PasshÃ¶he': 'Passhöhe',
        'PasshÃ¶he': 'Passhöhe',
        'Anstoﬂ': 'Anstoß',
        'AnstoÃŸ': 'Anstoß',
        'AnstoÃ': 'Anstoß'
    }
    
    # Korrigiere Spaltennamen
    corrected_columns = []
    for col in df.columns:
        if col in column_mapping:
            corrected_columns.append(column_mapping[col])
            st.info(f"Spaltenname korrigiert: '{col}' -> '{column_mapping[col]}'")
        else:
            corrected_columns.append(col)
    
    df.columns = corrected_columns
    return df
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
        
        # Neue Spalten aus der Possession Summary als Labels hinzufügen
        if 'Receiving Leg' in row and pd.notna(row['Receiving Leg']):
            receiving_leg_label = ET.SubElement(instance, "label")
            ET.SubElement(receiving_leg_label, "group").text = "Receiving Leg"
            ET.SubElement(receiving_leg_label, "text").text = str(row['Receiving Leg'])
        
        if 'Release Foot Zone' in row and pd.notna(row['Release Foot Zone']):
            release_foot_zone_label = ET.SubElement(instance, "label")
            ET.SubElement(release_foot_zone_label, "group").text = "Release Foot Zone"
            ET.SubElement(release_foot_zone_label, "text").text = str(row['Release Foot Zone'])
        
        if 'Release Velocity' in row and pd.notna(row['Release Velocity']):
            release_velocity_label = ET.SubElement(instance, "label")
            ET.SubElement(release_velocity_label, "group").text = "Release Velocity"
            ET.SubElement(release_velocity_label, "text").text = str(row['Release Velocity'])
        
        if 'Releasing Leg' in row and pd.notna(row['Releasing Leg']):
            releasing_leg_label = ET.SubElement(instance, "label")
            ET.SubElement(releasing_leg_label, "group").text = "Releasing Leg"
            ET.SubElement(releasing_leg_label, "text").text = str(row['Releasing Leg'])
        
        if 'Time to Release' in row and pd.notna(row['Time to Release']):
            time_to_release_label = ET.SubElement(instance, "label")
            ET.SubElement(time_to_release_label, "group").text = "Time to Release"
            ET.SubElement(time_to_release_label, "text").text = str(row['Time to Release'])
        
        if 'Position' in row and pd.notna(row['Position']):
            position_label = ET.SubElement(instance, "label")
            ET.SubElement(position_label, "group").text = "Position"
            ET.SubElement(position_label, "text").text = str(row['Position'])
        
        if 'passed_from_Position' in row and pd.notna(row['passed_from_Position']):
            passed_from_position_label = ET.SubElement(instance, "label")
            ET.SubElement(passed_from_position_label, "group").text = "Passed From Position"
            ET.SubElement(passed_from_position_label, "text").text = str(row['passed_from_Position'])
        
        if 'passed_to_Position' in row and pd.notna(row['passed_to_Position']):
            passed_to_position_label = ET.SubElement(instance, "label")
            ET.SubElement(passed_to_position_label, "group").text = "Passed To Position"
            ET.SubElement(passed_to_position_label, "text").text = str(row['passed_to_Position'])
        
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
        
        # Neue Spalten als Labels hinzufügen
        new_columns = {
            'Team': 'Team',
            'Halbzeit': 'Halbzeit', 
            'Gegnerdruck': 'Gegnerdruck',
            'Outcome': 'Outcome',
            'Passhöhe': 'Passhöhe',
            'Situation': 'Situation',
            'Aktionstyp': 'Aktionstyp'
        }
        
        for col, group_name in new_columns.items():
            if col in row and pd.notna(row[col]):
                new_label = ET.SubElement(instance, "label")
                ET.SubElement(new_label, "group").text = group_name
                ET.SubElement(new_label, "text").text = str(row[col])
        
        # Legacy-Spalten (falls noch vorhanden)
        legacy_columns = ['Distance', 'Type']
        for col in legacy_columns:
            if col in row and pd.notna(row[col]):
                legacy_label = ET.SubElement(instance, "label")
                ET.SubElement(legacy_label, "group").text = col
                ET.SubElement(legacy_label, "text").text = str(row[col])
    
    # XML als String zurückgeben
    return ET.tostring(root, encoding='unicode')


def load_player_database():
    """
    Lädt die Spieler-Datenbank aus einer lokalen JSON-Datei.
    Erstellt eine neue Datei, wenn sie nicht existiert.
    """
    import os
    import json
    
    db_path = "players_database.json"
    
    if os.path.exists(db_path):
        try:
            with open(db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            st.warning("Spieler-Datenbank konnte nicht geladen werden. Erstelle neue Datenbank.")
    
    # Standard-Struktur für neue Datenbank
    return {
        "players": {},
        "next_player_id": 1,
        "metadata": {
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_players": 0
        }
    }

def save_player_database(player_db):
    """
    Speichert die Spieler-Datenbank in eine lokale JSON-Datei.
    """
    import json
    from datetime import datetime
    
    db_path = "players_database.json"
    
    # Aktualisiere Metadaten
    player_db["metadata"]["last_updated"] = datetime.now().isoformat()
    player_db["metadata"]["total_players"] = len(player_db["players"])
    
    try:
        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(player_db, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Fehler beim Speichern der Spieler-Datenbank: {str(e)}")
        return False

def get_or_create_player_id(player_name, player_db):
    """
    Gibt die bestehende Player-ID zurück oder erstellt eine neue für den Spieler.
    """
    if not player_name or pd.isna(player_name) or str(player_name).strip() == "":
        return None
    
    player_name = str(player_name).strip()
    
    # Suche nach existierendem Spieler (case-insensitive)
    for existing_name, player_info in player_db["players"].items():
        if existing_name.lower() == player_name.lower():
            return player_info["player_id"]
    
    # Erstelle neue Player-ID mit 5 Stellen
    new_player_id = f"BVB_{player_db['next_player_id']:05d}"
    
    player_db["players"][player_name] = {
        "player_id": new_player_id,
        "display_name": player_name,
        "created": datetime.now().isoformat(),
        "matches_played": []
    }
    
    player_db["next_player_id"] += 1
    
    return new_player_id

def load_match_database():
    """
    Lädt die Match-Datenbank aus einer lokalen JSON-Datei.
    """
    import os
    import json
    
    db_path = "match_database.json"
    
    if os.path.exists(db_path):
        try:
            with open(db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            st.warning("Match-Datenbank konnte nicht geladen werden. Erstelle neue Datenbank.")
    
    return {
        "matches": {},
        "next_match_id": 1,
        "metadata": {
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_matches": 0
        }
    }

def save_match_database(match_db):
    """
    Speichert die Match-Datenbank in eine lokale JSON-Datei.
    """
    import json
    from datetime import datetime
    
    db_path = "match_database.json"
    
    # Aktualisiere Metadaten
    match_db["metadata"]["last_updated"] = datetime.now().isoformat()
    match_db["metadata"]["total_matches"] = len(match_db["matches"])
    
    try:
        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(match_db, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Fehler beim Speichern der Match-Datenbank: {str(e)}")
        return False

def extract_match_info_from_possession(possession_df):
    """
    Extrahiert Match-Informationen aus den Possession-Daten.
    """
    import re
    from datetime import datetime
    
    match_info = {
        'date': None,
        'team': None,
        'youth_level': None,
        'opponent': None,
        'venue': 'Unknown'
    }
    
    if possession_df is None or possession_df.empty:
        return match_info
    
    # Suche nach Datum in verschiedenen Spalten
    date_columns = ['Date', 'Datum', 'date', 'Match Date', 'Spieltag']
    for col in date_columns:
        if col in possession_df.columns:
            date_values = possession_df[col].dropna().unique()
            if len(date_values) > 0:
                try:
                    # Versuche verschiedene Datumsformate zu parsen
                    date_str = str(date_values[0])
                    
                    # Deutsche Formate: dd.mm.yyyy, dd.mm.yy
                    if re.match(r'\d{1,2}\.\d{1,2}\.\d{2,4}', date_str):
                        parts = date_str.split('.')
                        if len(parts[2]) == 2:  # 2-stelliges Jahr
                            parts[2] = '20' + parts[2]
                        match_info['date'] = f"{parts[2]}-{parts[1]:0>2}-{parts[0]:0>2}"
                    
                    # ISO Format: yyyy-mm-dd
                    elif re.match(r'\d{4}-\d{1,2}-\d{1,2}', date_str):
                        match_info['date'] = date_str
                    
                    # Amerikanisches Format: mm/dd/yyyy
                    elif re.match(r'\d{1,2}/\d{1,2}/\d{4}', date_str):
                        parts = date_str.split('/')
                        match_info['date'] = f"{parts[2]}-{parts[0]:0>2}-{parts[1]:0>2}"
                        
                except Exception:
                    continue
                    
                if match_info['date']:
                    break
    
    # Suche nach Team/Jugend-Informationen
    team_columns = ['Team', 'Mannschaft', 'Squad', 'team']
    for col in team_columns:
        if col in possession_df.columns:
            team_values = possession_df[col].dropna().unique()
            if len(team_values) > 0:
                team_str = str(team_values[0]).lower()
                
                # Erkenne Jugendmannschaften
                youth_patterns = {
                    'u19': r'u-?19|unter.?19|19er',
                    'u17': r'u-?17|unter.?17|17er',
                    'u16': r'u-?16|unter.?16|16er',
                    'u15': r'u-?15|unter.?15|15er',
                    'u14': r'u-?14|unter.?14|14er',
                    'u13': r'u-?13|unter.?13|13er',
                    'u12': r'u-?12|unter.?12|12er',
                    'u11': r'u-?11|unter.?11|11er',
                    'u10': r'u-?10|unter.?10|10er',
                    'u9': r'u-?9|unter.?9|9er',
                    'u8': r'u-?8|unter.?8|8er',
                    'u7': r'u-?7|unter.?7|7er'
                }
                
                for youth, pattern in youth_patterns.items():
                    if re.search(pattern, team_str):
                        match_info['youth_level'] = youth.upper() if youth else None
                        break
                
                # Team-Name extrahieren (BVB, Borussia Dortmund, etc.)
                if 'bvb' in team_str or 'dortmund' in team_str or 'borussia' in team_str:
                    match_info['team'] = 'BVB'
                else:
                    match_info['team'] = team_values[0]
                break
    
    # Suche nach Gegner-Informationen in verschiedenen Spalten
    opponent_columns = ['Opponent', 'Gegner', 'vs', 'Against', 'opponent']
    for col in opponent_columns:
        if col in possession_df.columns:
            opponent_values = possession_df[col].dropna().unique()
            if len(opponent_values) > 0:
                match_info['opponent'] = str(opponent_values[0])
                break
    
    return match_info

def create_or_get_match_id(match_info, match_db):
    """
    Erstellt eine neue Match-ID oder gibt eine bestehende zurück.
    Priorität: XML-Daten > Possession-Daten > CSV-Dateiname
    """
    from datetime import datetime
    import re
    
    extracted_info = None
    data_source = "unknown"
    
    # 1. Priorität: XML-Daten (am zuverlässigsten)
    if 'xml_data' in match_info:
        st.info("🔍 Extrahiere Match-Informationen aus XML-Daten...")
        extracted_info = extract_match_info_from_xml(match_info['xml_data'])
        data_source = "xml_session_info"
        
        if extracted_info and (extracted_info.get('date') or extracted_info.get('youth_level')):
            opponent_info = f" gegen {extracted_info.get('opponent', 'UNKNOWN')}" if extracted_info.get('opponent') else ""
            st.success(f"✅ XML-Daten erfolgreich extrahiert: {extracted_info.get('youth_level', 'N/A')} am {extracted_info.get('date', 'N/A')}{opponent_info}")
    
    # 2. Fallback: Possession-Daten
    elif 'possession_data' in match_info:
        st.info("📊 Extrahiere Match-Informationen aus Possession-Daten...")
        extracted_info = extract_match_info_from_possession(match_info['possession_data'])
        data_source = "possession_summary"
    
    # Wenn wir extrahierte Informationen haben, verwende diese
    if extracted_info:
        match_date = extracted_info.get('date')
        youth_level = extracted_info.get('youth_level', 'SENIOR')
        team = extracted_info.get('team', 'BVB')
        opponent = extracted_info.get('opponent')
        
        # Falls kein Gegner gefunden wurde, verwende Fallback
        if not opponent:
            opponent = 'UNKNOWN'
            st.info("ℹ️ Kein Gegner in den Daten gefunden - verwende 'UNKNOWN'")
        
        # Erstelle strukturierte Match-ID
        if match_date:
            # Format: YOUTH_TEAM-OPPONENT_DATE
            # Beispiel: U12_BVB-BMG_2023-11-04
            
            # Bereinige Namen für Dateisystem - mit Null-Checks
            clean_team = re.sub(r'[^\w]', '', (team or 'BVB').upper())
            clean_opponent = re.sub(r'[^\w]', '', (opponent or 'UNKNOWN').upper().replace(' ', ''))[:10]  # Max 10 Zeichen
            
            if youth_level and youth_level != 'SENIOR':
                match_identifier = f"{youth_level}_{clean_team}-{clean_opponent}_{match_date}"
            else:
                match_identifier = f"{clean_team}-{clean_opponent}_{match_date}"
        else:
            # Fallback ohne Datum
            today = datetime.now().strftime('%Y-%m-%d')
            clean_team = re.sub(r'[^\w]', '', (team or 'BVB').upper())
            match_identifier = f"{youth_level or 'SENIOR'}_{clean_team}-TRAINING_{today}"
    
    else:
        # 3. Fallback für alte CSV-basierte Methode
        csv_filename = match_info.get('csv_filename', '')
        data_source = "csv_filename"
        
        if csv_filename:
            base_name = csv_filename.replace('.csv', '').replace('.CSV', '')
            match_identifier = re.sub(r'_\d+_\d+$', '', base_name)
            if not match_identifier:
                match_identifier = base_name
        else:
            today = datetime.now().strftime("%Y-%m-%d")
            match_identifier = f"{match_info.get('type', 'training')}_{today}"
    
    # Prüfe, ob bereits ein Match mit diesem Identifier existiert
    for match_id, match_data in match_db["matches"].items():
        if match_data.get("identifier") == match_identifier:
            return match_id
    
    # Erstelle neue Match-ID basierend auf dem Identifier
    safe_identifier = re.sub(r'[^\w\-_.]', '_', match_identifier)
    new_match_id = safe_identifier
    
    # Prüfe, ob diese ID bereits existiert
    counter = 1
    original_id = new_match_id
    while new_match_id in match_db["matches"]:
        new_match_id = f"{original_id}_{counter}"
        counter += 1
    
    # Erstelle Match-Eintrag mit erweiterten Informationen
    match_entry = {
        "match_id": new_match_id,
        "identifier": match_identifier,
        "date": datetime.now().isoformat(),
        "type": match_info.get('type', 'match'),
        "venue": match_info.get('venue', 'BVB Training Ground'),
        "created": datetime.now().isoformat(),
        "players": [],
        "data_source": data_source
    }
    
    # Füge extrahierte Informationen hinzu falls verfügbar
    if extracted_info:
        match_entry.update({
            "match_date": extracted_info.get('date'),
            "youth_level": extracted_info.get('youth_level'),
            "team": extracted_info.get('team', 'BVB'),
            "opponent": opponent if 'opponent' in locals() else extracted_info.get('opponent')
        })
    else:
        match_entry.update({
            "original_filename": match_info.get('csv_filename', '')
        })
    
    match_db["matches"][new_match_id] = match_entry
    match_db["next_match_id"] += 1
    
    return new_match_id


def generate_consistent_event_id(timestamp, youth_level="U12"):
    """
    Generiert eine konsistente Event-ID basierend auf Youth-Level und Zeitstempel.
    Format: YOUTH_timestamp (z.B. U12_125.67, U17_89.23)
    
    Args:
        timestamp: Der Zeitstempel des Events (float)
        youth_level: Das Youth-Level (z.B. "U12", "U17", "SENIOR")
    
    Returns:
        str: Konsistente Event-ID im Format "YOUTH_timestamp"
    """
    # Bereinige Youth-Level falls nötig
    if not youth_level or youth_level == "Unknown":
        youth_level = "U12"
    
    # Stelle sicher dass Youth-Level korrekt formatiert ist
    if not youth_level.startswith(('U', 'SENIOR')):
        youth_level = f"U{youth_level}" if youth_level.isdigit() else "U12"
    
    # Formatiere Zeitstempel mit 2 Dezimalstellen
    timestamp_str = f"{float(timestamp):.5f}"
    
    return f"{youth_level}_{timestamp_str}"

def extract_youth_level_from_match_info(match_info):
    """
    Extrahiert das Youth-Level aus Match-Informationen.
    
    Args:
        match_info: Dictionary mit Match-Informationen
    
    Returns:
        str: Youth-Level (z.B. "U12", "U17", "SENIOR")
    """
    youth_level = match_info.get('youth_level', '')
    
    # Fallback-Mechanismen
    if not youth_level or youth_level == "Unknown":
        # Versuche aus anderen Feldern zu extrahieren
        team = match_info.get('team', '')
        if 'U12' in team:
            youth_level = 'U12'
        elif 'U13' in team:
            youth_level = 'U13'
        elif 'U14' in team:
            youth_level = 'U14'
        elif 'U15' in team:
            youth_level = 'U15'
        elif 'U16' in team:
            youth_level = 'U16'
        elif 'U17' in team:
            youth_level = 'U17'
        elif 'U19' in team:
            youth_level = 'U19'
        else:
            youth_level = 'U12'  # Default
    
    return youth_level

def create_structured_json_export_with_ids(df):
    """
    Erstellt JSON-Export mit der gleichen Struktur wie "Events herunterladen & im Archiv speichern".
    Verwendet flache Event-Struktur für Konsistenz und fügt Metadaten für JSON Merger hinzu.
    """
    import uuid
    import json
    from datetime import datetime
    
    # Lade Datenbanken
    player_db = load_player_database()
    match_db = load_match_database()
    
    # Match-Informationen extrahieren - GLEICHE LOGIK WIE "Events herunterladen & im Archiv speichern"
    xml_data = None
    possession_data = None
    csv_filename = ""
    
    # 1. Priorität: XML-Daten (am zuverlässigsten)
    if 'xml_file' in st.session_state and st.session_state.xml_file is not None:
        xml_data = st.session_state.xml_file
    
    # 2. Fallback: Possession-Daten
    elif 'possession_summary' in st.session_state and st.session_state.possession_summary is not None:
        possession_data = st.session_state.possession_summary
    
    # 3. Fallback: CSV-Dateiname
    if 'shot_plotter_file' in st.session_state:
        csv_filename = st.session_state.shot_plotter_file.name
    
    # Erstelle Match-Info basierend auf verfügbaren Daten (gleiche Priorität wie "Events herunterladen & im Archiv speichern")
    match_info = {
        'type': 'match',
        'venue': 'BVB Training Ground'
    }
    
    # Priorisiere XML-Daten
    if xml_data:
        match_info['xml_data'] = xml_data
    elif possession_data is not None:
        match_info['possession_data'] = possession_data
    else:
        match_info['csv_filename'] = csv_filename
    
    # Match ID erstellen/abrufen mit der gleichen Logik
    match_id = create_or_get_match_id(match_info, match_db)
    
    # Sammle alle Spieler für dieses Match (für playerInfo Metadaten)
    players_in_match = set()
    player_id_mapping = {}
    
    # Sammle alle Spielernamen aus verschiedenen Spalten
    player_columns = ['Player Name', 'passed_from', 'passed_to']
    for col in player_columns:
        if col in df.columns:
            unique_players = df[col].dropna().unique()
            for player_name in unique_players:
                if player_name and str(player_name).strip():
                    player_id = get_or_create_player_id(player_name, player_db)
                    if player_id:
                        player_id_mapping[player_name] = player_id
                        players_in_match.add(player_id)
    
    # Events aus df erstellen - GLEICHE STRUKTUR WIE "Events herunterladen & im Archiv speichern"
    events_data = []
    for idx, row in df.iterrows():
        # Extract player information
        player_name = row.get('Player Name', 'Unknown')
        
        # Create coordinates structure
        coordinates = {}
        if 'X' in row and pd.notna(row['X']):
            coordinates["start_x"] = float(row['X'])
        if 'Y' in row and pd.notna(row['Y']):
            coordinates["start_y"] = float(row['Y'])
        if 'X2' in row and pd.notna(row['X2']):
            coordinates["end_x"] = float(row['X2'])
        if 'Y2' in row and pd.notna(row['Y2']):
            coordinates["end_y"] = float(row['Y2'])
        
        # Create passing_network for database structure
        passing_network = {
            "passed_from": row.get('passed_from'),
            "passed_to": row.get('passed_to'),
            "passed_from_Position": row.get('passed_from_Position'),
            "passed_to_Position": row.get('passed_to_Position')
        }
        
        # Clean additional_data - remove coordinates, passing info, and redundant data
        additional_data = {}
        for col, value in row.items():
            if col not in ['Zeit', 'Team', 'Halbzeit', 'Player Name', 'passed_from', 'passed_to', 
                         'passed_from_Position', 'passed_to_Position', 'X', 'Y', 'X2', 'Y2', 
                         'Outcome', 'Aktionstyp', 'Position'] and pd.notna(value):
                additional_data[col] = str(value) if not isinstance(value, (int, float, bool)) else value
        
        # Determine action_type
        action_type = row.get('Aktionstyp', 'Pass')
        if isinstance(action_type, str):
            action_type = action_type.upper()
        if action_type != "PASS" and action_type != "LOSS":
            action_type = "PASS"
        
        # Generate consistent event ID using timestamp and youth level
        timestamp = row.get('Zeit', 0)
        match_entry = match_db["matches"].get(match_id, {})
        youth_level = extract_youth_level_from_match_info(match_entry)
        event_id = generate_consistent_event_id(timestamp, youth_level)
        
        event = {
            "event_id": event_id,
            "timestamp": timestamp,
            "player": player_name,
            "action_type": action_type,
            "start_x": coordinates.get("start_x", 0),
            "start_y": coordinates.get("start_y", 0),
            "end_x": coordinates.get("end_x", 0),
            "end_y": coordinates.get("end_y", 0),
            "outcome": row.get('Outcome', ''),
            "team": row.get('Team', ''),
            "half": row.get('Halbzeit', 1),
            "additional_data": additional_data,
            "passing_network": passing_network
        }
        
        # Bereinige Event für JSON
        event = clean_data_for_json(event)
        events_data.append(event)
    
    # Match-Metadaten aus der Match-Database verwenden - GLEICHE STRUKTUR
    match_entry = match_db["matches"].get(match_id, {})
    match_metadata = {
        "match_id": match_id,
        "date": match_entry.get('match_date', datetime.now().strftime('%Y-%m-%d')),
        "team": match_entry.get('team', 'BVB'),
        "opponent": match_entry.get('opponent', 'Unknown'),
        "youth_level": match_entry.get('youth_level', 'SENIOR'),
        "venue": match_entry.get('venue', 'Unknown'),
        "total_events": len(events_data)
    }
    # Bereinige Match-Metadaten für JSON
    match_metadata = clean_data_for_json(match_metadata)
    
    # Metadata für JSON Merger - FÜR KOMPATIBILITÄT MIT merge_structured_json_halves
    metadata = {
        "exportInfo": {
            "exportId": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "source": "Shot-Plotter & Playermaker Data Analysis",
            "version": "2.0",
            "format": "structured_football_data"
        },
        "playerInfo": {
            "totalPlayers": len(players_in_match),
            "playerIdMapping": player_id_mapping
        },
        "match_info": match_metadata
    }
    
    # Download-Datei erstellen - HYBRID STRUKTUR: Metadata für JSON Merger + Event-Struktur für Konsistenz
    event_export = {
        "metadata": metadata,
        "events": events_data,
        "export_metadata": {
            "exported_at": datetime.now().isoformat(),
            "total_events": len(events_data),
            "match_id": match_id
        }
    }
    
    return json.dumps(event_export, indent=2, ensure_ascii=False)

def extract_match_info_from_xml(xml_file_content):
    """
    Extrahiert Match-Informationen aus der XML-Datei.
    Sucht nach SESSION_INFO start_time, text (Gegner + Jugend) und Team-Informationen in instance codes.
    """
    import xml.etree.ElementTree as ET
    import re
    from datetime import datetime
    
    match_info = {
        'date': None,
        'team': None,
        'youth_level': None,
        'opponent': None,
        'venue': 'Unknown'
    }
    
    try:
        # Parse XML
        if hasattr(xml_file_content, 'read'):
            xml_content = xml_file_content.read()
            xml_file_content.seek(0)  # Reset für weitere Verwendung
        else:
            xml_content = xml_file_content
            
        root = ET.fromstring(xml_content)
        
        # 1. Extrahiere Informationen aus SESSION_INFO
        session_info = root.find('SESSION_INFO')
        if session_info is not None:
            # Datum extrahieren
            start_time_elem = session_info.find('start_time')
            if start_time_elem is not None:
                start_time_str = start_time_elem.text
                try:
                    # Parse: "2023-11-04 12:58:01.000000+0100"
                    dt = datetime.fromisoformat(start_time_str.replace('+0100', '+01:00'))
                    match_info['date'] = dt.strftime('%Y-%m-%d')
                except Exception as e:
                    st.warning(f"Konnte Datum nicht parsen: {start_time_str}")
            
            # Gegner und Jugendkategorie aus text-Element extrahieren
            # Format: "BMG U12" oder "Schalke U14" etc.
            text_elem = session_info.find('text')
            if text_elem is not None and text_elem.text:
                text_content = text_elem.text.strip()
                st.info(f"🔍 SESSION_INFO text gefunden: **{text_content}**")
                
                # Parse text für Jugendkategorie
                youth_patterns = {
                    'U19': r'u-?19|unter.?19|19er',
                    'U17': r'u-?17|unter.?17|17er',
                    'U16': r'u-?16|unter.?16|16er',
                    'U15': r'u-?15|unter.?15|15er',
                    'U14': r'u-?14|unter.?14|14er',
                    'U13': r'u-?13|unter.?13|13er',
                    'U12': r'u-?12|unter.?12|12er',
                    'U11': r'u-?11|unter.?11|11er',
                    'U10': r'u-?10|unter.?10|10er',
                    'U9': r'u-?9|unter.?9|9er',
                    'U8': r'u-?8|unter.?8|8er',
                    'U7': r'u-?7|unter.?7|7er'
                }
                
                text_lower = text_content.lower()
                for youth, pattern in youth_patterns.items():
                    if re.search(pattern, text_lower):
                        match_info['youth_level'] = youth
                        st.success(f"⚽ Jugendkategorie gefunden: **{youth}**")
                        break
                
                # Parse text für Gegner (alles außer der Jugendkategorie)
                opponent_text = text_content
                if match_info['youth_level']:
                    # Entferne alle Varianten der gefundenen Jugendkategorie
                    youth_level = match_info['youth_level']
                    patterns_to_remove = [
                        youth_level,  # U12
                        youth_level.lower(),  # u12
                        youth_level.replace('U', 'u-'),  # u-12
                        youth_level.replace('U', 'unter'),  # unter12
                        youth_level.replace('U', '') + 'er'  # 12er
                    ]
                    
                    for pattern in patterns_to_remove:
                        opponent_text = re.sub(r'\b' + re.escape(pattern) + r'\b', '', opponent_text, flags=re.IGNORECASE)
                
                # Bereinige den Gegner-Text
                opponent_text = re.sub(r'\s+', ' ', opponent_text.strip())  # Mehrfache Leerzeichen entfernen
                
                if opponent_text:
                    match_info['opponent'] = opponent_text
                    st.success(f"🆚 Gegner gefunden: **{opponent_text}**")
                else:
                    match_info['opponent'] = 'UNKNOWN'
                    st.info("ℹ️ Kein spezifischer Gegner erkannt - verwende 'UNKNOWN'")
        
        # 2. Extrahiere Team-Informationen aus instance codes (als Fallback oder Bestätigung)
        instances = root.find('ALL_INSTANCES')
        if instances is not None:
            for instance in instances.findall('instance'):
                code_elem = instance.find('code')
                if code_elem is not None:
                    code_text = code_elem.text.lower()
                    
                    # Suche nach Team-Informationen
                    # Beispiel: "Borussia Dortmund U12 team ball possession"
                    if 'team ball possession' in code_text or 'borussia dortmund' in code_text or 'bvb' in code_text:
                        # Team-Name
                        if 'borussia dortmund' in code_text or 'bvb' in code_text:
                            match_info['team'] = 'BVB'
                            st.success(f"🏆 Team gefunden: **BVB**")
                        
                        # Falls noch keine Jugendkategorie aus SESSION_INFO, versuche aus instance code
                        if not match_info['youth_level']:
                            youth_patterns = {
                                'U19': r'u-?19|unter.?19|19er',
                                'U17': r'u-?17|unter.?17|17er',
                                'U16': r'u-?16|unter.?16|16er',
                                'U15': r'u-?15|unter.?15|15er',
                                'U14': r'u-?14|unter.?14|14er',
                                'U13': r'u-?13|unter.?13|13er',
                                'U12': r'u-?12|unter.?12|12er',
                                'U11': r'u-?11|unter.?11|11er',
                                'U10': r'u-?10|unter.?10|10er',
                                'U9': r'u-?9|unter.?9|9er',
                                'U8': r'u-?8|unter.?8|8er',
                                'U7': r'u-?7|unter.?7|7er'
                            }
                            
                            for youth, pattern in youth_patterns.items():
                                if re.search(pattern, code_text):
                                    match_info['youth_level'] = youth
                                    st.info(f"⚽ Jugendkategorie aus instance code: **{youth}**")
                                    break
                        
                        # Wenn wir Team-Info gefunden haben, können wir aufhören
                        if match_info['team']:
                            break
        
        # Fallback: Setze Standard-Team wenn nicht gefunden
        if not match_info['team']:
            match_info['team'] = 'BVB'
            st.info("🏆 Kein Team gefunden - verwende Standard: **BVB**")
        
        return match_info
        
    except Exception as e:
        st.warning(f"Fehler beim Parsen der XML-Datei: {str(e)}")
        return match_info
# Event Database Functions
def load_event_database():
    """Lädt die Event-Datenbank von lokaler JSON-Datei"""
    try:
        if os.path.exists("event_database.json"):
            with open("event_database.json", "r", encoding='utf-8') as f:
                return json.load(f)
        else:
            # Neue Datenbank initialisieren
            return {
                "events_by_match": {},
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "total_matches": 0,
                    "total_events": 0
                }
            }
    except Exception as e:
        st.error(f"Fehler beim Laden der Event-Datenbank: {str(e)}")
        return {
            "events_by_match": {},
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_matches": 0,
                "total_events": 0
            }
        }

def save_event_database(event_db):
    """Speichert die Event-Datenbank in lokale JSON-Datei"""
    try:
        # Metadaten aktualisieren
        event_db["metadata"]["last_updated"] = datetime.now().isoformat()
        event_db["metadata"]["total_matches"] = len(event_db["events_by_match"])
        total_events = sum(len(match_data.get("events", [])) for match_data in event_db["events_by_match"].values())
        event_db["metadata"]["total_events"] = total_events
        
        # Bereinige Daten für JSON-Export
        clean_event_db = clean_data_for_json(event_db)
        
        with open("event_database.json", "w", encoding='utf-8') as f:
            json.dump(clean_event_db, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.warning(f"Fehler beim Speichern der Event-Datenbank: {str(e)}")
        return False

def add_or_replace_events_in_database(match_id, events_data, match_metadata, event_db):
    """
    Fügt Events zur Event-Datenbank hinzu oder aktualisiert bestehende Events basierend auf event_id.
    
    Neuer Ablauf mit konsistenten Event-IDs:
    1. Vergleich Match ID -> Zuordnung zu richtigem Spiel
    2. Wenn Spiel nicht existiert -> neues Spiel in database anlegen
    3. Für jedes Event: Event ID prüfen
       - Wenn Event ID bereits existiert -> altes Event löschen und durch neues ersetzen
       - Wenn Event ID nicht existiert -> neues Event hinzufügen
    """
    try:
        # Prüfe, ob bereits Events für diese Match-ID existieren
        existing_match = match_id in event_db["events_by_match"]
        
        if existing_match:
            st.info(f"🔄 Match-ID '{match_id}' existiert bereits - aktualisiere Events basierend auf Event-IDs...")
            existing_events = event_db["events_by_match"][match_id]["events"]
            old_event_count = len(existing_events)
            
            # Erstelle ein Dictionary der bestehenden Events basierend auf event_id
            existing_events_dict = {}
            for event in existing_events:
                event_id = event.get("event_id")
                if event_id:
                    existing_events_dict[event_id] = event
            
            # Verarbeite neue Events
            updated_events = 0
            new_events = 0
            
            for new_event in events_data:
                event_id = new_event.get("event_id")
                if not event_id:
                    st.warning(f"⚠️ Event ohne event_id übersprungen: {new_event}")
                    continue
                
                if event_id in existing_events_dict:
                    # Event ID existiert bereits -> ersetze das alte Event
                    existing_events_dict[event_id] = new_event
                    updated_events += 1
                    st.info(f"🔄 Event aktualisiert: {event_id}")
                else:
                    # Neue Event ID -> füge Event hinzu
                    existing_events_dict[event_id] = new_event
                    new_events += 1
                    st.info(f"➕ Neues Event hinzugefügt: {event_id}")
            
            # Konvertiere Dictionary zurück zu Liste
            final_events = list(existing_events_dict.values())
            
            # Aktualisiere den Match-Eintrag mit den neuen Events
            event_db["events_by_match"][match_id]["events"] = final_events
            event_db["events_by_match"][match_id]["last_updated"] = datetime.now().isoformat()
            
            # Update Match-Metadaten falls nötig
            event_db["events_by_match"][match_id]["match_info"] = match_metadata
            
            final_event_count = len(final_events)
            
            st.success(f"✅ Events aktualisiert für Match '{match_id}':")
            st.success(f"   📊 {updated_events} Events aktualisiert")
            st.success(f"   ➕ {new_events} neue Events hinzugefügt")
            st.success(f"   📈 Gesamt: {old_event_count} → {final_event_count} Events")
            
        else:
            st.info(f"🆕 Neue Match-ID '{match_id}' - erstelle neuen Eintrag...")
            
            # Erstelle neuen Match-Eintrag
            event_db["events_by_match"][match_id] = {
                "match_info": match_metadata,
                "events": events_data,
                "added_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
            
            final_event_count = len(events_data)
            st.success(f"✅ Neues Match erstellt: {final_event_count} Events für Match '{match_id}'")
        
        # Update globale Metadata
        event_db["metadata"]["total_events"] = sum(
            len(match_data["events"]) for match_data in event_db["events_by_match"].values()
        )
        event_db["metadata"]["total_matches"] = len(event_db["events_by_match"])
        event_db["metadata"]["last_updated"] = datetime.now().isoformat()
        
        return final_event_count if 'final_event_count' in locals() else len(events_data)
            
    except Exception as e:
        st.error(f"Fehler beim Speichern der Events: {str(e)}")
        return 0

def clean_data_for_json(data):
    """Bereinigt Daten für JSON-Export (NaN -> null, etc.)"""
    import math
    
    if isinstance(data, dict):
        return {k: clean_data_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, float) and math.isnan(data):
        return None
    elif data is None or str(data).lower() == 'nan':
        return None
    else:
        return data


# Definiere Tabs für den Workflow
tabs = st.tabs(["Daten hochladen", "Daten verarbeiten", "Ergebnisse", "📄 XML Merger", "📊 CSV Merger", "🔧 JSON Merger"])

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
                
                # Versuche verschiedene Kodierungen für deutsche Umlaute
                encodings_to_try = ['utf-8', 'iso-8859-1', 'windows-1252', 'latin-1']
                shot_plotter_df = None
                
                for encoding in encodings_to_try:
                    try:
                        st.session_state.shot_plotter_file.seek(0)  # Zurücksetzen für jeden Versuch
                        shot_plotter_df = pd.read_csv(st.session_state.shot_plotter_file, encoding=encoding)
                        st.success(f"CSV erfolgreich mit Kodierung '{encoding}' geladen")
                        break
                    except UnicodeDecodeError:
                        st.warning(f"Kodierung '{encoding}' funktioniert nicht, versuche nächste...")
                        continue
                    except Exception as e:
                        st.warning(f"Fehler mit Kodierung '{encoding}': {str(e)}")
                        continue
                
                if shot_plotter_df is None:
                    st.error("Konnte CSV mit keiner der versuchten Kodierungen laden")
                    st.session_state.shot_plotter_df = pd.DataFrame()
                else:
                    # Korrigiere Spaltennamen, falls nötig
                    shot_plotter_df = fix_column_names(shot_plotter_df)
                    
                    # Bereinige leere Strings
                    shot_plotter_df = clean_empty_strings(shot_plotter_df)
                    
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
                xml_df = pd.DataFrame([event.to_dict() for event in events])
                
                # Bereinige leere Strings
                xml_df = clean_empty_strings(xml_df)
                
                st.session_state.xml_events = xml_df
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
                
                # Bereinige leere Strings
                df = clean_empty_strings(df)
                
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
        main_columns = ['Player Name', 'Position', 'Zeit', 'passed_from', 'passed_from_Position', 'passed_to', 'passed_to_Position', 'Possession Type', 'X', 'Y']
        
        # Zeige die Hauptspalten zuerst in der Übersicht
        available_main_columns = [col for col in main_columns if col in merged_data.columns]
        other_columns = [col for col in merged_data.columns if col not in main_columns]
        display_columns = available_main_columns + other_columns
        
        # Übersicht
        st.subheader("Zusammengeführte Daten")
        st.write("Hauptspalten in der finalen Tabelle:")
        st.markdown("""
        - **Player Name**: Spielername aus der Possession Summary
        - **Position**: Spielerposition (TW, RV, LV, LIV, RIV, ZM, ZOM, LF, RF, ST) - durch manuelles Mapping zugeordnet
        - **Zeit**: Korrigierte Zeit aus XML/Possession Summary (synchronisiert mit Video)
        - **passed_from**: Ballabsender aus XML
        - **passed_from_Position**: Position des Ballabsenders - durch Mapping zugeordnet
        - **passed_to**: Ballempfänger aus XML
        - **passed_to_Position**: Position des Ballempfängers - durch Mapping zugeordnet
        - **Possession Type**: Art des Ballbesitzes aus der Possession Summary
        - **Receiving Leg**: Bein zum Empfangen des Balls aus der Possession Summary
        - **Release Foot Zone**: Zone des Abspielbeins aus der Possession Summary
        - **Release Velocity**: Geschwindigkeit beim Abspielen (m/sec) aus der Possession Summary
        - **Releasing Leg**: Bein zum Abspielen des Balls aus der Possession Summary
        - **Time to Release**: Zeit bis zum Abspielen (Sekunden) aus der Possession Summary
        - **X, Y**: Startkoordinaten aus dem Shot-Plotter (manuelle CSV)
        - **X2, Y2**: Endkoordinaten aus dem Shot-Plotter (manuelle CSV)
        - **Team**: Team-Information aus der CSV
        - **Halbzeit**: Halbzeit-Information aus der CSV
        - **Gegnerdruck**: Gegnerdruck-Kategorie aus der CSV
        - **Outcome**: Ergebnis des Passes aus der CSV
        - **Passhöhe**: Höhe des Passes aus der CSV
        - **Situation**: Spielsituation aus der CSV
        - **Aktionstyp**: Typ der Aktion aus der CSV
        """)
        
        # Anzeigen der zusammengeführten Daten
        st.dataframe(merged_data[display_columns])
        
        # Position-Mapping-Sektion
        st.subheader("Spieler-Position Mapping")
        st.write("Ordnen Sie jedem Spieler eine Position zu, um diese Information in der finalen Datei zu speichern.")
        
        # Extrahiere einzigartige Spieler
        if st.button("Spieler extrahieren und Position-Mapping starten", key="extract_players_btn"):
            unique_players = extract_unique_players(merged_data)
            st.info(f"🔍 DEBUG: {len(unique_players)} einzigartige Spieler gefunden: {sorted(list(unique_players))}")
            
            st.session_state.unique_players = unique_players
            st.session_state.show_position_mapping = True
            st.rerun()
        
        # Zeige Position-Mapping Interface wenn Spieler extrahiert wurden
        if st.session_state.get('show_position_mapping', False) and 'unique_players' in st.session_state:
            position_mapping = create_position_mapping_interface(st.session_state.unique_players)
            
            # Speichere das Mapping im session_state
            st.session_state.position_mapping = position_mapping
            
            # Button zum Anwenden des Position-Mappings
            if position_mapping and len(position_mapping) > 0:
                if st.button("Position-Mapping anwenden", key="apply_position_mapping_btn"):
                    # Füge Positionen zum merged_data hinzu
                    merged_data_with_positions = add_position_to_dataframe(merged_data, position_mapping)
                    st.session_state.merged_data = merged_data_with_positions
                    st.session_state.show_position_mapping = False  # Verstecke das Mapping-Interface
                    st.success(f"Positionen erfolgreich hinzugefügt! {len(position_mapping)} Spieler wurden gemappt.")
                    st.rerun()
        
        # Update merged_data reference after potential position mapping
        merged_data = st.session_state.merged_data
        
    
        
        # Export-Optionen
        st.subheader("Daten exportieren")
        
        # Definiere die Hauptspalten für den Export und stelle sicher, dass sie in der richtigen Reihenfolge sind
        main_columns = ['Player Name', 'Zeit', 'passed_from', 'passed_to', 'Possession Type', 'X', 'Y', 'X2', 'Y2', 'Team', 'Halbzeit', 'Gegnerdruck', 'Outcome', 'Passhöhe', 'Situation', 'Aktionstyp']
        available_main_columns = [col for col in main_columns if col in merged_data.columns]
        other_columns = [col for col in merged_data.columns if col not in main_columns]
        export_columns = available_main_columns + other_columns
        
        # Vorbereiten der Export-Daten mit richtiger Spaltenreihenfolge
        export_data = merged_data[export_columns].copy()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # CSV Export
            csv = export_data.to_csv(index=False, na_rep='Keine Angabe')
            st.download_button(
                label="Als CSV herunterladen",
                data=csv,
                file_name="merged_pass_data.csv",
                mime="text/csv"
            )
        
        with col2:
            # JSON Export Options
            st.markdown("**JSON Export & Event-Archiv**")
            
            # Enhanced Structured JSON Export with persistent IDs
            try:
                structured_json_with_ids = create_structured_json_export_with_ids(export_data)
                
                # Extrahiere Match-ID aus dem JSON für den Dateinamen
                try:
                    json_data = json.loads(structured_json_with_ids)
                    match_id_for_filename = json_data.get("metadata", {}).get("match_info", {}).get("match_id", "unknown_match")
                except:
                    match_id_for_filename = "unknown_match"
                
                st.download_button(
                    label="🆔 Strukturiertes JSON mit IDs herunterladen",
                    data=structured_json_with_ids,
                    file_name=f"{match_id_for_filename}.json",
                    mime="application/json",
                    help="Erweiterte JSON-Struktur mit konsistenten Spieler- und Match-IDs (BVB_00001, XML SESSION_INFO-basiert)"
                )
                
                # Event-Archivierung Button (NUR bei Klick speichern)
                if st.button("📚 Events herunterladen & im Archiv speichern", key="save_events_btn"):
                    try:
                        # Event-Archiv erstellen und lokal speichern
                        player_db = load_player_database()
                        match_db = load_match_database()
                        event_db = load_event_database()
                        
                        # Match-Informationen extrahieren - GLEICHE LOGIK WIE create_structured_json_export_with_ids
                        xml_data = None
                        possession_data = None
                        csv_filename = ""
                        
                        # 1. Priorität: XML-Daten (am zuverlässigsten)
                        if 'xml_file' in st.session_state and st.session_state.xml_file is not None:
                            xml_data = st.session_state.xml_file
                        
                        # 2. Fallback: Possession-Daten
                        elif 'possession_summary' in st.session_state and st.session_state.possession_summary is not None:
                            possession_data = st.session_state.possession_summary
                        
                        # 3. Fallback: CSV-Dateiname
                        if 'shot_plotter_file' in st.session_state:
                            csv_filename = st.session_state.shot_plotter_file.name
                        
                        # Erstelle Match-Info basierend auf verfügbaren Daten (gleiche Priorität wie create_structured_json_export_with_ids)
                        match_info = {
                            'type': 'match',
                            'venue': 'BVB Training Ground'
                        }
                        
                        # Priorisiere XML-Daten
                        if xml_data:
                            match_info['xml_data'] = xml_data
                        elif possession_data is not None:
                            match_info['possession_data'] = possession_data
                        else:
                            match_info['csv_filename'] = csv_filename
                        
                        # Match ID erstellen/abrufen mit der gleichen Logik
                        match_id = create_or_get_match_id(match_info, match_db)
                        
                        # *** FEHLENDE SPIELER-REGISTRIERUNG HINZUFÜGEN ***
                        # Sammle alle Spieler für dieses Match und registriere sie in der Player-Database
                        # (Gleiche Logik wie in create_structured_json_export_with_ids)
                        players_in_match = set()
                        player_id_mapping = {}
                        
                        # Sammle alle Spielernamen aus verschiedenen Spalten
                        player_columns = ['Player Name', 'passed_from', 'passed_to']
                        for col in player_columns:
                            if col in export_data.columns:
                                unique_players = export_data[col].dropna().unique()
                                for player_name in unique_players:
                                    if player_name and str(player_name).strip():
                                        player_id = get_or_create_player_id(player_name, player_db)
                                        if player_id:
                                            player_id_mapping[player_name] = player_id
                                            players_in_match.add(player_id)
                        
                        st.info(f"🆔 {len(players_in_match)} Spieler in Player-Database registriert: {list(player_id_mapping.keys())}")
                        
                        # Events aus export_data erstellen
                        events_data = []
                        for idx, row in export_data.iterrows():
                            # Extract player information
                            player_name = row.get('Player Name', 'Unknown')
                            
                            # *** FIX: Verwende Koordinaten DIREKT aus dem DataFrame (die bereits korrekt konvertiert wurden)
                            # Nicht neu erstellen - das DataFrame hat bereits die richtigen Koordinaten!
                            start_x = float(row.get('X', 0)) if pd.notna(row.get('X', 0)) else 0
                            start_y = float(row.get('Y', 0)) if pd.notna(row.get('Y', 0)) else 0
                            end_x = float(row.get('X2', 0)) if pd.notna(row.get('X2', 0)) else 0
                            end_y = float(row.get('Y2', 0)) if pd.notna(row.get('Y2', 0)) else 0
                            
                            # Create passInfo structure
                            pass_info = {}
                            if 'passed_from' in row and pd.notna(row['passed_from']):
                                pass_info["passFrom"] = str(row['passed_from'])
                                if 'passed_from_Position' in row and pd.notna(row['passed_from_Position']):
                                    pass_info["passFromPosition"] = str(row['passed_from_Position'])
                            
                            if 'passed_to' in row and pd.notna(row['passed_to']):
                                pass_info["passTo"] = str(row['passed_to'])
                                if 'passed_to_Position' in row and pd.notna(row['passed_to_Position']):
                                    pass_info["passToPosition"] = str(row['passed_to_Position'])
                            
                            # Create passing_network for database structure
                            passing_network = {
                                "passed_from": row.get('passed_from'),
                                "passed_to": row.get('passed_to'),
                                "passed_from_Position": row.get('passed_from_Position'),
                                "passed_to_Position": row.get('passed_to_Position')
                            }
                            
                            # Clean additional_data - remove coordinates, passing info, and redundant data (SAME as Tab 2)
                            additional_data = {}
                            for col, value in row.items():
                                if col not in ['Zeit', 'Team', 'Halbzeit', 'Player Name', 'passed_from', 'passed_to', 
                                             'passed_from_Position', 'passed_to_Position', 'X', 'Y', 'X2', 'Y2', 
                                             'Outcome', 'Aktionstyp', 'Position'] and pd.notna(value):
                                    additional_data[col] = str(value) if not isinstance(value, (int, float, bool)) else value
                            
                            # Determine action_type (SAME as Tab 2)
                            action_type = row.get('Aktionstyp', 'Pass')
                            if isinstance(action_type, str):
                                action_type = action_type.upper()
                            if action_type != "PASS" and action_type != "LOSS":
                                action_type = "PASS"
                            
                            # Generate consistent event ID using timestamp and youth level (SAME as Tab 2)
                            timestamp = row.get('Zeit', 0)
                            match_entry = match_db["matches"].get(match_id, {})
                            youth_level = extract_youth_level_from_match_info(match_entry)
                            event_id = generate_consistent_event_id(timestamp, youth_level)
                            
                            # Create event with EXACT same structure as Tab 2 - FIXE KOORDINATEN VERWENDEN
                            event = {
                                "event_id": event_id,
                                "timestamp": timestamp,
                                "player": player_name,
                                "action_type": action_type,
                                "start_x": start_x,  # ← FIX: Direkt aus DataFrame statt coordinates dict
                                "start_y": start_y,  # ← FIX: Direkt aus DataFrame
                                "end_x": end_x,      # ← FIX: Direkt aus DataFrame
                                "end_y": end_y,      # ← FIX: Direkt aus DataFrame
                                "outcome": row.get('Outcome', ''),
                                "team": row.get('Team', ''),
                                "half": row.get('Halbzeit', 1),
                                "additional_data": additional_data,
                                "passing_network": passing_network
                            }
                            
                            # Bereinige Event für JSON (SAME as Tab 2)
                            event = clean_data_for_json(event)
                            events_data.append(event)
                        
                        # Match-Metadaten aus der Match-Database verwenden
                        match_entry = match_db["matches"].get(match_id, {})
                        match_metadata = {
                            "match_id": match_id,
                            "date": match_entry.get('match_date', datetime.now().strftime('%Y-%m-%d')),
                            "team": match_entry.get('team', 'BVB'),
                            "opponent": match_entry.get('opponent', 'Unknown'),
                            "youth_level": match_entry.get('youth_level', 'SENIOR'),
                            "venue": match_entry.get('venue', 'Unknown'),
                            "total_events": len(events_data)
                        }
                        # Bereinige Match-Metadaten für JSON
                        match_metadata = clean_data_for_json(match_metadata)
                        
                        # Events zur Datenbank hinzufügen
                        new_events_count = add_or_replace_events_in_database(match_id, events_data, match_metadata, event_db)
                        
                        # Datenbanken speichern
                        save_event_database(event_db)
                        save_match_database(match_db)
                        save_player_database(player_db)
                        
                        # Download-Datei erstellen
                        event_export = {
                            "match_info": match_metadata,
                            "events": events_data,
                            "export_metadata": {
                                "exported_at": datetime.now().isoformat(),
                                "total_events": len(events_data),
                                "match_id": match_id
                            }
                        }
                        
                        # Erfolgreiche Speicherung anzeigen
                        st.success(f"✅ {new_events_count} neue Events im Archiv gespeichert!")
                        
                        # Download-Button für das JSON erstellen
                        st.download_button(
                            label="💾 JSON-Datei herunterladen",
                            data=json.dumps(event_export, indent=2, ensure_ascii=False),
                            file_name=f"events_{match_id}.json",
                            mime="application/json",
                            help=f"Events wurden im lokalen Event-Archiv gespeichert. {new_events_count} neue Events hinzugefügt.",
                            key="download_events_json"
                        )
                        
                    except Exception as e:
                        st.error(f"Fehler beim Speichern im Event-Archiv: {str(e)}")
                
                # Event-Archiv Status anzeigen (ohne zu speichern)
                try:
                    player_db = load_player_database()
                    match_db = load_match_database()
                    event_db = load_event_database()
                    
                    with st.expander("📊 Event-Archiv Status"):
                        col_arch1, col_arch2, col_arch3 = st.columns(3)
                        with col_arch1:
                            st.metric("Gespeicherte Matches", event_db["metadata"]["total_matches"])
                        with col_arch2:
                            st.metric("Registrierte Spieler", len(player_db.get("players", {})))
                        with col_arch3:
                            st.metric("Gesamt Events", event_db["metadata"]["total_events"])
                        
                        st.info("💡 Klicke auf '📚 Events herunterladen & im Archiv speichern' um die aktuellen Events zu speichern.")
                        
                except Exception as e:
                    st.warning(f"Fehler beim Laden der Archiv-Status: {str(e)}")
                
                # Zeige Info über das ID-System
                with st.expander("ℹ️ Info zum ID-System"):
                    st.markdown("""
                    **Konsistente ID-Verwaltung:**
                    
                    - **Spieler-IDs**: Format `BVB_XXXXX` (z.B. BVB_00001, BVB_00002) - 5-stellig
                    - **Event-IDs**: Format `YOUTH_timestamp` (z.B. U12_125.67, U17_89.23) - **NEU!**
                    - **Match-IDs**: Intelligente Erstellung mit Datenpriorität
                    
                    **🆔 Event-ID System (NEU - Konsistent über alle Tabs):**
                    
                    **Format:** `YOUTH_timestamp`
                    - **Youth-Level**: U12, U13, U14, U15, U16, U17, U19, SENIOR
                    - **Timestamp**: Zeitstempel mit 2 Dezimalstellen (z.B. 125.67 = 2:05.67)
                    
                    **Beispiele:**
                    - `U12_45.23` - U12-Event bei 45.23 Sekunden
                    - `U17_125.67` - U17-Event bei 2:05.67 (125.67 Sekunden)
                    - `SENIOR_1823.45` - Senior-Event bei 30:23.45
                    
                    **Vorteile des neuen Event-ID Systems:**
                    - ✅ **Einzigartig**: Zeitstempel sind praktisch unique
                    - ✅ **Konsistent**: Gleiche Methode in allen Tabs (Archivierung, JSON Export, Merger)
                    - ✅ **Aussagekräftig**: Youth-Level und Zeit direkt erkennbar
                    - ✅ **Sortierbar**: Chronologische Sortierung möglich
                    - ✅ **Kompakt**: Kürzer als UUIDs, aber eindeutig
                    - ✅ **Menschenlesbar**: Verständlich für Analysten
                    
                    **Ersetzt die alten Systeme:**
                    - ❌ **Alt**: `event_1`, `event_2`, `event_3` (nicht eindeutig)
                    - ❌ **Alt**: `d6a46c24-da07-4a83-83a9-8fe42ce1a479` (nicht lesbar)
                    - ✅ **Neu**: `U12_125.67` (eindeutig UND lesbar)
                    
                    **Match-ID Erstellung (Prioritätssystem):**
                    
                    **1. 🎯 XML SESSION_INFO (Höchste Priorität):**
                    - **Datum**: Aus `<SESSION_INFO><start_time>` extrahiert
                    - **Gegner**: Aus `<SESSION_INFO><opponent>` extrahiert (z.B. BMG, S04, FCB)
                    - **Jugend & Team**: Aus instance codes wie `"Borussia Dortmund U12 team ball possession"`
                    - **Beispiel XML**: 
                      ```xml
                      <SESSION_INFO>
                        <start_time>2023-11-04 12:58:01.000000+0100</start_time>
                        <opponent>BMG</opponent>
                      </SESSION_INFO>
                      ```
                    - **Team-Erkennung**: `Borussia Dortmund U12` → `U12_BVB`
                    
                    **2. 📊 Possession Summary (Fallback):**
                    - Datum automatisch erkannt (DD.MM.YYYY, YYYY-MM-DD, MM/DD/YYYY)
                    - Jugendmannschaft erkannt (U19, U17, U16, U15, U14, U13, U12, etc.)
                    - Team-Information (BVB, Borussia Dortmund automatisch erkannt)
                    - Gegner-Information falls in Daten vorhanden
                    
                    **3. 📁 CSV-Dateiname (Letzter Fallback):**
                    - Basierend auf Dateinamen-Struktur
                    - Automatische Viertel-Erkennung und -Zusammenführung
                    
                    **🆔 Match-ID Format:**
                    - `JUGEND_TEAM-GEGNER_DATUM`
                    - Beispiele:
                      - `U12_BVB-BMG_2023-11-04` (aus XML SESSION_INFO)
                      - `U17_BVB-S04_2023-12-15` (aus XML SESSION_INFO)
                      - `SENIOR_BVB-FCB_2024-01-20` (aus Possession/CSV)
                    
                    **Vorteile des gesamten ID-Systems:**
                    - ✅ **XML-Priorität**: Zuverlässigste Datenquelle (SESSION_INFO + instance codes)
                    - ✅ **Automatisch**: Keine manuelle Eingabe erforderlich
                    - ✅ **Intelligent**: Online-Suche für fehlende Gegner-Informationen
                    - ✅ **Robust**: Mehrfache Fallback-Mechanismen
                    - ✅ **Konsistent**: Eindeutige IDs über alle Analysen hinweg
                    - ✅ **Semantisch**: Alle IDs sind aussagekräftig und lesbar
                    - ✅ **Sicher**: Lokale Speicherung, keine sensiblen Daten in GitHub
                    
                    **Datenquellen-Hierarchie:**
                    1. **XML SESSION_INFO** (am zuverlässigsten)
                    2. **Possession Summary** (strukturierte Daten)
                    3. **CSV-Dateiname** (Fallback)
                    
                    **Fallback-Mechanismen:**
                    - Bei fehlenden XML-Daten: Possession-Daten als Basis
                    - Bei fehlenden Possession-Daten: CSV-Dateiname als Basis
                    - Bei fehlenden Datums-Informationen: Aktuelles Datum
                    - Bei Online-Suchen-Fehlern: Intelligente Vermutung basierend auf typischen Gegnern
                    
                    **Sicherheit:**
                    - Spielerdaten werden nur lokal gespeichert
                    - Keine sensiblen Daten in Git/GitHub
                    - Automatische .gitignore-Einträge
                    - Online-Suchen verwenden nur öffentliche Informationen
                    """)
                    
                    # Status der Datenbanken anzeigen
                    try:
                        player_db = load_player_database()
                        match_db = load_match_database()
                        
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.metric("Registrierte Spieler", len(player_db.get("players", {})))
                        with col_info2:
                            st.metric("Gespeicherte Matches", len(match_db.get("matches", {})))
                            
                    except Exception as e:
                        st.warning(f"Fehler beim Laden der Datenbank-Info: {str(e)}")
                        
            except Exception as e:
                st.error(f"Fehler beim Erstellen des ID-basierten JSON-Exports: {str(e)}")
                st.info("Der Standard-Export ist weiterhin verfügbar.")
        
        with col3:
            # Excel Export
            buffer = io.BytesIO()
            export_data.to_excel(buffer, index=False, na_rep='Keine Angabe')
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
                    - `Team`: Team-Information
                    - `Halbzeit`: Halbzeit-Information
                    - `Gegnerdruck`: Gegnerdruck-Kategorie
                    - `Outcome`: Ergebnis des Passes
                    - `Passhöhe`: Höhe des Passes
                    - `Situation`: Spielsituation
                    - `Aktionstyp`: Typ der Aktion
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

# XML Merger Functions
def parse_xml_file_merger(uploaded_file):
    """Parse uploaded XML file and return the root element"""
    try:
        content = uploaded_file.read()
        root = ET.fromstring(content)
        uploaded_file.seek(0)  # Reset file pointer
        return root, content
    except ET.ParseError as e:
        st.error(f"Fehler beim Parsen der XML-Datei {uploaded_file.name}: {e}")
        return None, None

def adjust_time_values_xml(element, time_offset):
    """Recursively adjust all time values in the XML by adding the offset"""
    for child in element:
        if child.tag in ['start', 'end'] and child.text:
            try:
                original_time = float(child.text)
                new_time = original_time + time_offset
                child.text = str(new_time)
            except ValueError:
                pass  # Skip if not a valid number
        
        # Recursively process child elements
        adjust_time_values_xml(child, time_offset)

def merge_xml_quarters(xml_data_list, quarter_offsets):
    """Merge multiple XML quarters into one"""
    if not xml_data_list:
        return None
    
    # Use the first XML as the base
    base_root = xml_data_list[0]['root']
    base_instances = base_root.find('ALL_INSTANCES')
    
    if base_instances is None:
        st.error("Keine ALL_INSTANCES Sektion in der ersten XML-Datei gefunden")
        return None
    
    # Get the highest ID from base instances
    max_id = 0
    for instance in base_instances.findall('instance'):
        id_element = instance.find('ID')
        if id_element is not None and id_element.text:
            try:
                max_id = max(max_id, int(id_element.text))
            except ValueError:
                pass
    
    # Adjust times in the base XML (first quarter)
    adjust_time_values_xml(base_instances, quarter_offsets[0])
    
    # Add instances from other quarters
    for i, xml_data in enumerate(xml_data_list[1:], 1):
        quarter_root = xml_data['root']
        quarter_instances = quarter_root.find('ALL_INSTANCES')
        
        if quarter_instances is None:
            st.warning(f"Keine ALL_INSTANCES in Viertel {i+1} gefunden, überspringe...")
            continue
        
        # Adjust times in this quarter
        adjust_time_values_xml(quarter_instances, quarter_offsets[i])
        
        # Add all instances from this quarter to the base
        for instance in quarter_instances.findall('instance'):
            # Update the ID to avoid conflicts
            max_id += 1
            id_element = instance.find('ID')
            if id_element is not None:
                id_element.text = str(max_id)
            
            # Add the instance to the base
            base_instances.append(instance)
    
    return base_root

def create_xml_download_file(merged_root, filename):
    """Create downloadable XML file"""
    xml_str = ET.tostring(merged_root, encoding='unicode', xml_declaration=True)
    return xml_str.encode('utf-8')

# Tab 4: XML Merger
with tabs[3]:
    st.header("XML-Dateien zusammenführen")
    st.markdown("**Führen Sie XML-Dateien verschiedener Spielviertel zu einer zusammenhängenden Datei zusammen**")
    
    # File upload section
    st.subheader("1. XML-Dateien hochladen")
    xml_merger_files = st.file_uploader(
        "Wählen Sie die XML-Dateien der Viertel aus",
        type=['xml'],
        accept_multiple_files=True,
        help="Laden Sie die XML-Dateien in der Reihenfolge der Viertel hoch (1. Viertel, 2. Viertel, etc.)",
        key="xml_merger_uploader"
    )

    if xml_merger_files:
        st.success(f"{len(xml_merger_files)} XML-Datei(en) hochgeladen")
        
        # Parse XML files
        xml_merger_data = []
        valid_xml_files = True
        
        for i, file in enumerate(xml_merger_files):
            st.write(f"**Viertel {i+1}:** {file.name}")
            root, content = parse_xml_file_merger(file)
            if root is not None:
                xml_merger_data.append({
                    'root': root,
                    'filename': file.name,
                    'quarter': i+1
                })
            else:
                valid_xml_files = False
                break
        
        if valid_xml_files and xml_merger_data:
            st.subheader("2. Startzeiten für Viertel konfigurieren")
            st.markdown("Geben Sie die Startzeit für jedes Viertel in Sekunden an:")
            
            xml_quarter_offsets = []
            
            # Create columns for time inputs
            cols = st.columns(min(len(xml_merger_data), 4))
            
            for i, data in enumerate(xml_merger_data):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    st.subheader(f"Viertel {i+1}")
                    st.write(f"📁 {data['filename']}")
                    
                    if i == 0:
                        # First quarter starts at 0
                        offset = st.number_input(
                            f"Startzeit (Sek)",
                            min_value=0.0,
                            value=0.0,
                            step=1.0,
                            key=f"xml_merger_offset_{i}"
                        )
                    else:
                        # Suggest offset based on previous quarter
                        suggested_offset = sum(xml_quarter_offsets) + 900  # 15 minutes default
                        offset = st.number_input(
                            f"Startzeit (Sek)",
                            min_value=0.0,
                            value=float(suggested_offset),
                            step=1.0,
                            key=f"xml_merger_offset_{i}",
                            help=f"Vorschlag: {suggested_offset} Sek (15 Min pro Viertel)"
                        )
                    
                    xml_quarter_offsets.append(offset)
                    
                    # Show time in minutes for better understanding
                    minutes = int(offset // 60)
                    seconds = int(offset % 60)
                    st.caption(f"⏰ {minutes}:{seconds:02d} Min")
            
            # Preview section
            st.subheader("3. Vorschau")
            
            preview_cols = st.columns(len(xml_merger_data))
            
            for i, (data, offset) in enumerate(zip(xml_merger_data, xml_quarter_offsets)):
                with preview_cols[i]:
                    st.subheader(f"Viertel {i+1}")
                    instances = data['root'].find('ALL_INSTANCES')
                    if instances is not None:
                        instance_count = len(instances.findall('instance'))
                        st.metric("Anzahl Events", instance_count)
                        st.metric("Startzeit", f"{int(offset//60)}:{int(offset%60):02d}")
            
            # Merge button
            st.subheader("4. Zusammenführen")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                xml_merge_button = st.button("🔄 XML-Dateien zusammenführen", type="primary", key="xml_merger_merge")
            
            with col2:
                xml_output_filename = st.text_input(
                    "Dateiname für zusammengeführte XML",
                    value="merged_match.xml",
                    help="Name der resultierenden XML-Datei",
                    key="xml_merger_filename"
                )
            
            if xml_merge_button:
                with st.spinner("Führe XML-Dateien zusammen..."):
                    try:
                        merged_root = merge_xml_quarters(xml_merger_data, xml_quarter_offsets)
                        
                        if merged_root is not None:
                            # Create download file
                            xml_content = create_xml_download_file(merged_root, xml_output_filename)
                            
                            st.success("✅ XML-Dateien erfolgreich zusammengeführt!")
                            
                            # Show statistics
                            merged_instances = merged_root.find('ALL_INSTANCES')
                            if merged_instances is not None:
                                total_instances = len(merged_instances.findall('instance'))
                                st.metric("Gesamte Events in der zusammengeführten Datei", total_instances)
                            
                            # Download button
                            st.download_button(
                                label="📥 Zusammengeführte XML herunterladen",
                                data=xml_content,
                                file_name=xml_output_filename,
                                mime="application/xml",
                                key="xml_merger_download"
                            )
                            
                            # Show success message with file info
                            st.info(f"Die Datei '{xml_output_filename}' wurde erfolgreich erstellt und kann heruntergeladen werden.")
                            
                    except Exception as e:
                        st.error(f"Fehler beim Zusammenführen: {str(e)}")

    else:
        st.info("👆 Laden Sie XML-Dateien hoch, um zu beginnen")
    
    # Help section
    st.markdown("---")
    with st.expander("ℹ️ Hilfe und Beispiele"):
        st.markdown("""
        ### Wie funktioniert der XML-Merger?
        
        **Funktionsweise:**
        - Erwartet XML-Dateien mit `ALL_INSTANCES` Elementen
        - Passt `start` und `end` Zeitangaben an
        - Führt alle `instance` Elemente zusammen
        - Verhindert ID-Konflikte durch automatische Neunummerierung
        
        ### Beispiel Startzeiten:
        - **1. Viertel**: 0 Sekunden (0:00)
        - **2. Viertel**: 900 Sekunden (15:00) 
        - **3. Viertel**: 1800 Sekunden (30:00)
        - **4. Viertel**: 2700 Sekunden (45:00)
        
        ### XML-Struktur:
        Die App erwartet XML-Dateien mit folgender Grundstruktur:
        ```xml
        <file>
          <ALL_INSTANCES>
            <instance>
              <ID>1</ID>
              <start>16.87</start>
              <end>44.26</end>
              <code>...</code>
              <label>...</label>
            </instance>
          </ALL_INSTANCES>
        </file>
        ```
        """) 

# CSV Merger Functions
def parse_csv_file_merger(uploaded_file):
    """Parse uploaded CSV file and return DataFrame"""
    try:
        # Try different encodings
        content = uploaded_file.read()
        uploaded_file.seek(0)
        
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin-1')
        
        uploaded_file.seek(0)
        return df
    except Exception as e:
        st.error(f"Fehler beim Parsen der CSV-Datei {uploaded_file.name}: {e}")
        return None

def adjust_time_values_csv_merger(df, time_offset, time_column='Zeit'):
    """Adjust time values in CSV DataFrame"""
    df_copy = df.copy()
    if time_column in df_copy.columns:
        try:
            df_copy[time_column] = pd.to_numeric(df_copy[time_column], errors='coerce') + time_offset
        except Exception as e:
            st.error(f"Fehler beim Anpassen der Zeitwerte: {e}")
    return df_copy

def merge_csv_quarters_merger(csv_data_list, quarter_offsets, time_column='Zeit'):
    """Merge multiple CSV quarters into one"""
    if not csv_data_list:
        return None
    
    merged_dataframes = []
    
    for i, csv_data in enumerate(csv_data_list):
        df = csv_data['dataframe'].copy()
        
        # Adjust times
        df_adjusted = adjust_time_values_csv_merger(df, quarter_offsets[i], time_column)
        merged_dataframes.append(df_adjusted)
    
    # Concatenate all dataframes
    merged_df = pd.concat(merged_dataframes, ignore_index=True)
    
    # Sort by time column if it exists
    if time_column in merged_df.columns:
        merged_df = merged_df.sort_values(by=time_column).reset_index(drop=True)
    
    return merged_df

def create_csv_download_file_merger(merged_df, filename):
    """Create downloadable CSV file"""
    csv_string = merged_df.to_csv(index=False)
    return csv_string.encode('utf-8')

# JSON Merger Functions
def parse_json_file_merger(uploaded_file):
    """Parse uploaded JSON file"""
    try:
        content = uploaded_file.read()
        data = json.loads(content)
        uploaded_file.seek(0)
        return data
    except json.JSONDecodeError as e:
        st.error(f"Fehler beim Parsen der JSON-Datei {uploaded_file.name}: {e}")
        return None

def adjust_time_values_json_merger(data, time_offset, time_field='Zeit'):
    """Adjust time values in JSON data - handles both simple lists and structured JSON"""
    adjusted_data = []
    
    # Handle different JSON structures
    events_list = []
    if isinstance(data, list):
        events_list = data
    elif isinstance(data, dict):
        # Check for common event structures
        if 'events' in data:
            events_list = data['events']
        elif 'data' in data:
            events_list = data['data']
        else:
            # If it's a single event object, treat it as a list
            events_list = [data]
    else:
        return adjusted_data
    
    for item in events_list:
        if not isinstance(item, dict):
            continue
            
        adjusted_item = item.copy()
        
        # Try different possible time field names
        time_fields_to_check = [time_field, 'Zeit', 'time', 'timestamp']
        
        for field in time_fields_to_check:
            if field in adjusted_item:
                try:
                    adjusted_item[field] = float(adjusted_item[field]) + time_offset
                    break  # Found and adjusted time field, break out of loop
                except (ValueError, TypeError):
                    continue  # Try next field
        
        adjusted_data.append(adjusted_item)
    
    return adjusted_data

def merge_json_quarters_merger(json_data_list, quarter_offsets, time_field='Zeit'):
    """Merge multiple JSON quarters into one - handles different JSON structures and merges playerInfo"""
    if not json_data_list:
        return None
    
    merged_data = []
    merged_player_info = {
        "totalPlayers": 0,
        "playerIdMapping": {}
    }
    
    # Debug: Log the structure of input files
    st.info(f"🔍 DEBUG: Verarbeite {len(json_data_list)} JSON-Dateien...")
    
    for i, json_data in enumerate(json_data_list):
        data = json_data['data']
        filename = json_data['filename']
        
        st.info(f"🔍 DEBUG: Datei {i+1} ({filename}) - Typ: {type(data)}")
        
        # Extract playerInfo if available
        if isinstance(data, dict) and 'playerInfo' in data:
            player_info = data['playerInfo']
            if 'playerIdMapping' in player_info:
                # Merge playerIdMapping from this JSON
                for player_name, player_id in player_info['playerIdMapping'].items():
                    if player_name not in merged_player_info['playerIdMapping']:
                        merged_player_info['playerIdMapping'][player_name] = player_id
                st.info(f"🔍 DEBUG: Datei {i+1} - PlayerInfo gefunden mit {len(player_info.get('playerIdMapping', {}))} Spielern")
            else:
                st.info(f"🔍 DEBUG: Datei {i+1} - PlayerInfo vorhanden, aber kein playerIdMapping")
        else:
            st.info(f"🔍 DEBUG: Datei {i+1} - Keine PlayerInfo gefunden")
        
        # Extract events from data
        events_to_process = []
        if isinstance(data, list):
            events_to_process = data
            st.info(f"🔍 DEBUG: Datei {i+1} - Liste mit {len(events_to_process)} Events")
        elif isinstance(data, dict):
            if 'events' in data:
                events_to_process = data['events']
                st.info(f"🔍 DEBUG: Datei {i+1} - Events-Struktur mit {len(events_to_process)} Events")
            elif 'data' in data:
                events_to_process = data['data']
                st.info(f"🔍 DEBUG: Datei {i+1} - Data-Struktur mit {len(events_to_process)} Events")
            else:
                events_to_process = [data]
                st.info(f"🔍 DEBUG: Datei {i+1} - Einzelnes Event-Objekt")
        
        # Collect player names from events if no playerInfo available
        if not merged_player_info['playerIdMapping']:
            players_from_events = set()
            for event in events_to_process:
                if isinstance(event, dict):
                    # Try different player name fields
                    player_field = event.get('Player Name', event.get('player', event.get('Player')))
                    
                    # Handle different player field formats
                    player_name = None
                    if isinstance(player_field, dict):
                        # If player field is a dictionary, extract the actual name
                        player_name = player_field.get('playerName', player_field.get('name', str(player_field)))
                    elif isinstance(player_field, str):
                        player_name = player_field
                    elif player_field is not None:
                        player_name = str(player_field)
                    
                    if player_name and str(player_name).strip() and str(player_name).strip().lower() != 'unknown':
                        clean_name = str(player_name).strip()
                        players_from_events.add(clean_name)
            
            # Create playerIdMapping from events if none exists
            for player_name in players_from_events:
                if player_name not in merged_player_info['playerIdMapping']:
                    # Create a simple ID (könnte später durch echte Player-IDs ersetzt werden)
                    merged_player_info['playerIdMapping'][player_name] = f"PLAYER_{len(merged_player_info['playerIdMapping']) + 1}"
            
            st.info(f"🔍 DEBUG: Datei {i+1} - Aus Events extrahiert: {len(players_from_events)} Spieler: {sorted(list(players_from_events))}")
        
        # Also collect players from this specific file even if playerIdMapping already exists
        else:
            # Always collect players from each file (for substitutions)
            players_from_events = set()
            for event in events_to_process:
                if isinstance(event, dict):
                    # Try different player name fields
                    player_field = event.get('Player Name', event.get('player', event.get('Player')))
                    
                    # Handle different player field formats
                    player_name = None
                    if isinstance(player_field, dict):
                        # If player field is a dictionary, extract the actual name
                        player_name = player_field.get('playerName', player_field.get('name', str(player_field)))
                    elif isinstance(player_field, str):
                        player_name = player_field
                    elif player_field is not None:
                        player_name = str(player_field)
                    
                    if player_name and str(player_name).strip() and str(player_name).strip().lower() != 'unknown':
                        clean_name = str(player_name).strip()
                        players_from_events.add(clean_name)
            
            # Add all new players to the mapping
            new_players_count = 0
            for player_name in players_from_events:
                if player_name not in merged_player_info['playerIdMapping']:
                    merged_player_info['playerIdMapping'][player_name] = f"PLAYER_{len(merged_player_info['playerIdMapping']) + 1}"
                    new_players_count += 1
            
            st.info(f"🔍 DEBUG: Datei {i+1} - Spieler gefunden: {len(players_from_events)}, neue Spieler: {new_players_count}")
            st.info(f"🔍 DEBUG: Datei {i+1} - Alle Spieler: {sorted(list(players_from_events))}")
            if new_players_count > 0:
                new_players = [name for name in players_from_events if name in merged_player_info['playerIdMapping'] and any(merged_player_info['playerIdMapping'][name].endswith(str(x)) for x in range(len(merged_player_info['playerIdMapping']) - new_players_count + 1, len(merged_player_info['playerIdMapping']) + 1))]
                st.info(f"🔍 DEBUG: Datei {i+1} - Neue Spieler: {sorted(new_players)}")
        
        # Adjust times
        adjusted_data = adjust_time_values_json_merger(events_to_process, quarter_offsets[i], time_field)
        
        # Add player names to events if they're missing or fix dictionary player names
        for event in adjusted_data:
            if isinstance(event, dict):
                # Fix Player Name field if it's a dictionary
                if 'Player Name' in event and isinstance(event['Player Name'], dict):
                    player_dict = event['Player Name']
                    clean_name = player_dict.get('playerName', player_dict.get('name', str(player_dict)))
                    event['Player Name'] = clean_name
                
                # If event has playerId but no player name, try to resolve it
                if 'playerId' in event and 'Player Name' not in event:
                    player_id = event['playerId']
                    # Find player name by ID
                    for name, id_val in merged_player_info['playerIdMapping'].items():
                        if id_val == player_id:
                            event['Player Name'] = name
                            break
                
                # If event has no Player Name but has other player fields, try to use them
                if 'Player Name' not in event:
                    if 'player' in event:
                        player_field = event['player']
                        if isinstance(player_field, dict):
                            event['Player Name'] = player_field.get('playerName', player_field.get('name', str(player_field)))
                        else:
                            event['Player Name'] = str(player_field)
                    elif 'Player' in event:
                        player_field = event['Player']
                        if isinstance(player_field, dict):
                            event['Player Name'] = player_field.get('playerName', player_field.get('name', str(player_field)))
                        else:
                            event['Player Name'] = str(player_field)
        
        merged_data.extend(adjusted_data)
    
    # Update total players count
    merged_player_info['totalPlayers'] = len(merged_player_info['playerIdMapping'])
    
    st.info(f"🔍 DEBUG: Gesamt nach Merge - {merged_player_info['totalPlayers']} Spieler: {sorted(list(merged_player_info['playerIdMapping'].keys()))}")
    
    # Sort by time field if it exists
    if merged_data:
        # Try different possible time field names for sorting
        time_fields_to_check = [time_field, 'Zeit', 'time', 'timestamp']
        sort_field = None
        
        for field in time_fields_to_check:
            if merged_data[0] and field in merged_data[0]:
                sort_field = field
                break
        
        if sort_field:
            try:
                merged_data.sort(key=lambda x: float(x.get(sort_field, 0)))
            except (ValueError, TypeError):
                pass  # Skip sorting if time values are not numeric
    
    # Return merged data with playerInfo attached
    result = {
        'events': merged_data,
        'playerInfo': merged_player_info,
        'metadata': {
            'source': 'JSON_MERGER',
            'total_files': len(json_data_list),
            'total_events': len(merged_data),
            'merged_at': datetime.now().isoformat()
        }
    }
    
    st.info(f"🔍 DEBUG: Rückgabe-Struktur - Events: {len(result['events'])}, PlayerInfo totalPlayers: {result['playerInfo']['totalPlayers']}")
    
    return result

def create_json_download_file_merger(merged_data, filename):
    """Create downloadable JSON file - handles new structure with events and playerInfo"""
    json_string = json.dumps(merged_data, indent=2, ensure_ascii=False)
    return json_string.encode('utf-8')

# 5. CSV Merger Tab
with tabs[4]:
    st.markdown("## 📊 CSV Merger")
    st.markdown("Lade mehrere CSV-Dateien von Vierteln hoch und führe sie zu einer zusammen.")
    
    # File uploader for CSV files
    csv_uploaded_files = st.file_uploader(
        "CSV-Dateien hochladen",
        type=['csv'],
        accept_multiple_files=True,
        key="csv_merger_files"
    )
    
    if csv_uploaded_files:
        st.success(f"{len(csv_uploaded_files)} CSV-Dateien hochgeladen!")
        
        # Quarter start times configuration
        st.markdown("### ⏰ Viertel-Startzeiten konfigurieren")
        col1, col2, col3, col4 = st.columns(4)
        
        csv_q1_start = col1.number_input("1. Viertel Start (min)", value=0.0, step=0.1, key="csv_q1")
        csv_q2_start = col2.number_input("2. Viertel Start (min)", value=15.0, step=0.1, key="csv_q2")
        csv_q3_start = col3.number_input("3. Viertel Start (min)", value=30.0, step=0.1, key="csv_q3")
        csv_q4_start = col4.number_input("4. Viertel Start (min)", value=45.0, step=0.1, key="csv_q4")
        
        csv_quarter_starts = [csv_q1_start, csv_q2_start, csv_q3_start, csv_q4_start]
        
        # Time column configuration
        csv_time_column = st.text_input("Name der Zeit-Spalte", value="Zeit", key="csv_time_col")
        
        # Parse CSV files
        st.markdown("### 📊 Datenvorschau")
        csv_data_list = []
        
        for i, uploaded_file in enumerate(csv_uploaded_files):
            st.markdown(f"**Datei {i+1}: {uploaded_file.name}**")
            
            df = parse_csv_file_merger(uploaded_file)
            if df is not None:
                csv_data_list.append({
                    'filename': uploaded_file.name,
                    'dataframe': df
                })
                
                # Show preview
                st.dataframe(df.head(), use_container_width=True)
                st.markdown(f"📈 **Zeilen:** {len(df)}, **Spalten:** {len(df.columns)}")
                
                if csv_time_column in df.columns:
                    time_range = f"{df[csv_time_column].min():.1f} - {df[csv_time_column].max():.1f}"
                    st.markdown(f"⏰ **Zeitbereich:** {time_range} Minuten")
            else:
                st.error(f"Fehler beim Laden von {uploaded_file.name}")
        
        # Merge CSV files
        if csv_data_list:
            if st.button("📊 CSV-Dateien zusammenführen", key="merge_csv_btn"):
                with st.spinner("Führe CSV-Dateien zusammen..."):
                    # Calculate quarter offsets
                    csv_quarter_offsets = []
                    for i in range(len(csv_data_list)):
                        if i < len(csv_quarter_starts):
                            csv_quarter_offsets.append(csv_quarter_starts[i])
                        else:
                            csv_quarter_offsets.append(0)
                        
                    # Merge CSV quarters
                    merged_csv_df = merge_csv_quarters_merger(csv_data_list, csv_quarter_offsets, csv_time_column)
                    
                    if merged_csv_df is not None:
                        st.success("✅ CSV-Dateien erfolgreich zusammengeführt!")
                        
                        # Show merged data preview
                        st.markdown("### 📊 Zusammengeführte Daten")
                        st.dataframe(merged_csv_df.head(20), use_container_width=True)
                        st.markdown(f"📈 **Gesamtzeilen:** {len(merged_csv_df)}")
                        
                        if csv_time_column in merged_csv_df.columns:
                            time_range = f"{merged_csv_df[csv_time_column].min():.1f} - {merged_csv_df[csv_time_column].max():.1f}"
                            st.markdown(f"⏰ **Gesamtzeitbereich:** {time_range} Minuten")
                        
                        # Download button
                        output_filename = "merged_quarters.csv"
                        csv_download_data = create_csv_download_file_merger(merged_csv_df, output_filename)
                        
                        st.download_button(
                            label="📥 Zusammengeführte CSV-Datei herunterladen",
                            data=csv_download_data,
                            file_name=output_filename,
                            mime="text/csv",
                            key="download_merged_csv"
                        )
                    else:
                        st.error("❌ Fehler beim Zusammenführen der CSV-Dateien")

# 6. JSON Merger Tab
with tabs[5]:
    st.markdown("## 🔧 JSON Merger für Event-Daten")
    st.markdown("**Führe strukturierte JSON-Dateien von verschiedenen Halbzeiten zusammen**")
    
    # Helper Functions für strukturierten JSON-Merger
    def parse_structured_json_file(uploaded_file):
        """Parse structured JSON file with event data"""
        try:
            content = uploaded_file.read()
            data = json.loads(content)
            uploaded_file.seek(0)
            return data
        except json.JSONDecodeError as e:
            st.error(f"Fehler beim Parsen der JSON-Datei {uploaded_file.name}: {e}")
            return None
    
    def validate_structured_json(data):
        """Validate that JSON has the expected structure"""
        if not isinstance(data, dict):
            return False, "JSON muss ein Objekt sein"
        
        if 'metadata' not in data:
            return False, "JSON muss 'metadata' enthalten"
        
        if 'events' not in data:
            return False, "JSON muss 'events' enthalten"
        
        if not isinstance(data['events'], list):
            return False, "'events' muss eine Liste sein"
        
        return True, "JSON-Struktur ist gültig"
    
    def merge_structured_json_halves(json1, json2, time_offset_second_half=45.0):
        """
        Merge two structured JSON files representing different halves of a match
        """
        # Validate inputs
        valid1, msg1 = validate_structured_json(json1)
        valid2, msg2 = validate_structured_json(json2)
        
        if not valid1:
            st.error(f"Erste JSON-Datei: {msg1}")
            return None
        
        if not valid2:
            st.error(f"Zweite JSON-Datei: {msg2}")
            return None
        
        # Extract match IDs to verify they're the same match
        match_id_1 = json1.get('match_info', {}).get('match_id', 'unknown_1')
        match_id_2 = json2.get('match_info', {}).get('match_id', 'unknown_2')
        
        # Create merged structure
        merged_json = {
            "metadata": {
                "exportInfo": {
                    "exportId": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                    "source": "Shot-Plotter & Playermaker Data Analysis - Merged Halves",
                    "version": "2.0",
                    "format": "structured_football_data_merged",
                    "originalFiles": [
                        json1.get('metadata', {}).get('match_info', {}),
                        json2.get('metadata', {}).get('match_info', {})
                    ]
                },
                "match_info": {},
                "playerInfo": {
                    "totalPlayers": 0,
                    "playerIdMapping": {}
                }
            },
            "events": []
        }
        
        # Merge match info (prefer first half's basic info, but combine specific details)
        match_info_1 = json1.get('match_info', {})
        match_info_2 = json2.get('match_info', {})
        
        merged_match_info = match_info_1.copy()
        merged_match_info.update({
            "timestamp": datetime.now().isoformat(),
            "totalEvents": len(json1.get('events', [])) + len(json2.get('events', [])),
            "dataSource": "merged_halves"
            
        })
        
        merged_json["metadata"]["match_info"] = merged_match_info
        
        # Merge player info - combine all unique players
        player_mapping_1 = json1.get('metadata', {}).get('playerInfo', {}).get('playerIdMapping', {})
        player_mapping_2 = json2.get('metadata', {}).get('playerInfo', {}).get('playerIdMapping', {})
        
        # Combine player mappings, ensuring no ID conflicts
        merged_player_mapping = player_mapping_1.copy()
        used_ids = set(player_mapping_1.values())
        
        for player_name, player_id in player_mapping_2.items():
            if player_name not in merged_player_mapping:
                # Check if this ID is already used
                if player_id in used_ids:
                    # Generate new unique ID
                    base_num = len(merged_player_mapping) + 1
                    new_id = f"BVB_{base_num:05d}"
                    while new_id in used_ids:
                        base_num += 1
                        new_id = f"BVB_{base_num:05d}"
                    merged_player_mapping[player_name] = new_id
                    used_ids.add(new_id)
                    st.info(f"🔄 Spieler '{player_name}': ID-Konflikt gelöst ({player_id} → {new_id})")
                else:
                    merged_player_mapping[player_name] = player_id
                    used_ids.add(player_id)
            else:
                # Player already exists, check if IDs match
                existing_id = merged_player_mapping[player_name]
                if existing_id != player_id:
                    st.warning(f"⚠️ Spieler '{player_name}': Unterschiedliche IDs ({existing_id} vs {player_id}), behalte erste ID")
        
        merged_json["metadata"]["playerInfo"]["playerIdMapping"] = merged_player_mapping
        merged_json["metadata"]["playerInfo"]["totalPlayers"] = len(merged_player_mapping)
        
        # Extract youth level from match metadata for consistent IDs
        match_info = merged_json.get('metadata', {}).get('match_info', {})
        youth_level = extract_youth_level_from_match_info(match_info)
        
        # Merge events
        # First half events (no time adjustment needed)
        events_1 = json1.get('events', [])
        merged_events = []
        
        for i, event in enumerate(events_1):
            merged_event = event.copy()
            merged_event["eventIndex"] = i + 1
            
            # Generate consistent event ID using timestamp and youth level
            timestamp = merged_event.get("timestamp", 0)
            event_id = generate_consistent_event_id(timestamp, youth_level)
            merged_event["eventId"] = event_id
            
            merged_events.append(merged_event)
        
        # Second half events (adjust timestamps)
        events_2 = json2.get('events', [])
        for i, event in enumerate(events_2):
            merged_event = event.copy()
            merged_event["eventIndex"] = len(merged_events) + 1
            
            # Adjust timestamp for second half first
            adjusted_timestamp = None
            if "timestamp" in merged_event:
                try:
                    original_timestamp = float(merged_event["timestamp"])
                    adjusted_timestamp = original_timestamp + (time_offset_second_half * 60)  # Convert minutes to seconds
                    merged_event["timestamp"] = adjusted_timestamp
                    
                    # Also adjust minute/second if they exist
                    if "minute" in merged_event:
                        merged_event["minute"] = int(adjusted_timestamp // 60)
                    if "second" in merged_event:
                        merged_event["second"] = int(adjusted_timestamp % 60)
                except (ValueError, TypeError):
                    st.warning(f"⚠️ Konnte Timestamp für Event {i+1} der zweiten Halbzeit nicht anpassen")
                    adjusted_timestamp = merged_event.get("timestamp", 0)
            
            # Generate consistent event ID using adjusted timestamp and youth level
            timestamp_for_id = adjusted_timestamp if adjusted_timestamp is not None else merged_event.get("timestamp", 0)
            event_id = generate_consistent_event_id(timestamp_for_id, youth_level)
            merged_event["eventId"] = event_id
            
            merged_events.append(merged_event)
        
        merged_json["events"] = merged_events
        
        return merged_json
    
    def update_databases_with_merged_json(merged_json):
        """Update all databases with the merged JSON data - using the SAME logic as 'Events herunterladen & im Archiv speichern'"""
        try:
            # Load databases SAME as Tab 2
            player_db = load_player_database()
            match_db = load_match_database()
            event_db = load_event_database()
            
            # IMPORTANT: Convert merged JSON events back to DataFrame format (like export_data in Tab 2)
            events_list = merged_json.get('events', [])
            
            # Convert JSON events to DataFrame format that matches export_data structure
            rows_for_df = []
            for event in events_list:
                # Extract player information - use the correct structure
                player_name = event.get("player", "Unknown")  # ← FIX: Direkt aus Event (nicht aus player Sub-Objekt)
                
                # *** FIX: Extrahiere Koordinaten DIREKT aus dem Event (nicht aus coordinates Sub-Objekt)
                # Das JSON hat die Koordinaten auf der obersten Event-Ebene gespeichert!
                start_x = event.get("start_x", 0)
                start_y = event.get("start_y", 0)
                end_x = event.get("end_x", 0)
                end_y = event.get("end_y", 0)
                
                # *** FIX: Extrahiere Pass-Info DIREKT aus dem Event (nicht aus passInfo Sub-Objekt)
                # Das JSON hat die passing_network auf der obersten Event-Ebene gespeichert!
                passing_network = event.get("passing_network", {})
                
                # *** FIX: Extrahiere Additional-Data DIREKT aus dem Event (nicht aus additionalData Sub-Objekt)
                # Das JSON hat die additional_data auf der obersten Event-Ebene gespeichert!
                additional_data = event.get("additional_data", {})
                
                # Create row that matches export_data structure
                row = {
                    'Zeit': event.get("timestamp", 0),
                    'Team': event.get("team", ''),              # ← FIX: Direkt aus Event
                    'Halbzeit': event.get("half", 1),           # ← FIX: Direkt aus Event (nicht period)
                    'Player Name': player_name,
                    'X': start_x,   # ← FIX: Direkt aus Event statt coordinates dict
                    'Y': start_y,   # ← FIX: Direkt aus Event
                    'X2': end_x,    # ← FIX: Direkt aus Event
                    'Y2': end_y,    # ← FIX: Direkt aus Event
                    'Outcome': event.get("outcome", ''),        # ← FIX: Direkt aus Event
                    'Aktionstyp': event.get("action_type", 'Pass'), # ← FIX: Direkt aus Event
                    'Position': '',  # ← Position wird nicht im JSON gespeichert
                    'passed_from': passing_network.get('passed_from', ''),     # ← FIX: Aus passing_network
                    'passed_to': passing_network.get('passed_to', ''),         # ← FIX: Aus passing_network
                    'passed_from_Position': passing_network.get('passed_from_Position', ''), # ← FIX: Aus passing_network
                    'passed_to_Position': passing_network.get('passed_to_Position', '')      # ← FIX: Aus passing_network
                }
                
                # *** FIX: Füge alle additional_data-Felder hinzu (sie sind direkt im Event verfügbar)
                for key, value in additional_data.items():
                    if key not in row:
                        row[key] = value
                
                rows_for_df.append(row)
            
            # Create DataFrame equivalent to export_data
            export_data = pd.DataFrame(rows_for_df)
            
            st.info(f"🔍 DEBUG: Erstellt DataFrame mit {len(export_data)} Events aus JSON")
            
            # SAME Match-Info extraction logic as Tab 2 - use data from merged JSON to reconstruct original sources
            match_info_from_json = merged_json.get('metadata', {}).get('match_info', {})
            
            # *** FIX: Verwende die originalen Match-Metadaten aus dem JSON statt das rekonstruierte DataFrame ***
            # Das rekonstruierte DataFrame enthält keine originalen Match-Informationen (Datum, Gegner, etc.)
            
            # Reconstruct match_info with SAME structure as Tab 2 - aber mit originalen Metadaten
            match_info = {
                'type': 'match',
                'venue': match_info_from_json.get('venue', 'BVB Training Ground')
            }
            
            # FIX: Erstelle Mock-Possession-Daten mit den originalen Match-Informationen
            # So kann extract_match_info_from_possession die richtigen Daten finden
            
            # WICHTIG: Die Youth Level Info muss in der Team-Spalte stehen (nicht separat)
            # extract_match_info_from_possession sucht nach Patterns wie "U12", "U17" etc. IN der Team-Spalte
            youth_level = match_info_from_json.get('youth_level', 'SENIOR')
            team_name = match_info_from_json.get('team', 'BVB')
            
            # Kombiniere Youth Level mit Team-Name für die Team-Spalte (wie es extract_match_info_from_possession erwartet)
            if youth_level and youth_level != 'SENIOR':
                team_with_youth = f"{youth_level} {team_name}"
            else:
                team_with_youth = team_name
            
            mock_possession_df = pd.DataFrame({
                'Date': [match_info_from_json.get('date', '')],
                'Team': [team_with_youth],  # ← FIX: Youth Level IN der Team-Spalte
                'Opponent': [match_info_from_json.get('opponent', 'Unknown')]
            })
            
            # Verwende die Mock-Daten statt das rekonstruierte DataFrame
            match_info['possession_data'] = mock_possession_df
            
            # SAME Match ID creation logic as Tab 2
            match_id = create_or_get_match_id(match_info, match_db)
            
            st.info(f"🔍 DEBUG: Match-ID erstellt/gefunden: {match_id}")
            
            # *** FEHLENDE SPIELER-REGISTRIERUNG HINZUFÜGEN (GLEICHE LOGIK WIE IN TAB 2) ***
            # Sammle alle Spieler für dieses Match und registriere sie in der Player-Database
            players_in_match = set()
            player_id_mapping = {}
            
            # Sammle alle Spielernamen aus verschiedenen Spalten
            player_columns = ['Player Name', 'passed_from', 'passed_to']
            for col in player_columns:
                if col in export_data.columns:
                    unique_players = export_data[col].dropna().unique()
                    for player_name in unique_players:
                        if player_name and str(player_name).strip():
                            player_id = get_or_create_player_id(player_name, player_db)
                            if player_id:
                                player_id_mapping[player_name] = player_id
                                players_in_match.add(player_id)
            
            st.info(f"🆔 MERGED JSON: {len(players_in_match)} Spieler in Player-Database registriert: {list(player_id_mapping.keys())}")
            
            # SAME Events creation logic as Tab 2 - copied exactly from "Events herunterladen & im Archiv speichern"
            events_data = []
            for idx, row in export_data.iterrows():
                # Extract player information
                player_name = row.get('Player Name', 'Unknown')
                
                # *** FIX: Verwende Koordinaten DIREKT aus dem DataFrame (die bereits korrekt konvertiert wurden)
                # Nicht neu erstellen - das DataFrame hat bereits die richtigen Koordinaten!
                start_x = float(row.get('X', 0)) if pd.notna(row.get('X', 0)) else 0
                start_y = float(row.get('Y', 0)) if pd.notna(row.get('Y', 0)) else 0
                end_x = float(row.get('X2', 0)) if pd.notna(row.get('X2', 0)) else 0
                end_y = float(row.get('Y2', 0)) if pd.notna(row.get('Y2', 0)) else 0
                
                # Create passInfo structure
                pass_info = {}
                if 'passed_from' in row and pd.notna(row['passed_from']):
                    pass_info["passFrom"] = str(row['passed_from'])
                    if 'passed_from_Position' in row and pd.notna(row['passed_from_Position']):
                        pass_info["passFromPosition"] = str(row['passed_from_Position'])
                
                if 'passed_to' in row and pd.notna(row['passed_to']):
                    pass_info["passTo"] = str(row['passed_to'])
                    if 'passed_to_Position' in row and pd.notna(row['passed_to_Position']):
                        pass_info["passToPosition"] = str(row['passed_to_Position'])
                
                # Create passing_network for database structure
                passing_network = {
                    "passed_from": row.get('passed_from'),
                    "passed_to": row.get('passed_to'),
                    "passed_from_Position": row.get('passed_from_Position'),
                    "passed_to_Position": row.get('passed_to_Position')
                }
                
                # Clean additional_data - remove coordinates, passing info, and redundant data (SAME as Tab 2)
                additional_data = {}
                for col, value in row.items():
                    if col not in ['Zeit', 'Team', 'Halbzeit', 'Player Name', 'passed_from', 'passed_to', 
                                 'passed_from_Position', 'passed_to_Position', 'X', 'Y', 'X2', 'Y2', 
                                 'Outcome', 'Aktionstyp', 'Position'] and pd.notna(value):
                        additional_data[col] = str(value) if not isinstance(value, (int, float, bool)) else value
                
                # Determine action_type (SAME as Tab 2)
                action_type = row.get('Aktionstyp', 'Pass')
                if isinstance(action_type, str):
                    action_type = action_type.upper()
                if action_type != "PASS" and action_type != "LOSS":
                    action_type = "PASS"
                
                # Generate consistent event ID using timestamp and youth level (SAME as Tab 2)
                timestamp = row.get('Zeit', 0)
                match_entry = match_db["matches"].get(match_id, {})
                youth_level = extract_youth_level_from_match_info(match_entry)
                event_id = generate_consistent_event_id(timestamp, youth_level)
                
                # Create event with EXACT same structure as Tab 2 - FIXE KOORDINATEN VERWENDEN
                event = {
                    "event_id": event_id,
                    "timestamp": timestamp,
                    "player": player_name,
                    "action_type": action_type,
                    "start_x": start_x,  # ← FIX: Direkt aus DataFrame statt coordinates dict
                    "start_y": start_y,  # ← FIX: Direkt aus DataFrame
                    "end_x": end_x,      # ← FIX: Direkt aus DataFrame
                    "end_y": end_y,      # ← FIX: Direkt aus DataFrame
                    "outcome": row.get('Outcome', ''),
                    "team": row.get('Team', ''),
                    "half": row.get('Halbzeit', 1),
                    "additional_data": additional_data,
                    "passing_network": passing_network
                }
                
                # Bereinige Event für JSON (SAME as Tab 2)
                event = clean_data_for_json(event)
                events_data.append(event)
            
            # SAME Match-Metadaten extraction as Tab 2
            match_entry = match_db["matches"].get(match_id, {})
            match_metadata = {
                "match_id": match_id,
                "date": match_entry.get('match_date', datetime.now().strftime('%Y-%m-%d')),
                "team": match_entry.get('team', 'BVB'),
                "opponent": match_entry.get('opponent', 'Unknown'),
                "youth_level": match_entry.get('youth_level', 'SENIOR'),
                "venue": match_entry.get('venue', 'Unknown'),
                "total_events": len(events_data)
            }
            # Bereinige Match-Metadaten für JSON (SAME as Tab 2)
            match_metadata = clean_data_for_json(match_metadata)
            
            # SAME Events database update as Tab 2
            new_events_count = add_or_replace_events_in_database(match_id, events_data, match_metadata, event_db)
            
            # SAME Database saving as Tab 2
            save_event_database(event_db)
            save_match_database(match_db)
            save_player_database(player_db)
            
            st.success(f"✅ {new_events_count} Events im Event-Archiv gespeichert! (Gleiche Logik wie Tab 2)")
            st.info(f"📊 Match-ID: {match_id}")
            st.info(f"📊 Total Events verarbeitet: {len(events_data)}")
            
            # Return dictionary with expected format for UI
            return {
                "success": True,
                "players_updated": len(set(row.get('Player Name', 'Unknown') for _, row in export_data.iterrows())),
                "events_count": new_events_count,
                "match_id": match_id,
                "player_saved": True,
                "match_saved": True,
                "event_saved": True,
                "metadata": {
                    "match_info": {
                        "match_id": match_id
                    }
                }
            }
            
        except Exception as e:
            st.error(f"Fehler beim Aktualisieren der Datenbanken: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "players_updated": 0,
                "events_count": 0,
                "match_id": "unknown",
                "player_saved": False,
                "match_saved": False,
                "event_saved": False
            }
    
    # Main UI for JSON Merger
    st.subheader("1. JSON-Dateien hochladen")
    st.info("💡 Diese Funktion ist speziell für strukturierte Event-JSON-Dateien aus der Shot-Plotter-Analyse gedacht.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Erste Halbzeit JSON**")
        json_file_1 = st.file_uploader(
            "Erste Halbzeit (wird nicht zeitlich angepasst)",
            type=['json'],
            key="json_merger_file_1",
            help="JSON-Datei der ersten Halbzeit oder des ersten Zeitabschnitts"
        )
    
    with col2:
        st.markdown("**Zweite Halbzeit JSON**")
        json_file_2 = st.file_uploader(
            "Zweite Halbzeit (Zeiten werden angepasst)",
            type=['json'],
            key="json_merger_file_2",
            help="JSON-Datei der zweiten Halbzeit oder des zweiten Zeitabschnitts"
        )
    
    # Time offset configuration
    st.subheader("2. Zeitanpassung konfigurieren")
    col_time1, col_time2 = st.columns(2)
    
    with col_time1:
        time_offset = st.number_input(
            "Zeitoffset für zweite Halbzeit (Minuten)",
            min_value=0.0,
            max_value=120.0,
            value=45.0,
            step=1.0,
            help="Zeit in Minuten, die zu allen Events der zweiten Halbzeit addiert wird"
        )
    
    with col_time2:
        st.metric("Zeitoffset in Sekunden", f"{time_offset * 60:.0f} s")
    
    # Process files if both are uploaded
    if json_file_1 and json_file_2:
        st.subheader("3. JSON-Dateien verarbeiten")
        
        # Parse JSON files
        with st.spinner("JSON-Dateien werden geladen..."):
            json_data_1 = parse_structured_json_file(json_file_1)
            json_data_2 = parse_structured_json_file(json_file_2)
        
        if json_data_1 and json_data_2:
            # Show file information
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.markdown("**Erste Datei:**")
                match_info_1 = json_data_1.get('metadata', {}).get('match_info', {})
                st.write(f"**Match-ID:** {match_info_1.get('match_id', 'Unbekannt')}")
                st.write(f"**Events:** {len(json_data_1.get('events', []))}")
                st.write(f"**Spieler:** {json_data_1.get('metadata', {}).get('playerInfo', {}).get('totalPlayers', 0)}")
                st.write(f"**Dateiname:** {match_info_1.get('originalFilename', 'Unbekannt')}")
            
            with col_info2:
                st.markdown("**Zweite Datei:**")
                match_info_2 = json_data_2.get('metadata', {}).get('match_info', {})
                st.write(f"**Match-ID:** {match_info_2.get('match_id', 'Unbekannt')}")
                st.write(f"**Events:** {len(json_data_2.get('events', []))}")
                st.write(f"**Spieler:** {json_data_2.get('metadata', {}).get('playerInfo', {}).get('totalPlayers', 0)}")
                st.write(f"**Dateiname:** {match_info_2.get('originalFilename', 'Unbekannt')}")
            
            # Check if match IDs are compatible
            match_id_1 = match_info_1.get('match_id', '').split('_')[0:3]  # Extract base match info
            match_id_2 = match_info_2.get('match_id', '').split('_')[0:3]
            
            if match_id_1 == match_id_2:
                st.success("✅ Match-IDs sind kompatibel - selbes Spiel erkannt!")
            else:
                st.warning("⚠️ Unterschiedliche Match-IDs - möglicherweise verschiedene Spiele!")
            
            # Merge button
            if st.button("🔧 JSON-Dateien zusammenführen", key="merge_json_btn"):
                with st.spinner("JSON-Dateien werden zusammengeführt..."):
                    merged_json = merge_structured_json_halves(json_data_1, json_data_2, time_offset)
                
                if merged_json:
                    # Store merged data in session state immediately
                    st.session_state['merged_json_data'] = merged_json
                    st.session_state['json_merge_completed'] = True
                    st.success("✅ JSON-Dateien erfolgreich zusammengeführt!")
        
        # Check if we have merged data (either just created or from session state)
        if st.session_state.get('merged_json_data') and st.session_state.get('json_merge_completed'):
            merged_json = st.session_state['merged_json_data']
            
            # Show merge statistics
            st.subheader("4. Merge-Statistiken")
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            
            with col_stats1:
                total_events = len(merged_json.get('events', []))
                events_1 = len(json_data_1.get('events', [])) if 'json_data_1' in locals() else 0
                events_2 = len(json_data_2.get('events', [])) if 'json_data_2' in locals() else 0
                if events_1 > 0 and events_2 > 0:
                    st.metric("Gesamt Events", total_events, f"{events_1} + {events_2}")
                else:
                    st.metric("Gesamt Events", total_events)
            
            with col_stats2:
                total_players = merged_json.get('metadata', {}).get('playerInfo', {}).get('totalPlayers', 0)
                st.metric("Einzigartige Spieler", total_players)
            
            with col_stats3:
                match_id = merged_json.get('metadata', {}).get('match_info', {}).get('match_id', 'unknown')
                st.metric("Match-ID", match_id)
            
            # Player mapping details
            with st.expander("👥 Spieler-Mapping Details"):
                player_mapping = merged_json.get('metadata', {}).get('playerInfo', {}).get('playerIdMapping', {})
                if player_mapping:
                    mapping_df = pd.DataFrame([
                        {'Spieler': name, 'Player-ID': player_id}
                        for name, player_id in player_mapping.items()
                    ])
                    st.dataframe(mapping_df, use_container_width=True)
                else:
                    st.info("Keine Spieler-Mappings verfügbar")
            
            # Download section
            st.subheader("5. Download & Datenbank-Update")
            
            col_download1, col_download2 = st.columns(2)
            
            with col_download1:
                # Download merged JSON
                merged_json_str = json.dumps(merged_json, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 Zusammengeführte JSON herunterladen",
                    data=merged_json_str,
                    file_name=f"merged_{match_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    help="Lädt die zusammengeführte JSON-Datei herunter"
                )
            
            with col_download2:
                # Update databases button
                if st.button("🗄️ Datenbanken aktualisieren", key="update_db_btn"):
                    with st.spinner("Datenbanken werden aktualisiert..."):
                        result = update_databases_with_merged_json(merged_json)
                    
                    # Store update result in session state
                    st.session_state['db_update_result'] = result
                    st.session_state['db_update_completed'] = True
            
            # Show database update results if available
            if st.session_state.get('db_update_completed') and st.session_state.get('db_update_result'):
                result = st.session_state['db_update_result']
                
                if result.get("success", False):
                    st.success("✅ Datenbanken erfolgreich aktualisiert!")
                    
                    # Detailed results
                    col_result1, col_result2, col_result3 = st.columns(3)
                    with col_result1:
                        st.metric("Neue/Aktualisierte Spieler", result.get("players_updated", 0))
                    with col_result2:
                        st.metric("Events gespeichert", result.get("events_count", 0))
                    with col_result3:
                        st.metric("Match-ID", result.get("metadata", {}).get("match_info", {}).get("match_id", "unknown"))
                    
                    # Save status details
                    with st.expander("🔍 Speicher-Details anzeigen"):
                        col_save1, col_save2, col_save3 = st.columns(3)
                        
                        with col_save1:
                            if result.get("player_saved", False):
                                st.success("✅ Spieler-DB gespeichert")
                            else:
                                st.error("❌ Spieler-DB Fehler")
                        
                        with col_save2:
                            if result.get("match_saved", False):
                                st.success("✅ Match-DB gespeichert")
                            else:
                                st.error("❌ Match-DB Fehler")
                        
                        with col_save3:
                            if result.get("event_saved", False):
                                st.success("✅ Event-DB gespeichert")
                            else:
                                st.error("❌ Event-DB Fehler")
                    
                    # Button to clear the session and start over
                    if st.button("🔄 Neue Analyse starten", key="clear_merge_session"):
                        for key in ['merged_json_data', 'json_merge_completed', 'db_update_result', 'db_update_completed']:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
                    
                else:
                    st.error(f"❌ Fehler beim Aktualisieren der Datenbanken: {result.get('error', 'Unbekannter Fehler')}")
                    
                    # Show debug information if available
                    if "error" in result:
                        with st.expander("🔍 Fehler-Details anzeigen"):
                            st.code(result["error"])
                    
                    # Button to retry
                    if st.button("🔄 Erneut versuchen", key="retry_db_update"):
                        if 'db_update_result' in st.session_state:
                            del st.session_state['db_update_result']
                        if 'db_update_completed' in st.session_state:
                            del st.session_state['db_update_completed']
                        st.rerun()
    
    else:
        st.info("📂 Bitte laden Sie beide JSON-Dateien hoch, um mit dem Merge-Prozess zu beginnen.")
    
    # Database status section
    st.subheader("📊 Datenbank-Status")
    try:
        col_db1, col_db2, col_db3 = st.columns(3)
        
        with col_db1:
            player_db = load_player_database()
            st.metric("Registrierte Spieler", len(player_db.get("players", {})))
        
        with col_db2:
            match_db = load_match_database()
            st.metric("Gespeicherte Matches", len(match_db.get("matches", {})))
        
        with col_db3:
            event_db = load_event_database()
            st.metric("Gesamt Events", event_db.get("metadata", {}).get("total_events", 0))
        
        # Show recent matches
        with st.expander("🏆 Letzte Matches"):
            match_db = load_match_database()
            if match_db.get("matches"):
                recent_matches = []
                for match_id, match_info in match_db["matches"].items():
                    # Sichere Datum-Extraktion mit Fallback
                    match_date = match_info.get("match_date")
                    if match_date is None or match_date == "":
                        match_date = match_info.get("date", "Unbekannt")
                    if match_date is None:
                        match_date = "Unbekannt"
                    
                    recent_matches.append({
                        "Match-ID": match_id,
                        "Datum": match_date,
                        "Team": match_info.get("team", "Unbekannt"),
                        "Gegner": match_info.get("opponent", "Unbekannt"),
                        "Jugend": match_info.get("youth_level", "Unbekannt"),
                        "Events": match_info.get("total_events", 0)
                    })
                
                # Sichere Sortierung nach Datum (behandelt None und "Unbekannt")
                def safe_date_sort(match):
                    date_value = match["Datum"]
                    if date_value is None or date_value == "Unbekannt":
                        return "0000-00-00"  # Sehr frühes Datum für unbekannte Daten
                    return str(date_value)
                
                recent_matches.sort(key=safe_date_sort, reverse=True)
                matches_df = pd.DataFrame(recent_matches[:10])  # Show last 10 matches
                st.dataframe(matches_df, use_container_width=True)
            else:
                st.info("Noch keine Matches in der Datenbank gespeichert")
                
    except Exception as e:
        st.warning(f"Fehler beim Laden des Datenbank-Status: {str(e)}")
    
    # Help section
    with st.expander("ℹ️ Hilfe zum JSON Merger"):
        st.markdown("""
        **Verwendung des JSON Mergers:**
        
        1. **JSON-Dateien hochladen**: Laden Sie zwei strukturierte JSON-Dateien hoch, die verschiedene Halbzeiten/Zeitabschnitte desselben Spiels repräsentieren
        
        2. **Zeitanpassung**: Konfigurieren Sie den Zeitoffset für die zweite Halbzeit (Standard: 45 Minuten)
        
        3. **Zusammenführung**: Die App führt automatisch zusammen:
           - **Events**: Alle Events beider Dateien mit angepassten Zeitstempeln
           - **Spieler-IDs**: Vermeidet Duplikate und löst ID-Konflikte
           - **Metadaten**: Kombiniert Match-Informationen
        
        4. **Download**: Erhalten Sie die zusammengeführte JSON-Datei
        
        5. **Datenbank-Update**: Aktualisieren Sie die lokalen Datenbanken mit den neuen Daten
        
        **Erwartete JSON-Struktur:**
        ```json
        {
          "metadata": {
            "match_info": { ... },
            "playerInfo": {
              "playerIdMapping": { ... }
            }
          },
          "events": [ ... ]
        }
        ```
        
        **Vorteile:**
        - ✅ Automatische Zeitanpassung
        - ✅ Intelligente Spieler-ID-Verwaltung
        - ✅ Datenbank-Integration
        - ✅ Validierung der JSON-Struktur
        - ✅ Conflict-Resolution für doppelte Spieler
        """)




