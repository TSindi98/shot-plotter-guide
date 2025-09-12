#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Saison-Hinzufüger für event_database.json
==========================================

Dieses Skript fügt ein 'season' Feld zu allen Match-Metadaten in der event_database.json hinzu.
Die Saison-Regel ist:
- Saison 23_24: 2023-07-01 bis 2024-06-30
- Saison 24_25: 2024-07-01 bis 2025-06-30
- usw.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

def determine_season_from_date(date_str):
    """
    #Bestimmt die Saison basierend auf dem Datum.
    
    #Args:
    #    date_str (str): Datum im Format 'YYYY-MM-DD'
        
    #Returns:
    #    str: Saison im Format 'YY_YY' (z.B. '23_24') oder None bei Fehler
    """
    if not date_str:
        return None
    
    try:
        match_date = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Bestimme Saison basierend auf dem Jahr
        # Wenn das Datum zwischen Juli und Dezember ist, ist es die Saison des aktuellen Jahres
        # Wenn das Datum zwischen Januar und Juni ist, ist es die Saison des vorherigen Jahres
        if match_date.month >= 7:  # Juli bis Dezember
            season_year = match_date.year
        else:  # Januar bis Juni
            season_year = match_date.year - 1
            
        # Erstelle Saison-String im Format "23_24"
        season_start = str(season_year)[-2:]  # Letzte 2 Ziffern des Jahres
        season_end = str(season_year + 1)[-2:]  # Letzte 2 Ziffern des nächsten Jahres
        
        return f"{season_start}_{season_end}"
        
    except Exception as e:
        print(f"⚠️  Fehler beim Parsen des Datums '{date_str}': {e}")
        return None

def add_season_to_database(json_file_path, backup=True):
    """
    #Fügt Saison-Informationen zu allen Matches in der event_database.json hinzu.
    
    #Args:
    #    json_file_path (str): Pfad zur event_database.json
    #    backup (bool): Ob eine Backup-Datei erstellt werden soll
        
    #Returns:
    #    bool: True wenn erfolgreich, False bei Fehler
    """
    
    # Prüfe ob Datei existiert
    if not Path(json_file_path).exists():
        print(f"❌ Fehler: Datei '{json_file_path}' nicht gefunden!")
        return False
    
    try:
        # Lade JSON-Datei
        print(f"📂 Lade {json_file_path}...")
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Prüfe ob die erwartete Struktur vorhanden ist
        if 'events_by_match' not in data:
            print("❌ Fehler: 'events_by_match' nicht in der JSON-Datei gefunden!")
            return False
        
        # Erstelle Backup falls gewünscht
        if backup:
            backup_path = json_file_path.replace('.json', '_backup.json')
            print(f"💾 Erstelle Backup: {backup_path}")
            with open(backup_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
        
        # Zähler für Statistiken
        total_matches = 0
        updated_matches = 0
        matches_without_date = 0
        matches_with_season = 0
        
        print(f"\n🔄 Verarbeite {len(data['events_by_match'])} Matches...")
        
        # Durch alle Matches iterieren
        for match_id, match_data in data['events_by_match'].items():
            total_matches += 1
            
            # Prüfe ob match_info vorhanden ist
            if 'match_info' not in match_data:
                print(f"⚠️  Match '{match_id}' hat keine match_info - überspringe...")
                continue
            
            match_info = match_data['match_info']
            
            # Prüfe ob bereits eine Saison vorhanden ist
            if 'season' in match_info:
                matches_with_season += 1
                print(f"   ✅ {match_id}: Saison bereits vorhanden ({match_info['season']})")
                continue
            
            # Hole das Datum aus den Match-Informationen
            match_date = match_info.get('date')
            
            if not match_date:
                matches_without_date += 1
                print(f"   ⚠️  {match_id}: Kein Datum gefunden")
                continue
            
            # Bestimme Saison basierend auf dem Datum
            season = determine_season_from_date(match_date)
            
            if season:
                # Füge Saison zu den Match-Informationen hinzu
                match_info['season'] = season
                updated_matches += 1
                print(f"   ➕ {match_id}: Saison '{season}' hinzugefügt (Datum: {match_date})")
            else:
                print(f"   ❌ {match_id}: Konnte Saison nicht bestimmen (Datum: {match_date})")
        
        # Speichere aktualisierte JSON-Datei
        print(f"\n💾 Speichere aktualisierte {json_file_path}...")
        with open(json_file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        
        # Zeige Statistiken
        print(f"\n📊 STATISTIKEN:")
        print(f"   • Gesamte Matches: {total_matches}")
        print(f"   • Matches mit Saison hinzugefügt: {updated_matches}")
        print(f"   • Matches mit bereits vorhandener Saison: {matches_with_season}")
        print(f"   • Matches ohne Datum: {matches_without_date}")
        print(f"   • Erfolgreich aktualisiert: {updated_matches}/{total_matches}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Fehler beim Lesen der JSON-Datei: {e}")
        return False
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        return False

def fix_unsuccessful_pass_targets(json_file_path, backup=True):
    """
    Korrigiert passed_to Werte für nicht erfolgreiche Pässe.
    
    Setzt passed_to auf None für alle Events mit outcome 'Nicht Erfolgreich',
    da ein nicht erfolgreicher Pass logischerweise keinen Empfänger haben kann.
    
    Args:
        json_file_path (str): Pfad zur JSON-Datei
        backup (bool): Ob ein Backup erstellt werden soll
        
    Returns:
        bool: True bei Erfolg, False bei Fehler
    """
    print(f"🔧 Starte Korrektur der passed_to Werte für nicht erfolgreiche Pässe...")
    
    try:
        # Lade JSON-Daten
        file_path = Path(json_file_path)
        if not file_path.exists():
            print(f"❌ Datei nicht gefunden: {json_file_path}")
            return False
        
        print(f"📖 Lade Daten aus {json_file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Erstelle Backup falls gewünscht
        if backup:
            backup_path = file_path.parent / f"{file_path.stem}_backup{file_path.suffix}"
            print(f"💾 Erstelle Backup: {backup_path}")
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Statistiken
        total_events = 0
        fixed_events = 0
        problematic_events = 0
        
        # Gehe durch alle Matches
        if 'events_by_match' not in data:
            print("❌ Keine 'events_by_match' Struktur gefunden!")
            return False
        
        print(f"🔍 Analysiere {len(data['events_by_match'])} Matches...")
        
        for match_id, match_data in data['events_by_match'].items():
            if 'events' not in match_data:
                continue
                
            for event in match_data['events']:
                # Nur PASS Events bearbeiten
                if event.get('action_type') != 'PASS':
                    continue
                    
                total_events += 1
                outcome = event.get('outcome', '')
                passed_to = event.get('passing_network', {}).get('passed_to')
                
                # Prüfe auf problematische Kombination
                if outcome == 'Nicht Erfolgreich' and passed_to is not None and passed_to != 'None' and passed_to != '':
                    # Markiere als problematisch vor der Korrektur
                    problematic_events += 1
                    
                    # Korrigiere den Wert
                    if 'passing_network' not in event:
                        event['passing_network'] = {}
                    event['passing_network']['passed_to'] = None
                    fixed_events += 1
                    
                    if fixed_events <= 5:  # Zeige erste 5 Korrekturen als Beispiel
                        player = event.get('player', 'Unbekannt')
                        timestamp = event.get('timestamp', 0)
                        print(f"  🔧 Korrektur: {player} @ {timestamp:.2f}s - passed_to: '{passed_to}' → None")
        
        print(f"\n📊 KORREKTUR-STATISTIKEN:")
        print(f"📈 Gesamt Pass-Events analysiert: {total_events}")
        print(f"⚠️  Problematische Events gefunden: {problematic_events}")
        print(f"✅ Events korrigiert: {fixed_events}")
        
        if fixed_events == 0:
            print(f"🎉 Keine Korrekturen nötig - alle Daten sind bereits konsistent!")
            return True
        
        # Speichere korrigierte Daten
        print(f"💾 Speichere korrigierte Daten...")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Korrektur erfolgreich abgeschlossen!")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON-Parsing-Fehler: {e}")
        return False
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        return False

def remove_passes_with_invalid_releasing_leg(json_file_path, backup=True):
    """
    Entfernt Pässe mit NaN/None/ungültigen Werten in 'Releasing Leg'
    
    Args:
        json_file_path (str): Pfad zur JSON-Datei
        backup (bool): Erstelle Backup vor Änderung
    
    Returns:
        bool: True wenn erfolgreich, False bei Fehler
    """
    try:
        # Backup erstellen falls gewünscht
        if backup:
            backup_path = json_file_path.replace('.json', '_backup_before_releasing_leg_cleanup.json')
            print(f"🔄 Erstelle Backup: {backup_path}")
            import shutil
            shutil.copy2(json_file_path, backup_path)
        
        # JSON-Datei laden
        print(f"📂 Lade JSON-Datei: {json_file_path}")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_events = 0
        removed_events = 0
        
        print(f"🔍 Analysiere Events auf ungültige 'Releasing Leg' Werte...")
        
        for match_id, match_data in data['events_by_match'].items():
            if 'events' not in match_data:
                continue
            
            # Events filtern - behalte nur die mit gültigen Releasing Leg Werten
            valid_events = []
            
            for event in match_data['events']:
                # Nur PASS Events prüfen
                if event.get('action_type') != 'PASS':
                    valid_events.append(event)  # Behalte Nicht-Pass Events
                    continue
                    
                total_events += 1
                
                # Prüfe Releasing Leg in additional_data
                additional_data = event.get('additional_data', {})
                releasing_leg = additional_data.get('Releasing Leg')
                
                # Prüfe auf ungültige Werte
                if releasing_leg is None or releasing_leg == '' or str(releasing_leg).lower() in ['nan', 'none', 'null']:
                    removed_events += 1
                    if removed_events <= 5:  # Zeige erste 5 als Beispiel
                        player = event.get('player', 'Unbekannt')
                        timestamp = event.get('timestamp', 0)
                        print(f"  🗑️  Entferne: {player} @ {timestamp:.2f}s - Releasing Leg: '{releasing_leg}'")
                else:
                    valid_events.append(event)  # Behalte Event mit gültigem Releasing Leg
            
            # Ersetze Events durch gefilterte Liste
            match_data['events'] = valid_events
        
        print(f"\n📊 BEREINIGUNG-STATISTIKEN:")
        print(f"📈 Gesamt Pass-Events analysiert: {total_events}")
        print(f"🗑️  Events mit ungültigem Releasing Leg entfernt: {removed_events}")
        print(f"✅ Events behalten: {total_events - removed_events}")
        
        if removed_events == 0:
            print(f"🎉 Keine Bereinigung nötig - alle Releasing Leg Werte sind gültig!")
            return True
        
        # Speichere bereinigte Daten
        print(f"💾 Speichere bereinigte Daten...")
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Bereinigung erfolgreich abgeschlossen!")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON-Parsing-Fehler: {e}")
        return False
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        return False

def main():
    """
    Hauptfunktion
    """
    print("🏈 SAISON-HINZUFÜGER FÜR EVENT DATABASE")
    print("=" * 50)
    
    # Standard-Pfad zur event_database.json
    json_file_path = 'event_database.json'
    
    # Prüfe ob Pfad als Kommandozeilenargument übergeben wurde
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
    
    print(f"🎯 Ziel-Datei: {json_file_path}")
    
    # Führe Saison-Hinzufügung durch
    success = add_season_to_database(json_file_path, backup=True)
    
    if success:
        print(f"\n✅ Erfolgreich abgeschlossen!")
        print(f"📁 Die Datei '{json_file_path}' wurde aktualisiert.")
        print(f"💾 Ein Backup wurde erstellt: '{json_file_path.replace('.json', '_backup.json')}'")
    else:
        print(f"\n❌ Fehler beim Verarbeiten der Datei!")
        sys.exit(1)

def add_pass_sequence_ids(json_file_path, backup=True):
    """
    Fügt Passfolgen-IDs zu Pass-Events hinzu.
    
    VERWENDET DIE AKTUELLE LOGIK AUS pass_sequences.py!
    
    Erstellt für jede Passfolge eine eindeutige ID im Format: DDMMYYYY###
    Beispiel: 02042025003 = 02.04.2025, Sequenz 3
    
    Fügt außerdem die Position des Passes innerhalb der Sequenz hinzu.
    
    Args:
        json_file_path (str): Pfad zur JSON-Datei
        backup (bool): Ob ein Backup erstellt werden soll
    """
    try:
        # Importiere die aktuellste Sequenz-Logik aus pass_sequences.py
        import sys
        import os
        # Korrekte Pfad-Berechnung: von Datenmerging/ zu shot-plotter/Passanalyse/
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Datenmerging/
        shot_plotter_dir = os.path.dirname(current_dir)  # shot-plotter/
        passanalyse_path = os.path.join(shot_plotter_dir, 'Passanalyse')
        sys.path.insert(0, passanalyse_path)
        print(f"🔍 Versuche Import aus: {passanalyse_path}")
        
        try:
            from modules.pass_sequences import (
                identify_pass_sequences,
                calculate_time_difference,
                check_network_consistency
            )
            print("✅ Sequenz-Logik aus pass_sequences.py importiert")
        except ImportError as e:
            print(f"❌ Konnte pass_sequences.py nicht importieren: {e}")
            print("💡 Verwende vereinfachte lokale Logik als Fallback")
            return add_pass_sequence_ids_fallback(json_file_path, backup)
        
        # JSON-Datei laden
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        total_sequences = 0
        total_passes = 0
        
        # Durch alle Matches iterieren
        for match_id, match_data in data.get("events_by_match", {}).items():
            events = match_data.get("events", [])
            match_info = match_data.get("match_info", {})
            
            # Datum aus match_info extrahieren
            date_str = match_info.get('date', '2025-01-01')
            try:
                # Format: YYYY-MM-DD zu DDMMYYYY
                date_parts = date_str.split('-')
                if len(date_parts) == 3:
                    year, month, day = date_parts
                    date_id = f"{day.zfill(2)}{month.zfill(2)}{year}"
                else:
                    date_id = "01012025"  # Fallback
            except:
                date_id = "01012025"  # Fallback
            
            print(f"🔍 Verarbeite Match {match_id} - Datum: {date_str} -> {date_id}")
            
            # Nur Pass-Events filtern und nach Halbzeit + Zeit sortieren
            pass_events = [event for event in events if event.get("action_type") == "PASS"]
            # WICHTIG: Sortiere zuerst nach Halbzeit, dann nach Zeit
            # Das verhindert Zeitstempel-Kollisionen zwischen Halbzeiten
            pass_events.sort(key=lambda x: (x.get('half', 1), x.get('timestamp', 0)))
            
            if len(pass_events) == 0:
                print(f"   ⚠️  Keine Pass-Events in Match {match_id}")
                continue
            
            # Debug: Zeige Halbzeit-Verteilung
            half1_count = len([e for e in pass_events if e.get('half', 1) == 1])
            half2_count = len([e for e in pass_events if e.get('half', 1) == 2])
            print(f"   📋 Nach Halbzeit sortiert: 1. HZ: {half1_count}, 2. HZ: {half2_count}")
            
            # Optional: Timestamp-Anpassung für U15-Spiele (2. Halbzeit)
            if 'U15' in match_id and half2_count > 0:
                # Finde das erste Event der 2. Halbzeit
                half2_events = [e for e in pass_events if e.get('half', 1) == 2]
                if half2_events:
                    first_half2_timestamp = min(e.get('timestamp', 0) for e in half2_events)
                    
                    # Wenn erste Event der 2. HZ < 2000s, addiere 35min (2100s)
                    if first_half2_timestamp < 2000:
                        adjustment = 35 * 60  # 2100 seconds
                        print(f"   🕐 U15 Timestamp-Anpassung: +{adjustment}s für 2. Halbzeit")
                        
                        for event in half2_events:
                            old_timestamp = event.get('timestamp', 0)
                            event['timestamp'] = old_timestamp + adjustment
                        
                        # Nach Anpassung neu sortieren
                        pass_events.sort(key=lambda x: (x.get('half', 1), x.get('timestamp', 0)))
                        print(f"   ✅ Timestamps angepasst: 2. HZ jetzt {first_half2_timestamp:.1f}s → {first_half2_timestamp + adjustment:.1f}s")
            
            # Alle Pass-Events mit leeren pass_sequence initialisieren
            for event in pass_events:
                event['pass_sequence'] = {
                    "sequence_id": "0",
                    "position_in_sequence": None,
                    "sequence_length": None
                }
            
            # Konvertiere zu DataFrame für pass_sequences.py Kompatibilität
            import pandas as pd
            
            # Erstelle DataFrame mit den nötigen Spalten für pass_sequences.py
            df_events = []
            for event in pass_events:
                df_event = {
                    'timestamp': event.get('timestamp', 0),
                    'player': event.get('player', ''),
                    'outcome': event.get('outcome', ''),
                    'passed_to': event.get('passing_network', {}).get('passed_to', ''),
                    'passed_from': event.get('passing_network', {}).get('passed_from', ''),
                    'Time to Release': event.get('additional_data', {}).get('Time to Release', 0),
                    'X': event.get('start_x', 0),  # Einheitliche Koordinaten-Namen
                    'Y': event.get('start_y', 0),
                    'X2': event.get('end_x', 0),
                    'Y2': event.get('end_y', 0),
                    'match_id': match_id,
                    '_original_event': event  # Referenz auf das Original-Event
                }
                df_events.append(df_event)
            
            df = pd.DataFrame(df_events)
            
            # Verwende die echte pass_sequences.py Logik!
            print(f"   🧠 Verwende pass_sequences.py Logik für {len(df)} Pässe...")
            sequences_df = identify_pass_sequences(df)
            
            if len(sequences_df) == 0:
                print(f"   ⚠️  Keine Sequenzen in Match {match_id} identifiziert")
                continue
            
            # Wende die gefundenen Sequenzen auf die Original-Events an
            for _, sequence in sequences_df.iterrows():
                pass_indices = sequence['pass_indices']
                sequence_number = sequence['sequence_id']
                sequence_id = f"{date_id}{sequence_number:03d}"  # z.B. 15032025005
                
                # Weise allen Pässen einer Sequenz die gleiche ID zu
                for pos, pass_idx in enumerate(pass_indices):
                    original_event = df.iloc[pass_idx]['_original_event']
                    original_event['pass_sequence']['sequence_id'] = sequence_id
                    original_event['pass_sequence']['position_in_sequence'] = pos + 1  # 1-basiert
                    original_event['pass_sequence']['sequence_length'] = len(pass_indices)
                
                total_sequences += 1
                total_passes += len(pass_indices)
            
            print(f"   ✅ {len(sequences_df)} Sequenzen mit {sum(len(seq['pass_indices']) for _, seq in sequences_df.iterrows())} Pässen identifiziert")
        
        # Korrigierte Datei speichern
        with open(json_file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Passfolgen-IDs mit pass_sequences.py Logik hinzugefügt!")
        print(f"📊 {total_sequences} Passfolgen identifiziert")
        print(f"🏃 {total_passes} Pässe in Sequenzen eingeordnet")
        print(f"💾 Datei gespeichert: {json_file_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fehler beim Hinzufügen der Passfolgen-IDs: {e}")
        import traceback
        traceback.print_exc()
        return False

def add_pass_sequence_ids_fallback(json_file_path, backup=True):
    """
    Fallback-Funktion mit der ursprünglichen vereinfachten Logik
    """
    try:
        
        # JSON-Datei laden
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        total_sequences = 0
        total_passes = 0
        
        # Durch alle Matches iterieren
        for match_id, match_data in data.get("events_by_match", {}).items():
            events = match_data.get("events", [])
            match_info = match_data.get("match_info", {})
            
            # Datum aus match_info extrahieren
            date_str = match_info.get('date', '2025-01-01')
            try:
                # Format: YYYY-MM-DD zu DDMMYYYY
                date_parts = date_str.split('-')
                if len(date_parts) == 3:
                    year, month, day = date_parts
                    date_id = f"{day.zfill(2)}{month.zfill(2)}{year}"
                else:
                    date_id = "01012025"  # Fallback
            except:
                date_id = "01012025"  # Fallback
            
            print(f"🔍 Verarbeite Match {match_id} - Datum: {date_str} -> {date_id} (Fallback-Logik)")
            
            # Nur Pass-Events filtern und nach Halbzeit + Zeit sortieren (Fallback)
            pass_events = [event for event in events if event.get("action_type") == "PASS"]
            # WICHTIG: Sortiere zuerst nach Halbzeit, dann nach Zeit
            pass_events.sort(key=lambda x: (x.get('half', 1), x.get('timestamp', 0)))
            
            # Alle Pass-Events mit leeren pass_sequence initialisieren
            for event in pass_events:
                if 'pass_sequence' not in event:
                    event['pass_sequence'] = {
                        "sequence_id": "0",
                        "position_in_sequence": None,
                        "sequence_length": None
                    }
            
            # Passfolgen identifizieren (vereinfachte Logik)
            sequence_id = 1
            current_sequence = []
            
            for i, event in enumerate(pass_events):
                # Beginne neue Sequenz oder füge zu bestehender hinzu
                if len(current_sequence) == 0:
                    current_sequence = [i]
                else:
                    # Prüfe ob aktueller Pass zur bestehenden Sequenz gehört
                    last_idx = current_sequence[-1]
                    last_event = pass_events[last_idx]
                    
                    # Zeitliche Kontinuität (max. 10 Sekunden)
                    time_diff = calculate_time_difference_simple(last_event, event)
                    
                    # Passing Network Konsistenz
                    network_consistent = check_network_consistency_simple(last_event, event)
                    
                    # Letzter Pass war erfolgreich
                    last_pass_successful = last_event.get('outcome', '') == 'Erfolgreich'
                    
                    if (time_diff <= 10.0 and network_consistent and last_pass_successful):
                        current_sequence.append(i)
                    else:
                        # Beende aktuelle Sequenz und starte neue
                        if len(current_sequence) >= 2:
                            assign_sequence_ids(current_sequence, pass_events, date_id, sequence_id)
                            total_sequences += 1
                            total_passes += len(current_sequence)
                            sequence_id += 1
                        current_sequence = [i]
                
                # Prüfe ob aktueller Pass eine Sequenz beendet
                if (event.get('outcome', '') == 'Nicht Erfolgreich' or 
                    not event.get('passing_network', {}).get('passed_to')):
                    if len(current_sequence) >= 2:
                        assign_sequence_ids(current_sequence, pass_events, date_id, sequence_id)
                        total_sequences += 1
                        total_passes += len(current_sequence)
                        sequence_id += 1
                    current_sequence = []
            
            # Behandle letzte Sequenz
            if len(current_sequence) >= 2:
                assign_sequence_ids(current_sequence, pass_events, date_id, sequence_id)
                total_sequences += 1
                total_passes += len(current_sequence)
        
        # Korrigierte Datei speichern
        with open(json_file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Passfolgen-IDs hinzugefügt! (Fallback-Logik)")
        print(f"📊 {total_sequences} Passfolgen identifiziert")
        print(f"🏃 {total_passes} Pässe in Sequenzen eingeordnet")
        print(f"💾 Datei gespeichert: {json_file_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fehler beim Hinzufügen der Passfolgen-IDs: {e}")
        return False

def assign_sequence_ids(sequence_indices, pass_events, date_id, sequence_number):
    """Weist allen Pässen einer Sequenz die gleiche ID zu"""
    sequence_id = f"{date_id}{sequence_number:03d}"  # z.B. 02042025003
    
    for pos, idx in enumerate(sequence_indices):
        event = pass_events[idx]
        # Überschreibe pass_sequence Informationen (wurde bereits initialisiert)
        event['pass_sequence']['sequence_id'] = sequence_id
        event['pass_sequence']['position_in_sequence'] = pos + 1  # 1-basiert
        event['pass_sequence']['sequence_length'] = len(sequence_indices)

def calculate_time_difference_simple(last_event, current_event):
    """Vereinfachte Zeitdifferenz-Berechnung für die ID-Generierung"""
    last_timestamp = last_event.get('timestamp', 0)
    current_timestamp = current_event.get('timestamp', 0)
    current_time_to_release = current_event.get('additional_data', {}).get('Time to Release', 0)
    
    # Ballabgabe des letzten Passes
    last_ball_release = last_timestamp
    
    # Ballannahme des aktuellen Passes  
    current_ball_reception = current_timestamp - current_time_to_release
    
    return current_ball_reception - last_ball_release

def check_network_consistency_simple(last_event, current_event):
    """Vereinfachte Netzwerk-Konsistenz-Prüfung für die ID-Generierung"""
    last_passing = last_event.get('passing_network', {})
    current_passing = current_event.get('passing_network', {})
    
    # Bedingung 1: passed_to von letztem Pass = player von aktuellem Pass
    condition1 = last_passing.get('passed_to', '') == current_event.get('player', '')
    
    # Bedingung 2: player von letztem Pass = passed_from von aktuellem Pass  
    condition2 = last_event.get('player', '') == current_passing.get('passed_from', '')
    
    return condition1 and condition2

if __name__ == "__main__":
    main() 


