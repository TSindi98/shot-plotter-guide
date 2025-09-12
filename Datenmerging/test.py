#!/usr/bin/env python3
"""
Test Script für Passsequenz-Analyse
Liest event_database.json ein und gibt Start- und Endkoordinaten jeder Sequenz aus
"""

import json
import sys
from collections import defaultdict

def analyze_pass_sequences(json_file_path='event_database.json'):
    """
    Analysiert Passsequenzen und gibt Start- und Endkoordinaten aus
    
    Args:
        json_file_path (str): Pfad zur event_database.json
    """
    try:
        # JSON-Datei laden
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        print("🔍 PASS-SEQUENZ ANALYSE")
        print("=" * 50)
        
        # Sammle alle Sequenzen
        sequences = defaultdict(list)
        total_sequences = 0
        total_passes = 0
        
        # Durch alle Matches iterieren
        for match_id, match_data in data.get("events_by_match", {}).items():
            events = match_data.get("events", [])
            match_info = match_data.get("match_info", {})
            
            print(f"\n📊 Match: {match_id}")
            print(f"   Datum: {match_info.get('date', 'N/A')}")
            
            # Nur Pass-Events mit Sequenz-Informationen
            pass_events = []
            for event in events:
                if (event.get("action_type") == "PASS" and 
                    event.get("pass_sequence", {}).get("sequence_id") not in [None, "0", ""]):
                    pass_events.append(event)
            
            if not pass_events:
                print("   ⚠️  Keine Passsequenzen gefunden")
                continue
            
            # Sortiere nach Halbzeit und Zeit
            pass_events.sort(key=lambda x: (x.get('half', 1), x.get('timestamp', 0)))
            
            # Gruppiere nach sequence_id
            match_sequences = defaultdict(list)
            for event in pass_events:
                seq_id = event['pass_sequence']['sequence_id']
                match_sequences[seq_id].append(event)
            
            print(f"   📋 {len(match_sequences)} Sequenzen gefunden")
            
            # Analysiere jede Sequenz
            for seq_id, seq_events in match_sequences.items():
                # Sortiere Events innerhalb der Sequenz nach Position
                seq_events.sort(key=lambda x: x['pass_sequence']['position_in_sequence'])
                
                if len(seq_events) < 2:
                    continue
                
                # Erster und letzter Pass
                first_pass = seq_events[0]
                last_pass = seq_events[-1]
                
                # Koordinaten extrahieren
                start_x = first_pass.get('start_x', 0)
                start_y = first_pass.get('start_y', 0)
                end_x = last_pass.get('end_x', 0)
                end_y = last_pass.get('end_y', 0)
                
                # Sequenz-Informationen
                seq_length = len(seq_events)
                duration = last_pass.get('timestamp', 0) - first_pass.get('timestamp', 0)
                
                # Spieler der Sequenz
                players = [event.get('player', 'Unknown') for event in seq_events]
                unique_players = list(dict.fromkeys(players))  # Behält Reihenfolge bei, entfernt Duplikate
                
                print(f"\n   🎯 Sequenz {seq_id}:")
                print(f"      Länge: {seq_length} Pässe")
                print(f"      Dauer: {duration:.1f}s")
                print(f"      Spieler: {', '.join(unique_players[:3])}{'...' if len(unique_players) > 3 else ''}")
                print(f"      Start: ({start_x:.1f}, {start_y:.1f}) - {first_pass.get('player', 'Unknown')}")
                print(f"      Ende:  ({end_x:.1f}, {end_y:.1f}) - {last_pass.get('player', 'Unknown')}")
                
                # Für Gesamtstatistik
                total_sequences += 1
                total_passes += len(seq_events)
        
        # Gesamtstatistik
        print(f"\n📈 GESAMTSTATISTIK:")
        print(f"   Sequenzen: {total_sequences}")
        print(f"   Pässe in Sequenzen: {total_passes}")
        
        return True
        
    except FileNotFoundError:
        print(f"❌ Datei nicht gefunden: {json_file_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ JSON-Fehler: {e}")
        return False
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        return False

def analyze_specific_sequences(json_file_path='event_database.json', target_sequences=None):
    """
    Analysiert spezifische Sequenzen detailliert
    
    Args:
        json_file_path (str): Pfad zur event_database.json
        target_sequences (list): Liste von Sequenz-IDs zum Analysieren
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        print("🎯 DETAILLIERTE SEQUENZ-ANALYSE")
        print("=" * 50)
        
        found_sequences = 0
        
        for match_id, match_data in data.get("events_by_match", {}).items():
            events = match_data.get("events", [])
            
            for event in events:
                if (event.get("action_type") == "PASS" and 
                    event.get("pass_sequence", {}).get("sequence_id") in target_sequences):
                    
                    seq_id = event['pass_sequence']['sequence_id']
                    if seq_id not in target_sequences:
                        continue
                    
                    found_sequences += 1
                    print(f"\n🔍 Sequenz {seq_id} gefunden in Match {match_id}")
                    
                    # Sammle alle Events dieser Sequenz
                    seq_events = []
                    for e in events:
                        if (e.get("action_type") == "PASS" and 
                            e.get("pass_sequence", {}).get("sequence_id") == seq_id):
                            seq_events.append(e)
                    
                    # Sortiere nach Position
                    seq_events.sort(key=lambda x: x['pass_sequence']['position_in_sequence'])
                    
                    print(f"   📊 {len(seq_events)} Pässe in dieser Sequenz:")
                    for i, seq_event in enumerate(seq_events):
                        pos = seq_event['pass_sequence']['position_in_sequence']
                        player = seq_event.get('player', 'Unknown')
                        start_x = seq_event.get('start_x', 0)
                        start_y = seq_event.get('start_y', 0)
                        end_x = seq_event.get('end_x', 0)
                        end_y = seq_event.get('end_y', 0)
                        timestamp = seq_event.get('timestamp', 0)
                        
                        print(f"      Pass {pos}: {player}")
                        print(f"         Start: ({start_x:.1f}, {start_y:.1f})")
                        print(f"         Ende:  ({end_x:.1f}, {end_y:.1f})")
                        print(f"         Zeit:  {timestamp:.1f}s")
                        print()
        
        if found_sequences == 0:
            print(f"❌ Keine der angeforderten Sequenzen gefunden: {target_sequences}")
        
        return found_sequences > 0
        
    except Exception as e:
        print(f"❌ Fehler bei detaillierter Analyse: {e}")
        return False

def main():
    """Hauptfunktion"""
    print("🧪 PASS-SEQUENZ TEST SCRIPT")
    print("=" * 50)
    
    # Prüfe Kommandozeilenargumente
    if len(sys.argv) > 1:
        if sys.argv[1] == "detail":
            # Detaillierte Analyse für spezifische Sequenzen
            target_sequences = ["15032025003"]  # Deine problematische Sequenz
            if len(sys.argv) > 2:
                target_sequences = sys.argv[2].split(',')
            
            print(f"🎯 Detaillierte Analyse für Sequenzen: {target_sequences}")
            analyze_specific_sequences(target_sequences=target_sequences)
        else:
            print(f"❌ Unbekannter Parameter: {sys.argv[1]}")
            print("💡 Verfügbare Optionen:")
            print("   python test.py          - Standard-Analyse aller Sequenzen")
            print("   python test.py detail   - Detaillierte Analyse der problematischen Sequenz")
            print("   python test.py detail 15032025003,15032025004  - Spezifische Sequenzen")
    else:
        # Standard-Analyse
        print("📊 Standard-Analyse aller Passsequenzen")
        analyze_pass_sequences()

if __name__ == "__main__":
    main()