#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug Script für Passsequenz-Analyse
=====================================

Analysiert warum Passsequenzen unterbrochen werden.
"""

import json
from utils import calculate_time_difference_simple, check_network_consistency_simple

def debug_sequence_break(json_file_path, target_event_id=None):
    """
    Debuggt warum eine spezifische Sequenz unterbrochen wird
    
    Args:
        json_file_path (str): Pfad zur JSON-Datei
        target_event_id (str): Event ID die analysiert werden soll (optional)
    """
    
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    print("🔍 PASSSEQUENZ DEBUG-ANALYSE")
    print("=" * 50)
    
    for match_id, match_data in data.get("events_by_match", {}).items():
        events = match_data.get("events", [])
        
        # Nur Pass-Events filtern und nach Zeit sortieren
        pass_events = [event for event in events if event.get("action_type") == "PASS"]
        pass_events.sort(key=lambda x: x.get('timestamp', 0))
        
        print(f"\n📊 Match: {match_id}")
        print(f"🏃 Pass-Events: {len(pass_events)}")
        
        # Finde Events mit bestehenden sequence_ids oder target_event_id
        events_to_analyze = []
        
        if target_event_id:
            # Finde spezifisches Event
            target_idx = None
            for i, event in enumerate(pass_events):
                if event.get('event_id') == target_event_id:
                    target_idx = i
                    break
            
            if target_idx is not None:
                # Analysiere Event und seine Nachbarn
                start_idx = max(0, target_idx - 2)
                end_idx = min(len(pass_events), target_idx + 3)
                events_to_analyze = list(range(start_idx, end_idx))
                print(f"🎯 Analysiere Event {target_event_id} und Umgebung (Index {target_idx})")
        else:
            # Finde Events mit bestehenden Sequence IDs
            for i, event in enumerate(pass_events):
                sequence_info = event.get('pass_sequence', {})
                if sequence_info and sequence_info.get('sequence_id', '0') != '0':
                    events_to_analyze.extend([i-1, i, i+1])
        
        # Entferne Duplikate und ungültige Indizes
        events_to_analyze = sorted(list(set([i for i in events_to_analyze if 0 <= i < len(pass_events)])))
        
        if not events_to_analyze:
            print("ℹ️  Keine relevanten Events für Analyse gefunden")
            continue
        
        print(f"\n🔬 Analysiere {len(events_to_analyze)} Events:")
        
        # Detailanalyse
        for i in range(len(events_to_analyze) - 1):
            current_idx = events_to_analyze[i]
            next_idx = events_to_analyze[i + 1]
            
            current_event = pass_events[current_idx]
            next_event = pass_events[next_idx]
            
            print(f"\n📋 Event {current_idx} → {next_idx}")
            print(f"   Spieler: {current_event.get('player', 'N/A')} → {next_event.get('player', 'N/A')}")
            print(f"   Zeit: {current_event.get('timestamp', 0):.2f}s → {next_event.get('timestamp', 0):.2f}s")
            
            # Bestehende Sequenz-IDs anzeigen
            current_seq = current_event.get('pass_sequence', {})
            next_seq = next_event.get('pass_sequence', {})
            print(f"   Bestehende Seq-ID: {current_seq.get('sequence_id', 'N/A')} → {next_seq.get('sequence_id', 'N/A')}")
            
            # Kriterien prüfen
            time_diff = calculate_time_difference_simple(current_event, next_event)
            network_consistent = check_network_consistency_simple(current_event, next_event)
            last_pass_successful = current_event.get('outcome', '') == 'Erfolgreich'
            
            print(f"   ⏱️  Zeitdifferenz: {time_diff:.3f}s (max 8.0s) → {'✅' if time_diff <= 8.0 else '❌'}")
            print(f"   🔗 Netzwerk-Konsistenz: {'✅' if network_consistent else '❌'}")
            print(f"   ✅ Letzter Pass erfolgreich: {'✅' if last_pass_successful else '❌'}")
            
            # Detailierte Netzwerk-Analyse
            if not network_consistent:
                current_passing = current_event.get('passing_network', {})
                next_passing = next_event.get('passing_network', {})
                
                condition1 = current_passing.get('passed_to', '') == next_event.get('player', '')
                condition2 = current_event.get('player', '') == next_passing.get('passed_from', '')
                
                print(f"      🔗 Bedingung 1 (passed_to → player): '{current_passing.get('passed_to', '')}' == '{next_event.get('player', '')}' → {'✅' if condition1 else '❌'}")
                print(f"      🔗 Bedingung 2 (player → passed_from): '{current_event.get('player', '')}' == '{next_passing.get('passed_from', '')}' → {'✅' if condition2 else '❌'}")
            
            # Zeitberechnung Details
            if time_diff > 8.0 or time_diff < 0:
                current_timestamp = current_event.get('timestamp', 0)
                next_timestamp = next_event.get('timestamp', 0)
                next_time_to_release = next_event.get('additional_data', {}).get('Time to Release', 0)
                
                last_ball_release = current_timestamp
                current_ball_reception = next_timestamp - next_time_to_release
                
                print(f"      ⏱️  Details: Release {last_ball_release:.3f}s → Reception {current_ball_reception:.3f}s")
                print(f"      ⏱️  Next timestamp: {next_timestamp:.3f}s, Time to Release: {next_time_to_release:.3f}s")
            
            # Gesamtbewertung
            sequence_continues = (time_diff <= 8.0 and network_consistent and last_pass_successful)
            print(f"   🏃 Sequenz geht weiter: {'✅' if sequence_continues else '❌'}")
            
            if not sequence_continues:
                print(f"   ⚠️  SEQUENZ WIRD UNTERBROCHEN!")
                reasons = []
                if time_diff > 8.0: reasons.append("Zeitlimit überschritten")
                if not network_consistent: reasons.append("Netzwerk inkonsistent")
                if not last_pass_successful: reasons.append("Vorheriger Pass nicht erfolgreich")
                print(f"   🚫 Gründe: {', '.join(reasons)}")

def analyze_existing_sequences(json_file_path):
    """Analysiert bereits vorhandene Sequenz-IDs"""
    
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    print("📊 ANALYSE BESTEHENDER SEQUENZ-IDs")
    print("=" * 50)
    
    total_passes = 0
    passes_with_sequences = 0
    unique_sequences = set()
    
    for match_id, match_data in data.get("events_by_match", {}).items():
        events = match_data.get("events", [])
        pass_events = [event for event in events if event.get("action_type") == "PASS"]
        
        match_passes = len(pass_events)
        match_sequenced = 0
        
        for event in pass_events:
            total_passes += 1
            sequence_info = event.get('pass_sequence', {})
            
            if sequence_info and sequence_info.get('sequence_id', '0') != '0':
                passes_with_sequences += 1
                match_sequenced += 1
                unique_sequences.add(sequence_info.get('sequence_id'))
        
        if match_sequenced > 0:
            print(f"📋 {match_id}: {match_sequenced}/{match_passes} Pässe in Sequenzen")
    
    print(f"\n📈 GESAMT-STATISTIK:")
    print(f"   • Gesamte Pässe: {total_passes}")
    print(f"   • Pässe mit Sequenz-ID: {passes_with_sequences}")
    print(f"   • Einzigartige Sequenzen: {len(unique_sequences)}")
    print(f"   • Abdeckung: {passes_with_sequences/total_passes*100:.1f}%")

if __name__ == "__main__":
    import sys
    
    json_file_path = 'event_database.json'
    target_event_id = None
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "existing":
            analyze_existing_sequences(json_file_path)
            sys.exit(0)
        else:
            target_event_id = sys.argv[1]
    
    if len(sys.argv) > 2:
        json_file_path = sys.argv[2]
    
    debug_sequence_break(json_file_path, target_event_id)