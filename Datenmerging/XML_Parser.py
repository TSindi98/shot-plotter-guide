from dataclasses import dataclass
from typing import Optional, List
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
import os

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
    action_type: str  # 'PASS', 'LOSS', or 'POTENTIAL_SHOT'

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

def parse_playermaker_data(xml_path: str) -> List[BallPossessionEvent]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
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
        event = BallPossessionEvent(
            event_id=int(instance.find('ID').text),
            start_time=float(instance.find('start').text),
            end_time=float(instance.find('end').text),
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
    return sorted(events, key=lambda x: x.start_time)

def save_to_csv(events: List[BallPossessionEvent], output_path: str):
    """Speichert die Events in einer CSV-Datei"""
    df = pd.DataFrame([event.to_dict() for event in events])
    df.to_csv(output_path, index=False)
    return df

def main():
    # Eingabedatei
    input_file = "Beispiel Durchgang 3.5.xml"
    
    # Überprüfen ob die Datei existiert
    if not os.path.exists(input_file):
        print(f"Fehler: Die Datei {input_file} wurde nicht gefunden!")
        return
    
    # Daten parsen
    print("Parsing XML-Datei...")
    events = parse_playermaker_data(input_file)
    
    # Statistiken ausgeben
    print(f"\nGefundene Events: {len(events)}")
    print(f"Zeitspanne: {min(e.start_time for e in events):.2f}s bis {max(e.end_time for e in events):.2f}s")
    
    # Aktionstypen zählen
    action_counts = {}
    for event in events:
        action_counts[event.action_type] = action_counts.get(event.action_type, 0) + 1
    
    print("\nAktionstypen:")
    for action, count in action_counts.items():
        print(f"{action}: {count}")
    
    # CSV erstellen
    output_file = "parsed_events.csv"
    df = save_to_csv(events, output_file)
    print(f"\nDaten wurden in {output_file} gespeichert")
    
    # Erste paar Zeilen anzeigen
    print("\nErste 5 Zeilen der Daten:")
    print(df.head())

if __name__ == "__main__":
    main()
