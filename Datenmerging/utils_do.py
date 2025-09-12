#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Utilities Executor
============================

Führt verschiedene Utility-Funktionen für die Event-Database aus.
Diese Datei dient als Ausführungs-Script für spezifische Korrekturen und Änderungen.

Verwendung:
    python utils_do.py
"""

import sys
import os
from pathlib import Path

# Importiere Utility-Funktionen
from utils import (
    add_season_to_database,
    fix_unsuccessful_pass_targets,
    determine_season_from_date,
    add_pass_sequence_ids,
    remove_passes_with_invalid_releasing_leg
)

def main():
    """
    Hauptfunktion - Führt die gewünschte Utility-Funktion aus
    """
    print("🛠️  DATABASE UTILITIES EXECUTOR")
    print("=" * 50)
    
    # Standard-Pfad zur event_database.json
    json_file_path = 'event_database.json'
    
    # Prüfe ob alternative Datei als Kommandozeilenargument übergeben wurde
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
    
    # Prüfe ob Datei existiert
    if not Path(json_file_path).exists():
        print(f"❌ Datei nicht gefunden: {json_file_path}")
        print(f"💡 Verfügbare Dateien im aktuellen Verzeichnis:")
        for file in Path('.').glob('*.json'):
            print(f"   - {file.name}")
        sys.exit(1)
    
    print(f"🎯 Ziel-Datei: {json_file_path}")
    print()
    
    # Zeige verfügbare Funktionen
    print("📋 Verfügbare Utility-Funktionen:")
    print("1. 🔧 Korrigiere passed_to Werte für nicht erfolgreiche Pässe")
    print("2. 🏈 Füge Saison-Informationen hinzu")
    print("3. 🏃 Füge Passfolgen-IDs hinzu")
    print("4. 🗑️  Entferne Pässe mit ungültigem Releasing Leg")
    print("5. ❌ Beenden")
    print()
    
    # Benutzereingabe
    while True:
        try:
            choice = input("Wählen Sie eine Funktion (1-5): ").strip()
            
            if choice == "1":
                print("\n🔧 KORREKTUR: Nicht erfolgreiche Pässe")
                print("-" * 40)
                success = fix_unsuccessful_pass_targets(json_file_path, backup=True)
                break
                
            elif choice == "2":
                print("\n🏈 SAISON-HINZUFÜGUNG")
                print("-" * 40)
                success = add_season_to_database(json_file_path, backup=True)
                break
                
            elif choice == "3":
                print("\n🏃 PASSFOLGEN-IDS HINZUFÜGEN")
                print("-" * 40)
                print("⚠️  Dies fügt 'pass_sequence' Felder zu allen Pass-Events hinzu!")
                confirm = input("Möchten Sie fortfahren? (j/n): ").strip().lower()
                
                if confirm in ['j', 'ja', 'y', 'yes']:
                    success = add_pass_sequence_ids(json_file_path, backup=True)
                else:
                    print("❌ Abgebrochen")
                    success = False
                break
                
            elif choice == "4":
                print("\n🗑️  BEREINIGUNG: Ungültige Releasing Leg Werte")
                print("-" * 45)
                confirm = input("⚠️  WARNUNG: Dies wird Pässe mit ungültigen 'Releasing Leg' Werten dauerhaft löschen!\n   Fortfahren? (j/N): ").strip().lower()
                if confirm in ['j', 'ja', 'y', 'yes']:
                    success = remove_passes_with_invalid_releasing_leg(json_file_path, backup=True)
                else:
                    print("❌ Abgebrochen")
                    success = False
                break
                
            elif choice == "5":
                print("👋 Programm beendet.")
                sys.exit(0)
                
            else:
                print("❌ Ungültige Eingabe. Bitte wählen Sie 1, 2, 3, 4 oder 5.")
                continue
                
        except KeyboardInterrupt:
            print("\n\n👋 Programm durch Benutzer abgebrochen.")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Eingabefehler: {e}")
            continue
    
    # Ergebnis
    if success:
        print(f"\n✅ Operation erfolgreich abgeschlossen!")
        print(f"📁 Die Datei '{json_file_path}' wurde aktualisiert.")
        
        # Zeige Backup-Info
        backup_path = Path(json_file_path).parent / f"{Path(json_file_path).stem}_backup{Path(json_file_path).suffix}"
        if backup_path.exists():
            print(f"💾 Backup erstellt: '{backup_path}'")
    else:
        print(f"\n❌ Operation fehlgeschlagen!")
        sys.exit(1)

def quick_fix_unsuccessful_passes():
    """
    Schnelle Ausführung der passed_to Korrektur ohne Menü
    """
    print("🔧 SCHNELLE KORREKTUR: Nicht erfolgreiche Pässe")
    print("=" * 50)
    
    json_file_path = 'event_database.json'
    
    # Im quick Modus ist argv[1] = "quick", argv[2] = Dateiname
    if len(sys.argv) > 2:
        json_file_path = sys.argv[2]
    
    success = fix_unsuccessful_pass_targets(json_file_path, backup=True)
    
    if success:
        print(f"\n✅ Korrektur erfolgreich!")
    else:
        print(f"\n❌ Korrektur fehlgeschlagen!")
        sys.exit(1)

if __name__ == "__main__":
    # Prüfe ob als "quick" Modus aufgerufen
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_fix_unsuccessful_passes()
    else:
        main()