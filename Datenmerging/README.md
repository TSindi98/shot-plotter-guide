# ⚽ Fußball-Datenanalyse Suite

Eine umfassende Streamlit-Anwendung für die Analyse und Verarbeitung von Fußballdaten mit mehreren integrierten Tools.

## 🚀 Installation & Start

### Option 1: Mit Conda (empfohlen)
1. **Neue Conda-Umgebung erstellen:**
   ```bash
   conda create -n shot-plotter python=3.12 -y
   conda activate shot-plotter
   ```

2. **Dependencies installieren:**
   ```bash
   cd Datenmerging
   pip install -r requirements.txt
   ```

3. **App starten:**
   ```bash
   streamlit run streamlit_app.py
   ```

### Option 2: Mit virtualenv
1. **Virtuelle Umgebung erstellen:**
   ```bash
   cd Datenmerging
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   # oder: venv\Scripts\activate  # Windows
   ```

2. **Dependencies installieren:**
   ```bash
   pip install -r requirements.txt
   ```

3. **App starten:**
   ```bash
   streamlit run streamlit_app.py
   ```

### Option 3: Global (nicht empfohlen)
```bash
cd Datenmerging
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**🌐 Im Browser öffnen:**
Die App öffnet sich automatisch unter `http://localhost:8501`

## 📖 Funktionen

### 1. 🔄 Daten hochladen & verarbeiten
- **Shot-Plotter CSV**: Upload und Verarbeitung von Positionsdaten
- **Playermaker XML**: Integration von Passdaten 
- **Playermaker Possession**: Zeitdaten aus Excel/CSV

### 2. 📊 Datenanalyse & Visualisierung
- **Automatisches Merging**: Intelligente Zusammenführung der verschiedenen Datenquellen
- **Interaktive Visualisierungen**: Plotly-basierte Charts und Diagramme
- **Feldansicht**: Visualisierung der Spielfeldpositionen
- **Passnetzwerke**: Analyse der Passverbindungen zwischen Spielern
- **Zeitliche Verteilung**: Analyse der Events über die Spielzeit

### 3. 📥 Export-Funktionen
- **CSV Export**: Herunterladbare CSV-Dateien mit "Keine Angabe" für fehlende Werte
- **Excel Export**: Formatierte Excel-Dateien für weitere Analyse
- **Sportscode XML**: Export für Sportscode-Videoplattform

### 4. 📄 XML Merger
- **Viertel-Zusammenführung**: Führt XML-Dateien verschiedener Spielviertel zusammen
- **Zeitanpassung**: Konfigurierbare Startzeiten für jedes Viertel
- **ID-Verwaltung**: Automatische Vermeidung von ID-Konflikten
- **Vorschau**: Übersicht der Events pro Viertel vor dem Merge

## 🛠️ Verwendung

### Datenverarbeitung (Tabs 1-3)
1. **Dateien hochladen**: Alle drei Datenquellen (CSV, XML, Excel) laden
2. **Automatische Verarbeitung**: Die App verarbeitet und bereinigt die Daten automatisch
3. **Visualisierung**: Interaktive Dashboards für die Datenanalyse
4. **Export**: Download der zusammengeführten Daten in verschiedenen Formaten

### XML Merger (Tab 4)
1. **XML-Dateien hochladen**: Upload der Viertel-XML-Dateien in chronologischer Reihenfolge
2. **Startzeiten konfigurieren**: 
   - 1. Viertel: 0 Sekunden (0:00 Min)
   - 2. Viertel: 900 Sekunden (15:00 Min)  
   - 3. Viertel: 1800 Sekunden (30:00 Min)
   - 4. Viertel: 2700 Sekunden (45:00 Min)
3. **Vorschau**: Überprüfung der Event-Anzahl pro Viertel
4. **Zusammenführen**: Merge zu einer einzigen XML-Datei
5. **Download**: Herunterladbare zusammengeführte XML-Datei

## 📋 Unterstützte Dateiformate

### Eingabe:
- **CSV**: Shot-Plotter Positionsdaten, Playermaker Possession
- **XML**: Playermaker Passdaten, XML-Merger für Sportscode
- **Excel**: Playermaker Possession Zeitdaten

### Export:
- **CSV**: Zusammengeführte Daten mit deutschen Labels
### 1. XML-Dateien hochladen
- Laden Sie die XML-Dateien der einzelnen Viertel hoch
- Die Dateien sollten in der Reihenfolge der Viertel hochgeladen werden (1., 2., 3., 4. Viertel)

### 2. Startzeiten konfigurieren
- Geben Sie für jedes Viertel eine Startzeit in Sekunden an
- **Beispiel für ein Standard-Fußballspiel:**
  - 1. Viertel: 0 Sekunden (0:00 Min)
  - 2. Viertel: 900 Sekunden (15:00 Min)  
  - 3. Viertel: 1800 Sekunden (30:00 Min)
  - 4. Viertel: 2700 Sekunden (45:00 Min)

### 3. Zusammenführen & Herunterladen
- Klicken Sie auf "XML-Dateien zusammenführen"
- Laden Sie die resultierende XML-Datei herunter

## 🔧 Funktionsweise

Die Anwendung:
1. **Parst** alle hochgeladenen XML-Dateien
2. **Addiert** die konfigurierten Startzeiten zu allen `start` und `end` Zeitangaben im jeweiligen Viertel
3. **Führt** alle `instance` Elemente in einer einzigen `ALL_INSTANCES` Sektion zusammen
4. **Aktualisiert** die IDs um Konflikte zu vermeiden
5. **Erstellt** eine downloadbare XML-Datei

## 📋 Unterstützte XML-Struktur

Die App erwartet XML-Dateien mit folgender Grundstruktur:

```xml
<file>
  <SESSION_INFO>
    <start_time>...</start_time>
  </SESSION_INFO>
  <ALL_INSTANCES>
    <instance>
      <ID>1</ID>
      <start>16.871537353515624</start>
      <end>44.26008544921875</end>
      <code>...</code>
      <label>...</label>
    </instance>
    <!-- weitere instances -->
  </ALL_INSTANCES>
  <!-- weitere Sektionen -->
</file>
```

## ⚡ Features

- ✅ **Drag & Drop Upload** für XML-Dateien
- ✅ **Flexible Zeitkonfiguration** für jedes Viertel
- ✅ **Automatische ID-Verwaltung** zur Vermeidung von Konflikten
- ✅ **Vorschau** der Anzahl Events pro Viertel
- ✅ **Download** der zusammengeführten XML-Datei
- ✅ **Fehlerbehandlung** für ungültige XML-Dateien
- ✅ **Benutzerfreundliche Oberfläche** mit Zeit-Anzeige in Min:Sek

## 🛠️ Technische Details

- **Framework:** Streamlit
- **XML-Verarbeitung:** Python `xml.etree.ElementTree`
- **Unterstützte Dateiformate:** XML
- **Browser-Kompatibilität:** Alle modernen Browser 