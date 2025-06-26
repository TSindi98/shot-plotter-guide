# 🚀 Schnelle Einrichtung - Shot Plotter Datenmerging

## Ein-Kommando Setup (empfohlen)

```bash
# 1. Conda-Umgebung erstellen und aktivieren
conda create -n shot-plotter python=3.12 -y && conda activate shot-plotter

# 2. In den richtigen Ordner wechseln
cd Datenmerging

# 3. Dependencies installieren
pip install -r requirements.txt

# 4. Streamlit-App starten
streamlit run streamlit_app.py
```

## Täglich verwenden

```bash
conda activate shot-plotter
cd Datenmerging
streamlit run streamlit_app.py
```

## Alias für einfache Nutzung (optional)

```bash
# Einmalig einrichten:
echo 'alias shot-plotter="conda activate shot-plotter && cd $(git rev-parse --show-toplevel)/Datenmerging"' >> ~/.zshrc
source ~/.zshrc

# Dann einfach verwenden:
shot-plotter
streamlit run streamlit_app.py
```

## Troubleshooting

**Problem**: `conda: command not found`
**Lösung**: [Anaconda installieren](https://www.anaconda.com/download)

**Problem**: `ModuleNotFoundError`  
**Lösung**: `pip install -r requirements.txt` erneut ausführen

**Problem**: Port bereits belegt
**Lösung**: `streamlit run streamlit_app.py --server.port 8502` 