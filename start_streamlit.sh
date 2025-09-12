#!/bin/bash
cd "$(dirname "$0")"
source shot-plotter-venv/bin/activate
cd Passanalyse
streamlit run main.py