# TMICC Risk UI (from scratch)

## What this does
- Reads **Probability** and **Severity** dropdown options (with numeric codes) from the Excel sheet **Risk Matrix**
- Lets you enter: **Line → Machine → Operation category → Phase(s) → Task → Step → Hazard**
- Stores the **numeric codes** for Probability/Severity and looks up the **Risk Rating** from the P×S grid
- Saves everything locally in `tmicc_risk.sqlite` (SQLite DB)

## Setup (macOS)
1) Open this folder in VS Code.
2) Put your spreadsheet next to `app.py` and name it **UMS.xlsx**.
3) In VS Code Terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If VS Code doesn't use your venv automatically:
- Cmd+Shift+P → **Python: Select Interpreter** → choose `.venv`
