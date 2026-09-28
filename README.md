# MedSim Patient Simulator

A local, LLM-driven simulated-patient training tool for clinicians.

## Quick start

See [START.md](START.md) for the full walkthrough.

```bash
pip install -r requirements.txt
ollama pull gemma3:12b
python medsim.py --case cases/appendicitis.json --model gemma3:12b
```

Then open **http://localhost:8000**.

## How it works

- **Case Editor** shows patient details — edit any field before starting
- Click **Start Encounter** — vitals are randomized slightly each session
- You type as the **clinician** — the patient responds in real time via your local Ollama
- Type orders naturally ("let's get a CBC and UA") — the EMR detects them and shows results
- Issuing a disposition ("I'm going to transfer you to the ED") confirms via modal and closes the session
- A **Session Summary** screen shows the full transcript, orders, and the ground-truth diagnosis

New cases: copy `cases/appendicitis.json`, edit the fields, load with `--case cases/yourcase.json`.
