# MedSim Patient Simulator — Quick Start

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Make sure Ollama is running with Gemma 4

```bash
# Pull the model if you haven't already
ollama pull gemma4:12b-mlx

# Ollama starts automatically on Mac, but to verify:
ollama list
```

## 3. Run the simulator

```bash
# From the medsim/ folder:
python medsim.py --case cases/appendicitis.json --model gemma4:12b-mlx
```

Then open your browser to: **http://localhost:8000**

## 4. How it works

- The **Case Editor** shows patient details — you can edit any field before starting
- Click **Start Encounter** — vitals are randomized slightly each session
- You type as the **clinician** — the patient responds in real time via your local Ollama
- Type orders naturally ("let's get a CBC and UA") — the EMR detects them and shows results
- When you issue a disposition ("I'm going to transfer you to the ED"), a modal confirms and closes the session
- A **Session Summary** screen shows the full transcript, orders, and the ground truth diagnosis

## Options

```
--case    Path to case JSON (default: cases/appendicitis.json)
--model   Ollama model name (default: gemma4:12b-mlx)
--port    Port to serve on (default: 8000)
```

## Adding new cases

Copy `cases/appendicitis.json` and edit the fields. Load with `--case cases/yourcase.json`.
