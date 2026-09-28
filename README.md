# MedSim Patient Simulator

A simulated-patient training tool for clinicians. You play the **clinician**; a
locally-run language model plays the **patient** in a realistic urgent-care
encounter: you take a history, order tests, get results, and disposition the
patient. At the end you see the full transcript and the ground-truth diagnosis.

Everything runs on your own machine — the web server, the patient brain, the
models. No cloud accounts, no API keys, no patient data ever leaves the Mac.

## What this was built for

This project is designed to run **locally on a Mac mini** (Apple Silicon). The
idea:

- **Ollama** runs the patient as a large language model on the Mac's own
  GPU/unified memory — no cloud inference, no per-token cost, works offline.
- **Python (FastAPI + Uvicorn)** serves a small web app on the same machine;
  you open it in a browser at `http://localhost:8000`.
- Because the "patient" is just a model with a carefully written system prompt,
  you can create new clinical cases by editing a JSON file — no code changes.

## Setup

### 1. Install Ollama

Ollama is the local model runner. Download it for Mac from
[ollama.com](https://ollama.com) (or `brew install ollama` if you use
Homebrew), then launch it — it runs in the background and serves models at
`http://localhost:11434`.

### 2. Download a model

A *model* is the actual AI brain (billions of parameters). Ollama downloads
them from its model library with `ollama pull`:

```bash
ollama pull gemma4:12b-mlx
```

This project defaults to **`gemma4:12b-mlx`** (see `DEFAULT_MODEL` in
`medsim.py`). A few things worth knowing:

- **What the name means:** `gemma3:12b` = Google's Gemma 3 model, 12 billion
  parameters. `gemma4:12b-mlx` = the Gemma 4 12B model packaged in **MLX**
  format.
- **What is MLX?** Apple's machine-learning framework for Apple Silicon Macs.
  MLX models run directly on the Mac mini's GPU/unified memory, which makes
  them faster and more memory-efficient on a Mac than the standard builds.
  That's why the MLX variant is the default here.
- **Size:** a 12B model is roughly 8–20 GB on disk depending on quantization
  (the compressed numeric precision). Make sure the Mac has room.
- **Switching models:** whatever you pulled, pass its exact name with
  `--model`, e.g. `--model qwen3:8b`. The name must match what
  `ollama list` shows.

Verify the model is ready:

```bash
ollama list        # shows downloaded models
```

### 3. Install the Python dependencies

```bash
pip install -r requirements.txt
```

That's just three packages:

| Package   | Why                     |
|-----------|-------------------------|
| `fastapi` | The web server framework |
| `uvicorn` | Runs the FastAPI server  |
| `httpx`   | Talks to Ollama over HTTP |

### 4. Run it

From the project folder:

```bash
python medsim.py --case cases/appendicitis.json --model gemma4:12b-mlx
```

Options:

| Flag      | Default                    | Meaning                              |
|-----------|----------------------------|--------------------------------------|
| `--case`  | `cases/appendicitis.json`  | Which clinical case to load          |
| `--model` | `gemma4:12b-mlx`           | Ollama model name (must be pulled)   |
| `--port`  | `8000`                     | Port for the web UI                  |

Then open **http://localhost:8000** in a browser.

### 5. Do an encounter

1. **Case Editor** — review the patient's details; vitals shown here get a
   small random variation applied per session so every run differs slightly.
   Click **Start Encounter**.
2. **Encounter** — the patient opens with one of several randomized opening
   lines. You type as the clinician. The left **EMR panel** shows vitals and
   accumulates test results.
3. **Order tests naturally** — type *"let's get a CBC and UA"* and the app
   detects the order keywords and returns the scripted lab result.
4. **Disposition** — type something like *"I'm going to transfer you to the
   ED"*; a confirmation modal appears, you pick the disposition type, and the
   session closes.
5. **Session Summary** — the full transcript, your orders, and the
   **ground-truth diagnosis** (revealed only at the end, so the patient never
   "knows" it during the encounter).

Each finished session is saved under `output/<session-id>/` as both
`session.json` (structured data) and `session.txt` (readable transcript).

## How the code works, step by step

### The moving parts

| File | Role |
|------|------|
| `medsim.py` | Everything backend: web server, session logic, EMR rules, Ollama calls |
| `ui.html` | The entire frontend (single page: case editor, encounter chat, summary) |
| `cases/*.json` | Clinical case definitions — patient, history, vitals, detractors, scripted lab results, ground truth |
| `requirements.txt` | `fastapi`, `uvicorn`, `httpx` |

### Startup

1. `argparse` reads `--case`, `--model`, `--port`.
2. The case JSON is loaded into memory as `case_data`.
3. Uvicorn starts the FastAPI app on `0.0.0.0:<port>`.
4. `GET /` serves `ui.html`. `GET /api/case` hands the frontend the case
   details (patient, history — but *not* the ground truth or oracle results).

### Starting a session

5. Clicking **Start Encounter** calls `POST /api/session/start`, which builds a
   `Session` object:
   - a random 8-character session id,
   - **jittered vitals**: each vital gets a small random offset from its base
     value, within the ranges defined in the case file (so HR might be 94 ± 6),
   - a **random presentation variant**: one of several opening lines, so the
     patient doesn't start identically every time.
6. The frontend then opens a **WebSocket** to `/ws` — one persistent,
   two-way connection for the whole encounter. The browser and server exchange
   small JSON messages (`clinician_message`, `token`, `emr_results`,
   `patient_start`, `patient_end`, `disposition_detected`, …).

### One turn of conversation

When you send a message, the server runs a pipeline *before* the patient
replies:

7. **Order detection** (`detect_orders`): regex patterns (`ORDER_PATTERNS`)
   scan your text for things like "CBC", "urinalysis", "CT abdomen". Matches
   become orders.
8. **Oracle results** (`get_oracle_result`): test results come from the case
   JSON's `oracle_results`, *not* from the LLM — they're deterministic and
   clinically authored, so the case behaves the same every run. Abnormal
   results are flagged, and both the chat and the EMR panel update instantly.
9. **Disclosure tracking** (`detect_disclosure`): if you tell the patient
   something ("your labs show…"), it's recorded so the patient stays
   consistent with what it "knows" you've told it.
10. **Disposition detection** (`detect_disposition`): phrases like
    "discharge", "transfer to ED", "admit" trigger the confirmation modal
    instead of a patient reply.
11. **Patient reply** (`stream_patient_response`): your message is appended to
    the conversation history, which plus a generated **system prompt** is sent
    to Ollama's `/api/chat` endpoint. Tokens stream back one at a time over
    the WebSocket, so the reply types out live in the browser.

### The system prompt — how the patient "thinks"

`build_patient_system_prompt()` assembles the patient's instructions from the
case file:

- identity and chief complaint,
- full history (HPI, PMH, meds, allergies, social, family),
- how it physically feels right now,
- **detractors**: real-but-misleading features (e.g. urinary symptoms that
  suggest a UTI, no fever) that it only reveals *if asked the right question* —
  this is what makes the case diagnostically interesting,
- facts you've disclosed so far,
- behavioral rules: doesn't know its diagnosis, answers only what's asked, no
  medical jargon, short replies, never breaks character.

### Ending a session

12. Confirming a disposition sets `session.active = False` and writes
    `output/<session-id>/session.json` + `session.txt` (transcript, vitals,
    orders, disposition, duration).
13. `GET /api/session/summary` feeds the summary view — including the ground
    truth, which is only revealed here.

### Design notes

- **Single session in memory** — no save/resume; closing the browser ends the
  encounter (this matches the original PRD).
- **Deterministic medicine, generative conversation** — labs/imaging are
  scripted per case; the *dialogue* is the LLM. This keeps cases
  reproducible while conversations feel human.
- **The model never sees the diagnosis** — it's only in the case JSON on the
  server, so the patient can't accidentally leak it.

## The Ollama piece, in more depth

Ollama is doing one job here: **being the patient**. `medsim.py` talks to it
like this:

```python
OLLAMA_URL = "http://localhost:11434/api/chat"
```

- On every patient turn, the server POSTs `{"model": ..., "messages": [...],
  "stream": true}` — a standard chat-completions-style payload: a system
  message (the patient persona) plus the full conversation history.
- With `"stream": true`, Ollama returns the reply as a stream of JSON lines,
  each carrying a few tokens. `medsim.py` forwards each chunk over the
  WebSocket as a `token` message, which is why the patient's words appear
  progressively instead of all at once.
- The request has a 60-second timeout; a 12B model on a Mac mini typically
  answers in seconds.

Why local instead of a cloud API:

- **Privacy** — clinical training scenarios stay on the machine.
- **Cost** — no per-token billing; run unlimited sessions.
- **Offline** — works with no internet once the model is downloaded.
- **Latency** — localhost round-trips, no network hop.

If you want to experiment: any chat-capable Ollama model works — just pull it
and pass `--model <name>`. Smaller models (e.g. 8B) answer faster but
role-play less convincingly; larger ones are slower but stay in character
better.

## Adding new cases

Copy `cases/appendicitis.json` and edit the fields:

- `patient`, `history` — demographics, HPI, PMH, meds, allergies, social/family
- `vitals.base` / `vitals.variation` — base values and per-session jitter ranges
- `presentation_variants` — randomized opening lines
- `physical_exam_perceived` — what the patient feels on exam
- `detractors` — the misleading-but-real features that make the case tricky
- `persona` — personality and answering style
- `oracle_results` — scripted result text + `is_abnormal` flag per orderable test
- `ground_truth_diagnosis` — revealed only on the summary screen

Load it with `--case cases/yourcase.json`. New order keywords go in
`ORDER_PATTERNS` in `medsim.py` (regex → order key), with display labels in
`format_order_key`.

## License

MIT — see [LICENSE](LICENSE). Use it, fork it, teach with it.
