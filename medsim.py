"""
MedSim Patient Simulator — v1.0
Per PRD v1.2 Final

Run:
    pip install fastapi uvicorn httpx
    python medsim.py --case cases/appendicitis.json

Then open: http://localhost:8000
"""

import argparse
import json
import random
import re
import time
import uuid
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/chat"
#DEFAULT_MODEL = "gemma3:12b"  # change to match your pulled model name
DEFAULT_MODEL = "gemma4:12b-mlx"  # change to match your pulled model name
# Order detection keywords → category
ORDER_PATTERNS = {
    "cbc":               [r"\bcbc\b", r"complete blood count", r"blood count"],
    "bmp":               [r"\bbmp\b", r"basic metabolic", r"chem\s*7", r"electrolytes"],
    "ua":                [r"\bua\b", r"urinalysis", r"urine analysis", r"urine test"],
    "bhcg":              [r"bhcg", r"hcg", r"pregnancy test", r"beta hcg", r"beta-hcg"],
    "lipase":            [r"\blipase\b", r"amylase"],
    "rlq_ultrasound":    [r"rlq\s*ultrasound", r"right lower quadrant ultrasound",
                          r"appendix ultrasound", r"ultrasound.*append",
                          r"append.*ultrasound", r"abdominal ultrasound", r"\bultrasound\b", r"\bus\b"],
    "ct_abdomen_pelvis": [r"ct\s*(of\s*)?(abdomen|abd|belly|stomach)",
                          r"ct\s*a/?p\b", r"cat scan", r"computed tomography",
                          r"ct\s*scan"],
    "xray_abdomen":      [r"x-?ray.*abdomen", r"abdominal\s*x-?ray",
                          r"kub\b", r"flat\s*plate"],
}

DISPOSITION_PATTERNS = [
    r"discharge", r"send home", r"go home",
    r"transfer.*ed", r"transfer.*emergency", r"send.*ed",
    r"call.*surgery", r"surgery consult", r"admit",
    r"call.*ems", r"\bems\b", r"call.*ambulance",
    r"refer.*surgeon", r"refer.*ed",
]

# ─────────────────────────────────────────────────────────────────────────────
# Session state (single session in memory — per PRD: no save/resume)
# ─────────────────────────────────────────────────────────────────────────────
class Session:
    def __init__(self, case: dict):
        self.session_id = str(uuid.uuid4())[:8]
        self.started_at = datetime.now().isoformat()
        self.case = case
        self.session_vitals = self._apply_variation()
        self.selected_variant = random.choice(case["presentation_variants"])
        self.conversation_history: list[dict] = []  # {role, content}
        self.turns: list[dict] = []                  # display log
        self.orders: list[dict] = []
        self.disclosed_to_patient: list[str] = []
        self.disposition: Optional[dict] = None
        self.start_time = time.time()
        self.active = True

    def _apply_variation(self) -> dict:
        base = self.case["vitals"]["base"]
        var  = self.case["vitals"]["variation"]
        def jitter(key):
            lo, hi = var[key]
            return round(base[key] + random.uniform(lo, hi), 1)
        return {
            "hr":           int(jitter("hr")),
            "bp_systolic":  int(jitter("bp_systolic")),
            "bp_diastolic": base["bp_diastolic"],
            "rr":           int(jitter("rr")),
            "temp_c":       round(jitter("temp_c"), 1),
            "spo2":         base["spo2"],
        }

    def build_patient_system_prompt(self) -> str:
        c = self.case
        p = c["patient"]
        h = c["history"]
        v = self.session_vitals
        detractor_text = "\n".join(
            f"- {d['description']}" for d in c["detractors"]
        )
        disclosed_text = (
            "\n".join(f"- {d}" for d in self.disclosed_to_patient)
            if self.disclosed_to_patient
            else "  (none yet)"
        )
        return f"""You are {p['name']}, a {p['age']}-year-old {p['sex'].lower()} patient in an urgent care clinic.

CHIEF COMPLAINT: {p['chief_complaint']}

YOUR HISTORY (know this but don't volunteer it all at once):
- HPI: {h['hpi']}
- Past medical history: {h['pmh']}
- Medications: {', '.join(h['medications'])}
- Allergies: {h['allergies']}
- Social: {h['social_history']}
- Family history: {h['family_history']}

HOW YOU PHYSICALLY FEEL RIGHT NOW:
- {c['physical_exam_perceived']}
- Your actual temperature is {v['temp_c']}°C. You do NOT feel feverish.
- Your heart is beating a little fast — you feel anxious, not specifically your heart racing.

YOUR OPENING LINE (use this first time you speak):
"{self.selected_variant}"

YOUR PERSONALITY:
{c['persona']}

IMPORTANT BEHAVIORAL RULES:
1. You do NOT know your diagnosis. You're a patient, not a doctor.
2. Answer only what was asked. Don't volunteer everything at once.
3. Describe symptoms in plain, everyday language — no medical terms.
4. The following features are REAL and you experience them — but only reveal them if the clinician asks the RIGHT question:
{detractor_text}
5. FACTS THE CLINICIAN HAS TOLD YOU (respond consistently with these):
{disclosed_text}
6. Keep answers short — 1 to 4 sentences. You're uncomfortable and a little worried.
7. Never break character. You are a real patient in a clinic."""

    def add_turn(self, speaker: str, text: str):
        turn = {
            "turn_number": len(self.turns) + 1,
            "speaker": speaker,
            "text": text,
            "timestamp": datetime.now().isoformat(),
        }
        self.turns.append(turn)
        return turn

    def duration_seconds(self) -> int:
        return int(time.time() - self.start_time)


# ─────────────────────────────────────────────────────────────────────────────
# EMR logic
# ─────────────────────────────────────────────────────────────────────────────
def detect_orders(text: str, case: dict) -> list[str]:
    """Return list of order keys found in clinician text."""
    found = []
    lower = text.lower()
    for key, patterns in ORDER_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, lower):
                found.append(key)
                break
    return found


def detect_disposition(text: str) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in DISPOSITION_PATTERNS)


def detect_disclosure(text: str) -> Optional[str]:
    """
    Detect when clinician is sharing information with the patient.
    Returns a summary string if disclosure detected, else None.
    """
    disclosure_triggers = [
        r"your (labs?|results?|test|ultrasound|ct|x-?ray)",
        r"(looks like|appears|seems|i think|i believe|you have|you may have|you might have)",
        r"(we found|shows?|revealed?|came back)",
        r"(your wbc|your blood|your urine)",
    ]
    lower = text.lower()
    if any(re.search(t, lower) for t in disclosure_triggers):
        return text.strip()
    return None


def get_oracle_result(order_key: str, case: dict) -> dict:
    """
    Get results for an ordered test.
    Uses pre-defined oracle results from case file (per PRD).
    Could be replaced with live Ollama call for dynamic results.
    """
    results = case.get("oracle_results", {})
    if order_key in results:
        return results[order_key]
    return {
        "result_text": f"Results for {order_key}: Pending — no data available for this test in current case.",
        "is_abnormal": False,
    }


def format_order_key(key: str) -> str:
    labels = {
        "cbc": "CBC with differential",
        "bmp": "BMP / Basic Metabolic Panel",
        "ua": "Urinalysis",
        "bhcg": "Beta-hCG (pregnancy test)",
        "lipase": "Lipase",
        "rlq_ultrasound": "RLQ Ultrasound",
        "ct_abdomen_pelvis": "CT Abdomen/Pelvis",
        "xray_abdomen": "Abdominal X-Ray (KUB)",
    }
    return labels.get(key, key.upper())


# ─────────────────────────────────────────────────────────────────────────────
# Ollama calls
# ─────────────────────────────────────────────────────────────────────────────
async def stream_patient_response(session: Session, websocket: WebSocket, model: str):
    """Send clinician input to Ollama, stream patient response token by token."""
    messages = [
        {"role": "system", "content": session.build_patient_system_prompt()},
        *session.conversation_history,
    ]

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            OLLAMA_URL,
            json={"model": model, "messages": messages, "stream": True},
        ) as resp:
            full_text = ""
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                content = chunk.get("message", {}).get("content", "")
                if content:
                    full_text += content
                    await websocket.send_json({"type": "token", "content": content})

                if chunk.get("done"):
                    break

    return full_text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="MedSim")
session: Optional[Session] = None
model_name = DEFAULT_MODEL
case_data: Optional[dict] = None


@app.get("/", response_class=HTMLResponse)
async def root():
    ui_path = Path(__file__).parent / "ui.html"
    return HTMLResponse(ui_path.read_text())


@app.get("/api/case")
async def get_case():
    if not case_data:
        return {"error": "No case loaded"}
    return {
        "case_name": case_data["case_name"],
        "patient": case_data["patient"],
        "history": case_data["history"],
    }


@app.get("/api/session")
async def get_session():
    if not session:
        return {"active": False}
    return {
        "active": session.active,
        "session_id": session.session_id,
        "vitals": session.session_vitals,
        "variant": session.selected_variant,
        "turns": len(session.turns),
        "orders": session.orders,
        "disposition": session.disposition,
    }


@app.post("/api/session/start")
async def start_session():
    global session
    if not case_data:
        return {"error": "No case loaded"}
    session = Session(case_data)
    return {
        "session_id": session.session_id,
        "vitals": session.session_vitals,
        "variant": session.selected_variant,
    }


@app.get("/api/session/summary")
async def get_summary():
    if not session:
        return {"error": "No session"}
    return {
        "session_id": session.session_id,
        "case_name": session.case["case_name"],
        "patient": session.case["patient"],
        "vitals": session.session_vitals,
        "variant": session.selected_variant,
        "turns": session.turns,
        "orders": session.orders,
        "disposition": session.disposition,
        "duration_seconds": session.duration_seconds(),
        "ground_truth": session.case["ground_truth_diagnosis"],
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global session
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            # ── clinician sends a message ──────────────────────────────────
            if msg_type == "clinician_message":
                if not session or not session.active:
                    await websocket.send_json({"type": "error", "content": "No active session."})
                    continue

                clinician_text = data.get("text", "").strip()
                if not clinician_text:
                    continue

                # Log clinician turn
                session.add_turn("clinician", clinician_text)

                # ── Detect orders ──────────────────────────────────────────
                new_orders = detect_orders(clinician_text, session.case)
                emr_events = []

                for order_key in new_orders:
                    # Don't duplicate orders already processed
                    already = any(o["type"] == order_key for o in session.orders)
                    if already:
                        continue

                    result = get_oracle_result(order_key, session.case)
                    order_record = {
                        "type": order_key,
                        "label": format_order_key(order_key),
                        "turn_detected": len(session.turns),
                        "result": result["result_text"],
                        "is_abnormal": result["is_abnormal"],
                    }
                    session.orders.append(order_record)
                    emr_events.append(order_record)

                # Send EMR events to UI before patient responds
                if emr_events:
                    await websocket.send_json({
                        "type": "emr_results",
                        "orders": emr_events,
                    })

                # ── Detect disclosures ─────────────────────────────────────
                disclosure = detect_disclosure(clinician_text)
                if disclosure:
                    session.disclosed_to_patient.append(disclosure)

                # ── Detect disposition ─────────────────────────────────────
                if detect_disposition(clinician_text):
                    await websocket.send_json({
                        "type": "disposition_detected",
                        "text": clinician_text,
                    })
                    # Disposition will be confirmed by client sending disposition_confirm
                    # Add clinician message to history but don't get patient response yet
                    session.conversation_history.append({
                        "role": "user",
                        "content": clinician_text,
                    })
                    continue

                # ── Add to conversation history ────────────────────────────
                session.conversation_history.append({
                    "role": "user",
                    "content": clinician_text,
                })

                # ── Stream patient response ────────────────────────────────
                await websocket.send_json({"type": "patient_start"})
                patient_text = await stream_patient_response(session, websocket, model_name)
                await websocket.send_json({"type": "patient_end"})

                # Add patient response to history and log
                session.conversation_history.append({
                    "role": "assistant",
                    "content": patient_text,
                })
                session.add_turn("patient", patient_text)

            # ── Disposition confirmed ──────────────────────────────────────
            elif msg_type == "disposition_confirm":
                if not session:
                    continue
                session.disposition = {
                    "type": data.get("disposition_type", "unspecified"),
                    "clinician_text": data.get("text", ""),
                    "turn_issued": len(session.turns),
                    "timestamp": datetime.now().isoformat(),
                }
                session.active = False

                # Write output files
                output_dir = Path("output") / session.session_id
                output_dir.mkdir(parents=True, exist_ok=True)
                _write_output(session, output_dir)

                await websocket.send_json({
                    "type": "session_closed",
                    "session_id": session.session_id,
                    "output_dir": str(output_dir),
                })

            # ── Disposition cancelled (clinician said "no, not yet") ───────
            elif msg_type == "disposition_cancel":
                await websocket.send_json({"type": "disposition_cancelled"})

    except WebSocketDisconnect:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Output writing
# ─────────────────────────────────────────────────────────────────────────────
def _write_output(s: Session, output_dir: Path):
    # JSON
    json_data = {
        "session_id": s.session_id,
        "timestamp": s.started_at,
        "case_config": s.case,
        "session_vitals": s.session_vitals,
        "selected_variant": s.selected_variant,
        "turns": s.turns,
        "orders": s.orders,
        "disclosed_to_patient": s.disclosed_to_patient,
        "disposition": s.disposition,
        "metadata": {
            "total_turns": len(s.turns),
            "duration_seconds": s.duration_seconds(),
            "model_name": model_name,
            "end_reason": "disposition",
        },
    }
    (output_dir / "session.json").write_text(json.dumps(json_data, indent=2))

    # Plain text
    v = s.session_vitals
    lines = [
        f"=== MedSim Patient Simulator | Session: {s.session_id} | {s.started_at[:16]} ===",
        f"Case: {s.case['case_name']} | Patient: {s.case['patient']['name']}, {s.case['patient']['age']}{s.case['patient']['sex'][0]}",
        f"Vitals: HR {v['hr']} | BP {v['bp_systolic']}/{v['bp_diastolic']} | RR {v['rr']} | Temp {v['temp_c']}°C | SpO2 {v['spo2']}%",
        f"Presentation variant: \"{s.selected_variant}\"",
        "─" * 70,
        "",
    ]
    for t in s.turns:
        speaker = "CLINICIAN" if t["speaker"] == "clinician" else "PATIENT"
        lines.append(f"[Turn {t['turn_number']}] {speaker}: {t['text']}")
        lines.append("")

    if s.orders:
        lines.append("─" * 70)
        lines.append("ORDERS & RESULTS:")
        for o in s.orders:
            flag = " ⚠ ABNORMAL" if o["is_abnormal"] else ""
            lines.append(f"  [{o['label']}]{flag}: {o['result']}")
        lines.append("")

    if s.disposition:
        lines.append("─" * 70)
        lines.append(f"DISPOSITION: {s.disposition.get('type', 'unspecified')}")
        lines.append(f"Clinician said: {s.disposition.get('clinician_text', '')}")

    lines.extend([
        "",
        f"Total turns: {len(s.turns)} | Duration: {s.duration_seconds()}s",
        "=== END OF SESSION ===",
    ])

    (output_dir / "session.txt").write_text("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedSim Patient Simulator")
    parser.add_argument("--case", default="cases/appendicitis.json",
                        help="Path to case JSON file")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Ollama model name (default: gemma3:12b)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to serve on (default: 8000)")
    args = parser.parse_args()

    case_path = Path(args.case)
    if not case_path.exists():
        print(f"ERROR: Case file not found: {case_path}")
        exit(1)

    case_data = json.loads(case_path.read_text())
    model_name = args.model

    print(f"\n🏥 MedSim Patient Simulator")
    print(f"   Case:  {case_data['case_name']}")
    print(f"   Model: {model_name}")
    print(f"   Open:  http://localhost:{args.port}\n")

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
