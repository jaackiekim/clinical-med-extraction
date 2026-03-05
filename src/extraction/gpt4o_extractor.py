"""
GPT-4o extraction pipeline for medication name and dose extraction.
Supports zero-shot and few-shot prompting modes.
Outputs standardized drug-dose pairs matching regex baseline schema.
"""

import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── 1. Few-shot examples ──────────────────────────────────────────────────────
# These are synthetic examples covering all three drug classes.
# In production these will be replaced with real n2c2 annotated examples.

# Few-shot examples drawn from n2c2 2018 Track 2 training set (notes
# 101779, 105954, 100187). Training notes only — no test contamination.
FEW_SHOT_EXAMPLES = [
    {
        # n2c2 train note 105954 — standard oral medications, clean list format
        "note": """Medications on Admission:
Benicar 20/12.5 mg once daily
ranitidine 150 mg once daily
simvastatin 20 mg once daily
aspirin 325 mg once daily
Darvocet p.r.n. for abdominal discomfort""",
        "extractions": [
            {"drug": "benicar", "dose": "20/12.5 mg"},
            {"drug": "ranitidine", "dose": "150 mg"},
            {"drug": "simvastatin", "dose": "20 mg"},
            {"drug": "aspirin", "dose": "325 mg"},
            {"drug": "darvocet", "dose": None}
        ]
    },
    {
        # n2c2 train note 101779 — oncology context, BSA-dosed chemo,
        # mixed with supportive medications
        "note": """Medications on Admission:
Amoxicillin-Pot Clavulanate 500-125 mg PO Q8H
Gabapentin 300 mg PO HS
Lorazepam 0.5 mg Tablet PO Q4H
Acyclovir 800 mg PO Q8H
Methadone 30mg PO QAM
Morphine 15 mg PO Q4H prn
Omeprazole 20 mg PO DAILY
Prednisone 20 mg PO daily
Furosemide 40 mg PO DAILY
Acetaminophen 650 mg PO Q4H prn""",
        "extractions": [
            {"drug": "amoxicillin-pot clavulanate", "dose": "500-125 mg"},
            {"drug": "gabapentin", "dose": "300 mg"},
            {"drug": "lorazepam", "dose": "0.5 mg"},
            {"drug": "acyclovir", "dose": "800 mg"},
            {"drug": "methadone", "dose": "30mg"},
            {"drug": "morphine", "dose": "15 mg"},
            {"drug": "omeprazole", "dose": "20 mg"},
            {"drug": "prednisone", "dose": "20 mg"},
            {"drug": "furosemide", "dose": "40 mg"},
            {"drug": "acetaminophen", "dose": "650 mg"}
        ]
    },
    {
        # n2c2 train note 100187 — discharge medications, numbered list format,
        # includes PRN and range doses
        "note": """Discharge Medications:
1. Fluoxetine 10 mg Capsule Sig: Three (3) Capsule PO DAILY
2. Risperidone 1 mg Tablet Sig: Three (3) Tablet PO HS
3. Bupropion 150 mg Tablet Sustained Release Sig: One (1) PO BID
4. Calcium Carbonate 500 mg Tablet PO BID
5. Fluticasone-Salmeterol 250-50 mcg/Dose Disk Inhalation BID
6. Midodrine 5 mg Tablet PO TID
7. Clonazepam 1 mg Tablet PO QHS prn anxiety
8. Oxycodone 5 mg Tablet PO Q4-6H prn pain
9. Acetaminophen 325 mg Tablet PO Q4-6H prn""",
        "extractions": [
            {"drug": "fluoxetine", "dose": "10 mg"},
            {"drug": "risperidone", "dose": "1 mg"},
            {"drug": "bupropion", "dose": "150 mg"},
            {"drug": "calcium carbonate", "dose": "500 mg"},
            {"drug": "fluticasone-salmeterol", "dose": "250-50 mcg/Dose"},
            {"drug": "midodrine", "dose": "5 mg"},
            {"drug": "clonazepam", "dose": "1 mg"},
            {"drug": "oxycodone", "dose": "5 mg"},
            {"drug": "acetaminophen", "dose": "325 mg"}
        ]
    },
    {
        # Allergy-only note — model should return empty list
        "note": """Allergies:
Keflex / Penicillins / Erythromycin Base

History of Present Illness:
Patient presents with shortness of breath. No current medications listed.""",
        "extractions": []
    },
    {
        # Oncology BSA/AUC dosing — the critical hard case
        "note": """Hospital Course:
Patient received Rituximab 375mg/m2 IV on day 1.
Cyclophosphamide 750mg/m2 and Doxorubicin 20mg/m2 given on day 1
of R-CHOP cycle. Dexamethasone 20mg PO given as premedication.""",
        "extractions": [
            {"drug": "rituximab", "dose": "375mg/m2"},
            {"drug": "cyclophosphamide", "dose": "750mg/m2"},
            {"drug": "doxorubicin", "dose": "20mg/m2"},
            {"drug": "dexamethasone", "dose": "20mg"}
        ]
    }
]


# ── 2. Prompt builders ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a clinical NLP system that extracts medication information from clinical notes.

Your task: Extract all (drug_name, dose) pairs from the clinical note provided.

Rules:
- Extract the drug name exactly as written, normalized to lowercase
- Extract the dose as written, including units (mg, mcg, g, AUC, mg/m2, mg/kg)
- For range doses, extract the full range (e.g. "5-10 mg")
- For AUC-based dosing, include "AUC" in the dose (e.g. "AUC 5")
- For BSA-based dosing, include the unit (e.g. "175 mg/m2")
- Do NOT extract drugs mentioned only as allergies or past history
- Do NOT hallucinate doses that are not explicitly stated
- If a drug is mentioned without a dose, include it with dose as null

Respond ONLY with a JSON array. No explanation, no markdown, no code blocks.
Example format: [{"drug": "metformin", "dose": "500 mg"}, {"drug": "aspirin", "dose": null}]
If no medications found, respond with: []"""


def build_zero_shot_prompt(note_text: str) -> list:
    """Zero-shot: system prompt + note only."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Clinical note:\n\n{note_text}"}
    ]


def build_few_shot_prompt(note_text: str) -> list:
    """Few-shot: system prompt + 5 examples + note."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for example in FEW_SHOT_EXAMPLES:
        messages.append({
            "role": "user",
            "content": f"Clinical note:\n\n{example['note']}"
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps(example["extractions"])
        })

    messages.append({
        "role": "user",
        "content": f"Clinical note:\n\n{note_text}"
    })

    return messages


# ── 3. API call with retry ────────────────────────────────────────────────────

def call_gpt4o(messages: list, model: str = "gpt-4o",
               max_retries: int = 3) -> str:
    """Call OpenAI API with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,  # Deterministic output for evaluation
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"API error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


# ── 4. Output parser ──────────────────────────────────────────────────────────

def parse_extractions(raw_response: str, source: str) -> list:
    """
    Parse GPT-4o JSON response into standardized drug-dose pairs.
    Handles malformed JSON gracefully.
    """
    try:
        # Strip markdown code blocks if model ignores instructions
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]

        extractions = json.loads(cleaned)

        results = []
        for item in extractions:
            dose_string = item.get("dose")
            results.append({
                "drug": item.get("drug", "").lower().strip(),
                "dose_string": dose_string,
                "dose_value": None,  # GPT-4o returns full string; parsed in eval
                "dose_unit": None,
                "dose_type": None,
                "source": source
            })
        return results

    except json.JSONDecodeError:
        print(f"Failed to parse response: {raw_response[:100]}")
        return []


# ── 5. Main extraction functions ──────────────────────────────────────────────

def extract_zero_shot(note_text: str, model: str = "gpt-4o") -> list:
    messages = build_zero_shot_prompt(note_text)
    raw = call_gpt4o(messages, model=model)
    return parse_extractions(raw, source=f"{model}_zero_shot")


def extract_few_shot(note_text: str, model: str = "gpt-4o") -> list:
    messages = build_few_shot_prompt(note_text)
    raw = call_gpt4o(messages, model=model)
    return parse_extractions(raw, source=f"{model}_few_shot")


def extract_mini_zero_shot(note_text: str) -> list:
    return extract_zero_shot(note_text, model="gpt-4o-mini")


# ── 6. Smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        "Patient is on metformin 500 mg twice daily and lisinopril 10 mg daily.",
        "Carboplatin AUC 5 administered on day 1 of each cycle.",
        "Methotrexate 15 mg/m2 weekly for rheumatoid arthritis.",
        "Oxycodone 5-10 mg q4-6h PRN pain.",
        "Patient has history of penicillin allergy. No current medications.",
    ]

    print("Testing GPT-4o zero-shot...")
    for text in test_cases:
        print(f"\nInput: {text}")
        results = extract_zero_shot(text)
        for r in results:
            print(f"  -> {r}")

    print("\n\nTesting GPT-4o-mini zero-shot...")
    for text in test_cases[:2]:  # Just first two to save cost
        print(f"\nInput: {text}")
        results = extract_mini_zero_shot(text)
        for r in results:
            print(f"  -> {r}")