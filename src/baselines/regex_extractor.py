"""
Regex + RxNorm baseline extractor for medication name and dose extraction.
Serves as the rule-based performance floor for comparison against LLM models.
"""

import re
import csv
from pathlib import Path


# ── 1. Load RxNorm drug vocabulary ──────────────────────────────────────────

def load_rxnorm_drugs(rxnconso_path: str) -> set:
    """
    Extract all unique drug names from RXNCONSO.RRF.
    We keep only rows where the source (column 11) is RXNORM
    and term type (column 12) is IN (ingredient) or BN (brand name).
    This filters ~244k rows down to the clinically relevant subset.
    """
    drug_names = set()
    with open(rxnconso_path, "r", encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("|")
            if len(cols) < 15:
                continue
            source = cols[11]   # SAB column
            term_type = cols[12]  # TTY column
            name = cols[14].lower().strip()  # STR column
            if source == "RXNORM" and term_type in ("IN", "BN", "PIN"):
                if len(name) > 2:  # filter out noise
                    drug_names.add(name)
    return drug_names


# ── 2. Dose patterns ─────────────────────────────────────────────────────────

# Standard dose: 500 mg, 10.5 mcg, 2 g, 20 mEq
STANDARD_DOSE = re.compile(
    r'\b(\d+\.?\d*)\s*(mg|mcg|microgram|g|gram|mEq|meq|units?|IU|mmol)\b',
    re.IGNORECASE
)

# Range dose (PRN): 5-10 mg, 1-2 mg
RANGE_DOSE = re.compile(
    r'\b(\d+\.?\d*\s*[-–]\s*\d+\.?\d*)\s*(mg|mcg|g|units?)\b',
    re.IGNORECASE
)

# BSA-based dosing (oncology): 15 mg/m2, 75 mg/m²
BSA_DOSE = re.compile(
    r'\b(\d+\.?\d*)\s*(mg|mcg|g)\s*/\s*m[2²]\b',
    re.IGNORECASE
)

# AUC-based dosing (oncology): AUC 5, AUC 6, AUC5
AUC_DOSE = re.compile(
    r'\bAUC\s*(\d+\.?\d*)\b',
    re.IGNORECASE
)

# Weight-based dosing: 1.5 mg/kg
WEIGHT_DOSE = re.compile(
    r'\b(\d+\.?\d*)\s*(mg|mcg|g)\s*/\s*kg\b',
    re.IGNORECASE
)


def extract_doses(text: str) -> list:
    """Extract all dose mentions from text with their type."""
    doses = []

    for m in AUC_DOSE.finditer(text):
        doses.append({
            "dose_string": m.group(0),
            "dose_value": m.group(1),
            "dose_unit": "AUC",
            "dose_type": "auc",
            "span": m.span()
        })

    for m in BSA_DOSE.finditer(text):
        doses.append({
            "dose_string": m.group(0),
            "dose_value": m.group(1),
            "dose_unit": f"{m.group(2)}/m2",
            "dose_type": "bsa",
            "span": m.span()
        })

    for m in WEIGHT_DOSE.finditer(text):
        doses.append({
            "dose_string": m.group(0),
            "dose_value": m.group(1),
            "dose_unit": f"{m.group(2)}/kg",
            "dose_type": "weight_based",
            "span": m.span()
        })

    for m in RANGE_DOSE.finditer(text):
        doses.append({
            "dose_string": m.group(0),
            "dose_value": m.group(1),
            "dose_unit": m.group(2),
            "dose_type": "range",
            "span": m.span()
        })

    for m in STANDARD_DOSE.finditer(text):
        doses.append({
            "dose_string": m.group(0),
            "dose_value": m.group(1),
            "dose_unit": m.group(2),
            "dose_type": "standard",
            "span": m.span()
        })

    return doses


# ── 3. Drug name matcher ──────────────────────────────────────────────────────

def extract_drug_names(text: str, drug_vocab: set) -> list:
    """
    Find drug name mentions by scanning text tokens against RxNorm vocab.
    O(n) in text length rather than O(vocab_size * text_length).
    Handles 1, 2, and 3-token drug names (e.g. tylenol with codeine).
    """
    found = []
    text_lower = text.lower()
    tokens = []
    for chunk in re.finditer(r'\S+', text_lower):
        tokens.append((chunk.group(), chunk.start(), chunk.end()))

    for i in range(len(tokens)):
        for n in (3, 2, 1):
            if i + n > len(tokens):
                continue
            phrase = " ".join(t[0] for t in tokens[i:i+n])
            if phrase in drug_vocab:
                phrase_end = tokens[i+n-1][2]
                found.append({"drug": phrase, "span": (tokens[i][1], phrase_end)})
                break

    return found


# ── 4. Drug-dose linker ───────────────────────────────────────────────────────

def link_drug_dose(drug_mentions: list, dose_mentions: list,
                   window: int = 100) -> list:
    """
    Link each drug mention to the nearest dose mention within a character window.
    Clinical notes typically express dose immediately after drug name.
    Window of 100 chars captures 'metformin 500 mg twice daily' patterns.
    """
    results = []

    for drug in drug_mentions:
        drug_end = drug["span"][1]
        best_dose = None
        best_distance = window + 1

        for dose in dose_mentions:
            dose_start = dose["span"][0]
            # Dose should appear AFTER drug name, within window
            distance = dose_start - drug_end
            if 0 <= distance <= window:
                if distance < best_distance:
                    best_distance = distance
                    best_dose = dose

        results.append({
            "drug": drug["drug"],
            "dose_string": best_dose["dose_string"] if best_dose else None,
            "dose_value": best_dose["dose_value"] if best_dose else None,
            "dose_unit": best_dose["dose_unit"] if best_dose else None,
            "dose_type": best_dose["dose_type"] if best_dose else None,
            "source": "regex"
        })

    return results


# ── 5. Main extraction function ───────────────────────────────────────────────

def extract_medications(text: str, drug_vocab: set) -> list:
    """
    Full pipeline: find drugs, find doses, link them.
    Returns list of standardized drug-dose pairs.
    """
    drug_mentions = extract_drug_names(text, drug_vocab)
    dose_mentions = extract_doses(text)
    return link_drug_dose(drug_mentions, dose_mentions)


# ── 6. Quick smoke test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    RXNCONSO = "data/raw/rxnorm/rrf/RXNCONSO.RRF"

    print("Loading RxNorm vocabulary...")
    vocab = load_rxnorm_drugs(RXNCONSO)
    print(f"Loaded {len(vocab):,} drug names")

    # Test sentences covering all three drug classes
    test_cases = [
        "Patient is on metformin 500 mg twice daily and lisinopril 10 mg daily.",
        "Carboplatin AUC 5 administered on day 1 of each cycle.",
        "Methotrexate 15 mg/m2 weekly for rheumatoid arthritis.",
        "Oxycodone 5-10 mg q4-6h PRN pain.",
        "Vancomycin 1.25 g IV q8h, renally adjusted.",
    ]

    for text in test_cases:
        print(f"\nInput: {text}")
        results = extract_medications(text, vocab)
        for r in results:
            print(f"  -> {r}")