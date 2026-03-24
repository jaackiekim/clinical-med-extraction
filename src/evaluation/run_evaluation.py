"""
Evaluation runner: regex baseline against n2c2 2018 Track 2 test set.
Produces drug-name F1 and drug+strength F1, stratified by drug class.
"""

import sys
import os
import json
from pathlib import Path
from difflib import SequenceMatcher

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.parse_n2c2 import parse_directory
from src.baselines.regex_extractor import load_rxnorm_drugs, extract_medications

# ── Drug class taxonomy ───────────────────────────────────────────────────────

from src.evaluation.evaluator import classify_drug


# ── Matching logic ────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    if text is None:
        return ""
    return " ".join(text.lower().strip().split())

def fuzzy_match(a: str, b: str, threshold: float = 0.75) -> bool:
    if not a or not b:
        return False
    a, b = normalize(a), normalize(b)
    if a == b:
        return True
    return SequenceMatcher(None, a, b).ratio() >= threshold

def drug_matches(pred_drug: str, gold_drug: str) -> bool:
    return fuzzy_match(pred_drug, gold_drug)

def strength_matches(pred_strength, gold_strength) -> bool:
    if gold_strength is None:
        return True  # no gold strength = not evaluated
    if pred_strength is None:
        return False
    return fuzzy_match(pred_strength, gold_strength)


# ── Per-note evaluation ───────────────────────────────────────────────────────

def evaluate_note(gold_records, pred_results, note_text):
    """
    For each gold drug mention, check if extractor found it (drug F1).
    For gold mentions WITH strength, also check if strength was correct.
    Returns per-record match results with drug class labels.
    """
    results = []

    for gold in gold_records:
        gold_drug = normalize(gold.drug_text)
        gold_strength = normalize(gold.strength_text) if gold.strength else None
        drug_class = classify_drug(gold_drug, note_text)

        # Find best matching prediction
        best_pred = None
        for pred in pred_results:
            if drug_matches(pred["drug"], gold_drug):
                best_pred = pred
                break

        drug_tp = 1 if best_pred is not None else 0

        # Strength evaluation only for gold records that have strength
        if gold_strength:
            pred_strength = normalize(best_pred.get("dose_string")) if best_pred else None
            strength_tp = 1 if (best_pred and strength_matches(pred_strength, gold_strength)) else 0
        else:
            strength_tp = None  # not evaluated

        results.append({
            "gold_drug": gold_drug,
            "gold_strength": gold_strength,
            "drug_class": drug_class,
            "drug_found": drug_tp,
            "strength_correct": strength_tp,
        })

    # Collect false positives: predictions with no matching gold
    fp_drugs = []
    for pred in pred_results:
        matched = any(drug_matches(pred["drug"], normalize(g.drug_text)) for g in gold_records)
        if not matched:
            fp_drugs.append(pred["drug"])

    return results, fp_drugs


# ── Aggregate metrics ─────────────────────────────────────────────────────────

def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def aggregate_results(all_results, all_fp):
    classes = ["standard_oral", "oncology", "prn", "other", "overall"]
    metrics = {c: {"drug_tp": 0, "drug_fp": 0, "drug_fn": 0,
                   "str_tp": 0, "str_fp": 0, "str_fn": 0} for c in classes}

    for note_results, fp in zip(all_results, all_fp):
        for r in note_results:
            cls = r["drug_class"]
            for c in [cls, "overall"]:
                if r["drug_found"]:
                    metrics[c]["drug_tp"] += 1
                else:
                    metrics[c]["drug_fn"] += 1
                if r["strength_correct"] is not None:
                    if r["strength_correct"]:
                        metrics[c]["str_tp"] += 1
                    else:
                        metrics[c]["str_fn"] += 1
        for fp_drug in fp:
            fp_cls = classify_drug(fp_drug)
            metrics[fp_cls]["drug_fp"] += 1
            metrics["overall"]["drug_fp"] += 1

    print(f"\n{'='*65}")
    print(f"{'Drug Class':<12} {'Drug P':>7} {'Drug R':>7} {'Drug F1':>8} {'Str F1':>8} {'N':>6}")
    print(f"{'-'*65}")

    for cls in classes:
        m = metrics[cls]
        dp, dr, df = compute_f1(m["drug_tp"], m["drug_fp"], m["drug_fn"])
        sp, sr, sf = compute_f1(m["str_tp"], m["drug_fp"], m["str_fn"])
        n = m["drug_tp"] + m["drug_fn"]
        print(f"{cls:<12} {dp:>7.3f} {dr:>7.3f} {df:>8.3f} {sf:>8.3f} {n:>6}")

    print(f"{'='*65}")
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BASE = Path(__file__).resolve().parents[2]
    TEST_DIR = BASE / "data/raw/n2c2_2018_track2/test"
    RXNCONSO = BASE / "data/raw/rxnorm/rrf/RXNCONSO.RRF"

    print("Loading RxNorm vocabulary...")
    vocab = load_rxnorm_drugs(str(RXNCONSO))
    print(f"Loaded {len(vocab):,} drug names")

    print("Parsing n2c2 gold annotations...")
    corpus = parse_directory(str(TEST_DIR))
    print(f"Loaded {len(corpus)} notes")

    all_results = []
    all_fp = []
    skipped = 0

    for note_id, gold_records in sorted(corpus.items()):
        txt_path = TEST_DIR / f"{note_id}.txt"
        if not txt_path.exists():
            skipped += 1
            continue

        with open(txt_path, encoding="latin-1") as f:
            note_text = f.read()

        pred_results = extract_medications(note_text, vocab)
        note_results, fp = evaluate_note(gold_records, pred_results, note_text)
        all_results.append(note_results)
        all_fp.append(fp)

    print(f"\nEvaluated {len(all_results)} notes ({skipped} skipped)")
    print("\n=== REGEX BASELINE RESULTS ===")
    aggregate_results(all_results, all_fp)
