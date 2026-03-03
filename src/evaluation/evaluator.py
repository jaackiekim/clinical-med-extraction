"""
Evaluation framework for medication extraction.
Computes lenient micro-averaged F1 on drug-dose pairs.
Stratifies performance by drug class and error taxonomy.
"""

import re
import json
import pandas as pd
from pathlib import Path


# ── 1. Drug class taxonomy ────────────────────────────────────────────────────

DRUG_CLASSES = {
    "oncology": {
        "carboplatin", "cisplatin", "oxaliplatin", "paclitaxel", "docetaxel",
        "doxorubicin", "cyclophosphamide", "methotrexate", "fluorouracil",
        "gemcitabine", "vincristine", "vinblastine", "etoposide", "irinotecan",
        "topotecan", "capecitabine", "temozolomide", "bevacizumab", "rituximab",
        "trastuzumab", "pembrolizumab", "nivolumab", "atezolizumab", "letrozole",
        "anastrozole", "tamoxifen", "leuprolide", "bicalutamide"
    },
    "standard_oral": {
        "metformin", "lisinopril", "atorvastatin", "simvastatin", "amlodipine",
        "metoprolol", "carvedilol", "losartan", "hydrochlorothiazide", "furosemide",
        "aspirin", "clopidogrel", "warfarin", "apixaban", "rivaroxaban",
        "levothyroxine", "omeprazole", "pantoprazole", "sertraline", "escitalopram",
        "fluoxetine", "amoxicillin", "azithromycin", "prednisone", "gabapentin",
        "tramadol", "albuterol", "montelukast", "insulin", "glipizide"
    },
    "prn": {
        "oxycodone", "hydrocodone", "morphine", "fentanyl", "hydromorphone",
        "lorazepam", "diazepam", "alprazolam", "zolpidem", "quetiapine",
        "haloperidol", "ondansetron", "promethazine", "diphenhydramine",
        "acetaminophen", "ibuprofen", "ketorolac", "nitroglycerin"
    }
}


def classify_drug(drug_name: str) -> str:
    """Assign drug to class. Returns 'other' if not in taxonomy."""
    name = drug_name.lower().strip()
    for cls, drugs in DRUG_CLASSES.items():
        if name in drugs:
            return cls
    return "other"


# ── 2. Lenient matching ───────────────────────────────────────────────────────

def normalize_drug(name: str) -> str:
    """Normalize drug name for lenient matching."""
    return name.lower().strip()


def normalize_dose(dose: str) -> str:
    """
    Normalize dose string for lenient matching.
    Handles spacing variations: '500mg' == '500 mg'
    Handles case: 'AUC 5' == 'auc 5'
    """
    if dose is None:
        return None
    dose = dose.lower().strip()
    # Remove spaces between number and unit: '500 mg' -> '500mg'
    dose = re.sub(r'(\d)\s+(mg|mcg|g|auc|meq|units?)', r'\1\2', dose)
    return dose


def doses_match(pred_dose: str, gold_dose: str) -> bool:
    """
    Lenient dose matching.
    Exact match after normalization.
    Could be extended to allow ±10% numeric tolerance.
    """
    if pred_dose is None and gold_dose is None:
        return True
    if pred_dose is None or gold_dose is None:
        return False
    return normalize_dose(pred_dose) == normalize_dose(gold_dose)


def drugs_match(pred_drug: str, gold_drug: str) -> bool:
    """Lenient drug name matching after normalization."""
    return normalize_drug(pred_drug) == normalize_drug(gold_drug)


# ── 3. Error taxonomy ─────────────────────────────────────────────────────────

def classify_error(pred: dict, gold: dict, match_type: str) -> str:
    """
    Classify extraction error type.
    match_type: 'tp', 'fp', 'fn'
    """
    if match_type == "tp":
        return "correct"

    if match_type == "fn":
        return "missing_drug"

    if match_type == "fp":
        # Check if drug exists in gold but dose is wrong
        gold_drugs = {normalize_drug(g["drug"]) for g in gold}
        pred_drug_norm = normalize_drug(pred["drug"])

        if pred_drug_norm in gold_drugs:
            # Drug found, but dose didn't match
            gold_for_drug = [g for g in gold
                           if normalize_drug(g["drug"]) == pred_drug_norm]
            if gold_for_drug:
                gold_dose = gold_for_drug[0].get("dose_string")
                pred_dose = pred.get("dose_string")
                if pred_dose is None and gold_dose is not None:
                    return "missing_dose"
                elif pred_dose is not None and gold_dose is None:
                    return "hallucinated_dose"
                else:
                    # Both present but don't match - check if unit wrong
                    if gold_dose and pred_dose:
                        gold_unit = re.findall(
                            r'(mg/m2|mg/kg|auc|mg|mcg|g)', 
                            gold_dose.lower()
                        )
                        pred_unit = re.findall(
                            r'(mg/m2|mg/kg|auc|mg|mcg|g)',
                            pred_dose.lower()
                        )
                        if gold_unit and pred_unit and gold_unit != pred_unit:
                            return "wrong_dose_unit"
                    return "wrong_dose_value"
        else:
            return "hallucinated_drug"

    return "unknown"


# ── 4. Core F1 computation ────────────────────────────────────────────────────

def compute_f1(predictions: list, gold_standard: list) -> dict:
    """
    Compute lenient micro-averaged F1 on drug-dose pairs.
    A prediction is a TP if both drug name and dose match a gold annotation.

    Args:
        predictions: list of {"drug": str, "dose_string": str} dicts
        gold_standard: list of {"drug": str, "dose_string": str} dicts

    Returns:
        dict with precision, recall, f1, tp, fp, fn counts
    """
    gold_matched = [False] * len(gold_standard)
    tp, fp = 0, 0
    errors = []

    for pred in predictions:
        matched = False
        for i, gold in enumerate(gold_standard):
            if not gold_matched[i]:
                if (drugs_match(pred["drug"], gold["drug"]) and
                        doses_match(pred.get("dose_string"),
                                   gold.get("dose_string"))):
                    tp += 1
                    gold_matched[i] = True
                    matched = True
                    break

        if not matched:
            fp += 1
            error_type = classify_error(pred, gold_standard, "fp")
            errors.append({
                "type": error_type,
                "pred_drug": pred["drug"],
                "pred_dose": pred.get("dose_string"),
                "drug_class": classify_drug(pred["drug"])
            })

    fn = sum(1 for m in gold_matched if not m)
    for i, matched in enumerate(gold_matched):
        if not matched:
            errors.append({
                "type": "missing_drug",
                "gold_drug": gold_standard[i]["drug"],
                "gold_dose": gold_standard[i].get("dose_string"),
                "drug_class": classify_drug(gold_standard[i]["drug"])
            })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "errors": errors
    }


# ── 5. Stratified evaluation ──────────────────────────────────────────────────

def evaluate_stratified(predictions: list, gold_standard: list) -> dict:
    """
    Compute F1 overall and broken down by drug class.
    This is your core research contribution — not just aggregate F1
    but where failures concentrate.
    """
    results = {
        "overall": compute_f1(predictions, gold_standard)
    }

    for drug_class in ["oncology", "standard_oral", "prn", "other"]:
        class_preds = [p for p in predictions
                      if classify_drug(p["drug"]) == drug_class]
        class_gold = [g for g in gold_standard
                     if classify_drug(g["drug"]) == drug_class]

        if class_gold:
            results[drug_class] = compute_f1(class_preds, class_gold)
        else:
            results[drug_class] = None

    return results


# ── 6. Smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate predictions vs gold standard
    gold = [
        {"drug": "metformin", "dose_string": "500 mg"},
        {"drug": "lisinopril", "dose_string": "10 mg"},
        {"drug": "carboplatin", "dose_string": "AUC 5"},
        {"drug": "methotrexate", "dose_string": "15 mg/m2"},
        {"drug": "oxycodone", "dose_string": "5-10 mg"},
    ]

    # Perfect predictions
    perfect_preds = gold.copy()

    # Imperfect predictions - simulating model errors
    imperfect_preds = [
        {"drug": "metformin", "dose_string": "500 mg"},      # correct
        {"drug": "lisinopril", "dose_string": "20 mg"},      # wrong dose value
        {"drug": "carboplatin", "dose_string": "AUC 5"},     # correct
        {"drug": "methotrexate", "dose_string": "15 mg"},    # wrong unit (mg not mg/m2)
        # oxycodone missing entirely                          # missing drug
        {"drug": "aspirin", "dose_string": "81 mg"},         # hallucinated drug
    ]

    print("=== Perfect predictions ===")
    results = evaluate_stratified(perfect_preds, gold)
    for cls, metrics in results.items():
        if metrics:
            print(f"{cls}: F1={metrics['f1']:.3f} "
                  f"P={metrics['precision']:.3f} "
                  f"R={metrics['recall']:.3f}")

    print("\n=== Imperfect predictions ===")
    results = evaluate_stratified(imperfect_preds, gold)
    for cls, metrics in results.items():
        if metrics:
            print(f"{cls}: F1={metrics['f1']:.3f} "
                  f"P={metrics['precision']:.3f} "
                  f"R={metrics['recall']:.3f}")
            if metrics.get("errors"):
                for e in metrics["errors"]:
                    print(f"  ERROR: {e}")