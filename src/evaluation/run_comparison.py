"""
Full model comparison table.
Evaluates all five conditions against n2c2 2018 Track 2 test set:
  - Regex baseline
  - GPT-4o zero-shot
  - GPT-4o few-shot
  - BioMistral zero-shot
  - BioMistral few-shot

Reads from existing cache directories. Does not make any API calls.

Usage:
    cd ~/clinical-med-extraction
    python src/evaluation/run_comparison.py

Output:
    Terminal table + results/comparison.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.parse_n2c2 import parse_directory
from src.baselines.regex_extractor import load_rxnorm_drugs, extract_medications
from src.evaluation.run_evaluation import evaluate_note, aggregate_results
from src.evaluation.evaluator import classify_drug

BASE       = Path(__file__).resolve().parents[2]
TEST_DIR   = BASE / "data/raw/n2c2_2018_track2/test"
RXNCONSO   = BASE / "data/raw/rxnorm/rrf/RXNCONSO.RRF"
GPT4O_DIR  = BASE / "data/processed/gpt4o_cache"
BM_DIR     = BASE / "data/processed/biomistral_cache"
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_cache(cache_dir, note_id, mode):
    path = cache_dir / f"{note_id}_{mode}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def run_cached_model(label, cache_dir, mode, corpus):
    """Evaluate a model whose predictions are already cached."""
    all_results, all_fp = [], []
    missing = 0

    for note_id, gold_records in sorted(corpus.items()):
        txt_path = TEST_DIR / f"{note_id}.txt"
        if not txt_path.exists():
            continue

        pred_results = load_cache(cache_dir, note_id, mode)
        if pred_results is None:
            # Missing cache entry — treat as empty prediction (counts as FN)
            pred_results = []
            missing += 1

        note_text = txt_path.read_text(encoding="latin-1")
        note_results, fp = evaluate_note(gold_records, pred_results, note_text)
        all_results.append(note_results)
        all_fp.append(fp)

    if missing > 0:
        print(f"  WARNING [{label}]: {missing} notes had no cache entry (treated as empty)")

    return all_results, all_fp


def run_regex(corpus):
    """Run regex baseline live (no cache — it's fast)."""
    print("  Loading RxNorm vocabulary...")
    vocab = load_rxnorm_drugs(str(RXNCONSO))
    print(f"  Loaded {len(vocab):,} drug names")

    all_results, all_fp = [], []
    for note_id, gold_records in sorted(corpus.items()):
        txt_path = TEST_DIR / f"{note_id}.txt"
        if not txt_path.exists():
            continue
        note_text = txt_path.read_text(encoding="latin-1")
        pred_results = extract_medications(note_text, vocab)
        note_results, fp = evaluate_note(gold_records, pred_results, note_text)
        all_results.append(note_results)
        all_fp.append(fp)
    return all_results, all_fp


def compute_metrics(all_results, all_fp):
    """
    Compute precision, recall, F1 per drug class.
    Returns dict: {class: {precision, recall, f1, n}}
    """
    classes = ["standard_oral", "oncology", "prn", "other", "overall"]
    metrics = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}

    for note_results, fp_list in zip(all_results, all_fp):
        for r in note_results:
            cls = r["drug_class"]
            for c in [cls, "overall"]:
                if r["drug_found"]:
                    metrics[c]["tp"] += 1
                else:
                    metrics[c]["fn"] += 1
        for fp_drug in fp_list:
            fp_cls = classify_drug(fp_drug)
            metrics[fp_cls]["fp"] += 1
            metrics["overall"]["fp"] += 1

    out = {}
    for cls in classes:
        tp = metrics[cls]["tp"]
        fp = metrics[cls]["fp"]
        fn = metrics[cls]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        n = tp + fn
        out[cls] = {"precision": p, "recall": r, "f1": f1, "n": n}
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Parsing n2c2 gold annotations...")
    corpus = parse_directory(str(TEST_DIR))
    print(f"Loaded {len(corpus)} notes\n")

    # Define all conditions to evaluate
    # Each entry: (display_label, callable that returns (all_results, all_fp))
    conditions = [
        ("Regex baseline",      lambda: run_regex(corpus)),
        ("GPT-4o zero-shot",    lambda: run_cached_model("GPT-4o zero-shot", GPT4O_DIR, "zero_shot", corpus)),
        ("GPT-4o few-shot",     lambda: run_cached_model("GPT-4o few-shot",  GPT4O_DIR, "few_shot",  corpus)),
        ("BioMistral zero-shot",lambda: run_cached_model("BioMistral zero-shot", BM_DIR, "zero_shot", corpus)),
        ("BioMistral few-shot", lambda: run_cached_model("BioMistral few-shot",  BM_DIR, "few_shot",  corpus)),
    ]

    # Check which caches exist upfront — warn clearly if BioMistral not ready
    if not BM_DIR.exists() or not any(BM_DIR.glob("*.json")):
        print("WARNING: BioMistral cache not found at data/processed/biomistral_cache/")
        print("         Run Colab notebook first, then:")
        print("         mkdir -p data/processed/biomistral_cache/")
        print("         unzip ~/Downloads/biomistral_cache.zip -d data/processed/biomistral_cache/")
        print("         Skipping BioMistral conditions for now.\n")
        conditions = conditions[:3]  # run only regex + GPT-4o

    all_metrics = {}
    classes = ["overall", "standard_oral", "oncology", "prn", "other"]

    for label, runner in conditions:
        print(f"Running: {label}...")
        results, fp = runner()
        all_metrics[label] = compute_metrics(results, fp)
        print(f"  Done.\n")

    # ── Print comparison table ────────────────────────────────────────────────
    col_w = 22
    metric_labels = ["P", "R", "F1"]

    print("\n" + "=" * 110)
    print("FULL MODEL COMPARISON — Drug F1 by subgroup")
    print("=" * 110)

    for cls in classes:
        n = list(all_metrics.values())[0][cls]["n"]
        print(f"\n{cls.upper()} (N={n})")
        print(f"  {'Model':<24} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'-'*56}")
        for label, metrics in all_metrics.items():
            m = metrics[cls]
            print(f"  {label:<24} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f}")

    print("\n" + "=" * 110)

    # ── Save to JSON ──────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Key findings summary ──────────────────────────────────────────────────
    print("\nKEY FINDINGS:")
    if "GPT-4o zero-shot" in all_metrics and "GPT-4o few-shot" in all_metrics:
        zs_onc = all_metrics["GPT-4o zero-shot"]["oncology"]["recall"]
        fs_onc = all_metrics["GPT-4o few-shot"]["oncology"]["recall"]
        print(f"  GPT-4o oncology recall: zero-shot={zs_onc:.3f}, few-shot={fs_onc:.3f} "
              f"(delta={fs_onc-zs_onc:+.3f})")
    if "Regex baseline" in all_metrics:
        reg_onc_r = all_metrics["Regex baseline"]["oncology"]["recall"]
        reg_onc_p = all_metrics["Regex baseline"]["oncology"]["precision"]
        print(f"  Regex oncology: precision={reg_onc_p:.3f}, recall={reg_onc_r:.3f} "
              f"(high precision / low recall pattern)")
    if "BioMistral zero-shot" in all_metrics:
        bm_zs = all_metrics["BioMistral zero-shot"]["overall"]["f1"]
        gpt_zs = all_metrics["GPT-4o zero-shot"]["overall"]["f1"]
        print(f"  Overall F1: GPT-4o zero-shot={gpt_zs:.3f}, BioMistral zero-shot={bm_zs:.3f} "
              f"(delta={bm_zs-gpt_zs:+.3f})")


if __name__ == "__main__":
    main()
