"""
GPT-4o evaluation runner for medication extraction.
Runs zero-shot and few-shot on n2c2 2018 test set.
Produces stratified F1 results matching regex baseline format.
Caches API responses to avoid re-running on interruption.
"""

import json
import time
from pathlib import Path
import sys
sys.path.insert(0, '/Users/jackiekim/clinical-med-extraction')

from src.data.parse_n2c2 import parse_directory
from src.extraction.gpt4o_extractor import extract_zero_shot, extract_few_shot
from src.evaluation.run_evaluation import evaluate_note, aggregate_results

TEST_DIR  = Path("data/raw/n2c2_2018_track2/test")
CACHE_DIR = Path("data/processed/gpt4o_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_cache(note_id: str, mode: str):
    path = CACHE_DIR / f"{note_id}_{mode}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_cache(note_id: str, mode: str, results: list):
    path = CACHE_DIR / f"{note_id}_{mode}.json"
    path.write_text(json.dumps(results))


def run_mode(mode: str, corpus: dict):
    extractor = extract_zero_shot if mode == "zero_shot" else extract_few_shot
    all_results, all_fp = [], []
    skipped, cached, called = 0, 0, 0

    note_ids = sorted(corpus.keys())
    total = len(note_ids)

    for i, note_id in enumerate(note_ids):
        gold_records = corpus[note_id]
        txt_path = TEST_DIR / f"{note_id}.txt"

        if not txt_path.exists():
            skipped += 1
            continue

        # Check cache first
        cached_result = load_cache(note_id, mode)
        if cached_result is not None:
            pred_results = cached_result
            cached += 1
        else:
            note_text = txt_path.read_text(encoding="latin-1")
            try:
                pred_results = extractor(note_text)
                save_cache(note_id, mode, pred_results)
                called += 1
                # Polite rate limiting
                time.sleep(0.5)
            except Exception as e:
                print(f"  ERROR on {note_id}: {e}")
                pred_results = []

        note_text = txt_path.read_text(encoding="latin-1")
        note_results, fp = evaluate_note(gold_records, pred_results, note_text)
        all_results.append(note_results)
        all_fp.append(fp)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{total} notes processed "
                  f"(cached: {cached}, API calls: {called})")

    print(f"\nDone: {len(all_results)} notes "
          f"({skipped} skipped, {cached} cached, {called} API calls)")
    print(f"\n=== GPT-4o {mode.upper()} RESULTS ===")
    aggregate_results(all_results, all_fp)


if __name__ == "__main__":
    print("Parsing n2c2 gold annotations...")
    corpus = parse_directory(str(TEST_DIR))
    print(f"Loaded {len(corpus)} notes\n")

    print("=" * 50)
    print("MODE: ZERO-SHOT")
    print("=" * 50)
    run_mode("zero_shot", corpus)

    print("\n")
    print("=" * 50)
    print("MODE: FEW-SHOT")
    print("=" * 50)
    run_mode("few_shot", corpus)
