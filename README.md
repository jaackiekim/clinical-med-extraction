# Clinical Medication Extraction Evaluation

## Research Question

When medication extraction systems process clinical notes, do their errors
distribute randomly across drug types or do they concentrate
systematically in specific categories? And if errors are systematic, what
does that mean for the validity of downstream pharmacoepidemiology studies
built on that extracted data?

## Motivation

Real-world evidence (RWE) generation depends on accurate structured
extraction from unstructured clinical notes. At health tech companies,
extracted medication information feeds directly into cohort definitions,
treatment timelines, and drug safety analyses. A model that extracts drug
names at F1 = 0.74 overall but collapses to F1 = 0.14 on chemotherapy
agents does not just produce noisy data, it introduces differential
misclassification that can systematically bias study conclusions in exactly
the cases where dosing errors carry the most clinical risk.

Prior work (Richter-Pechanski et al. 2025, Shao et al. 2025) benchmarks
LLMs on medication extraction using aggregate F1 scores. Neither study asks
whether errors are random or systematic, nor connects extraction failure
patterns to downstream analytical risk. This project fills that gap.

## Dataset

**n2c2 2018 Track 2** — 303 training notes, 202 test notes with BRAT
annotations for Drug, Strength, Dosage, Route, Frequency, Duration, Form,
Reason, and ADE entities plus relation annotations linking attributes to
drug mentions.

Corpus statistics (test set, 202 notes):

| Statistic | Value |
|---|---|
| Drug mentions | 10,575 |
| With Strength annotation | 4,082 (38.6%) |
| With Dosage annotation | 2,617 (24.7%) |
| Discontinuous spans | 613 (5.8%) |


## Models Evaluated

| Model | Dataset | Type | Status |
|---|---|---|---|
| Regex + RxNorm | n2c2 2018 | Rule-based baseline | Complete |
| GPT-4o (zero-shot) | MTSamples | Proprietary LLM | In progress |
| GPT-4o (5-shot) | MTSamples | Proprietary LLM | In progress |
| BioMistral | n2c2 2018 | Domain-adapted open LLM | Planned |

## Preliminary Results — Regex Baseline (n2c2 2018 Test Set)

Evaluated on 202 notes, 10,575 drug mentions. Lenient matching with fuzzy
drug name normalization (SequenceMatcher >= 0.75).

| Drug Class | Drug P | Drug R | Drug F1 | Strength F1 | N |
|---|---|---|---|---|---|
| Standard | 0.871 | 0.618 | 0.723 | 0.595 | 8,746 |
| Oncology | 0.083 | 0.474 | 0.141 | 0.000 | 152 |
| PRN | 0.598 | 0.708 | 0.648 | 0.489 | 1,677 |
| **Overall** | 0.893 | 0.630 | **0.739** | **0.626** | 10,575 |

### What this shows

The aggregate Drug F1 of 0.739 obscures a near-total failure on oncology
drugs (F1 = 0.141) and a complete failure on oncology strength extraction
(F1 = 0.000). This is the core finding: aggregate metrics hide clinically
meaningful stratified failure patterns.

The oncology result is structurally expected. Regex patterns designed for
standard `drug 500 mg` notation cannot handle AUC-based dosing (carboplatin
AUC 5) or BSA-based dosing (paclitaxel 175 mg/m2), which are the dominant
dosing conventions in chemotherapy. The question this project asks of
GPT-4o and BioMistral is whether LLMs can close that gap and whether
their failures, if any, are distributed differently.

PRN strength F1 of 0.489 reflects a separate structural challenge: range
doses (oxycodone 5-10 mg q4h PRN) require recognizing that the dose is
conditional and expressed as an interval, not a single value.

## Evaluation Framework

### Primary Metric
Lenient micro-averaged F1 on drug mentions and drug-strength pairs.
A drug match is correct if predicted and gold drug names match with fuzzy
normalization. A strength match is correct only for gold records that carry
a Strength annotation (61.4% of drug mentions have no gold strength and are
excluded from strength evaluation).

### Drug Class Taxonomy
- **Standard:** Cardiovascular, anti-infective, and general oral medications
  with standard mg/mcg/g dosing (lisinopril, metoprolol, vancomycin)
- **Oncology:** Chemotherapy agents with AUC-based, BSA-based, or
  weight-based dosing (carboplatin, paclitaxel, cisplatin, methotrexate)
- **PRN:** As-needed medications with range doses and conditional frequency
  (oxycodone PRN, lorazepam PRN, morphine PRN)

### Error Taxonomy
Each extraction failure is classified as:
- Missing drug (drug not found in note)
- Drug found, strength missing
- Drug found, strength value wrong
- Drug found, strength unit wrong (e.g. mg vs mg/m2)
- Hallucinated dose (dose returned for drug not in gold)

## Comparison to Prior Work

| Study | Models | Drug | Dose | Stratified | Downstream Framing |
|---|---|---|---|---|---|
| Richter-Pechanski 2025 | Llama 3.1 70B (fine-tuned) | Yes | Yes | No | No |
| Shao et al. 2025 | GPT-4o, Llama, others | Yes | No | No | No |
| **This work** | GPT-4o, BioMistral, Regex | Yes | Yes | Yes | Yes |

## Repository Structure
```
clinical-med-extraction/
├── data/
│   ├── raw/              # n2c2 and RxNorm files (gitignored, DUA protected)
│   └── processed/        # Cleaned and formatted inputs
├── src/
│   ├── data/             # BRAT annotation parser (parse_n2c2.py)
│   ├── baselines/        # Regex + RxNorm extractor
│   ├── extraction/       # GPT-4o extraction pipeline
│   └── evaluation/       # Stratified F1 runner, error classification
├── notebooks/            # Analysis and figures
└── results/              # Output CSVs, error analysis tables
```

## Setup and Reproduction
```
git clone https://github.com/jaackiekim/clinical-med-extraction.git
cd clinical-med-extraction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Data access requires:
- n2c2 2018 Track 2: Request access via https://portal.dbmi.hms.harvard.edu
- RxNorm: Download RXNCONSO.RRF from https://www.nlm.nih.gov/research/umls/rxnorm/

To run the regex baseline:
```
python3 src/evaluation/run_evaluation.py
```

## References

- Richter-Pechanski et al. (2025). Medication information extraction using
  local large language models. Journal of Biomedical Informatics.
- Shao et al. (2025). Scalable Medication Extraction and Discontinuation
  Identification from EHRs Using Large Language Models.
- Henry et al. (2020). 2018 n2c2 shared task on adverse drug events and
  medication extraction. JAMIA.
