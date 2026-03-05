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

## Results (n2c2 2018 Test Set, 202 notes, 10,575 drug mentions)

Lenient matching with fuzzy drug name normalization (SequenceMatcher >= 0.75).

### Drug F1 by model and drug class

| Drug Class | Regex | GPT-4o Zero-Shot | GPT-4o Few-Shot |
|---|---|---|---|
| Standard | 0.723 | 0.737 | 0.592 |
| Oncology | 0.141 | 0.354 | 0.223 |
| PRN | 0.648 | 0.846 | 0.786 |
| **Overall** | **0.739** | **0.755** | **0.627** |

### Strength F1 by model and drug class

| Drug Class | Regex | GPT-4o Zero-Shot | GPT-4o Few-Shot |
|---|---|---|---|
| Standard | 0.595 | 0.847 | 0.808 |
| Oncology | 0.000 | 0.276 | 0.213 |
| PRN | 0.489 | 0.861 | 0.844 |
| **Overall** | **0.626** | **0.858** | **0.826** |

### What this shows

Aggregate Drug F1 across models spans a narrow range (0.627-0.755), which
would suggest broadly comparable performance. Stratified evaluation reveals
a different picture: oncology Drug F1 ranges from 0.141 to 0.354, and
oncology Strength F1 ranges from 0.000 to 0.276. No model achieves
acceptable oncology strength extraction. This is the core finding:
aggregate metrics obscure systematic failure concentrated in exactly the
drug class where dosing errors carry the highest clinical risk.

GPT-4o zero-shot substantially improves strength extraction overall
(Str F1: 0.626 -> 0.858) driven by standard and PRN classes where
contextual understanding outperforms regex pattern matching. The oncology
gap persists: GPT-4o zero-shot recovers drug detection (0.141 -> 0.354)
but strength extraction remains poor (0.000 -> 0.276), consistent with
GPT-4o recognizing chemo drug names but failing to reliably normalize
AUC-based and BSA-based dosing notation.

Few-shot prompting degrades recall across all classes (overall Drug R:
0.610 -> 0.460), likely because in-context examples bias extraction toward
structured medication list formatting, causing the model to miss drugs
mentioned in narrative prose sections of discharge notes. Precision
remains high (0.986), but the recall penalty dominates.

BioMistral evaluation in progress.

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
