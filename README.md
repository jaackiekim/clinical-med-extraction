# Clinical Medication Extraction Evaluation

## Research Question

When large language models extract medication names and doses from clinical 
notes, do their errors distribute randomly across drug types, or do they 
concentrate systematically in specific categories? And if errors are 
systematic, what does that mean for the validity of downstream 
pharmacoepidemiology studies built on that extracted data?

## Motivation

Real-world evidence (RWE) generation depends on accurate structured 
extraction from unstructured clinical notes. At major health tech companies, extracted medication information feeds directly into cohort 
definitions, treatment timelines, and drug safety analyses. A model that 
extracts drug names at 90% F1 but dose at 67% F1 on chemotherapy agents 
does not just produce noisy data, it introduces differential 
misclassification that can systematically bias study conclusions.

Prior work (Richter-Pechanski et al. 2025, Shao et al. 2025) benchmarks 
LLMs on medication extraction using aggregate F1 scores. Neither study asks 
whether errors are random or systematic, nor connects extraction failure 
patterns to downstream analytical risk. This project fills that gap.

## Dataset

### Data Access Status
- MIMIC-III: Access request submitted (PhysioNet credentialing complete)
- n2c2 2018 Track 2: Portal currently closed; 
  direct outreach to dataset authors in progress

## Models Evaluated

| Model | Type | Purpose |
|---|---|---|
| Regex + RxNorm | Rule-based baseline | Performance floor, no ML |
| GPT-4o zero-shot | Proprietary LLM | Primary model, zero training data |
| GPT-4o few-shot (5-shot) | Proprietary LLM | Tests in-context learning |
| GPT-4o mini zero-shot | Proprietary LLM | Cost-performance tradeoff |

## Evaluation Framework

### Primary Metric
Lenient micro-averaged F1 on drug-dose pairs, matching the official n2c2 
2018 evaluation protocol. A prediction is correct if the predicted drug 
name and dose are both correct and correctly linked to each other.

### Stratification
Performance broken down across three drug categories:

**Standard oral medications** (cardiovascular: lisinopril, metoprolol, 
atorvastatin) — simple mg dosing, high frequency in training data, 
expected baseline performance

**Oncology and chemotherapy** (carboplatin AUC 5, methotrexate 15mg/m²) 
— non-standard pharmacokinetic units, rare notation, highest clinical 
risk if wrong, directly relevant to RWE in oncology

**PRN medications** (oxycodone 5-10mg q4h PRN, lorazepam 1-2mg PRN) 
— range doses and conditional frequency, structural ambiguity rather 
than notation unfamiliarity

### Error Taxonomy
Each extraction failure is classified as one of:
- Missing dose (drug found, dose not extracted)
- Wrong dose value (numeric error)
- Wrong dose unit (mg vs mg/m² vs AUC)
- Wrong drug-dose linkage (dose extracted but linked to wrong drug)
- Hallucinated dose (dose extracted for drug not in note)

## Key Findings

*To be completed after experiments*

## Comparison to Prior Work

| Study | Models | Drug Name | Dose | Stratification | Downstream Framing |
|---|---|---|---|---|---|
| Richter-Pechanski 2025 | Llama (fine-tuned) | Yes | Yes | No | No |
| Shao et al. 2025 | GPT-4o, Llama, others | Yes | No | No | No |
| This work | GPT-4o, GPT-4o mini, Regex | Yes | Yes | Yes | Yes |

## Repository Structure

```
clinical-med-extraction/
├── data/
│   ├── raw/              # n2c2 and MIMIC files (gitignored, DUA protected)
│   └── processed/        # Cleaned and formatted inputs
├── annotations/          # Gold standard labels
├── src/
│   ├── extraction/       # GPT-4o and baseline extraction pipelines
│   ├── evaluation/       # Lenient F1 scoring, error classification
│   └── baselines/        # Regex + RxNorm baseline
├── notebooks/            # Analysis and figures
└── results/              # Output CSVs, error analysis tables
```

## Setup and Reproduction

```bash
git clone https://github.com/jaackiekim/clinical-med-extraction.git
cd clinical-med-extraction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```
Data access requires:
- PhysioNet credentialing for MIMIC-III 
  (https://physionet.org/content/mimiciii/1.4/)

## References

- Richter-Pechanski et al. (2025). Medication information extraction using 
  local large language models. Journal of Biomedical Informatics.
- Shao et al. (2025). Scalable Medication Extraction and Discontinuation 
  Identification from EHRs Using Large Language Models.
- Henry et al. (2020). 2018 n2c2 shared task on adverse drug events and 
  medication extraction. JAMIA.