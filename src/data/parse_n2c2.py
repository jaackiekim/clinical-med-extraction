import os
import re
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class Span:
    tag_id: str
    entity_type: str
    text: str
    positions: list = field(default_factory=list)

    @property
    def start(self):
        return self.positions[0][0]

    @property
    def end(self):
        return self.positions[-1][1]

    @property
    def is_discontinuous(self):
        return len(self.positions) > 1


@dataclass
class MedicationRecord:
    note_id: str
    drug: Span
    strength: Optional[Span] = None
    dosage: Optional[Span] = None
    route: Optional[Span] = None
    frequency: Optional[Span] = None
    duration: Optional[Span] = None
    form: Optional[Span] = None
    reason: Optional[Span] = None
    ade: Optional[Span] = None

    @property
    def drug_text(self):
        return self.drug.text.lower().strip()

    @property
    def strength_text(self):
        return self.strength.text.lower().strip() if self.strength else None

    def to_dict(self):
        return {
            "note_id": self.note_id,
            "drug": self.drug_text,
            "drug_start": self.drug.start,
            "drug_end": self.drug.end,
            "strength": self.strength_text,
            "strength_start": self.strength.start if self.strength else None,
            "strength_end": self.strength.end if self.strength else None,
            "dosage": self.dosage.text.lower() if self.dosage else None,
            "route": self.route.text.lower() if self.route else None,
            "frequency": self.frequency.text.lower() if self.frequency else None,
            "duration": self.duration.text.lower() if self.duration else None,
            "form": self.form.text.lower() if self.form else None,
            "reason": self.reason.text.lower() if self.reason else None,
            "ade": self.ade.text.lower() if self.ade else None,
        }


def parse_ann_file(ann_path):
    note_id = Path(ann_path).stem
    spans = {}
    relations = []

    with open(ann_path, encoding="latin-1") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line.startswith("T"):
            continue
        try:
            tag_id, tag_meta, tag_text = line.split("\t", 2)
        except ValueError:
            continue
        parts = tag_meta.split(" ")
        entity_type = parts[0]
        position_str = " ".join(parts[1:])
        positions = []
        for segment in position_str.split(";"):
            segment = segment.strip()
            coords = segment.split(" ")
            try:
                start = int(coords[0])
                end = int(coords[-1])
                positions.append((start, end))
            except (ValueError, IndexError):
                continue
        if not positions:
            continue
        spans[tag_id] = Span(tag_id=tag_id, entity_type=entity_type,
                             text=tag_text.strip(), positions=positions)

    for line in lines:
        line = line.strip()
        if not line.startswith("R"):
            continue
        try:
            rel_id, rel_meta = line.split("\t", 1)
        except ValueError:
            continue
        rel_parts = rel_meta.split(" ")
        if len(rel_parts) != 3:
            continue
        rel_type = rel_parts[0]
        arg1_id = rel_parts[1].split(":")[1]
        arg2_id = rel_parts[2].split(":")[1]
        relations.append((rel_type, arg1_id, arg2_id))

    records = {}
    for tag_id, span in spans.items():
        if span.entity_type == "Drug":
            records[tag_id] = MedicationRecord(note_id=note_id, drug=span)

    ATTR_MAP = {
        "Strength-Drug": "strength",
        "Dosage-Drug": "dosage",
        "Route-Drug": "route",
        "Frequency-Drug": "frequency",
        "Duration-Drug": "duration",
        "Form-Drug": "form",
        "Reason-Drug": "reason",
        "ADE-Drug": "ade",
    }

    for rel_type, attr_tag_id, drug_tag_id in relations:
        if rel_type not in ATTR_MAP:
            continue
        if drug_tag_id not in records:
            continue
        if attr_tag_id not in spans:
            continue
        attr_field = ATTR_MAP[rel_type]
        attr_span = spans[attr_tag_id]
        if getattr(records[drug_tag_id], attr_field) is None:
            setattr(records[drug_tag_id], attr_field, attr_span)

    return list(records.values())


def parse_directory(ann_dir):
    ann_dir = Path(ann_dir)
    results = {}
    ann_files = sorted(f for f in ann_dir.glob("*.ann") if not f.name.startswith("."))
    if not ann_files:
        raise FileNotFoundError(f"No .ann files found in {ann_dir}")
    for ann_path in ann_files:
        records = parse_ann_file(str(ann_path))
        results[ann_path.stem] = records
    return results


def print_corpus_stats(corpus):
    total_notes = len(corpus)
    total_drugs = sum(len(recs) for recs in corpus.values())
    with_strength = sum(1 for recs in corpus.values() for r in recs if r.strength is not None)
    with_dosage = sum(1 for recs in corpus.values() for r in recs if r.dosage is not None)
    discontinuous = sum(1 for recs in corpus.values() for r in recs if r.drug.is_discontinuous)
    print(f"Notes:                {total_notes}")
    print(f"Drug mentions:        {total_drugs}")
    print(f"  With Strength:      {with_strength} ({100*with_strength/total_drugs:.1f}%)")
    print(f"  With Dosage:        {with_dosage} ({100*with_dosage/total_drugs:.1f}%)")
    print(f"  Discontinuous span: {discontinuous} ({100*discontinuous/total_drugs:.1f}%)")
    print(f"Drugs w/o Strength:   {total_drugs - with_strength}")


if __name__ == "__main__":
    import json
    test_dir = os.path.expanduser(
        "~/clinical-med-extraction/data/raw/n2c2_2018_track2/test"
    )
    print(f"Parsing {test_dir} ...\n")
    corpus = parse_directory(test_dir)
    print("=== Corpus Statistics ===")
    print_corpus_stats(corpus)
    first_note_id = sorted(corpus.keys())[0]
    print(f"\n=== First 5 records from note {first_note_id} ===")
    for rec in corpus[first_note_id][:5]:
        print(json.dumps(rec.to_dict(), indent=2))
