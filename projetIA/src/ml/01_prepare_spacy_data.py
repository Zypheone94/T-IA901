import json
from pathlib import Path
import spacy
from spacy.tokens import DocBin

IN_PATH = Path("data/ner/dataset_spacy_generated.json")
OUT_DIR = Path("data/spacy")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = {"DEP", "ARR"}

def _clean_spans(spans):
    """
    Supprime les spans None + garde uniquement des spans non-chevauchants.
    On garde le plus long en priorité (souvent mieux pour les noms composés).
    """
    spans = [s for s in spans if s is not None]
    spans = sorted(spans, key=lambda s: (-(s.end - s.start), s.start))

    kept = []
    used_tokens = set()
    for s in spans:
        tok_range = set(range(s.start, s.end))
        if tok_range & used_tokens:
            continue
        kept.append(s)
        used_tokens |= tok_range

    kept = sorted(kept, key=lambda s: s.start)
    return kept

def convert_split(nlp, samples, out_path: Path):
    db = DocBin(store_user_data=True)

    n_total = 0
    n_skipped = 0
    n_overlap_fixed = 0

    for text, ann in samples:
        n_total += 1
        doc = nlp.make_doc(text)

        raw_spans = []
        for start, end, label in ann.get("entities", []):
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            raw_spans.append(span)

        spans = _clean_spans(raw_spans)

        if len(spans) != len([s for s in raw_spans if s is not None]):
            n_overlap_fixed += 1

        try:
            doc.ents = spans
        except ValueError:
            n_skipped += 1
            continue

        db.add(doc)

    db.to_disk(out_path)
    print(f" écrit: {out_path}")
    print(f"   exemples: {len(samples)} | ajoutés: {len(db)} | skip: {n_skipped} | overlaps/align fix: {n_overlap_fixed}")

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable: {IN_PATH}")

    data = json.loads(IN_PATH.read_text(encoding="utf-8"))

    nlp = spacy.blank("fr")
    for lab in LABELS:
        nlp.vocab.strings.add(lab)

    convert_split(nlp, data["train"], OUT_DIR / "train.spacy")
    convert_split(nlp, data["val"],   OUT_DIR / "dev.spacy")
    convert_split(nlp, data["test"],  OUT_DIR / "test.spacy")

if __name__ == "__main__":
    main()