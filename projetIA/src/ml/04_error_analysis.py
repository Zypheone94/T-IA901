import json
from pathlib import Path
from collections import Counter
import spacy
import matplotlib.pyplot as plt

DATASET_PATH = Path("data/ner/dataset_spacy_generated.json")
MODEL_PATH = Path("models/spacy_ner_dep_arr/model-best")  # ou model-last
OUT_PLOT = Path("report/plots/error_types.png")

LABELS = {"DEP", "ARR"}


def load_test_examples(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["test"]  # list of [text, {"entities":[[start,end,label], ...]}]


def ents_to_set(entities):
    # entities: [[start,end,label], ...]
    return {(s, e, lab) for s, e, lab in entities if lab in LABELS}


def spans_to_set(doc):
    return {(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents if ent.label_ in LABELS}


def overlaps(a, b):
    # a=(s,e,lab), b=(s,e,lab)
    return not (a[1] <= b[0] or b[1] <= a[0])


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable: {DATASET_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}")

    nlp = spacy.load(MODEL_PATH)
    test = load_test_examples(DATASET_PATH)

    counts = Counter()
    total = 0

    for text, ann in test:
        gold = ents_to_set(ann.get("entities", []))
        doc = nlp(text)
        pred = spans_to_set(doc)

        total += 1

        if gold == pred:
            counts["perfect"] += 1
            continue

        # missed: gold not found at all (exact span+label)
        missed = [g for g in gold if g not in pred]

        # spurious: pred not in gold
        spurious = [p for p in pred if p not in gold]

        # wrong label: same span exists but label differs
        wrong_label = 0
        for (gs, ge, glab) in missed:
            for (ps, pe, plab) in pred:
                if gs == ps and ge == pe and glab != plab:
                    wrong_label += 1

        # partial span overlap (overlap but not exact)
        partial_span = 0
        for g in missed:
            for p in pred:
                if overlaps(g, p) and (g[0], g[1]) != (p[0], p[1]):
                    partial_span += 1
                    break

        # We count error types (phrase-level: if present at least once)
        if missed:
            counts["missed"] += 1
        if spurious:
            counts["spurious"] += 1
        if wrong_label > 0:
            counts["wrong_label"] += 1
        if partial_span > 0:
            counts["partial_span"] += 1

    # Plot: rate per type
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)

    keys = ["missed", "wrong_label", "partial_span", "spurious", "perfect"]
    vals = [counts[k] / total for k in keys]

    plt.figure()
    plt.title("Répartition des types d'erreurs (taux) - spaCy NER DEP/ARR")
    plt.ylim(0, 1)
    plt.bar(keys, vals)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=200)
    print(f" Graph sauvegardé : {OUT_PLOT}")
    print("📌 Stats phrases:")
    for k in keys:
        print(f"  - {k}: {counts[k]} / {total} = {counts[k]/total:.3f}")


if __name__ == "__main__":
    main()