import json
from pathlib import Path
from collections import defaultdict
import spacy
import matplotlib.pyplot as plt

DATASET_PATH = Path("data/ner/dataset_spacy_generated.json")
MODEL_PATH = Path("models/spacy_ner_dep_arr/model-best")  # ou model-last
OUT_PLOT = Path("report/plots/accuracy_by_length.png")

LABELS = {"DEP", "ARR"}

BUCKETS = [
    (0, 30, "0-30"),
    (31, 60, "31-60"),
    (61, 90, "61-90"),
    (91, 120, "91-120"),
    (121, 10**9, "120+"),
]


def bucket_of(n):
    for a, b, name in BUCKETS:
        if a <= n <= b:
            return name
    return "unknown"


def ents_to_set(entities):
    return {(s, e, lab) for s, e, lab in entities if lab in LABELS}


def pred_to_set(doc):
    return {(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents if ent.label_ in LABELS}


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable: {DATASET_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}")

    nlp = spacy.load(MODEL_PATH)
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    test = data["test"]

    # stats per bucket only for "trip" (gold has 2 entities DEP+ARR)
    stats = defaultdict(lambda: {"n": 0, "correct": 0})

    for text, ann in test:
        gold = ann.get("entities", [])
        gold_set = ents_to_set(gold)

        # on ne calcule l'accuracy que si on a DEP+ARR (2 entités)
        if len(gold_set) != 2:
            continue

        L = len(text)
        b = bucket_of(L)

        doc = nlp(text)
        pred_set = pred_to_set(doc)

        stats[b]["n"] += 1
        if pred_set == gold_set:
            stats[b]["correct"] += 1

    # plot
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)

    labels = [name for _, _, name in BUCKETS if name in stats]
    accs = [(stats[name]["correct"] / stats[name]["n"]) if stats[name]["n"] > 0 else 0 for name in labels]

    plt.figure()
    plt.title("Accuracy par bucket de longueur (phrases trip) - spaCy")
    plt.ylim(0, 1)
    plt.bar(labels, accs)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=200)

    print(f" Graph sauvegardé : {OUT_PLOT}")
    print(" Détails par bucket (trip seulement):")
    for name in labels:
        n = stats[name]["n"]
        c = stats[name]["correct"]
        print(f"  - {name}: {c}/{n} = {(c/n if n else 0):.3f}")


if __name__ == "__main__":
    main()