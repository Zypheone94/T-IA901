import spacy
from spacy.tokens import DocBin
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


MODEL_PATH = "models/spacy_ner_dep_arr/model-best"
TEST_PATH = "data/spacy/test.spacy"

LABELS = ["invalid", "valid"]


def is_valid(ents):
    """
    Une phrase est VALID si elle contient
    au moins une ville DEP et une ville ARR
    """
    has_dep = any(ent.label_ == "DEP" for ent in ents)
    has_arr = any(ent.label_ == "ARR" for ent in ents)

    return has_dep and has_arr


def main():

    print("Chargement du modèle...")
    nlp = spacy.load(MODEL_PATH)

    print("Chargement dataset test...")
    doc_bin = DocBin().from_disk(TEST_PATH)
    docs = list(doc_bin.get_docs(nlp.vocab))

    y_true = []
    y_pred = []

    print("Calcul valid / invalid ...")

    for gold_doc in docs:

        text = gold_doc.text

        # label réel
        true_label = "valid" if is_valid(gold_doc.ents) else "invalid"

        # prédiction
        pred_doc = nlp(text)
        pred_label = "valid" if is_valid(pred_doc.ents) else "invalid"

        y_true.append(true_label)
        y_pred.append(pred_label)

    print("Calcul matrice de confusion...")

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    print(cm)

    fig, ax = plt.subplots(figsize=(6,5))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=LABELS
    )

    disp.plot(ax=ax, cmap="viridis", values_format="d")

    ax.set_title("Validity confusion (gold vs pred)")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("True label")

    plt.tight_layout()

    Path("report/plots").mkdir(parents=True, exist_ok=True)

    plt.savefig("report/plots/confusion_matrix_validity.png")

    print("Image sauvegardée : report/plots/confusion_matrix_validity.png")


if __name__ == "__main__":
    main()