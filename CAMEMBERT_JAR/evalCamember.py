from pathlib import Path
import json
import re
import unicodedata

import pandas as pd
import matplotlib.pyplot as plt

MODEL_NAME = "ROBERTA-V2"
MODEL_DIR = Path("./ROBERTA-V2")
METRICS_JSON = MODEL_DIR / "test_metrics_ROBERTA.json"
PRED_CSV = MODEL_DIR / "test_predictions_ROBERTA.csv"
OUT_DIR = MODEL_DIR / "plots_ROBERTA"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def norm(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip().lower().replace("’", "'")
    s = strip_accents(s)
    for ch in ["-", "_", "/", "\\", ",", ";", ":", ".", "(", ")", "[", "]", "{", "}", "|"]:
        s = s.replace(ch, " ")
    s = s.replace("'", " ")
    s = " ".join(s.split())
    return s


def plot_confusion_2x2(cm, labels, title, out_path: Path):
    plt.figure()
    plt.imshow(cm)
    plt.xticks([0, 1], labels, rotation=15, ha="right")
    plt.yticks([0, 1], labels)
    plt.title(title)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center")
    savefig(out_path)


def main():
    metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    df = pd.read_csv(PRED_CSV)
    for col in ["gold_dep", "gold_arr", "pred_dep", "pred_arr", "ok", "len"]:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante dans {PRED_CSV}: {col}")

    df["ok"] = df["ok"].astype(int)
    df["gold_is_valid"] = (df["gold_dep"].fillna("").astype(str).str.upper() != "INVALID").astype(int)
    df["pred_is_valid"] = (
        (df["pred_dep"].fillna("").astype(str).str.upper() != "INVALID")
        & (df["pred_arr"].fillna("").astype(str).str.strip() != "")
    ).astype(int)
    keys = ["overall", "exact_od_valid", "dep_acc_valid", "arr_acc_valid", "invalid_acc"]
    vals = [float(metrics.get(k, 0.0)) for k in keys]

    plt.figure()
    plt.bar(keys, vals)
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.title("ROBERTA – métriques principales (test)")
    savefig(OUT_DIR / "metrics_bar.png")
    bins = [0, 30, 60, 90, 120, 999999]
    labels = ["0-30", "31-60", "61-90", "91-120", "120+"]
    df["len_bucket"] = pd.cut(df["len"], bins=bins, labels=labels, include_lowest=True)

    acc_by_len = df.groupby("len_bucket", observed=True)["ok"].mean()
    n_by_len = df.groupby("len_bucket", observed=True)["ok"].size()

    plt.figure()
    plt.bar(acc_by_len.index.astype(str), acc_by_len.values)
    plt.ylim(0, 1)
    plt.title("ROBERTA – accuracy par bucket de longueur (test)")
    savefig(OUT_DIR / "acc_by_length.png")
    plt.figure()
    plt.hist(df["len"].dropna().values, bins=30)
    plt.title("ROBERTA – distribution des longueurs (test)")
    savefig(OUT_DIR / "sentence_len_hist.png")
    valid_acc = df[df["gold_is_valid"] == 1]["ok"].mean()
    invalid_acc = df[df["gold_is_valid"] == 0]["ok"].mean()

    plt.figure()
    plt.bar(["gold_valid", "gold_invalid"], [valid_acc, invalid_acc])
    plt.ylim(0, 1)
    plt.title("ROBERTA – accuracy: phrases valides vs invalides (test)")
    savefig(OUT_DIR / "acc_valid_vs_invalid.png")
    gold = df["gold_is_valid"].astype(int)
    pred = df["pred_is_valid"].astype(int)

    tn = int(((gold == 0) & (pred == 0)).sum())
    fp = int(((gold == 0) & (pred == 1)).sum())
    fn = int(((gold == 1) & (pred == 0)).sum())
    tp = int(((gold == 1) & (pred == 1)).sum())

    plot_confusion_2x2(
        [[tn, fp], [fn, tp]],
        labels=["invalid", "valid"],
        title="ROBERTA – matrice de validité (gold vs pred)",
        out_path=OUT_DIR / "validity_confusion.png"
    )
    dfv = df[df["gold_is_valid"] == 1].copy()
    dfv["gold_dep_n"] = dfv["gold_dep"].apply(norm)
    dfv["gold_arr_n"] = dfv["gold_arr"].apply(norm)
    dfv["pred_dep_n"] = dfv["pred_dep"].apply(norm)
    dfv["pred_arr_n"] = dfv["pred_arr"].apply(norm)

    dfv["dep_ok"] = (dfv["gold_dep_n"] == dfv["pred_dep_n"]).astype(int)
    dfv["arr_ok"] = (dfv["gold_arr_n"] == dfv["pred_arr_n"]).astype(int)

    cats = dfv["dep_ok"].map({1: "DEP_OK", 0: "DEP_KO"}) + " & " + dfv["arr_ok"].map({1: "ARR_OK", 0: "ARR_KO"})
    counts = cats.value_counts()

    plt.figure()
    plt.bar(counts.index.astype(str), counts.values)
    plt.xticks(rotation=25, ha="right")
    plt.title("ROBERTA – qualité d'extraction (phrases valides)")
    savefig(OUT_DIR / "extraction_quality_valid.png")
    def gold_class(row):
        return "INVALID" if int(row["gold_is_valid"]) == 0 else "EXACT"

    def pred_class(row):
        if int(row["pred_is_valid"]) == 0:
            return "INVALID"
        if int(row["gold_is_valid"]) == 0:
            return "NONE"
        dep_ok = int(row.get("dep_ok", 0))
        arr_ok = int(row.get("arr_ok", 0))
        if dep_ok == 1 and arr_ok == 1:
            return "EXACT"
        if dep_ok == 1 and arr_ok == 0:
            return "DEP_ONLY"
        if dep_ok == 0 and arr_ok == 1:
            return "ARR_ONLY"
        return "NONE"

    dfa = df.copy()
    dfa["gold_dep_n"] = dfa["gold_dep"].apply(norm)
    dfa["gold_arr_n"] = dfa["gold_arr"].apply(norm)
    dfa["pred_dep_n"] = dfa["pred_dep"].apply(norm)
    dfa["pred_arr_n"] = dfa["pred_arr"].apply(norm)

    mask_valid = dfa["gold_is_valid"] == 1
    dfa["dep_ok"] = 0
    dfa["arr_ok"] = 0
    dfa.loc[mask_valid, "dep_ok"] = (dfa.loc[mask_valid, "gold_dep_n"] == dfa.loc[mask_valid, "pred_dep_n"]).astype(int)
    dfa.loc[mask_valid, "arr_ok"] = (dfa.loc[mask_valid, "gold_arr_n"] == dfa.loc[mask_valid, "pred_arr_n"]).astype(int)

    dfa["gold_cls"] = dfa.apply(gold_class, axis=1)
    dfa["pred_cls"] = dfa.apply(pred_class, axis=1)

    cls_labels = ["INVALID", "EXACT", "DEP_ONLY", "ARR_ONLY", "NONE"]
    cm = pd.crosstab(dfa["gold_cls"], dfa["pred_cls"], rownames=["gold"], colnames=["pred"]).reindex(
        index=cls_labels, columns=cls_labels, fill_value=0
    )

    plt.figure()
    plt.imshow(cm.values)
    plt.xticks(range(len(cls_labels)), cls_labels, rotation=20, ha="right")
    plt.yticks(range(len(cls_labels)), cls_labels)
    plt.title("ROBERTA – confusion (INVALID/EXACT/DEP_ONLY/ARR_ONLY/NONE)")
    for i in range(len(cls_labels)):
        for j in range(len(cls_labels)):
            plt.text(j, i, int(cm.values[i, j]), ha="center", va="center")
    savefig(OUT_DIR / "extraction_confusion_5x5.png")

    cm_norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    plt.figure()
    plt.imshow(cm_norm.values)
    plt.xticks(range(len(cls_labels)), cls_labels, rotation=20, ha="right")
    plt.yticks(range(len(cls_labels)), cls_labels)
    plt.title("ROBERTA – confusion normalisée (row-wise)")
    for i in range(len(cls_labels)):
        for j in range(len(cls_labels)):
            plt.text(j, i, f"{cm_norm.values[i, j]:.2f}", ha="center", va="center")
    savefig(OUT_DIR / "extraction_confusion_5x5_norm.png")

    out = {
        "model": "ROBERTA TokenClassification",
        "test_metrics_od": metrics,
        "computed_from_test_predictions_csv": {
            "n_examples": int(len(df)),
            "overall_ok_mean": float(df["ok"].mean()),
            "valid_acc": float(valid_acc),
            "invalid_acc": float(invalid_acc),
            "validity_confusion_2x2": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "acc_by_length_bucket": {
                str(k): {"acc": float(acc_by_len.loc[k]), "n": int(n_by_len.loc[k])}
                for k in acc_by_len.index
            },
            "extraction_quality_valid_counts": {str(k): int(v) for k, v in counts.to_dict().items()},
            "confusion_5x5_labels": cls_labels,
            "confusion_5x5": cm.to_dict(),
        },
    }

    (OUT_DIR / "metrics_results_ROBERTA.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"[OK] Plots + JSON écrits dans: {OUT_DIR}")


if __name__ == "__main__":
    main()