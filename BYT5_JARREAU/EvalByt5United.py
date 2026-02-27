from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import unicodedata
from typing import Optional

SUMMARY = Path("eval_summary.json")
PER_EX  = Path("eval_per_example.csv")

OUT_DIR = Path("plots_byt5")
OUT_DIR.mkdir(exist_ok=True)
LIMIT_N = None

def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def maybe_get(d: dict, key: str, default=None):
    return d[key] if (d is not None and key in d) else default

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def norm(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip().lower()
    s = strip_accents(s)
    s = s.replace("’", "'")
    for ch in ["-", "_", "/", "\\", ",", ";", ":", ".", "(", ")", "[", "]", "{", "}", "|"]:
        s = s.replace(ch, " ")
    s = s.replace("'", " ")
    s = " ".join(s.split())
    return s

def infer_pred_is_valid(df: pd.DataFrame) -> pd.Series:
    if "pred_is_valid" in df.columns:
        return df["pred_is_valid"].astype(int)

    if "raw_pred" in df.columns:
        rp = df["raw_pred"].fillna("").astype(str).str.strip()
        return (~rp.str.upper().str.startswith("INVALID")).astype(int)

    dep_col = "pred_dep_norm" if "pred_dep_norm" in df.columns else None
    arr_col = "pred_arr_norm" if "pred_arr_norm" in df.columns else None

    if dep_col or arr_col:
        dep_ok = df[dep_col].fillna("").astype(str).str.strip().ne("") if dep_col else False
        arr_ok = df[arr_col].fillna("").astype(str).str.strip().ne("") if arr_col else False
        if dep_col:
            dep_ok = dep_ok & (~df[dep_col].fillna("").astype(str).str.upper().str.contains("INVALID"))
        if arr_col:
            arr_ok = arr_ok & (~df[arr_col].fillna("").astype(str).str.upper().str.contains("INVALID"))
        return (dep_ok | arr_ok).astype(int)

    return pd.Series([1] * len(df), index=df.index, dtype=int)

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
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    df = pd.read_csv(PER_EX)

    if LIMIT_N is not None:
        df = df.head(int(LIMIT_N)).copy()
    if "ok" in df.columns:
        df["ok"] = df["ok"].astype(int)
    if "gold_is_valid" in df.columns:
        df["gold_is_valid"] = df["gold_is_valid"].astype(int)

    df["pred_is_valid_inferred"] = infer_pred_is_valid(df).astype(int)
    metrics = ["overall", "exact_od_valid", "dep_acc_valid", "arr_acc_valid", "invalid_acc"]
    fallback_overall = df["ok"].mean() if "ok" in df.columns else None
    vals = [
        maybe_get(summary, "overall", fallback_overall),
        maybe_get(summary, "exact_od_valid", None),
        maybe_get(summary, "dep_acc_valid", None),
        maybe_get(summary, "arr_acc_valid", None),
        maybe_get(summary, "invalid_acc", None),
    ]

    plt.figure()
    plt.bar(metrics, [0 if v is None else v for v in vals])
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.title("ByT5 – métriques principales")
    savefig(OUT_DIR / "metrics_bar.png")
    if "error_type" in df.columns:
        counts = df["error_type"].fillna("UNKNOWN").value_counts().sort_values(ascending=False)

        plt.figure()
        plt.bar(counts.index.astype(str), counts.values)
        plt.xticks(rotation=25, ha="right")
        plt.title("ByT5 – répartition des types d'erreurs (counts)")
        savefig(OUT_DIR / "error_types.png")

        plt.figure()
        rates = counts / counts.sum()
        plt.bar(rates.index.astype(str), rates.values)
        plt.ylim(0, 1)
        plt.xticks(rotation=25, ha="right")
        plt.title("ByT5 – répartition des types d'erreurs (taux)")
        savefig(OUT_DIR / "error_types_rate.png")
    by_len = None
    if "sentence_len" in df.columns and "ok" in df.columns:
        bins = [0, 30, 60, 90, 120, 999999]
        labels = ["0-30", "31-60", "61-90", "91-120", "120+"]
        df["len_bucket"] = pd.cut(df["sentence_len"], bins=bins, labels=labels, include_lowest=True)
        ok_by_len = df.groupby("len_bucket", observed=True)["ok"].mean()
        n_by_len  = df.groupby("len_bucket", observed=True)["ok"].size()
        by_len = {str(k): {"acc": float(ok_by_len.loc[k]), "n": int(n_by_len.loc[k])} for k in ok_by_len.index}
        plt.figure()
        plt.bar(ok_by_len.index.astype(str), ok_by_len.values)
        plt.ylim(0, 1)
        plt.title("ByT5 – accuracy par bucket de longueur")
        savefig(OUT_DIR / "acc_by_length.png")
        plt.figure()
        plt.hist(df["sentence_len"].dropna().values, bins=30)
        plt.title("ByT5 – distribution des longueurs de phrase")
        savefig(OUT_DIR / "sentence_len_hist.png")

    valid_acc = None
    invalid_acc = None
    if "gold_is_valid" in df.columns and "ok" in df.columns:
        valid_acc = df[df["gold_is_valid"] == 1]["ok"].mean()
        invalid_acc = df[df["gold_is_valid"] == 0]["ok"].mean()

        plt.figure()
        plt.bar(["gold_valid", "gold_invalid"], [valid_acc, invalid_acc])
        plt.ylim(0, 1)
        plt.title("ByT5 – accuracy: phrases valides vs invalides")
        savefig(OUT_DIR / "acc_valid_vs_invalid.png")
    validity_cm = None
    if "gold_is_valid" in df.columns:
        gold = df["gold_is_valid"].astype(int)
        pred = df["pred_is_valid_inferred"].astype(int)

        tn = int(((gold == 0) & (pred == 0)).sum())
        fp = int(((gold == 0) & (pred == 1)).sum())
        fn = int(((gold == 1) & (pred == 0)).sum())
        tp = int(((gold == 1) & (pred == 1)).sum())

        validity_cm = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

        plot_confusion_2x2(
            [[tn, fp], [fn, tp]],
            labels=["invalid", "valid"],
            title="ByT5 – matrice de validité (gold vs pred)",
            out_path=OUT_DIR / "validity_confusion.png"
        )
    extraction_stats = None
    confusion_5x5 = None

    required_cols = {"gold_is_valid", "pred_dep_norm", "pred_arr_norm", "gold_dep", "gold_arr"}
    if required_cols.issubset(set(df.columns)):
        df["gold_dep_n"] = df["gold_dep"].apply(norm)
        df["gold_arr_n"] = df["gold_arr"].apply(norm)
        df["pred_dep_n"] = df["pred_dep_norm"].apply(norm)
        df["pred_arr_n"] = df["pred_arr_norm"].apply(norm)
        df["dep_ok"] = 0
        df["arr_ok"] = 0
        mask_valid = (df["gold_is_valid"] == 1)
        df.loc[mask_valid, "dep_ok"] = (df.loc[mask_valid, "gold_dep_n"] == df.loc[mask_valid, "pred_dep_n"]).astype(int)
        df.loc[mask_valid, "arr_ok"] = (df.loc[mask_valid, "gold_arr_n"] == df.loc[mask_valid, "pred_arr_n"]).astype(int)
        dfv = df[mask_valid].copy()
        cats = (
            dfv["dep_ok"].map({1: "DEP_OK", 0: "DEP_KO"}) + " & " +
            dfv["arr_ok"].map({1: "ARR_OK", 0: "ARR_KO"})
        )
        counts = cats.value_counts()
        extraction_stats = {str(k): int(v) for k, v in counts.to_dict().items()}

        plt.figure()
        plt.bar(counts.index.astype(str), counts.values)
        plt.xticks(rotation=25, ha="right")
        plt.title("ByT5 – qualité d'extraction (phrases valides)")
        savefig(OUT_DIR / "extraction_quality_valid.png")

        def gold_class_fine(row):
            return "INVALID" if int(row["gold_is_valid"]) == 0 else "EXACT"
        def pred_class(row):
            if int(row["pred_is_valid_inferred"]) == 0:
                return "INVALID"
            if int(row["gold_is_valid"]) == 0:
                return "NONE"
            dep_ok = int(row["dep_ok"])
            arr_ok = int(row["arr_ok"])
            if dep_ok == 1 and arr_ok == 1:
                return "EXACT"
            if dep_ok == 1 and arr_ok == 0:
                return "DEP_ONLY"
            if dep_ok == 0 and arr_ok == 1:
                return "ARR_ONLY"
            return "NONE"

        df["gold_cls"] = df.apply(gold_class_fine, axis=1)
        df["pred_cls"] = df.apply(pred_class, axis=1)

        labels = ["INVALID", "EXACT", "DEP_ONLY", "ARR_ONLY", "NONE"]

        cm = pd.crosstab(df["gold_cls"], df["pred_cls"], rownames=["gold"], colnames=["pred"]).reindex(
            index=labels, columns=labels, fill_value=0
        )
        confusion_5x5 = cm

        plt.figure()
        plt.imshow(cm.values)
        plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
        plt.yticks(range(len(labels)), labels)
        plt.title("ByT5 – confusion matrix (INVALID/EXACT/DEP_ONLY/ARR_ONLY/NONE)")
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, int(cm.values[i, j]), ha="center", va="center")
        savefig(OUT_DIR / "extraction_confusion_5x5.png")
        cm_norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
        plt.figure()
        plt.imshow(cm_norm.values)
        plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
        plt.yticks(range(len(labels)), labels)
        plt.title("ByT5 – confusion matrix normalisée (row-wise)")
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, f"{cm_norm.values[i, j]:.2f}", ha="center", va="center")
        savefig(OUT_DIR / "extraction_confusion_5x5_norm.png")
    out = {
        "model": "ByT5",
        "main_metrics_from_summary": {k: v for k, v in zip(metrics, vals)},
        "computed_from_per_example": {
            "n_examples": int(len(df)),
            "overall_ok_mean": float(df["ok"].mean()) if "ok" in df.columns else None,
            "valid_acc": float(valid_acc) if valid_acc is not None else None,
            "invalid_acc": float(invalid_acc) if invalid_acc is not None else None,
            "validity_confusion": validity_cm,
        },
        "by_length_bucket": by_len,
        "error_type_counts": None,
        "extraction_quality_valid_counts": extraction_stats,
        "confusion_5x5_labels": ["INVALID", "EXACT", "DEP_ONLY", "ARR_ONLY", "NONE"],
    }

    if "error_type" in df.columns:
        err_counts = df["error_type"].fillna("UNKNOWN").value_counts().to_dict()
        out["error_type_counts"] = {str(k): int(v) for k, v in err_counts.items()}
    if confusion_5x5 is not None:
        out["confusion_5x5"] = confusion_5x5.to_dict()

    (OUT_DIR / "metrics_results_byt5.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

if __name__ == "__main__":
    main()