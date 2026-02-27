import re
import json
import unicodedata
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


DATA_JSON = Path("dataset_ner_cities_split.json")
SPLIT = "test"

OUT_BASE = Path("baseline_eval")
OUT_BASE.mkdir(exist_ok=True)

PRED_OUTPUT  = OUT_BASE / "baseline_pred.txt"
EVAL_SUMMARY = OUT_BASE / "eval_summary.json"
EVAL_PER_EX  = OUT_BASE / "eval_per_example.csv"

PLOTS_DIR = OUT_BASE / "plots_baseline"
PLOTS_DIR.mkdir(exist_ok=True)

LIMIT_N = None
STOP_END = {
    "maintenant", "tout", "suite", "aujourd", "hui", "demain", "soir", "matin",
    "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
    "fin", "apres", "midi",
    "stp", "svp", "merci", "plait", "vous", "te", "s", "il",
    "?", "!", "..."
}

PATTERNS = [
    (re.compile(r"\bde\s+(.+?)\s+(?:a|à|vers)\s+(.+)\b"), ("O", "D")),
    (re.compile(r"\bdepuis\s+(.+?)\s+(?:a|à|vers)\s+(.+)\b"), ("O", "D")),
    (re.compile(r"\b(?:a|à|vers)\s+(.+?)\s+depuis\s+(.+)\b"), ("D", "O")),
    (re.compile(r"\bentre\s+(.+?)\s+et\s+(.+)\b"), ("O", "D")),
    (re.compile(r"(.+?)\s*->\s*(.+)"), ("O", "D")),
    (re.compile(r"\bau\s+depart\s+de\s+(.+?)\s+(?:pour|vers|a|à)\s+(.+)\b"), ("O", "D")),
]
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("’", "'")
    s = strip_accents(s)
    s = s.replace("->", " ARROW ")
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = s.replace("arrow", "->")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm_for_compare(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("’", "'")
    s = strip_accents(s.lower())
    s = re.sub(r"[-_/]", " ", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def trim_span(span: str) -> str:
    span = (span or "").strip()
    span = re.sub(r"\s+", " ", span)
    tokens = span.split()

    while tokens:
        t = tokens[-1]
        if t in STOP_END:
            tokens.pop()
            continue
        if re.fullmatch(r"\d{1,2}h(\d{2})?", t):
            tokens.pop()
            continue
        if re.fullmatch(r"\d{1,4}", t):
            tokens.pop()
            continue
        break

    span = " ".join(tokens).strip()
    span = re.sub(r"^(destination|depart)\s+", "", span).strip()
    span = re.sub(r"\s+(destination|depart)$", "", span).strip()
    return span

def is_meaningful(span: str) -> bool:
    return bool(span) and bool(re.search(r"[a-z]", span))

def extract_od(sentence_raw: str):
    s = norm_text(sentence_raw)
    if not any(p in s for p in (" de ", " depuis ", " vers ", " a ", "->", " entre ")):
        return ("INVALID", "")

    for rx, order in PATTERNS:
        m = rx.search(s)
        if not m:
            continue

        g1 = trim_span(m.group(1))
        g2 = trim_span(m.group(2))

        if order == ("O", "D"):
            o, d = g1, g2
        else:
            d, o = g1, g2

        if not is_meaningful(o) or not is_meaningful(d):
            return ("INVALID", "")
        if norm_for_compare(o) == norm_for_compare(d):
            return ("INVALID", "")

        return (o, d)

    return ("INVALID", "")

def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

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

def load_json_splits(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["train"], data["val"], data["test"]

def make_rows_from_split(rows, split_name: str):
    out = []
    for i, r in enumerate(rows):
        sid = f"{split_name}_{i}"
        sent = r.get("sentence", "") or ""
        dep = r.get("start_city")
        arr = r.get("end_city")
        if dep and arr:
            gold = (str(dep), str(arr))
        else:
            gold = ("INVALID", "")
        out.append((sid, sent, gold))
    return out

def evaluate_and_export(rows):
    total = 0
    invalid_total = invalid_ok = 0
    valid_total = exact_joint = dep_ok = arr_ok = 0
    pred_invalid_when_valid = pred_valid_when_invalid = 0

    per_rows = []

    def safe_div(a, b): return a / b if b else 0.0

    with PRED_OUTPUT.open("w", encoding="utf-8") as fpred:
        for sid, sent, (g_dep, g_arr) in rows:
            total += 1

            p_dep_raw, p_arr_raw = extract_od(sent)
            p_is_invalid = (p_dep_raw == "INVALID")
            g_is_invalid = (g_dep == "INVALID")

            gold_is_valid = 0 if g_is_invalid else 1
            pred_is_valid = 0 if p_is_invalid else 1

            raw_pred = "INVALID" if p_is_invalid else f"{p_dep_raw} ; {p_arr_raw}"
            p_dep_n = "" if p_is_invalid else norm_for_compare(p_dep_raw)
            p_arr_n = "" if p_is_invalid else norm_for_compare(p_arr_raw)

            if p_is_invalid:
                fpred.write(f"{sid},INVALID,\n")
            else:
                fpred.write(f"{sid},{p_dep_raw},{p_arr_raw}\n")
            if g_is_invalid:
                invalid_total += 1
                if p_is_invalid:
                    invalid_ok += 1
                    ok = 1
                    error_type = "ok_invalid"
                else:
                    pred_valid_when_invalid += 1
                    ok = 0
                    error_type = "gold_invalid_pred_valid"
            else:
                valid_total += 1
                g_dep_n = norm_for_compare(g_dep)
                g_arr_n = norm_for_compare(g_arr)

                if p_is_invalid:
                    pred_invalid_when_valid += 1
                    ok = 0
                    error_type = "gold_valid_pred_invalid"
                else:
                    d_ok = int(p_dep_n == g_dep_n)
                    a_ok = int(p_arr_n == g_arr_n)
                    dep_ok += d_ok
                    arr_ok += a_ok
                    if d_ok and a_ok:
                        exact_joint += 1
                        ok = 1
                        error_type = "ok_valid"
                    else:
                        ok = 0
                        error_type = "mismatch_valid"

            per_rows.append({
                "id": sid,
                "sentence": sent,
                "gold_dep": g_dep if not g_is_invalid else "INVALID",
                "gold_arr": g_arr if not g_is_invalid else "",
                "raw_pred": raw_pred,
                "pred_dep_norm": p_dep_n,
                "pred_arr_norm": p_arr_n,
                "gold_is_valid": gold_is_valid,
                "pred_is_valid": pred_is_valid,
                "ok": ok,
                "error_type": error_type,
                "sentence_len": len(sent),
            })

    invalid_acc    = safe_div(invalid_ok, invalid_total)
    exact_od_valid = safe_div(exact_joint, valid_total)
    dep_acc_valid  = safe_div(dep_ok, valid_total)
    arr_acc_valid  = safe_div(arr_ok, valid_total)
    overall        = safe_div(exact_joint + invalid_ok, total)

    summary = {
        "model": "BASELINE_PREPOSITIONS",
        "total": total,
        "valid_total": valid_total,
        "invalid_total": invalid_total,
        "overall": overall,
        "exact_od_valid": exact_od_valid,
        "dep_acc_valid": dep_acc_valid,
        "arr_acc_valid": arr_acc_valid,
        "invalid_acc": invalid_acc,
        "pred_invalid_when_valid": pred_invalid_when_valid,
        "pred_valid_when_invalid": pred_valid_when_invalid,
    }

    EVAL_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    df = pd.DataFrame(per_rows)
    df.to_csv(EVAL_PER_EX, index=False, encoding="utf-8")

    print("Summary:", EVAL_SUMMARY.resolve())
    print("Per-example CSV:", EVAL_PER_EX.resolve())
    print("Pred file:", PRED_OUTPUT.resolve())

    print("\n=== BASELINE (prépositions) ===")
    print(f"total={total} | valid={valid_total} | invalid={invalid_total}")
    print(f"overall={overall:.4f}")
    print(f"invalid_acc={invalid_acc:.4f}")
    print(f"exact_od_valid={exact_od_valid:.4f}")
    print(f"dep_acc_valid={dep_acc_valid:.4f}")
    print(f"arr_acc_valid={arr_acc_valid:.4f}")

    return summary, df

def make_plots(summary: dict, df: pd.DataFrame):
    metrics = ["overall", "exact_od_valid", "dep_acc_valid", "arr_acc_valid", "invalid_acc"]
    vals = [float(summary.get(m, 0.0)) for m in metrics]

    plt.figure()
    plt.bar(metrics, vals)
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.title("Baseline – métriques principales")
    savefig(PLOTS_DIR / "metrics_bar.png")
    counts = df["error_type"].fillna("UNKNOWN").value_counts().sort_values(ascending=False)
    plt.figure()
    plt.bar(counts.index.astype(str), counts.values)
    plt.xticks(rotation=25, ha="right")
    plt.title("Error type counts")
    savefig(PLOTS_DIR / "error_types.png")

    plt.figure()
    rates = counts / counts.sum()
    plt.bar(rates.index.astype(str), rates.values)
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.title("Error type rate")
    savefig(PLOTS_DIR / "error_types_rate.png")
    bins = [0, 30, 60, 90, 120, 999999]
    labels = ["0-30", "31-60", "61-90", "91-120", "120+"]
    df["len_bucket"] = pd.cut(df["sentence_len"], bins=bins, labels=labels, include_lowest=True)
    ok_by_len = df.groupby("len_bucket", observed=True)["ok"].mean()
    plt.figure()
    plt.bar(ok_by_len.index.astype(str), ok_by_len.values)
    plt.ylim(0, 1)
    plt.title("Accuracy by sentence length bucket")
    savefig(PLOTS_DIR / "acc_by_length.png")
    plt.figure()
    plt.hist(df["sentence_len"].dropna().values, bins=30)
    plt.title("Sentence length distribution")
    savefig(PLOTS_DIR / "sentence_len_hist.png")
    valid_acc = df[df["gold_is_valid"] == 1]["ok"].mean()
    invalid_acc = df[df["gold_is_valid"] == 0]["ok"].mean()
    plt.figure()
    plt.bar(["gold_valid", "gold_invalid"], [valid_acc, invalid_acc])
    plt.ylim(0, 1)
    plt.title("Accuracy: valid vs invalid")
    savefig(PLOTS_DIR / "acc_valid_vs_invalid.png")
    gold = df["gold_is_valid"].astype(int)
    pred = df["pred_is_valid"].astype(int)
    tn = int(((gold == 0) & (pred == 0)).sum())
    fp = int(((gold == 0) & (pred == 1)).sum())
    fn = int(((gold == 1) & (pred == 0)).sum())
    tp = int(((gold == 1) & (pred == 1)).sum())
    plot_confusion_2x2([[tn, fp], [fn, tp]], ["invalid", "valid"], "Validity confusion (gold vs pred)", PLOTS_DIR / "validity_confusion.png")

    metrics_results = {
        "model": summary.get("model", "BASELINE_PREPOSITIONS"),
        "main_metrics": {k: float(summary[k]) for k in ["overall", "exact_od_valid", "dep_acc_valid", "arr_acc_valid", "invalid_acc"]},
        "counts": {
            "total": int(summary.get("total", len(df))),
            "valid_total": int(summary.get("valid_total", (df["gold_is_valid"] == 1).sum())),
            "invalid_total": int(summary.get("invalid_total", (df["gold_is_valid"] == 0).sum())),
        },
        "error_type_counts": df["error_type"].value_counts().to_dict(),
    }
    (PLOTS_DIR / "metrics_results_baseline.json").write_text(
        json.dumps(metrics_results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

def main():
    if not DATA_JSON.exists():
        raise FileNotFoundError(f"Introuvable: {DATA_JSON.resolve()}")

    train_rows, val_rows, test_rows = load_json_splits(DATA_JSON)
    split_rows = {"train": train_rows, "val": val_rows, "test": test_rows}[SPLIT]

    rows = make_rows_from_split(split_rows, SPLIT)

    if LIMIT_N is not None:
        rows = rows[: int(LIMIT_N)]

    summary, df = evaluate_and_export(rows)
    make_plots(summary, df)

    print("\n Généré :")
    print(" -", EVAL_SUMMARY.resolve())
    print(" -", EVAL_PER_EX.resolve())
    print(" -", PRED_OUTPUT.resolve())
    print(" - Plots:", PLOTS_DIR.resolve())

if __name__ == "__main__":
    main()