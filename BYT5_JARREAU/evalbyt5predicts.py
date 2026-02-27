import re
import unicodedata
from pathlib import Path
import csv
import json

GOLD_OUTPUT = Path("test_output.txt")
TEST_INPUT  = Path("test_input.txt")
PRED_FILE   = Path("pred_test.txt")

OUT_SUMMARY = Path("eval_summary.json")
OUT_PER_EX  = Path("eval_per_example.csv")

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def norm(s: str) -> str:
    s = (s or "").strip().lower().replace("’", "'")
    s = strip_accents(s)
    s = re.sub(r"[^a-z0-9\s;']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_text_od(text: str):
    t = norm(text)
    if "invalid" in t and ";" not in t:
        return ("INVALID", "")
    if ";" not in t:
        return ("INVALID", "")
    left, right = t.split(";", 1)
    left, right = left.strip(), right.strip()
    if not left or not right:
        return ("INVALID", "")
    return (left, right)

def read_input_map(path: Path):
    if not path.exists():
        return {}
    m = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sid, sent = line.split(",", 1)
        m[sid.strip()] = sent.strip()
    return m

def read_gold_output(path: Path):
    m = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(",", 2)
        sid = parts[0].strip()
        dep = (parts[1] if len(parts) > 1 else "").strip()
        arr = (parts[2] if len(parts) > 2 else "").strip()
        if dep.upper() == "INVALID" or dep == "":
            m[sid] = ("INVALID", "")
        else:
            m[sid] = (dep, arr)
    return m

def read_pred(path: Path):
    m = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(",", 2)
        sid = parts[0].strip()

        if len(parts) == 1:
            pred_txt = ""
            pred = ("INVALID", "")
        elif len(parts) == 2:
            pred_txt = parts[1].strip()
            pred = split_text_od(pred_txt)
        else:
            dep = (parts[1] or "").strip()
            arr = (parts[2] or "").strip()
            pred_txt = f"{dep} ; {arr}"
            if dep.upper() == "INVALID" or dep == "":
                pred = ("INVALID", "")
            else:
                pred = (norm(dep), norm(arr))
        m[sid] = (pred_txt, pred)
    return m

def main():
    sent_map = read_input_map(TEST_INPUT)
    gold_map = read_gold_output(GOLD_OUTPUT)
    pred_map = read_pred(PRED_FILE)

    ids = sorted(set(gold_map.keys()) & set(pred_map.keys()))
    if not ids:
        print("[ERROR] Aucun id commun entre gold et pred. Vérifie les fichiers.")
        return

    total = len(ids)
    invalid_total = invalid_ok = 0
    valid_total = exact_joint = dep_ok = arr_ok = 0
    pred_invalid_when_valid = pred_valid_when_invalid = 0

    per_rows = []

    for sid in ids:
        g_dep, g_arr = gold_map[sid]
        raw_pred, (p_dep, p_arr) = pred_map[sid]
        sentence = sent_map.get(sid, "")

        g_is_invalid = (g_dep == "INVALID")
        p_is_invalid = (p_dep == "INVALID")

        if g_is_invalid:
            invalid_total += 1
            if p_is_invalid:
                invalid_ok += 1
                error_type = "ok_invalid"
                ok = 1
            else:
                pred_valid_when_invalid += 1
                error_type = "gold_invalid_pred_valid"
                ok = 0
        else:
            valid_total += 1
            g_dep_n, g_arr_n = norm(g_dep), norm(g_arr)

            if p_is_invalid:
                pred_invalid_when_valid += 1
                error_type = "gold_valid_pred_invalid"
                ok = 0
            else:
                dep_match = int(p_dep == g_dep_n)
                arr_match = int(p_arr == g_arr_n)
                dep_ok += dep_match
                arr_ok += arr_match

                if dep_match and arr_match:
                    exact_joint += 1
                    error_type = "ok_valid"
                    ok = 1
                else:
                    error_type = "mismatch_valid"
                    ok = 0

        per_rows.append([
            sid,
            sentence,
            g_dep, g_arr,
            raw_pred,
            p_dep, p_arr,
            int(not g_is_invalid),  # gold_is_valid
            int(not p_is_invalid),  # pred_is_valid
            ok,
            error_type,
            len(sentence) if sentence else 0
        ])

    exact_od_valid = exact_joint / max(1, valid_total)
    dep_acc_valid  = dep_ok / max(1, valid_total)
    arr_acc_valid  = arr_ok / max(1, valid_total)
    invalid_acc    = invalid_ok / max(1, invalid_total)
    overall        = (exact_joint + invalid_ok) / max(1, total)

    summary = {
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

    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with OUT_PER_EX.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "id","sentence","gold_dep","gold_arr","raw_pred",
            "pred_dep_norm","pred_arr_norm",
            "gold_is_valid","pred_is_valid","ok","error_type","sentence_len"
        ])
        w.writerows(per_rows)

    print(" Summary:", OUT_SUMMARY.resolve())
    print(" Per-example CSV:", OUT_PER_EX.resolve())
    print("=== METRICS ===")
    for k,v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
