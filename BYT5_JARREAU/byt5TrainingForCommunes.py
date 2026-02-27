import json
import re
import unicodedata
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

DATA_JSON = Path("dataset_ner_cities_split.json")
MODEL_NAME = "google/byt5-small"
OUT_DIR = "./model_byt5_from_json"

PREFIX = "extract_od: "

MAX_SOURCE_LEN = 128
MAX_TARGET_LEN = 48

EPOCHS = 2
BATCH_SIZE = 8
LR = 3e-4
SEED = 42
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("’", "'")
    s = strip_accents(s)
    s = re.sub(r"[^a-z0-9\s;']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_pred(text: str):
    t = norm(text)
    if "invalid" in t and ";" not in t:
        return ("INVALID", "")
    if ";" not in t:
        return ("INVALID", "")
    left, right = t.split(";", 1)
    left = left.strip()
    right = right.strip()
    if not left or not right:
        return ("INVALID", "")
    return (left, right)

def split_gold(text: str):
    return split_pred(text)

def load_json_splits(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["train"], data["val"], data["test"]

def to_pairs(rows, split_name: str):
    pairs = []
    for i, r in enumerate(rows):
        sent = r["sentence"]
        dep = r.get("start_city")
        arr = r.get("end_city")

        if dep and arr:
            tgt = f"{dep} ; {arr}"
        else:
            tgt = "INVALID"

        pairs.append({
            "id": f"{split_name}_{i}",
            "source": PREFIX + sent,
            "target": tgt
        })
    return pairs

def build_compute_metrics(tokenizer):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.asarray(preds)
        labels = np.asarray(labels)

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        if preds.ndim == 3:
            preds = preds.argmax(axis=-1)
        vocab = getattr(tokenizer, "vocab_size", None)
        preds = np.where(preds == -100, pad_id, preds)

        if vocab is not None:
            preds = np.where((preds < 0) | (preds >= vocab), pad_id, preds)
        labels_clean = np.where(labels != -100, labels, pad_id)

        pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        gold_texts = tokenizer.batch_decode(labels_clean, skip_special_tokens=True)
        tp = fp = fn = tn = 0
        for p_txt, g_txt in zip(pred_texts, gold_texts):
            p_dep, p_arr = split_pred(p_txt)
            g_dep, g_arr = split_gold(g_txt)

            gold_valid = (g_dep != "INVALID")
            pred_valid = (p_dep != "INVALID")

            if gold_valid and pred_valid:
                tp += 1
            elif gold_valid and not pred_valid:
                fn += 1
            elif (not gold_valid) and pred_valid:
                fp += 1
            else:
                tn += 1

        def safe_div(a, b):
            return a / b if b else 0.0

        prec_valid = safe_div(tp, tp + fp)
        rec_valid = safe_div(tp, tp + fn)
        f1_valid = safe_div(2 * prec_valid * rec_valid, prec_valid + rec_valid)
        tp_i, fp_i, fn_i, tn_i = tn, fn, fp, tp
        prec_inv = safe_div(tp_i, tp_i + fp_i)
        rec_inv = safe_div(tp_i, tp_i + fn_i)
        f1_inv = safe_div(2 * prec_inv * rec_inv, prec_inv + rec_inv)

        macro_f1 = (f1_valid + f1_inv) / 2.0

        total = len(gold_texts)

        invalid_total = 0
        invalid_ok = 0

        valid_total = 0
        exact_joint = 0
        dep_ok = 0
        arr_ok = 0
        pred_invalid_when_valid = 0
        pred_valid_when_invalid = 0

        for p_txt, g_txt in zip(pred_texts, gold_texts):
            p_dep, p_arr = split_pred(p_txt)
            g_dep, g_arr = split_gold(g_txt)

            g_is_invalid = (g_dep == "INVALID")
            p_is_invalid = (p_dep == "INVALID")

            if g_is_invalid:
                invalid_total += 1
                if p_is_invalid:
                    invalid_ok += 1
                else:
                    pred_valid_when_invalid += 1
            else:
                valid_total += 1
                if p_is_invalid:
                    pred_invalid_when_valid += 1
                    continue
                g_dep_n = norm(g_dep)
                g_arr_n = norm(g_arr)

                if p_dep == g_dep_n:
                    dep_ok += 1
                if p_arr == g_arr_n:
                    arr_ok += 1
                if (p_dep == g_dep_n) and (p_arr == g_arr_n):
                    exact_joint += 1
        exact_od_valid = exact_joint / max(1, valid_total)
        dep_acc_valid = dep_ok / max(1, valid_total)
        arr_acc_valid = arr_ok / max(1, valid_total)
        invalid_acc = invalid_ok / max(1, invalid_total)

        overall = (exact_joint + invalid_ok) / max(1, total)

        return {
            "overall": overall,
            "exact_od_valid": exact_od_valid,
            "dep_acc_valid": dep_acc_valid,
            "arr_acc_valid": arr_acc_valid,
            "invalid_acc": invalid_acc,
            "valid_total": valid_total,
            "invalid_total": invalid_total,
            "pred_invalid_when_valid": pred_invalid_when_valid,
            "pred_valid_when_invalid": pred_valid_when_invalid,
            "cm_tp_valid": tp,
            "cm_fp_valid": fp,
            "cm_fn_valid": fn,
            "cm_tn_valid": tn,
            "valid_precision": prec_valid,
            "valid_recall": rec_valid,
            "valid_f1": f1_valid,
            "invalid_precision": prec_inv,
            "invalid_recall": rec_inv,
            "invalid_f1": f1_inv,
            "macro_f1_valid_invalid": macro_f1,

        }

    return compute_metrics

train_rows, val_rows, test_rows = load_json_splits(DATA_JSON)
train_pairs = to_pairs(train_rows, "train")
val_pairs   = to_pairs(val_rows, "val")
test_pairs  = to_pairs(test_rows, "test")
train_ds = Dataset.from_list(train_pairs)
val_ds   = Dataset.from_list(val_pairs)
test_ds  = Dataset.from_list(test_pairs)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
def preprocess(batch):
    model_inputs = tokenizer(
        batch["source"],
        max_length=MAX_SOURCE_LEN,
        truncation=True,
    )
    labels = tokenizer(
        text_target=batch["target"],
        max_length=MAX_TARGET_LEN,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
val_tok   = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)
test_tok  = test_ds.map(preprocess, batched=True, remove_columns=test_ds.column_names)

collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

args = Seq2SeqTrainingArguments(
    output_dir=OUT_DIR,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,

    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    generation_num_beams=4,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="overall",
    greater_is_better=True,

    logging_steps=50,
    seed=SEED,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=build_compute_metrics(tokenizer),
)

trainer.train()

trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)


print("\n Évaluation finale sur TEST:")
test_metrics = trainer.evaluate(eval_dataset=test_tok, metric_key_prefix="test")
for k, v in test_metrics.items():
    print(f"{k}: {v}")

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
Path(OUT_DIR, "test_metrics.json").write_text(
    json.dumps(test_metrics, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

pred_out = trainer.predict(test_dataset=test_tok, max_length=MAX_TARGET_LEN, num_beams=4)

pred_ids = pred_out.predictions
label_ids = pred_out.label_ids
if isinstance(pred_ids, tuple):
    pred_ids = pred_ids[0]

pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
label_ids_clean = np.where(label_ids != -100, label_ids, pad_id)

pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
gold_texts = tokenizer.batch_decode(label_ids_clean, skip_special_tokens=True)
sentences = [p["source"].replace(PREFIX, "", 1) for p in test_pairs]
def is_ok(pred_txt: str, gold_txt: str) -> int:
    p_dep, p_arr = split_pred(pred_txt)
    g_dep, g_arr = split_gold(gold_txt)
    if g_dep == "INVALID":
        return int(p_dep == "INVALID")
    return int((p_dep == norm(g_dep)) and (p_arr == norm(g_arr)))

oks = [is_ok(p, g) for p, g in zip(pred_texts, gold_texts)]

df = pd.DataFrame({
    "sentence": sentences,
    "gold": gold_texts,
    "pred": pred_texts,
    "ok": oks,
    "len": [len(s) for s in sentences],
})

csv_path = Path(OUT_DIR, "test_predictions.csv")
df.to_csv(csv_path, index=False, encoding="utf-8")
print("CSV prédictions:", csv_path.resolve())

plots_dir = Path(OUT_DIR, "plots")
plots_dir.mkdir(exist_ok=True)
metrics_keys = ["test_overall", "test_exact_od_valid", "test_dep_acc_valid", "test_arr_acc_valid", "test_invalid_acc"]
vals = [float(test_metrics.get(k, 0.0)) for k in metrics_keys]

plt.figure()
plt.bar(metrics_keys, vals)
plt.ylim(0, 1)
plt.xticks(rotation=25, ha="right")
plt.title("ByT5 metrics (test)")
plt.tight_layout()
plt.savefig(plots_dir / "metrics_bar.png", dpi=160)
plt.close()
bins = [0, 30, 60, 90, 120, 99999]
labels = ["0-30", "31-60", "61-90", "91-120", "120+"]
df["len_bucket"] = pd.cut(df["len"], bins=bins, labels=labels, include_lowest=True)
acc_by_len = df.groupby("len_bucket")["ok"].mean()

plt.figure()
plt.bar(acc_by_len.index.astype(str), acc_by_len.values)
plt.ylim(0, 1)
plt.title("Accuracy by sentence length bucket (test)")
plt.tight_layout()
plt.savefig(plots_dir / "acc_by_length.png", dpi=160)
plt.close()

