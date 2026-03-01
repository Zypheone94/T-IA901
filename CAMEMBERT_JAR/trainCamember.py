import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch import nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

N_TRIALS = 20
OPTUNA_EPOCHS_MAX = 4
DATA_JSON = Path("dataset_ner_cities_split_v2.json")
MODEL_NAME = "ROBERTA-V2"
OUT_DIR = Path("./ROBERTA-V2")

MAX_LEN = 160

EPOCHS = 6
BATCH_SIZE = 16
LR = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42
LABEL_LIST = ["O", "B-DEP", "I-DEP", "B-ARR", "I-ARR"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}
def clean_text(s: str) -> str:
    s = (s or "")
    s = s.replace("\u2019", "'")  # ’ -> '
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
def expand_saint_variants(s: str) -> str:
    s = re.sub(r"\bst\.?\b", "saint", s)     # st / st.
    s = re.sub(r"\bste\.?\b", "sainte", s)   # ste / ste.
    return s

def key_for_match(s: str) -> str:
    s = (s or "").replace("\u2019", "'")
    s = strip_accents(s).lower()
    s = expand_saint_variants(s)
    out = []
    for ch in s:
        out.append(ch if (ch.isalnum() or ch in [" ", "'"]) else " ")
    return "".join(out)

def norm_for_compare(s: str) -> str:
    s = (s or "").strip().lower().replace("\u2019", "'")
    s = strip_accents(s)
    s = expand_saint_variants(s)
    s = re.sub(r"[^a-z0-9\s'_-]", " ", s)
    s = re.sub(r"[\s_-]+", " ", s).strip()
    return s

def find_city_spans(sentence: str, city: str) -> List[Tuple[int, int]]:

    if not city:
        return []

    sent_key = key_for_match(sentence)
    city_key = key_for_match(city)
    city_key = re.sub(r"\s+", " ", city_key).strip()
    if not city_key:
        return []
    pattern = re.escape(city_key).replace(r"\ ", r"\s+")
    pattern = r"(?<!\w)" + pattern + r"(?!\w)"

    spans = []
    for m in re.finditer(pattern, sent_key):
        spans.append((m.start(), m.end()))
    return spans

def pick_best_span(sentence: str, city: str, role: str) -> Optional[Tuple[int, int]]:
    spans = find_city_spans(sentence, city)
    if not spans:
        return None
    if len(spans) == 1:
        return spans[0]

    sent_key = key_for_match(sentence)

    def score(span: Tuple[int, int]) -> int:
        start, _ = span
        left = sent_key[max(0, start - 40):start]
        s = 0
        if role == "DEP":
            if re.search(r"\b(de|depuis|depart de)\s*$", left):
                s += 10
            if re.search(r"\b(a|vers|pour|jusque)\s*$", left):
                s -= 2
        else:
            if re.search(r"\b(a|vers|pour|jusque)\s*$", left):
                s += 10
            if re.search(r"\b(de|depuis|depart de)\s*$", left):
                s -= 2
        if role == "DEP":
            s += max(0, 5 - start // 40)
        else:
            s += max(0, 5 - (len(sentence) - start) // 40)
        return s

    spans_sorted = sorted(spans, key=score, reverse=True)
    return spans_sorted[0]

def overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])

def make_token_labels(
    tokenizer,
    sentence: str,
    dep_city: Optional[str],
    arr_city: Optional[str],
    max_len: int
) -> Dict:
    sentence = clean_text(sentence)

    dep_span = pick_best_span(sentence, dep_city or "", "DEP") if dep_city and arr_city else None
    arr_span = pick_best_span(sentence, arr_city or "", "ARR") if dep_city and arr_city else None

    enc = tokenizer(
        sentence,
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True,
    )

    offsets = enc["offset_mapping"]
    labels = []

    prev_entity = None  # "DEP"/"ARR"/None

    for (s, e) in offsets:
        if s == 0 and e == 0:
            labels.append(-100)  # special token
            prev_entity = None
            continue

        tok_span = (s, e)
        entity = None

        if dep_span and overlap(tok_span, dep_span):
            entity = "DEP"
        elif arr_span and overlap(tok_span, arr_span):
            entity = "ARR"

        if entity is None:
            labels.append(LABEL2ID["O"])
            prev_entity = None
        else:
            if prev_entity != entity:
                labels.append(LABEL2ID[f"B-{entity}"])
            else:
                labels.append(LABEL2ID[f"I-{entity}"])
            prev_entity = entity

    enc.pop("offset_mapping")
    enc["labels"] = labels
    return enc

def build_compute_metrics():
    from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

    def compute_metrics(p):
        logits, labels = p
        preds = np.argmax(logits, axis=-1)

        true_labels = []
        true_preds = []

        for pred_seq, lab_seq in zip(preds, labels):
            seq_labs = []
            seq_preds = []
            for pr, lb in zip(pred_seq, lab_seq):
                if lb == -100:
                    continue
                seq_labs.append(ID2LABEL[int(lb)])
                seq_preds.append(ID2LABEL[int(pr)])
            true_labels.append(seq_labs)
            true_preds.append(seq_preds)
        prec = precision_score(true_labels, true_preds)
        rec = recall_score(true_labels, true_preds)
        f1 = f1_score(true_labels, true_preds)
        rep = classification_report(true_labels, true_preds, output_dict=True, zero_division=0)
        f1_dep = rep.get("DEP", {}).get("f1-score", 0.0)
        f1_arr = rep.get("ARR", {}).get("f1-score", 0.0)

        return {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "f1_DEP": f1_dep,
            "f1_ARR": f1_arr,
        }

    return compute_metrics
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits

        weight = self.class_weights
        if weight is not None and hasattr(weight, "to"):
            weight = weight.to(logits.device)

        loss_fct = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
def compute_class_weights(train_tok: Dataset, num_labels: int) -> "np.ndarray":
    counts = np.zeros(num_labels, dtype=np.int64)
    for labs in train_tok["labels"]:
        for lb in labs:
            if lb == -100:
                continue
            counts[int(lb)] += 1
    counts = np.maximum(counts, 1)
    total = counts.sum()
    weights = total / (num_labels * counts.astype(np.float64))
    weights = weights / weights.mean()
    return weights

def extract_entity_spans_from_labels(sentence: str, offsets, label_ids: List[int]) -> Tuple[str, str]:
    dep_tokens = []
    arr_tokens = []
    for (s, e), lb in zip(offsets, label_ids):
        if s == 0 and e == 0:
            continue
        if lb == -100:
            continue
        tag = ID2LABEL.get(int(lb), "O")
        if tag.endswith("DEP"):
            dep_tokens.append((s, e))
        elif tag.endswith("ARR"):
            arr_tokens.append((s, e))

    def merge(spans):
        if not spans:
            return None
        s0 = min(s for s, _ in spans)
        e0 = max(e for _, e in spans)
        return (s0, e0)

    dep_span = merge(dep_tokens)
    arr_span = merge(arr_tokens)

    if not dep_span or not arr_span:
        return ("INVALID", "")

    dep = sentence[dep_span[0]:dep_span[1]].strip()
    arr = sentence[arr_span[0]:arr_span[1]].strip()
    if not dep or not arr:
        return ("INVALID", "")
    return (dep, arr)

def eval_od_metrics(tokenizer, test_rows, pred_label_ids) -> Tuple[dict, pd.DataFrame]:
    total = len(test_rows)
    invalid_total = invalid_ok = 0
    valid_total = 0
    exact_joint = dep_ok = arr_ok = 0
    pred_invalid_when_valid = pred_valid_when_invalid = 0
    records = []
    for i, row in enumerate(test_rows):
        sent = clean_text(row["sentence"])
        g_dep = row.get("start_city")
        g_arr = row.get("end_city")
        gold_valid = bool(g_dep and g_arr)
        enc = tokenizer(sent, truncation=True, max_length=MAX_LEN, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        labels_i = pred_label_ids[i].tolist()
        p_dep, p_arr = extract_entity_spans_from_labels(sent, offsets, labels_i)
        if not gold_valid:
            invalid_total += 1
            if p_dep == "INVALID":
                invalid_ok += 1
                ok = 1
            else:
                pred_valid_when_invalid += 1
                ok = 0
        else:
            valid_total += 1
            if p_dep == "INVALID":
                pred_invalid_when_valid += 1
                ok = 0
            else:
                g_dep_n = norm_for_compare(g_dep)
                g_arr_n = norm_for_compare(g_arr)
                p_dep_n = norm_for_compare(p_dep)
                p_arr_n = norm_for_compare(p_arr)

                dep_hit = int(p_dep_n == g_dep_n)
                arr_hit = int(p_arr_n == g_arr_n)
                joint_hit = int(dep_hit and arr_hit)

                dep_ok += dep_hit
                arr_ok += arr_hit
                exact_joint += joint_hit
                ok = joint_hit
        records.append({
            "sentence": sent,
            "gold_dep": g_dep if g_dep else "INVALID",
            "gold_arr": g_arr if g_arr else "",
            "pred_dep": p_dep,
            "pred_arr": p_arr,
            "ok": ok,
            "len": len(sent),
        })

    exact_od_valid = exact_joint / max(1, valid_total)
    dep_acc_valid = dep_ok / max(1, valid_total)
    arr_acc_valid = arr_ok / max(1, valid_total)
    invalid_acc = invalid_ok / max(1, invalid_total)
    overall = (exact_joint + invalid_ok) / max(1, total)
    metrics = {
        "overall": overall,
        "exact_od_valid": exact_od_valid,
        "dep_acc_valid": dep_acc_valid,
        "arr_acc_valid": arr_acc_valid,
        "invalid_acc": invalid_acc,
        "valid_total": valid_total,
        "invalid_total": invalid_total,
        "pred_invalid_when_valid": pred_invalid_when_valid,
        "pred_valid_when_invalid": pred_valid_when_invalid,
    }

    df = pd.DataFrame.from_records(records)
    return metrics, df

def load_json_splits(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["train"], data["val"], data["test"]

def to_rows(rows, split_name: str):
    out = []
    for i, r in enumerate(rows):
        out.append({
            "id": f"{split_name}_{i}",
            "sentence": clean_text(r["sentence"]),
            "start_city": r.get("start_city"),
            "end_city": r.get("end_city"),
        })
    return out

from transformers import set_seed

def main():
    set_seed(SEED)

    train_raw, val_raw, test_raw = load_json_splits(DATA_JSON)

    train_rows = to_rows(train_raw, "train")
    val_rows   = to_rows(val_raw, "val")
    test_rows  = to_rows(test_raw, "test")

    train_ds = Dataset.from_list(train_rows)
    val_ds   = Dataset.from_list(val_rows)
    test_ds  = Dataset.from_list(test_rows)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def preprocess(batch):
        encs = {"input_ids": [], "attention_mask": [], "labels": []}
        for sent, dep, arr in zip(batch["sentence"], batch["start_city"], batch["end_city"]):
            enc = make_token_labels(tokenizer, sent, dep, arr, MAX_LEN)
            encs["input_ids"].append(enc["input_ids"])
            encs["attention_mask"].append(enc["attention_mask"])
            encs["labels"].append(enc["labels"])
        return encs
    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    val_tok   = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)
    test_tok  = test_ds.map(preprocess, batched=True, remove_columns=test_ds.column_names)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    weights_np = compute_class_weights(train_tok, num_labels=len(LABEL_LIST))
    import torch
    class_weights = torch.tensor(weights_np, dtype=torch.float32)
    use_fp16 = torch.cuda.is_available()
    def model_init():
        return AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(LABEL_LIST),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base_args = TrainingArguments(
        output_dir=str(OUT_DIR / "optuna_runs"),
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        seed=SEED,
        fp16=use_fp16,
        disable_tqdm=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
    )

    tuner = WeightedTrainer(
        model_init=model_init,
        args=base_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        compute_metrics=build_compute_metrics(),
        class_weights=class_weights,
    )

    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 8e-5, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.15),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 2, OPTUNA_EPOCHS_MAX),
        }

    study_name = "camembert_od_ner"
    storage = f"sqlite:///{OUT_DIR / 'optuna_study.db'}"

    best_run = tuner.hyperparameter_search(
        backend="optuna",
        direction="maximize",
        n_trials=N_TRIALS,
        hp_space=hp_space,
        compute_objective=lambda metrics: metrics.get("eval_f1", 0.0),
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    print("\n[OPTUNA] Best eval_f1:", best_run.objective)
    print("[OPTUNA] Best hyperparams:", best_run.hyperparameters)
    best_hp = best_run.hyperparameters
    final_args = TrainingArguments(
        output_dir=str(OUT_DIR),
        learning_rate=float(best_hp["learning_rate"]),
        weight_decay=float(best_hp["weight_decay"]),
        warmup_ratio=float(best_hp["warmup_ratio"]),
        per_device_train_batch_size=int(best_hp["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(best_hp["per_device_train_batch_size"]),
        num_train_epochs=int(best_hp["num_train_epochs"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=SEED,
        fp16=use_fp16,
    )

    trainer = WeightedTrainer(
        model=model_init(),
        args=final_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        compute_metrics=build_compute_metrics(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights,
    )

    print("\n[FINAL] Training with best hyperparams...")
    trainer.train()
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print("\n[TEST] Predict...")
    pred_out = trainer.predict(test_tok)
    logits = pred_out.predictions
    pred_ids = np.argmax(logits, axis=-1)
    od_metrics, df = eval_od_metrics(tokenizer, test_rows, pred_ids)
    print("\n[TEST] OD metrics:")
    for k, v in od_metrics.items():
        print(f"{k}: {v}")
    (OUT_DIR / "test_metrics_od.json").write_text(
        json.dumps(od_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    df.to_csv(OUT_DIR / "test_predictions.csv", index=False, encoding="utf-8")

    plots_dir = OUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    keys = ["overall", "exact_od_valid", "dep_acc_valid", "arr_acc_valid", "invalid_acc"]
    vals = [float(od_metrics.get(k, 0.0)) for k in keys]
    plt.figure()
    plt.bar(keys, vals)
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.title("CamemBERT OD metrics (test)")
    plt.tight_layout()
    plt.savefig(plots_dir / "metrics_bar.png", dpi=160)
    plt.close()
if __name__ == "__main__":
    main()