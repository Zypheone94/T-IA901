#!/usr/bin/env python3
"""
Usage :
    python ner_chat.py --model ./camembert-ner-essai1
    python ner_chat.py --model ./camembert-ner-essai1 --phrase "je veux aller de Lyon à Paris"
"""

import argparse
import sys
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ============================================================
# ARGS
# ============================================================
parser = argparse.ArgumentParser(description="NER interactif — détection DEP/ARR")
parser.add_argument("--model", type=str, default="./camembert-ner-essai1",
                    help="Chemin vers le modèle fine-tuné")
parser.add_argument("--phrase", type=str, default=None,
                    help="Phrase unique à analyser (mode non-interactif)")
args = parser.parse_args()

# ============================================================
# CHARGEMENT
# ============================================================
print(f"\n Chargement du modèle depuis '{args.model}'...")
try:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = AutoModelForTokenClassification.from_pretrained(args.model)
except Exception as e:
    print(f" Impossible de charger le modèle : {e}")
    sys.exit(1)

ner_pipe = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="first"
)
print("✅ Modèle prêt.\n")

# ============================================================
# ANALYSE D'UNE PHRASE
# ============================================================
def analyze(text):
    results = ner_pipe(text)

    dep_entities = [r for r in results if r["entity_group"] == "DEP"]
    arr_entities = [r for r in results if r["entity_group"] == "ARR"]

    dep = dep_entities[0] if dep_entities else None
    arr = arr_entities[0] if arr_entities else None

    print(f"\n   '{text}'")
    print(f"   DEP : {dep['word'] if dep else '—'}"
          + (f"  (confiance : {dep['score']:.0%})" if dep else ""))
    print(f"   ARR : {arr['word'] if arr else '—'}"
          + (f"  (confiance : {arr['score']:.0%})" if arr else ""))

    if len(dep_entities) > 1:
        print(f"    Plusieurs DEP détectés : {[r['word'] for r in dep_entities]}")
    if len(arr_entities) > 1:
        print(f"    Plusieurs ARR détectés : {[r['word'] for r in arr_entities]}")

    return dep, arr

# ============================================================
# MODE PHRASE UNIQUE
# ============================================================
if args.phrase:
    analyze(args.phrase)
    print()
    sys.exit(0)

# ============================================================
# MODE INTERACTIF
# ============================================================
print("=" * 55)
print("  NER Interactif — Départ / Arrivée")
print("  Tape une phrase, appuie sur Entrée.")
print("  'q' ou Ctrl+C pour quitter.")
print("=" * 55)

while True:
    try:
        text = input("\n> ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n\nAu revoir !")
        break

    if not text:
        continue
    if text.lower() in ("q", "quit", "exit"):
        print("Au revoir !")
        break

    analyze(text)
