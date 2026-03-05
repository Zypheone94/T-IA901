#!/usr/bin/env python3
# Usage:
#   python app.py
#   python app.py --date 2026-03-10
#   python app.py --phrase "je veux aller de Lyon a Paris"

import argparse
import sys
import datetime
from pathlib import Path

# Chemin absolu vers ShortestPath 
THIS_FILE = Path(__file__).resolve()
# app.py est dans : T-IA901-clean/majid/src/
# ShortestPath est dans : T-IA901-clean/ShortestPath/
SHORTESTPATH_DIR = THIS_FILE.parent.parent.parent / "ShortestPath"

if not SHORTESTPATH_DIR.exists():
    print(f"Erreur : dossier ShortestPath introuvable à {SHORTESTPATH_DIR}")
    print("Verifie la structure du projet.")
    sys.exit(1)

sys.path.insert(0, str(SHORTESTPATH_DIR))
from shortestPathSNCF import plan_itinerary

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ============================================================
# ARGS
# ============================================================
parser = argparse.ArgumentParser(description="Chatbot itineraire SNCF")
parser.add_argument("--model",  type=str,
                    default=str(THIS_FILE.parent / "camembert-ner-essai1"))
parser.add_argument("--gtfs",   type=str,
                    default=str(SHORTESTPATH_DIR / "data" / "sncf_gtfs"))                  
parser.add_argument("--shp",    type=str,
    default=str(THIS_FILE.parent.parent / "data" / "communes-20220101-shp" / "communes-20220101.shp"))
parser.add_argument("--date",   type=str, default=None)
parser.add_argument("--phrase", type=str, default=None)
args = parser.parse_args()

date_str = args.date or datetime.date.today().isoformat()

# ============================================================
# CHARGEMENT NER
# ============================================================
print(f"\nChargement du modele NER depuis '{args.model}'...")
try:
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model_ner = AutoModelForTokenClassification.from_pretrained(args.model, local_files_only=True)
except Exception as e:
    print(f"Impossible de charger le modele NER : {e}")
    sys.exit(1)

ner_pipe = pipeline(
    "token-classification",
    model=model_ner,
    tokenizer=tokenizer,
    aggregation_strategy="first"
)
print("Modele NER pret.\n")


# ============================================================
# EXTRACTION DEP/ARR
# ============================================================
def extract_dep_arr(text):
    results = ner_pipe(text)
    deps = [r for r in results if r["entity_group"] == "DEP"]
    arrs = [r for r in results if r["entity_group"] == "ARR"]

    if not deps and not arrs:
        raise ValueError("Aucune ville detectee. Essaie : 'je veux aller de Lyon a Paris'")
    if not deps:
        raise ValueError(f"Depart non detecte. Arrivee trouvee : '{arrs[0]['word']}'")
    if not arrs:
        raise ValueError(f"Arrivee non detectee. Depart trouve : '{deps[0]['word']}'")

    return deps[0]["word"].strip(), arrs[0]["word"].strip(), deps[0]["score"], arrs[0]["score"]


# ============================================================
# PIPELINE COMPLET
# ============================================================
def process(text):
    try:
        dep, arr, dep_score, arr_score = extract_dep_arr(text)
    except ValueError as e:
        print(f"\n  {e}\n")
        return

    print(f"\n  Depart  : {dep}  ({dep_score:.0%})")
    print(f"  Arrivee : {arr}  ({arr_score:.0%})")
    print(f"  Date    : {date_str}\n")

    try:
        plan_itinerary(
            from_commune=dep,
            to_commune=arr,
            date_str=date_str,
            COMMUNES_SHP=args.shp,
            GTFS_DIR=args.gtfs,
        )
    except FileNotFoundError as e:
        print(f"Fichier manquant : {e}")
    except ValueError as e:
        print(f"Commune introuvable : {e}")
        print("Verifie l'orthographe ou essaie le nom officiel INSEE.")
    except Exception as e:
        print(f"Erreur : {type(e).__name__}: {e}")


# ============================================================
# MODE PHRASE UNIQUE
# ============================================================
if args.phrase:
    process(args.phrase)
    sys.exit(0)

# ============================================================
# MODE INTERACTIF
# ============================================================
print("=" * 55)
print("  Chatbot itineraire SNCF")
print(f"  Date : {date_str}  (tape 'date YYYY-MM-DD' pour changer)")
print("  'q' pour quitter.")
print("=" * 55 + "\n")

while True:
    try:
        text = input("> ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nAu revoir !")
        break

    if not text:
        continue
    if text.lower() in ("q", "quit", "exit"):
        print("Au revoir !")
        break

    if text.lower().startswith("date "):
        new_date = text[5:].strip()
        try:
            datetime.date.fromisoformat(new_date)
            date_str = new_date
            print(f"Date : {date_str}")
        except ValueError:
            print("Format invalide. Utilise : date YYYY-MM-DD")
        continue

    process(text)
    print()
