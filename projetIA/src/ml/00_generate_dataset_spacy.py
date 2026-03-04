import json
import random
import re
from pathlib import Path
from collections import Counter
import geopandas as gpd

SEED = 42

SHP_PATH = Path("data/communes/communes-20220101.shp")
OUT_PATH = Path("data/ner/dataset_spacy_generated.json")

# Taille dataset
N_TRAIN = 6000
N_VAL = 2000
N_TEST = 2000

# Répartition des types
WEIGHTS = {"trip": 0.75, "arr_only": 0.15, "no_city": 0.10}

# Probabilités d'augmentation
P_CASING_VARIATION = 0.20   # change casse (ex: "je vais a PARIS")
P_ADD_CONTEXT = 0.35        # ajoute un contexte (phrase longue)
P_ADD_STOPS = 0.20          # ajoute des villes étapes (stop1/stop2) en plus
P_ADD_SLANG = 0.35          # version familière (jvais, chui, etc.)
P_ADD_TYPO = 0.25           # introduire fautes contrôlées

# Fautes contrôlées : intensité
TYPO_MAX_EDITS = 1          # 1 = léger (recommandé au début)


templates_trip = [
    "Je veux aller de {dep} vers {arr}",
    "Je souhaite aller de {dep} à {arr}",
    "Comment aller de {dep} à {arr} ?",
    "Je pars de {dep} pour {arr}",
    "Trajet {dep} {arr}",
    "Itinéraire {dep} vers {arr}",
    "{dep} jusqu'à {arr}",
    "Aller de {dep} à {arr}",
    "Je suis à {dep} et je dois me rendre à {arr}",
    "Je décolle de {dep} direction {arr}",
    "Prochain départ de {dep} pour {arr}",
]

templates_trip_long = [
    "Demain matin, je dois partir de {dep} et aller à {arr}, tu peux m'aider ?",
    "Je vais voir un ami ce week-end : départ {dep}, arrivée {arr}, c'est quoi le meilleur trajet ?",
    "Je dois rejoindre {arr} en partant de {dep}, et si possible éviter les détours.",
    "Je suis actuellement à {dep} et j'ai un rendez-vous à {arr} dans l'après-midi.",
    "Pour un voyage prévu demain, je pars de {dep} et j'arrive à {arr}.",
]

templates_trip_with_stops = [
    "Je pars de {dep} et je vais à {arr} en passant par {stop1}",
    "Je veux aller de {dep} à {arr}, je passe d'abord par {stop1} puis {stop2}",
    "Trajet {dep} -> {stop1} -> {arr}",
]

templates_arr_only = [
    "Je vais à {arr}",
    "Gare de {arr}",
    "Direction {arr}",
    "Prochain train pour {arr}",
]

templates_arr_only_long = [
    "J'ai un rendez-vous et je dois aller à {arr}, tu peux me guider ?",
    "Je dois absolument être à {arr} ce soir, c'est quoi l'itinéraire ?",
    "Je veux rejoindre {arr} mais je ne sais pas quel trajet prendre.",
]

templates_no_city = [
    "Il fait beau aujourd'hui",
    "Le temps est à l'averse",
    "J'aime voyager",
    "Quelle heure est-il ?",
    "Tu peux m'aider ?",
    "Je suis fatigué aujourd'hui",
]

# Petites variantes familières (simple mais efficace)
slang_map = {
    r"\bje veux\b": "j'veux",
    r"\bje vais\b": "jvais",
    r"\bje pars\b": "jpars",
    r"\bje souhaite\b": "j'aimerais",
    r"\bje suis\b": "chui",
    r"\btu peux\b": "tu peux stp",
    r"\bcomment\b": "comment ça",
    r"\bprochain\b": "prochain",
    r"\bde\b": "de",  # on garde
}

def check_shapefile_complete(shp_path: Path):
    base = shp_path.with_suffix("")
    required = [base.with_suffix(".shp"), base.with_suffix(".shx"), base.with_suffix(".dbf")]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Shapefile incomplet. Il manque: " + ", ".join(missing) +
            f"\n Mets tous les fichiers du shapefile dans: {shp_path.parent}"
        )

def make_entities(text: str, dep: str | None, arr: str | None):
    """Retourne des entités spaCy au format [start, end, label]."""
    entities = []
    if dep:
        s = text.find(dep)
        if s != -1:
            entities.append([s, s + len(dep), "DEP"])
    if arr:
        s = text.find(arr)
        if s != -1:
            entities.append([s, s + len(arr), "ARR"])
    return entities

def apply_casing_variation(text: str) -> str:
    """Varie la casse (léger) : random upper/lower/title sur quelques mots."""
    if random.random() > P_CASING_VARIATION:
        return text
    words = text.split()
    if not words:
        return text
    # on modifie 1 ou 2 mots max
    k = 1 if len(words) < 6 else 2
    idxs = random.sample(range(len(words)), k=min(k, len(words)))
    for i in idxs:
        w = words[i]
        mode = random.choice(["upper", "lower", "title"])
        if mode == "upper":
            words[i] = w.upper()
        elif mode == "lower":
            words[i] = w.lower()
        else:
            words[i] = w.title()
    return " ".join(words)

def apply_slang(text: str) -> str:
    """Transforme certains patterns en version familière."""
    if random.random() > P_ADD_SLANG:
        return text
    out = text
    for pattern, repl in slang_map.items():
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    # petites contractions
    out = out.replace(" je ", " j'") if random.random() < 0.1 else out
    return out

def typo_one_edit(word: str) -> str:
    """Une faute légère sur un mot (swap, drop, replace)."""
    if len(word) < 4:
        return word
    choice = random.choice(["swap", "drop", "replace"])
    i = random.randint(1, len(word) - 2)

    if choice == "swap":
        # échange 2 lettres adjacentes
        chars = list(word)
        chars[i], chars[i+1] = chars[i+1], chars[i]
        return "".join(chars)

    if choice == "drop":
        return word[:i] + word[i+1:]

    # replace
    letters = "abcdefghijklmnopqrstuvwxyz"
    return word[:i] + random.choice(letters) + word[i+1:]

def apply_typos_controlled(text: str, protected_spans: list[tuple[int, int]]) -> tuple[str, bool]:
    """
    Ajoute des fautes MAIS sans toucher aux villes (DEP/ARR) pour ne pas casser les labels.
    protected_spans = [(start,end), ...] zones à ne pas modifier.
    """
    if random.random() > P_ADD_TYPO:
        return text, False

    # On découpe en tokens simples (mots)
    # Important : on ne modifie pas dans les spans des villes
    def is_protected(pos: int) -> bool:
        return any(s <= pos < e for s, e in protected_spans)

    chars = list(text)
    changed = False

    # On va faire 1 edit max par défaut
    n_edits = random.randint(1, TYPO_MAX_EDITS)
    attempts = 0
    max_attempts = 30

    while n_edits > 0 and attempts < max_attempts:
        attempts += 1
        # choisir un mot alphabétique
        matches = list(re.finditer(r"[A-Za-zÀ-ÖØ-öø-ÿ']{4,}", text))
        if not matches:
            break
        m = random.choice(matches)
        start, end = m.start(), m.end()

        # si le mot chevauche une zone protégée, on skip
        if any(is_protected(p) for p in range(start, end)):
            continue

        old = text[start:end]
        new = typo_one_edit(old)

        # appliquer remplacement
        text = text[:start] + new + text[end:]
        changed = True
        n_edits -= 1

    return text, changed

def pick_city(cities_pool):
    # simple helper
    return random.choice(cities_pool)

# -------------------------
# Génération
# -------------------------

def generate_one(cities_pool, city_train_pool, city_all_pool):
    """
    Retour:
      sample: [text, {"entities": [...]}]
      meta: dict (type, used_template_family, has_typo, length, etc.)
    """
    r = random.random()
    meta = {"kind": None, "template_family": None, "has_typo": False, "length": 0}

    # --- TRIP (DEP + ARR) ---
    if r < WEIGHTS["trip"]:
        dep = pick_city(cities_pool)
        arr = pick_city([c for c in cities_pool if c != dep])

        # parfois phrase longue
        if random.random() < P_ADD_CONTEXT:
            base_template = random.choice(templates_trip_long)
            meta["template_family"] = "trip_long"
        else:
            base_template = random.choice(templates_trip)
            meta["template_family"] = "trip"

        # parfois avec étapes (sans label supplémentaire, juste bruit)
        stop1 = stop2 = None
        if random.random() < P_ADD_STOPS:
            stop1 = pick_city([c for c in city_all_pool if c not in (dep, arr)])
            if random.random() < 0.5:
                stop2 = pick_city([c for c in city_all_pool if c not in (dep, arr, stop1)])
            base_template = random.choice(templates_trip_with_stops)
            meta["template_family"] = "trip_stops"

        if "{stop2}" in base_template and stop2 is None:
            # si template demande stop2 mais on en a pas, on repick
            stop2 = pick_city([c for c in city_all_pool if c not in (dep, arr, stop1)])

        text = base_template.format(dep=dep, arr=arr, stop1=stop1, stop2=stop2)

        # entities avant modifications (pour protéger spans)
        entities = make_entities(text, dep, arr)
        protected = [(s, e) for s, e, _ in entities]

        # variations
        text = apply_slang(text)
        text = apply_casing_variation(text)
        text, typo_changed = apply_typos_controlled(text, protected)
        meta["has_typo"] = typo_changed

        # recalcul entities après mods (car casing/typos peuvent changer indices)
        # IMPORTANT: si on modifie le texte, les positions peuvent bouger.
        # Ici, on protège les villes donc elles restent identiques -> find() marche.
        entities = make_entities(text, dep, arr)
        meta["kind"] = "trip"
        meta["length"] = len(text)

        return [text, {"entities": entities}], meta

    # --- ARR ONLY (1 ville) ---
    if r < WEIGHTS["trip"] + WEIGHTS["arr_only"]:
        arr = pick_city(cities_pool)

        if random.random() < P_ADD_CONTEXT:
            base_template = random.choice(templates_arr_only_long)
            meta["template_family"] = "arr_only_long"
        else:
            base_template = random.choice(templates_arr_only)
            meta["template_family"] = "arr_only"

        text = base_template.format(arr=arr)
        entities = make_entities(text, None, arr)
        protected = [(s, e) for s, e, _ in entities]

        text = apply_slang(text)
        text = apply_casing_variation(text)
        text, typo_changed = apply_typos_controlled(text, protected)
        meta["has_typo"] = typo_changed

        entities = make_entities(text, None, arr)
        meta["kind"] = "arr_only"
        meta["length"] = len(text)

        return [text, {"entities": entities}], meta

    # --- NO CITY ---
    text = random.choice(templates_no_city)
    text = apply_slang(text)
    text = apply_casing_variation(text)
    text, typo_changed = apply_typos_controlled(text, protected_spans=[])
    meta["has_typo"] = typo_changed
    meta["kind"] = "no_city"
    meta["template_family"] = "no_city"
    meta["length"] = len(text)

    return [text, {"entities": []}], meta


def generate_split(n_samples, cities_pool, city_train_pool, city_all_pool, stats_counter):
    out = []
    for _ in range(n_samples):
        sample, meta = generate_one(cities_pool, city_train_pool, city_all_pool)
        out.append(sample)

        stats_counter["kind"][meta["kind"]] += 1
        stats_counter["template_family"][meta["template_family"]] += 1
        stats_counter["typo"][str(meta["has_typo"])] += 1
        stats_counter["length_bucket"][bucket_length(meta["length"])] += 1

    return out

def bucket_length(n: int) -> str:
    if n <= 20:
        return "0-20"
    if n <= 40:
        return "21-40"
    if n <= 60:
        return "41-60"
    if n <= 90:
        return "61-90"
    return "90+"

def init_stats():
    return {
        "kind": Counter(),
        "template_family": Counter(),
        "typo": Counter(),
        "length_bucket": Counter(),
    }

def print_stats(title: str, stats):
    print(f"\n--- Stats {title} ---")
    print("Types:", dict(stats["kind"]))
    print("Templates:", dict(stats["template_family"]))
    print("Typos:", dict(stats["typo"]))
    print("Longueurs:", dict(stats["length_bucket"]))


def main():
    random.seed(SEED)

    if not SHP_PATH.exists():
        raise FileNotFoundError(f"Shapefile introuvable: {SHP_PATH}")
    check_shapefile_complete(SHP_PATH)

    gdf = gpd.read_file(SHP_PATH)
    if "nom" not in gdf.columns:
        raise ValueError(f"Colonne 'nom' absente. Colonnes dispo: {list(gdf.columns)}")

    cities = gdf["nom"].astype(str).tolist()
    random.shuffle(cities)

    split = int(0.9 * len(cities))
    cities_train_val = cities[:split]
    cities_test_only = cities[split:]

    print(f"Villes train/val: {len(cities_train_val)} | villes test only: {len(cities_test_only)}")

    # Stats
    stats_train = init_stats()
    stats_val = init_stats()
    stats_test = init_stats()

    train = generate_split(N_TRAIN, cities_train_val, cities_train_val, cities_train_val + cities_test_only, stats_train)
    val = generate_split(N_VAL, cities_train_val, cities_train_val, cities_train_val + cities_test_only, stats_val)

    # test mélange: moitié villes vues + moitié villes nouvelles
    test_a = generate_split(N_TEST // 2, cities_train_val, cities_train_val, cities_train_val + cities_test_only, stats_test)
    test_b = generate_split(N_TEST - len(test_a), cities_train_val + cities_test_only, cities_train_val, cities_train_val + cities_test_only, stats_test)
    test = test_a + test_b
    random.shuffle(test)

    dataset = {"train": train, "val": val, "test": test}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n Dataset spaCy généré:", OUT_PATH)
    print(f"   train={len(train)} | val={len(val)} | test={len(test)}")

    print_stats("TRAIN", stats_train)
    print_stats("VAL", stats_val)
    print_stats("TEST", stats_test)


if __name__ == "__main__":
    main()