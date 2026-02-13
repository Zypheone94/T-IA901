import geopandas as gpd
import random
import json
import re

# ======================================
# LOAD CITIES
# ======================================

content = gpd.read_file('../data/communes-20220101-shp/communes-20220101.shp')
cities = content["nom"].dropna().unique().tolist()

random.shuffle(cities)

split = int(0.9 * len(cities))
cities_train = cities[:split]
cities_test = cities[split:]

# ======================================
# LABELS
# ======================================

label2id = {
    "O": 0,
    "B-DEP": 1,
    "I-DEP": 2,
    "B-ARR": 3,
    "I-ARR": 4
}

# ===============================
# TEMPLATES
# ===============================

templates_dep_arr = [
    "Je veux aller de {dep} vers {arr}",
    "Je souhaite aller de {dep} à {arr}",
    "Comment aller de {dep} à {arr} ?",
    "Quel est le trajet entre {dep} et {arr} ?",
    "Je cherche un itinéraire de {dep} vers {arr}",
    "Comment aller à {arr} en partant de {dep} ?",
    "Je vais à {arr}, je pars de {dep}",
    "Cherche un billet pour {arr} au départ de {dep}",
    "Trajet pour {arr} depuis {dep}",
    "Est-ce qu'on peut rejoindre {arr} en venant de {dep} ?",
    "Itinéraire {dep} vers {arr}",
    "{dep} jusqu'à {arr}",
    "Aller de {dep} à {arr}",
    "Je suis à {dep} et je dois me rendre à {arr}",
    "Je décolle de {dep} direction {arr}",
    "Faut que j'aille à {arr}, je suis actuellement à {dep}",
    "Y'a quoi comme transport entre {dep} et {arr} ?",
    "je pars pour {arr} de {dep}",
    "Prochain départ de {dep} pour {arr}",
    "passer a {arr} après avoir été a {dep}",
    "partir d’{dep} pour finir a {arr}",
    "{dep} direction {arr}",
    "s'en aller de {dep} pour etre a {arr}",
    "je viens de {dep} pour {arr}",
    "je quitte {dep} pour {arr}",
    "je suis sur {dep} et je veux me rendre a {arr}",
    "rejoindre {arr} de {dep}",
    "je veux faire le touriste à {arr} mais je suis encore à {dep}",
    "{dep} je me rends à {arr}",
    "un train de {dep} a {arr}",
    "de {dep} a {arr}",
    "je veux me trouver a {arr} en venant de {dep}",
    "dis moi l'itinéraire pour aller a {arr} depuis {dep}",
    "de {dep} a {arr} comme un touriste italien",
    "{dep} vers {arr}",
    "{dep} pour aller a {arr}",
    "comment partir de {dep} pour visiter {arr}",
    "aller de {dep} a {arr} comme ca",
    "je veux me balader de {dep} a {arr}",
    "{dep} vers {arr} maintenant",
    "{dep} pour rejoindre {arr}",
    "sortir de {dep} a {arr}",
    "a {dep} et je m'en vais a {arr}",
    "je vais d’{dep} à {arr}",
    "je pars de {dep} pour {arr}",
    "je m’en vais de {dep} pour {arr}",
    "de {dep} à {arr} comme ca",
    "{dep} vers {arr} hop",
    "{dep} a {arr} maintenant",
    "{dep} jusqu’à {arr}",
    "je dois aller de {dep} à {arr}",
    "je pars de {dep} direction {arr}",
    "{dep} vers {arr} faut que je parte",
    "je veux bouger de {dep} à {arr}",
    "{dep} encore {arr}",
    "{dep} je dois aller à {arr}",
    "{dep} je veux rejoindre {arr}",
    "je file de {dep} à {arr}",
    "{dep} je veux aller à {arr}",
    "{dep} je pars pour {arr}",
    "{dep} je dois filer à {arr}",
]




templates_dep_arr = [
    "Je veux aller de {dep} vers {arr}",
    "Je souhaite aller de {dep} à {arr}",
    "Comment aller de {dep} à {arr} ?",
    "Je cherche un itinéraire de {dep} vers {arr}",
    "Je vais à {arr}, je pars de {dep}",
]

templates_arr_only = [
    "je vais à {arr}",
    "prochain train pour {arr}",
    "je dois aller à {arr}",
]

templates_dep_only = [
    "je suis à {dep}",
    "je pars de {dep}",
]

templates_city_no_role = [
    "{city} est magnifique",
    "j'ai visité {city} l'an dernier",
]

templates_no_city = [
    "il fait beau aujourd'hui",
    "je prépare mon sac",
    "je regarde les horaires",
]

# ======================================
# SIMPLE TOKENIZER (word-level)
# ======================================

def tokenize(text):
    # découpe mots + ponctuation
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

# ======================================
# BUILD BIO TAGS
# ======================================

def build_example(sentence, dep=None, arr=None):
    tokens = tokenize(sentence)
    ner_tags = ["O"] * len(tokens)

    def tag_city(city, prefix):
        city_tokens = tokenize(city)
        for i in range(len(tokens)):
            if tokens[i:i+len(city_tokens)] == city_tokens:
                ner_tags[i] = f"B-{prefix}"
                for j in range(1, len(city_tokens)):
                    ner_tags[i+j] = f"I-{prefix}"

    if dep:
        tag_city(dep, "DEP")

    if arr:
        tag_city(arr, "ARR")

    ner_tags_ids = [label2id[tag] for tag in ner_tags]

    return {
        "tokens": tokens,
        "ner_tags": ner_tags_ids
    }

# ======================================
# DATASET GENERATION
# ======================================

def generate_dataset(n_samples, cities_pool):
    dataset = []

    for _ in range(n_samples):
        mode = random.random()

        if mode < 0.5:
            dep = random.choice(cities_pool)
            arr = random.choice([c for c in cities_pool if c != dep])
            template = random.choice(templates_dep_arr)
            sentence = template.format(dep=dep, arr=arr)
            dataset.append(build_example(sentence, dep, arr))

        elif mode < 0.7:
            arr = random.choice(cities_pool)
            template = random.choice(templates_arr_only)
            sentence = template.format(arr=arr)
            dataset.append(build_example(sentence, arr=arr))

        elif mode < 0.85:
            dep = random.choice(cities_pool)
            template = random.choice(templates_dep_only)
            sentence = template.format(dep=dep)
            dataset.append(build_example(sentence, dep=dep))

        else:
            sentence = random.choice(templates_no_city)
            dataset.append(build_example(sentence))

    return dataset

# ======================================
# GENERATE DATA
# ======================================

train_data = generate_dataset(20000, cities_train)
val_data = generate_dataset(5000, cities_train)
test_data = generate_dataset(5000, cities_test)

dataset = {
    "train": train_data,
    "validation": val_data,
    "test": test_data
}

with open("dataset_camembert_ner.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print("Dataset generated")
print("Train:", len(train_data))
print("Validation:", len(val_data))
print("Test:", len(test_data))

