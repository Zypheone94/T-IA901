from transformers import AutoTokenizer
import geopandas as gpd
import random
import json

content = gpd.read_file('../data/communes-20220101.shp')
checkpoint = "camembert-base"
cities = content["nom"].to_numpy()

names_train = ["Maxime", "Jo", "Solange", "Clara", "Camille", "Alex",
               "Jean-Pierre", "Alain", "Naya", "Kevin", "Clemence",
               "Jessica", "Rob", "Mario"]

names_test_only = ["Paris", "Florence", "Jean-Marie", "Pierre-Edouard",
                   "Karl", "Copernicus", "Pierre-Alexandre", "Leopoldine",
                   "Roxane", "Ibanez", "Ezekiel", "Thor", "Kratos",
                   "Saint-Chéron", "George"]

random.shuffle(cities)
split_cities = int(0.9 * len(cities))
cities_train_val = cities[:split_cities]
cities_test_only = cities[split_cities:]

print(f"Villes pour train/val : {len(cities_train_val)}")
print(f"Villes réservées au test : {len(cities_test_only)}")

# Templates pour train/val
templates_complets_train = [
    "Je veux aller de {dep} vers {arr}",
    "Je souhaite aller de {dep} à {arr}",
    "Comment aller de {dep} à {arr} ?",
    "Je veux me rendre à {dep} de {arr}",
    "Quel est le trajet entre {dep} et {arr} ?",
    "Je cherche un itinéraire de {dep} vers {arr}",
    "Comment aller à {arr} en partant de {dep} ?",
    "Je vais à {arr}, je pars de {dep}",
    "Cherche un billet pour {arr} au départ de {dep}",
    "Trajet pour {arr} depuis {dep}",
    "Est-ce qu'on peut rejoindre {arr} en venant de {dep} ?",
    "Trajet {dep} {arr}",
    "Itinéraire {dep} vers {arr}",
    "{dep} jusqu'à {arr}",
    "Aller de {dep} à {arr}",
    "Liaison {dep}-{arr}",
    "Je suis à {dep} et je dois me rendre à {arr}",
    "Je décolle de {dep} direction {arr}",
    "Faut que j'aille à {arr}, je suis actuellement à {dep}",
    "Y'a quoi comme transport entre {dep} et {arr} ?",
    "je pars pour {arr} de {dep}",
    "Prochain départ de {dep} pour {arr}",
]

# Templates UNIQUEMENT pour test
templates_complets_test_only = [
    "Demain, je vais voir mon amie {name} qui habite à {arr}, je pars de {dep}",
    "Comment je me rends à {arr} depuis {dep} pour aller voir ma copine {name}",
    "Retour de {arr} vers {dep}",
    "Je passe par Le Mans, Angers et Nantes pour rejoindre {arr} depuis {dep}",
    "Je passe par Lyon pour aller de {dep} à {arr}",
    "Jvais à {arr} dpuis {dep}",
    "chui a {dep} jveu alé a {arr}",
    "Jsuis {dep}. Bon jvais à {arr}",
    "Mon pote {name} m'attend à {arr}, je pars de {dep}",
    "Vol {dep}-{arr}, c'est pour quand ?",
]

# Templates arr_seul pour train/val
templates_arr_seul_train = [
    "Je vais à {arr}",
    "Gare de {arr}",
]

# Templates arr_seul UNIQUEMENT pour test
templates_arr_seul_test_only = [
    "Nancy m'a dit d'aller à {arr}",
    "Faut qu'jaille à {arr}",
    "Direction {arr}",
    "Prochain train pour {arr}",
]

# Templates sans ville (communs à tous)
templates_sans_ville = [
    "Il fait beau aujourd'hui",
    "Le temps est à l'averse",
    "J'aime voyager",
    "Quelle heure est-il ?",
]


def create_ner_labels(sentence, start_city, end_city):
    labels = ['O'] * len(sentence)

    if start_city:
        dep_start = sentence.find(start_city)
        if dep_start != -1:
            labels[dep_start] = 'B-DEP'
            for i in range(dep_start + 1, dep_start + len(start_city)):
                labels[i] = 'I-DEP'

    if end_city:
        arr_start = sentence.find(end_city)
        if arr_start != -1:
            labels[arr_start] = 'B-ARR'
            for i in range(arr_start + 1, arr_start + len(end_city)):
                labels[i] = 'I-ARR'

    return labels


def align_labels_with_tokens(tokenizer, sentence, char_labels):
    encoding = tokenizer(sentence, return_offsets_mapping=True, truncation=True, max_length=128)

    label2id = {
        'O': 0,
        'B-DEP': 1,
        'I-DEP': 2,
        'B-ARR': 3,
        'I-ARR': 4
    }

    token_labels = []

    for idx, (start, end) in enumerate(encoding['offset_mapping']):
        if start == end == 0:
            token_labels.append(-100)
        else:
            token_labels.append(label2id[char_labels[start]])

    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'labels': token_labels
    }


def generate_samples(n_samples, cities_pool, names_pool, templates_complets,
                     templates_arr_seul, is_test=False):
    samples = []
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    for _ in range(n_samples):
        template_type = random.choices(
            ['complet', 'arr_seul', 'sans_ville'],
            weights=[0.7, 0.2, 0.1]
        )[0]

        if template_type == 'complet':
            dep = random.choice(cities_pool)
            arr = random.choice([v for v in cities_pool if v != dep])
            name = random.choice(names_pool)
            template = random.choice(templates_complets)

            if "{name}" in template:
                sentence = template.format(dep=dep, arr=arr, name=name)
            else:
                sentence = template.format(dep=dep, arr=arr)

        elif template_type == 'arr_seul':
            dep = None
            arr = random.choice(cities_pool)
            template = random.choice(templates_arr_seul)
            sentence = template.format(arr=arr)

        else:
            dep = None
            arr = None
            template = random.choice(templates_sans_ville)
            sentence = template

        char_labels = create_ner_labels(sentence, dep, arr)
        tokenized_data = align_labels_with_tokens(tokenizer, sentence, char_labels)

        samples.append({
            'sentence': sentence,
            'start_city': dep,
            'end_city': arr,
            'input_ids': tokenized_data['input_ids'],
            'attention_mask': tokenized_data['attention_mask'],
            'labels': tokenized_data['labels']
        })

    return samples


print("Génération du dataset...")

# Train : 6000 samples (60%) avec villes et templates train
train_data = generate_samples(
    6000,
    cities_train_val,
    names_train,
    templates_complets_train,
    templates_arr_seul_train
)

# Validation : 2000 samples (20%) avec villes et templates train
val_data = generate_samples(
    2000,
    cities_train_val,
    names_train,
    templates_complets_train,
    templates_arr_seul_train
)

# Test : 2000 samples (20%) avec :
test_samples_standard = generate_samples(
    1000,
    cities_train_val,
    names_train,
    templates_complets_train,
    templates_arr_seul_train
)

test_samples_nouveaux = generate_samples(
    1000,
    list(cities_train_val) + list(cities_test_only),
    names_train + names_test_only,
    templates_complets_train + templates_complets_test_only,
    templates_arr_seul_train + templates_arr_seul_test_only
)

test_data = test_samples_standard + test_samples_nouveaux
random.shuffle(test_data)

all_data = {
    'train': train_data,
    'val': val_data,
    'test': test_data
}

with open('dataset_ner_cities_split.json', 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print(f"\n✅ Dataset sauvegardé :")
print(f"   Train: {len(train_data)} samples (60%)")
print(f"   Val: {len(val_data)} samples (20%)")
print(f"   Test: {len(test_data)} samples (20%)")
print(f"   → dont {len(test_samples_nouveaux)} avec nouveaux patterns/villes")