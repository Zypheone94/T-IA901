import geopandas as gpd
import random
import json
import re
import unicodedata
from transformers import AutoTokenizer
CHECKPOINT = "camembert-base"
random.seed(42)
content = gpd.read_file('./data/communes-20220101.shp')
cities = content["nom"].dropna().unique().tolist()
random.shuffle(cities)

split_cities = int(0.9 * len(cities))
cities_train_val = cities[:split_cities]
cities_test_only = cities[split_cities:]

print(f"Villes pour train/val : {len(cities_train_val)}")
print(f"Villes réservées au test (Zero-Shot) : {len(cities_test_only)}")
names_train = [
    "Maxime", "Jo", "Solange", "Clara", "Camille", "Alex", "Jean-Pierre",
    "Alain", "Naya", "Kevin", "Clemence", "Jessica", "Rob", "Mario",
    "Paris", "Florence", "Jean-Marie", "Leopoldine", "George"  ]

names_test_only = ["Pierre-Edouard", "Karl", "Copernicus", "Roxane", "Ibanez", "Ezekiel", "Thor", "Kratos"]
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
    "Demain, je vais voir mon amie {name} qui habite à {arr}, je pars de {dep}",
    "Comment je me rends à {arr} depuis {dep} pour aller voir ma copine {name}",
    "Mon pote {name} m'attend à {arr}, je pars de {dep}",
    "Je passe par Lyon pour aller de {dep} à {arr}",
    "Jvais à {arr} dpuis {dep}",
    "chui a {dep} jveu alé a {arr}",
]
templates_complets_test_only = [
    "Je passe par Le Mans, Angers et Nantes pour rejoindre {arr} depuis {dep}",
    "Vol {dep}-{arr}, c'est pour quand ?",
    "Jsuis {dep}. Bon jvais à {arr}",
    "Retour de {arr} vers {dep}",
]

templates_arr_seul_train = [
    "Je vais à {arr}",
    "Gare de {arr}",
    "Nancy m'a dit d'aller à {arr}",
    "Faut qu'jaille à {arr}",
    "Direction {arr}",
]

templates_arr_seul_test_only = [
    "Prochain train pour {arr}",
]

templates_sans_ville = [
    "Il fait beau aujourd'hui",
    "Le temps est à l'averse",
    "J'aime voyager",
    "Quelle heure est-il ?",
    "Je vais au restaurant avec mon ami Maxime",
]


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


def apply_noise(sentence, dep, arr):
    if random.random() < 0.20:
        sentence = sentence.lower()
        if dep: dep = dep.lower()
        if arr: arr = arr.lower()

    if random.random() < 0.15:
        sentence = remove_accents(sentence)
        if dep: dep = remove_accents(dep)
        if arr: arr = remove_accents(arr)

    return sentence, dep, arr
def create_ner_labels(sentence, start_city, end_city):

    labels = ['O'] * len(sentence)

    def annotate_entity(entity, label_prefix):
        if not entity:
            return
        pattern = r'(?<!\w)' + re.escape(entity) + r'(?!\w)'
        match = re.search(pattern, sentence, flags=re.IGNORECASE)
        if not match:
            idx = sentence.lower().find(entity.lower())
            if idx != -1:
                start, end = idx, idx + len(entity)
            else:
                return
        else:
            start, end = match.span()

        labels[start] = f'B-{label_prefix}'
        for i in range(start + 1, end):
            labels[i] = f'I-{label_prefix}'

    annotate_entity(start_city, 'DEP')
    annotate_entity(end_city, 'ARR')

    return labels


def align_labels_with_tokens(tokenizer, sentence, char_labels):
    encoding = tokenizer(sentence, return_offsets_mapping=True, truncation=True, max_length=128)
    label2id = {'O': 0, 'B-DEP': 1, 'I-DEP': 2, 'B-ARR': 3, 'I-ARR': 4}
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
                     templates_arr_seul, tokenizer):
    samples = []

    for _ in range(n_samples):
        template_type = random.choices(
            ['complet', 'arr_seul', 'sans_ville'],
            weights=[0.7, 0.2, 0.1]
        )[0]

        dep, arr, name = None, None, None

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
            arr = random.choice(cities_pool)
            template = random.choice(templates_arr_seul)
            sentence = template.format(arr=arr)

        else:
            sentence = random.choice(templates_sans_ville)

        # Application du bruit (minuscules/accents)
        sentence, dep, arr = apply_noise(sentence, dep, arr)

        # Labellisation robuste
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


print("Chargement du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

print("Génération du dataset...")

train_data = generate_samples(
    6000,
    cities_train_val,
    names_train,
    templates_complets_train,
    templates_arr_seul_train,
    tokenizer
)

val_data = generate_samples(
    2000,
    cities_train_val,
    names_train,
    templates_complets_train,
    templates_arr_seul_train,
    tokenizer
)

test_samples_standard = generate_samples(
    1000,
    cities_train_val,
    names_train,
    templates_complets_train,
    templates_arr_seul_train,
    tokenizer
)

test_samples_nouveaux = generate_samples(
    1000,
    cities_test_only,
    names_train + names_test_only,
    templates_complets_train + templates_complets_test_only,
    templates_arr_seul_train + templates_arr_seul_test_only,
    tokenizer
)

test_data = test_samples_standard + test_samples_nouveaux
random.shuffle(test_data)

all_data = {
    'train': train_data,
    'val': val_data,
    'test': test_data
}

with open('dataset_ner_cities_split_v2.json', 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print(f"\n Dataset sauvegardé sous 'dataset_ner_cities_split_v2.json'")
print(f"   Train: {len(train_data)} samples")
print(f"   Val: {len(val_data)} samples")
print(f"   Test: {len(test_data)} samples")