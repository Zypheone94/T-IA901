import geopandas as gpd
import random
import json



content = gpd.read_file('../data/communes-20220101-shp/communes-20220101.shp')
cities = content["nom"].to_numpy()

random.shuffle(cities)
split_cities = int(0.9 * len(cities))
cities_train_val = cities[:split_cities]
cities_test_only = cities[split_cities:]

print(f"Villes train/val : {len(cities_train_val)}")
print(f"Villes test only : {len(cities_test_only)}")


names_train = [
    "Maxime", "Jo", "Solange", "Clara", "Camille", "Alex",
    "Jean-Pierre", "Alain", "Naya", "Kevin", "Clemence",
    "Jessica", "Rob", "Mario"
]

names_test_only = [
    "Paris", "Florence", "Jean-Marie", "Pierre-Edouard",
    "Karl", "Copernicus", "Pierre-Alexandre", "Leopoldine",
    "Roxane", "Ibanez", "Ezekiel", "Thor", "Kratos",
    "Saint-Chéron", "George"
]


# Templates pour train/val
templates_complets_train = [
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




def extract_entities(sentence, dep, arr):
    entities = []

    if dep:
        dep_lower = dep.lower()
        sent_lower = sentence.lower()
        start = sent_lower.find(dep_lower)
        if start != -1:
            entities.append((start, start + len(dep), "DEP"))

    if arr:
        arr_lower = arr.lower()
        sent_lower = sentence.lower()
        start = sent_lower.find(arr_lower)
        if start != -1:
            entities.append((start, start + len(arr), "ARR"))

    return entities



def random_case(text):
    mode = random.choice(["lower", "upper", "original"])
    if mode == "lower":
        return text.lower()
    elif mode == "upper":
        return text.upper()
    return text



def generate_samples(
    n_samples,
    cities_pool,
    names_pool,
    templates_complets,
    templates_arr_seul
):
    samples = []

    for _ in range(n_samples):
        template_type = random.choices(
            ["complet", "arr_seul", "sans_ville"],
            weights=[0.7, 0.2, 0.1]
        )[0]

        if template_type == "complet":
            dep = random.choice(cities_pool)
            arr = random.choice([c for c in cities_pool if c != dep])
            name = random.choice(names_pool)
            template = random.choice(templates_complets)

            if "{name}" in template:
                sentence = template.format(dep=dep, arr=arr, name=name)
                sentence = random_case(sentence)
            else:
                sentence = template.format(dep=dep, arr=arr)
                sentence = random_case(sentence)

        elif template_type == "arr_seul":
            dep = None
            arr = random.choice(cities_pool)
            template = random.choice(templates_arr_seul)
            sentence = template.format(arr=arr)
            sentence = random_case(sentence)

        else:
            dep = None
            arr = None
            sentence = random.choice(templates_sans_ville)
            sentence = random_case(sentence)

        entities = extract_entities(sentence, dep, arr)

        samples.append((
            sentence,
            {"entities": entities}
        ))

    return samples




print("Génération dataset spaCy...")

train_data = generate_samples(
    6000,
    cities_train_val,
    names_train,
    templates_complets_train,
    templates_arr_seul_train
)

val_data = generate_samples(
    2000,
    cities_train_val,
    names_train,
    templates_complets_train,
    templates_arr_seul_train
)

test_standard = generate_samples(
    1000,
    cities_train_val,
    names_train,
    templates_complets_train,
    templates_arr_seul_train
)

test_nouveaux = generate_samples(
    1000,
    list(cities_train_val) + list(cities_test_only),
    names_train + names_test_only,
    templates_complets_train + templates_complets_test_only,
    templates_arr_seul_train + templates_arr_seul_test_only
)

test_data = test_standard + test_nouveaux
random.shuffle(test_data)

dataset = {
    "train": train_data,
    "val": val_data,
    "test": test_data
}

with open("dataset_spacy_dep_arr_MAJUSCULE.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(" Dataset spaCy généré")
print(f"Train : {len(train_data)}")
print(f"Val   : {len(val_data)}")
print(f"Test  : {len(test_data)}")

