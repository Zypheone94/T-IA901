from load_communes import load_city_index
from city_ruler import build_city_nlp

COMMUNES_PATH = "data/communes/communes-20220101.shp"

def detect(sentence, nlp, city_index):
    from text_normalize import normalize_text
    norm = normalize_text(sentence)
    doc = nlp(norm)

    cities = []
    for ent in doc.ents:
        if ent.label_ == "CITY":
            key = normalize_text(ent.text)
            canonical = city_index.get(key)
            if canonical:
                cities.append(canonical)
    return cities

def main():
    city_index = load_city_index(COMMUNES_PATH)
    nlp = build_city_nlp(city_index)

    tests = [
        "Je veux aller de Paris à Lille",
        "Comment me rendre à Port-Boulet depuis Tours ?",
        "Je vais de saint etienne à bordeaux",
        "Trajet Marseille Cannes",
        "Il fait beau aujourd'hui",
        "je vais à bordeaux",
        "je pars de Saint-Tropez vers Nice",
    ]

    for t in tests:
        print("PHRASE:", t)
        print("VILLES:", detect(t, nlp, city_index))
        print("-" * 60)

if __name__ == "__main__":
    main()
