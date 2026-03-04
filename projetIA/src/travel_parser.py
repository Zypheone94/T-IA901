from text_normalize import normalize_text

DEP_WORDS = ["de", "depuis", "au depart de", "en partant de"]
ARR_WORDS = ["a", "vers", "pour", "direction", "jusqu a"]

def parse_trip(sentence: str, nlp, city_index: dict):
    """
    Retour:
      ("INVALID", None, None) ou ("VALID", departure, destination)
    Règle: il faut STRICTEMENT 2 villes (au minimum) pour être VALID.
    """
    norm = normalize_text(sentence)
    doc = nlp(norm)

    cities = []
    for ent in doc.ents:
        if ent.label_ == "CITY":
            key = normalize_text(ent.text)
            canonical = city_index.get(key)
            if canonical:
                cities.append((canonical, ent.start_char, ent.end_char))

    if len(cities) < 2:
        return "INVALID", None, None

    dep, arr = None, None

    for city, start, end in cities:
        left = norm[max(0, start - 35):start]
        right = norm[end:min(len(norm), end + 35)]

        
        if any(w in left for w in DEP_WORDS):
            dep = city

        if any(w in left for w in ARR_WORDS):
            arr = city

    if dep is None:
        dep = cities[0][0]
    if arr is None:
        arr = cities[-1][0]

    if dep == arr:
        return "INVALID", None, None

    return "VALID", dep, arr