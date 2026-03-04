import spacy

def build_city_nlp(city_index: dict):
    nlp = spacy.blank("fr")  # spaCy vide = pas de modèle ML, juste tokenizer + règles
    ruler = nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER"})

    patterns = []
    for norm_city in sorted(city_index.keys(), key=lambda s: len(s.split()), reverse=True):
        patterns.append({"label": "CITY", "pattern": norm_city})

    ruler.add_patterns(patterns)
    return nlp



