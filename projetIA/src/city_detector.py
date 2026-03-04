import spacy

nlp = spacy.load("fr_core_news_sm")

def detect_cities(sentence: str):
    """
    Prend une phrase en entrée
    Retourne la liste des villes détectées
    """
    doc = nlp(sentence)
    return [(ent.text, ent.label_) for ent in doc.ents]

if __name__ == "__main__":
    sentence = "Je veux aller de Paris à Lyon"
    cities = detect_cities(sentence)
    print(cities)
