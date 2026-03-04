import re
import unicodedata

DASHES = r"[\u2010\u2011\u2012\u2013\u2014\u2212\-]"  # tous les tirets possibles

def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")

def normalize_text(text: str) -> str:
    text = text.lower()
    # apostrophes / guillemets (optionnel)
    text = text.replace("’", "'").replace("`", "'")

    # tirets -> espace (Port-Boulet => "port boulet")
    text = re.sub(DASHES, " ", text)

    # enlever accents
    text = strip_accents(text)

    # garder lettres/chiffres + espaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # espaces multiples
    text = re.sub(r"\s+", " ", text).strip()
    return text
