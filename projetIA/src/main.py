from load_communes import load_city_index
from city_ruler import build_city_nlp
from travel_parser import parse_trip

COMMUNES_PATH = "data/communes/communes-20220101.shp"

def main():
    city_index = load_city_index(COMMUNES_PATH)
    nlp = build_city_nlp(city_index)

    tests = [
        "Je veux aller de Paris à Lille",
        "Comment me rendre à Port-Boulet depuis Tours ?",
        "je vais à bordeaux",
        "je pars de Saint-Tropez vers Nice",
    ]

    for s in tests:
        status, dep, arr = parse_trip(s, nlp, city_index)
        if status == "VALID":
            print(f"VALID | dep: {dep} -> arr: {arr} | {s}")
        else:
            print(f"INVALID | {s}")

if __name__ == "__main__":
    main()