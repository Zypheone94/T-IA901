import geopandas as gpd
from text_normalize import normalize_text

STOP_CITIES = {
    "vers", "pour", "direction", "jusqu", "jusqua",
    "de", "depuis", "a", "au", "aux", "en"
}

def load_city_index(shp_path: str) -> dict:
    gdf = gpd.read_file(shp_path)

    if "nom" not in gdf.columns:
        raise ValueError(f"Colonne 'nom' absente. Colonnes dispo: {list(gdf.columns)}")

    city_index = {}
    for city in gdf["nom"].astype(str):
        key = normalize_text(city)
        if not key:
            continue
        if key in STOP_CITIES:
            continue
        if key not in city_index:
            city_index[key] = city

    return city_index
