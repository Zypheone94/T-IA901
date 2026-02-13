import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

print("Chargement du modèle entraîné...")
model = AutoModelForTokenClassification.from_pretrained('./model_ner_cities')
tokenizer = AutoTokenizer.from_pretrained('./model_ner_cities')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

print(f"Modèle chargé sur {device}")
print("=" * 60)

# Mapping des labels
id2label = {
    0: 'O',
    1: 'B-DEP',
    2: 'I-DEP',
    3: 'B-ARR',
    4: 'I-ARR'
}


def extraire_villes(tokens, labels):
    """
    Extrait les villes de départ et d'arrivée depuis les tokens et labels
    """
    ville_depart = []
    ville_arrivee = []

    for token, label in zip(tokens, labels):
        # Ignorer les tokens spéciaux
        if token in ['<s>', '</s>', '<pad>']:
            continue

        # Nettoyer le token (enlever le _ de début qui représente l'espace)
        token_clean = token.replace('▁', ' ').strip()

        if label in ['B-DEP', 'I-DEP']:
            ville_depart.append(token_clean)
        elif label in ['B-ARR', 'I-ARR']:
            ville_arrivee.append(token_clean)

    # Joindre les tokens pour former les noms complets
    depart = ''.join(ville_depart).strip()
    arrivee = ''.join(ville_arrivee).strip()

    return depart, arrivee


def predire(phrase):
    """
    Prédit les villes de départ et d'arrivée dans une phrase
    """
    # Tokeniser
    inputs = tokenizer(phrase, return_tensors='pt').to(device)

    # Prédire
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)[0]

    # Récupérer tokens et labels
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    labels_pred = [id2label[p.item()] for p in predictions]

    # Afficher les détails
    print("\n📋 Analyse détaillée :")
    print("-" * 60)
    for token, label in zip(tokens, labels_pred):
        if token not in ['<s>', '</s>', '<pad>']:
            token_display = token.replace('▁', '_')  # Afficher _ pour les espaces
            color = ""
            if label.endswith('DEP'):
                color = "🔵"
            elif label.endswith('ARR'):
                color = "🔴"
            print(f"{color} {token_display:20s} -> {label}")

    # Extraire les villes
    depart, arrivee = extraire_villes(tokens, labels_pred)

    print("\n" + "=" * 60)
    print("🎯 RÉSULTAT :")
    print(f"   🔵 Ville de DÉPART  : {depart if depart else '❌ Non détectée'}")
    print(f"   🔴 Ville d'ARRIVÉE  : {arrivee if arrivee else '❌ Non détectée'}")
    print("=" * 60)


# ===== BOUCLE INTERACTIVE =====
print("\n🚀 Script de test interactif du modèle NER")
print("Tapez 'quit' ou 'exit' pour quitter\n")

while True:
    print("\n" + "=" * 60)
    phrase = input("💬 Entrez une phrase : ").strip()

    if phrase.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Au revoir !")
        break

    if not phrase:
        print("⚠️  Phrase vide, réessayez.")
        continue

    try:
        predire(phrase)
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        print("Réessayez avec une autre phrase.")