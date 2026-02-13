import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForTokenClassification, AutoTokenizer
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class NERDataset(Dataset):
    """
    Dataset pour le NER (Named Entity Recognition).
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long)
        }

def collate_fn(batch):
    """
    Fonction pour padder les séquences de longueurs différentes.
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'labels': labels_padded
    }

def evaluate(model, dataloader, device):
    """
    Évalue le modèle sur un dataset (validation ou test).
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=-1)

            for pred, label in zip(preds, labels):
                mask = label != -100
                all_preds.extend(pred[mask].cpu().numpy())
                all_labels.extend(label[mask].cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_preds, all_labels

print("=" * 70)
print("📂 CHARGEMENT DU DATASET")
print("=" * 70)

with open('dataset_ner_cities.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

if isinstance(dataset, dict) and 'train' in dataset:
    print("✅ Dataset avec clés détectées (train/val/test)")
    train_data = dataset['train']
    val_data = dataset.get('val', dataset.get('validation', []))
    test_data = dataset.get('test', [])

    if not val_data or not test_data:
        print("⚠️  Validation ou Test manquant, création automatique...")
        split_val = int(0.9 * len(train_data))
        remaining = train_data[split_val:]
        val_data = remaining[:len(remaining) // 2]
        test_data = remaining[len(remaining) // 2:]
        train_data = train_data[:split_val]
else:
    print("✅ Dataset sous forme de liste")
    data = dataset if isinstance(dataset, list) else list(dataset.values())[0]
    split_train = int(0.8 * len(data))
    split_val = int(0.9 * len(data))
    train_data = data[:split_train]
    val_data = data[split_train:split_val]
    test_data = data[split_val:]

print(f"✅ Train: {len(train_data)} exemples")
print(f"✅ Val:   {len(val_data)} exemples")
print(f"✅ Test:  {len(test_data)} exemples\n")

print("🔄 Création des DataLoaders avec padding...")

train_dataset = NERDataset(train_data)
val_dataset = NERDataset(val_data)
test_dataset = NERDataset(test_data)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=collate_fn
)

print("✅ DataLoaders créés\n")

print("=" * 70)
print("🤖 CHARGEMENT DU MODÈLE CAMEMBERT")
print("=" * 70)

checkpoint = "camembert-base"
num_labels = 5


tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForTokenClassification.from_pretrained(
    checkpoint,
    num_labels=num_labels
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"✅ Modèle chargé sur : {device}\n")

optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3

id2label = {0: 'O', 1: 'B-DEP', 2: 'I-DEP', 3: 'B-ARR', 4: 'I-ARR'}

print("=" * 70)
print("🚀 DÉBUT DE L'ENTRAÎNEMENT")
print("=" * 70)

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    print(f"\n📍 EPOCH {epoch + 1}/{num_epochs}")
    print("-" * 70)

    model.train()
    total_loss = 0

    for i, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"  Batch {i + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)
    val_loss, val_preds, val_labels = evaluate(model, val_loader, device)

    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)

    print(f"\n{'=' * 60}")
    print(f"📊 Résultats Epoch {epoch + 1}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"{'=' * 60}")

print("\n" + "=" * 70)
print("🎯 ÉVALUATION FINALE SUR LE TEST SET")
print("=" * 70)

test_loss, test_preds, test_labels = evaluate(model, test_loader, device)

print(f"\n📉 Test Loss: {test_loss:.4f}\n")
print("📊 Classification Report:")
print("-" * 70)

entities = ['O', 'B-DEP', 'I-DEP', 'B-ARR', 'I-ARR']

report = classification_report(
    test_labels,
    test_preds,
    target_names=entities,
    output_dict=True,
    zero_division=0
)

print(classification_report(
    test_labels,
    test_preds,
    target_names=entities,
    zero_division=0
))

print("\n" + "=" * 70)
print("📊 GÉNÉRATION DES GRAPHIQUES")
print("=" * 70)

print("\n1️⃣ Génération de la courbe de loss...")
plt.figure(figsize=(10, 6))
epochs_range = range(1, num_epochs + 1)
plt.plot(epochs_range, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=8)
plt.plot(epochs_range, val_losses, 'r-o', label='Validation Loss', linewidth=2, markersize=8)
plt.xlabel('Epochs', fontsize=13, fontweight='bold')
plt.ylabel('Loss', fontsize=13, fontweight='bold')
plt.title('Évolution de la Loss pendant l\'entraînement', fontsize=15, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
print("   ✅ loss_curve.png sauvegardé")
plt.close()

print("\n2️⃣ Génération des métriques par entité...")
f1_scores = [report[ent]['f1-score'] for ent in entities]
precisions = [report[ent]['precision'] for ent in entities]
recalls = [report[ent]['recall'] for ent in entities]

plt.figure(figsize=(14, 7))
x = np.arange(len(entities))
width = 0.25

bars1 = plt.bar(x - width, precisions, width, label='Precision', color='#66b3ff', edgecolor='black', linewidth=1.2)
bars2 = plt.bar(x, recalls, width, label='Recall', color='#ff9999', edgecolor='black', linewidth=1.2)
bars3 = plt.bar(x + width, f1_scores, width, label='F1-Score', color='#99ff99', edgecolor='black', linewidth=1.2)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.xlabel('Entités', fontsize=13, fontweight='bold')
plt.ylabel('Score', fontsize=13, fontweight='bold')
plt.title('Métriques par type d\'entité (Test Set)', fontsize=15, fontweight='bold')
plt.xticks(x, entities, rotation=0, fontsize=11)
plt.ylim(0, 1.15)
plt.legend(fontsize=12, loc='upper right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('metrics_per_entity.png', dpi=300, bbox_inches='tight')
print("   ✅ metrics_per_entity.png sauvegardé")
plt.close()

print("\n3️⃣ Génération de la matrice de confusion...")
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(11, 9))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=entities, yticklabels=entities,
            cbar_kws={'label': 'Nombre de prédictions'},
            linewidths=0.5, linecolor='gray',
            annot_kws={'fontsize': 12, 'fontweight': 'bold'})
plt.xlabel('Prédictions', fontsize=13, fontweight='bold')
plt.ylabel('Vraies valeurs', fontsize=13, fontweight='bold')
plt.title('Matrice de confusion (Test Set)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("   ✅ confusion_matrix.png sauvegardé")
plt.close()

print("\n4️⃣ Génération des métriques globales...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

metrics_names = ['Precision', 'Recall', 'F1-Score']
metrics_values = [
    report['macro avg']['precision'],
    report['macro avg']['recall'],
    report['macro avg']['f1-score']
]
colors = ['#66b3ff', '#ff9999', '#99ff99']

bars = ax1.bar(metrics_names, metrics_values, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Métriques Globales (Macro Avg)', fontsize=13, fontweight='bold')
ax1.set_ylim(0, 1.1)
ax1.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, metrics_values):
    ax1.text(bar.get_x() + bar.get_width() / 2., val + 0.02,
             f'{val:.3f}', ha='center', fontweight='bold', fontsize=11)

supports = [report[ent]['support'] for ent in entities]
bars2 = ax2.bar(entities, supports, color='lightcoral', edgecolor='black', linewidth=1.2)
ax2.set_xlabel('Entités', fontsize=12, fontweight='bold')
ax2.set_ylabel('Nombre d\'exemples', fontsize=12, fontweight='bold')
ax2.set_title('Distribution des entités dans le Test Set', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

for bar, val in zip(bars2, supports):
    ax2.text(bar.get_x() + bar.get_width() / 2., val + max(supports) * 0.01,
             f'{int(val)}', ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('metrics_global.png', dpi=300, bbox_inches='tight')
print("   ✅ metrics_global.png sauvegardé")
plt.close()

print("\n5️⃣ Sauvegarde des métriques en JSON...")
metrics_summary = {
    'test_loss': float(test_loss),
    'accuracy': float(report['accuracy']),
    'macro_avg': {
        'precision': float(report['macro avg']['precision']),
        'recall': float(report['macro avg']['recall']),
        'f1-score': float(report['macro avg']['f1-score'])
    },
    'weighted_avg': {
        'precision': float(report['weighted avg']['precision']),
        'recall': float(report['weighted avg']['recall']),
        'f1-score': float(report['weighted avg']['f1-score'])
    },
    'per_entity': {
        ent: {
            'precision': float(report[ent]['precision']),
            'recall': float(report[ent]['recall']),
            'f1-score': float(report[ent]['f1-score']),
            'support': int(report[ent]['support'])
        } for ent in entities
    },
    'training_history': {
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses]
    }
}

with open('metrics_results.json', 'w', encoding='utf-8') as f:
    json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
print("   ✅ metrics_results.json sauvegardé")

print("\n" + "=" * 70)
print("🎉 TOUS LES GRAPHIQUES ET MÉTRIQUES ONT ÉTÉ GÉNÉRÉS !")
print("=" * 70)
print("\n📁 Fichiers générés :")
print("   1. loss_curve.png - Évolution de la loss")
print("   2. metrics_per_entity.png - Métriques par entité")
print("   3. confusion_matrix.png - Matrice de confusion")
print("   4. metrics_global.png - Métriques globales et distribution")
print("   5. metrics_results.json - Toutes les métriques en JSON")

print("\n" + "=" * 70)
print("💾 SAUVEGARDE DU MODÈLE")
print("=" * 70)

model.save_pretrained('./model_ner_cities')
tokenizer.save_pretrained('./model_ner_cities')
print("✅ Modèle sauvegardé dans './model_ner_cities'")

print("\n" + "=" * 70)
print("🧪 TEST SUR UNE PHRASE EXEMPLE")
print("=" * 70)

test_sentence = "Je veux aller de Paris vers Lyon"
print(f"\nPhrase: {test_sentence}\n")

inputs = tokenizer(test_sentence, return_tensors='pt').to(device)

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)[0]

tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
labels_pred = [id2label[p.item()] for p in predictions]

print("Tokens et prédictions:")
print("-" * 40)
for token, label in zip(tokens, labels_pred):
    if token not in ['<s>', '</s>', '<pad>']:
        print(f"{token:15s} -> {label}")

print("\n" + "=" * 70)
print("✨ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
print("=" * 70)