import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

print("="*60)
print("EVALUASI MODEL")
print("="*60)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_dataset = datasets.ImageFolder('data/val', transform=val_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

class_names = val_dataset.classes
num_classes = len(class_names)

# Load model
print("\nMemuat model...")
model = torchvision.models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('models/best_model.pt', map_location=device))
model = model.to(device)
model.eval()
print("✓ Model loaded")

# Prediksi
print("\nMelakukan prediksi...")
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

accuracy = (all_preds == all_labels).sum() / len(all_labels) * 100
print(f"✓ Accuracy: {accuracy:.2f}%")

# Confusion Matrix
print("\n" + "="*60)
print("CONFUSION MATRIX")
print("="*60)

cm = confusion_matrix(all_labels, all_preds)
print(cm)

os.makedirs('results', exist_ok=True)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2f}%', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: results/confusion_matrix.png")

# Classification Report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)

report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print(report)

with open('results/classification_report.txt', 'w') as f:
    f.write("CLASSIFICATION REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(f"Overall Accuracy: {accuracy:.2f}%\n\n")
    f.write(report)

print("\n✓ Saved: results/classification_report.txt")

# Per-class accuracy
print("\n" + "="*60)
print("PER-CLASS ACCURACY")
print("="*60)

class_accuracies = []
for i, class_name in enumerate(class_names):
    class_correct = ((all_preds == i) & (all_labels == i)).sum()
    class_total = (all_labels == i).sum()
    class_acc = class_correct / class_total * 100 if class_total > 0 else 0
    class_accuracies.append(class_acc)
    print(f"{class_name}: {class_acc:.2f}% ({class_correct}/{class_total})")

# Bar chart
plt.figure(figsize=(10, 6))
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
bars = plt.bar(class_names, class_accuracies, color=colors)
plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12)
plt.ylim(0, 100)
plt.grid(axis='y', alpha=0.3)

for bar, acc in zip(bars, class_accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{acc:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('results/per_class_accuracy.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: results/per_class_accuracy.png")

print("\n" + "="*60)
print("✓ EVALUASI SELESAI!")
print("  - results/confusion_matrix.png")
print("  - results/classification_report.txt")
print("  - results/per_class_accuracy.png")
print("="*60)

plt.show()