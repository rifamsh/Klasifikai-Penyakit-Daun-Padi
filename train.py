import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
import time
import os
import shutil
from sklearn.model_selection import train_test_split

print("="*60)
print("TRAINING MODEL KLASIFIKASI PENYAKIT DAUN PADI")
print("="*60)

# ========== 1. SPLIT DATASET ==========
print("\n[1/4] Membagi dataset...")

source_dir = 'Rice_All'  # Folder dataset asli
dest_dir = 'data'        # Folder output

if not os.path.exists(dest_dir):
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
        
        # Buat folder dan copy
        for split, img_list in [('train', train_imgs), ('val', val_imgs)]:
            split_dir = os.path.join(dest_dir, split, class_name)
            os.makedirs(split_dir, exist_ok=True)
            for img in img_list:
                shutil.copy(os.path.join(class_path, img), os.path.join(split_dir, img))
        
        print(f"  {class_name}: {len(train_imgs)} train, {len(val_imgs)} val")
    
    print("✓ Dataset berhasil dibagi!")
else:
    print("✓ Dataset sudah ada, skip splitting")

# ========== 2. SETUP TRAINING ==========
print("\n[2/4] Setup training...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Transformasi
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Dataset & DataLoader
train_dataset = datasets.ImageFolder('data/train', transform=train_transform)
val_dataset = datasets.ImageFolder('data/val', transform=val_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

num_classes = len(train_dataset.classes)
print(f"Kelas: {train_dataset.classes}")
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# Model
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Unfreeze layer4 + fc (langsung fine-tuning)
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

# Optimizer & Loss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])

print("✓ Model ResNet18 siap (layer4 + fc trainable)")

# ========== 3. TRAINING ==========
print("\n[3/4] Mulai training...")
print("="*60)

os.makedirs('models', exist_ok=True)

num_epochs = 20
best_val_acc = 0.0
patience = 5
wait = 0

for epoch in range(num_epochs):
    # Train
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        train_total += labels.size(0)
        train_correct += preds.eq(labels).sum().item()
    
    train_loss = train_loss / train_total
    train_acc = 100.0 * train_correct / train_total
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            val_total += labels.size(0)
            val_correct += preds.eq(labels).sum().item()
    
    val_loss = val_loss / val_total
    val_acc = 100.0 * val_correct / val_total
    
    # Print
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.2f}%")
    
    # Save best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        wait = 0
        torch.save(model.state_dict(), 'models/best_model.pt')
        print(f"  ✓ Best model saved! ({val_acc:.2f}%)")
    else:
        wait += 1
        print(f"  No improvement ({wait}/{patience})")
    
    print("-"*60)
    
    if wait >= patience:
        print("Early stopping!")
        break

# ========== 4. SIMPAN INFO ==========
print("\n[4/4] Menyimpan informasi model...")

with open('models/model_info.txt', 'w') as f:
    f.write(f"Model: ResNet18 (Transfer Learning)\n")
    f.write(f"Classes: {train_dataset.classes}\n")
    f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
    f.write(f"Training samples: {len(train_dataset)}\n")
    f.write(f"Validation samples: {len(val_dataset)}\n")

print("="*60)
print("✓ TRAINING SELESAI!")
print(f"✓ Best Accuracy: {best_val_acc:.2f}%")
print(f"✓ Model tersimpan: models/best_model.pt")
print("="*60)