import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

print("="*60)
print("PREDIKSI PENYAKIT DAUN PADI")
print("="*60)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast']

# Informasi penyakit
disease_info = {
    'BrownSpot': {'nama': 'Bercak Cokelat', 'penyebab': 'Jamur Bipolaris oryzae'},
    'Healthy': {'nama': 'Sehat', 'penyebab': '-'},
    'Hispa': {'nama': 'Hispa', 'penyebab': 'Serangga Dicladispa armigera'},
    'LeafBlast': {'nama': 'Blas Daun', 'penyebab': 'Jamur Pyricularia oryzae'}
}

# Load model
print("\nMemuat model...")
model = torchvision.models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load('models/best_model.pt', map_location=device))
model = model.to(device)
model.eval()
print("✓ Model loaded")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class):
        output = self.model(input_tensor)
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(gradients, dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()

gradcam = GradCAM(model, model.layer4[-1].conv2)

# Fungsi prediksi
def predict_image(image_path):
    # Load
    original_img = Image.open(image_path).convert('RGB')
    img_tensor = transform(original_img).unsqueeze(0).to(device)
    
    # Prediksi
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = probs.max(1)
    
    pred_class = class_names[pred.item()]
    confidence = conf.item() * 100
    all_probs = probs[0].cpu().numpy() * 100
    
    # Grad-CAM
    cam = gradcam.generate_cam(img_tensor, pred.item())
    cam_resized = cv2.resize(cam, original_img.size)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.array(original_img), 0.6, heatmap, 0.4, 0)
    
    # Visualisasi
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].imshow(original_img)
    axes[0,0].set_title('Gambar Asli', fontsize=12, fontweight='bold')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(overlay)
    axes[0,1].set_title('Grad-CAM Overlay', fontsize=12, fontweight='bold')
    axes[0,1].axis('off')
    
    colors = ['green' if i == pred.item() else 'lightblue' for i in range(len(class_names))]
    axes[1,0].barh(class_names, all_probs, color=colors)
    axes[1,0].set_xlabel('Probabilitas (%)')
    axes[1,0].set_title('Probabilitas Kelas', fontsize=12, fontweight='bold')
    axes[1,0].set_xlim(0, 100)
    
    info = disease_info[pred_class]
    result_text = f"""
HASIL DIAGNOSA

Prediksi: {pred_class}
Nama: {info['nama']}
Confidence: {confidence:.2f}%

Penyebab:
{info['penyebab']}

Probabilitas:
"""
    for name, prob in zip(class_names, all_probs):
        result_text += f"  {name}: {prob:.1f}%\n"
    
    axes[1,1].axis('off')
    axes[1,1].text(0.1, 0.9, result_text, transform=axes[1,1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Diagnosis: {pred_class} ({confidence:.1f}%)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Simpan
    os.makedirs('predictions', exist_ok=True)
    filename = os.path.basename(image_path)
    output_path = f'predictions/result_{filename}'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    # Print
    print("\n" + "="*60)
    print("HASIL PREDIKSI")
    print("="*60)
    print(f"File: {image_path}")
    print(f"Prediksi: {pred_class} ({info['nama']})")
    print(f"Confidence: {confidence:.2f}%")
    print(f"\nProbabilitas:")
    for name, prob in zip(class_names, all_probs):
        print(f"  {name}: {prob:.2f}%")
    print("="*60)
    print(f"✓ Hasil disimpan: {output_path}")
    print("="*60)
    
    plt.show()

# Input
print("\n" + "="*60)
print("MASUKKAN PATH GAMBAR")
print("="*60)
print("Contoh: test_images/daun.jpg")
print("="*60)

image_path = input("\nPath gambar: ").strip().strip('"').strip("'")

if os.path.exists(image_path):
    predict_image(image_path)
else:
    print(f"\n❌ File tidak ditemukan: {image_path}")
