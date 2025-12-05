# Klasifikasi Penyakit Daun Padi

Sistem klasifikasi otomatis untuk mendeteksi penyakit daun padi menggunakan Deep Learning (ResNet18).

## Dataset
- 3,655 gambar daun padi
- 4 kelas: BrownSpot, Healthy, Hispa, LeafBlast
- Split: 80% training, 20% validation

## Requirements
```
pip install -r requirements.txt
```

## Cara Menggunakan

### 1. Training
```bash
python train.py
```

### 2. Evaluasi
```bash
python evaluate.py
```

### 3. Prediksi Gambar Baru
```bash
python predict.py
```

## Hasil
- **Accuracy:** 53.33%
- **Model:** ResNet18 (Transfer Learning)
- **Output:** Confusion matrix, classification report, Grad-CAM visualization

## Struktur Folder
```
├── Rice_All/          # Dataset
├── train.py           # Training script
├── evaluate.py        # Evaluation script
├── predict.py         # Prediction script
├── models/            # Saved models
├── results/           # Evaluation results
└── predictions/       # Prediction results
```

## Author
[Maulana Arif Hikmat Suci] - [Universitas Sebeleas April]