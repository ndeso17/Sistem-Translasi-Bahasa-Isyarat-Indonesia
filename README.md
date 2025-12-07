# Sistem Translasi Bahasa Isyarat Indonesia

Sistem **translasi Bahasa Isyarat Indonesia (BISINDO)** berbasis _Computer Vision_ yang menggunakan **Convolutional Neural Network (CNN)** dan **Transfer Learning** untuk mengenali alfabet BISINDO (A–Z) dari citra tangan.

Sistem ini dirancang untuk:

- Melatih model pengenalan alfabet BISINDO
- Melakukan inferensi/prediksi huruf dari citra input
- Menyediakan contoh augmentasi data untuk meningkatkan akurasi model

---

## 📁 Struktur Folder

```
Sistem-Translasi-Bahasa-Isyarat-Indonesia/
│
├── README.md
├── augmentation_samples.png
│   Contoh hasil augmentasi data citra
│
├── Citra BISINDO/
│   ├── A/
│   ├── B/
│   ├── C/
│   └── ... sampai Z/
│   Dataset asli citra BISINDO per alfabet
│
├── Data/
│   ├── A/
│   ├── B/
│   ├── C/
│   └── ... sampai Z/
│   Dataset hasil augmentasi yang siap digunakan untuk training
│
├── full_train.py
│   Script training model dengan konfigurasi penuh
│
├── medium_train.py
│   Script training model dengan konfigurasi lebih ringan
│
├── inference.py
│   Script untuk melakukan inferensi/prediksi dari citra input
│
└── utils_augmentation.py
    Script utilitas untuk melakukan augmentasi data citra
```

---

## 📦 Dataset

Dataset citra Bahasa Isyarat Indonesia (BISINDO) dapat diunduh melalui Kaggle:

🔗 [Download Dataset Citra BISINDO](https://www.kaggle.com/datasets/achmadnoer/alfabet-bisindo)

Pastikan dataset diekstrak dan ditempatkan pada folder `Citra BISINDO/` sesuai dengan struktur alfabet (A–Z).

---

## ▶️ Menjalankan Program

### 1️⃣ Instalasi Library

Pastikan berada di environment Python yang sesuai (disarankan menggunakan virtual environment):

```bash
pip install -r requirements.txt
```

### 2️⃣ Augmentasi Data

Proses ini akan menghasilkan data tambahan untuk meningkatkan variasi dataset dan membantu model belajar lebih baik.

```bash
python utils_augmentation.py
```

Contoh hasil augmentasi dapat dilihat pada file:

```
augmentation_samples.png
```

### 3️⃣ Training Model

Untuk melakukan training dengan konfigurasi penuh:

```bash
python full_train.py
```

**⏱️ Estimasi waktu training:**

- Per epoch: ± 3–5 menit
- 50 epoch: ± 2.5–4.5 jam
- Total rata-rata: ± 3–4 jam
- Dengan early stopping: bisa selesai ± 2 jam

**Alternatif training ringan:**

```bash
python medium_train.py
```

### 4️⃣ Inference / Prediksi

Gunakan script berikut untuk melakukan prediksi dari citra input:

```bash
python inference.py
```

Script ini akan menampilkan hasil klasifikasi huruf BISINDO dari citra yang diberikan.

---

## 📌 Catatan

- Pastikan struktur folder dataset sesuai dengan format alfabet (A–Z)
- Ukuran dan kualitas citra sangat mempengaruhi hasil prediksi
- Disarankan menggunakan GPU untuk mempercepat proses training
