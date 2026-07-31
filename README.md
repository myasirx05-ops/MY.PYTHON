# 📊 Data Mining — Supervised & Unsupervised Learning

<p align="center">
  <b>Ujian Akhir Semester (UAS) — Data Mining (SIF304)</b><br>
  Tahun Ajaran Genap 2025/2026
</p>

<p align="center">
  <a href="https://myasir-g2wern9svgrge5p3dtrwlt.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-red?style=for-the-badge" alt="Live Demo">
  </a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Machine%20Learning-Supervised%20%26%20Unsupervised-success?style=for-the-badge" alt="Machine Learning">
</p>

---

## 👨‍🎓 Identitas Mahasiswa

| Informasi          | Detail                               |
| ------------------ | ------------------------------------ |
| **Nama**           | M. Yasir                             |
| **NIM**            | `[NIM ANDA]`                         |
| **Kelas**          | `[KELAS ANDA]`                       |
| **Mata Kuliah**    | Data Mining (SIF304)                 |
| **Dosen Pengampu** | Teuku Rizky Noviandy, S.Kom., M.Kom. |
| **Tahun Ajaran**   | Genap 2025/2026                      |

---

## 📌 Tentang Project

Project ini merupakan implementasi akhir mata kuliah **Data Mining (SIF304)** yang menerapkan konsep **Machine Learning** melalui dua pendekatan utama:

* 🧠 **Supervised Learning** — Klasifikasi Risiko Diabetes
* 📍 **Unsupervised Learning** — Clustering Persebaran Gerai Kopi

Aplikasi dikembangkan menggunakan **Python** dan **Streamlit** sehingga pengguna dapat melakukan eksplorasi data, melihat performa model, melakukan visualisasi, serta mencoba prediksi secara interaktif melalui browser.

---

# 🧠 Part A — Klasifikasi Risiko Diabetes

### 🎯 Tujuan

Membangun model Machine Learning untuk memprediksi apakah seseorang memiliki risiko diabetes berdasarkan sejumlah parameter medis.

### 📂 Dataset

Project menggunakan **Pima Indians Diabetes Dataset** yang terdiri dari:

* 8 fitur medis
* 1 variabel target/label
* Data pasien untuk proses training dan testing model

### 🤖 Algoritma

Tiga algoritma klasifikasi digunakan untuk membandingkan performa model:

| Algoritma                    | Jenis               |
| ---------------------------- | ------------------- |
| 🔵 K-Nearest Neighbors (KNN) | Supervised Learning |
| 🟢 Gaussian Naive Bayes      | Supervised Learning |
| 🟠 Decision Tree             | Supervised Learning |

### 📈 Evaluasi Model

Performa masing-masing algoritma dibandingkan menggunakan beberapa metrik:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**

Selain itu, aplikasi menyediakan **Confusion Matrix** untuk melihat detail hasil prediksi model.

### ✨ Fitur

* 📊 Perbandingan performa 3 algoritma
* 📈 Accuracy, Precision, Recall, dan F1-Score
* 🔲 Confusion Matrix
* 🧑‍⚕️ Form input data pasien
* 🔮 Prediksi risiko diabetes secara interaktif
* ⚡ Hasil prediksi secara langsung

---

# ☕ Part B — Clustering Gerai Kopi

### 🎯 Tujuan

Menganalisis persebaran gerai kopi menggunakan metode **K-Means Clustering** untuk menemukan pola wilayah berdasarkan kepadatan pelanggan dan tingkat kompetisi.

Analisis ini dapat digunakan untuk membantu mengidentifikasi:

> 🔥 **Zona Ramai** — area dengan kepadatan pelanggan tinggi.

> ⚠️ **Zona Sepi** — area dengan kepadatan pelanggan rendah tetapi tingkat kompetisi relatif tinggi.

### 📂 Dataset

Dataset berisi beberapa parameter utama:

* 📍 Latitude
* 📍 Longitude
* 👥 Customer Density
* 🏪 Competition Level
* 📏 Distance to Center

Aplikasi juga mendukung **upload dataset CSV custom**.

### 🤖 Algoritma

**K-Means Clustering**

K-Means digunakan untuk mengelompokkan lokasi gerai kopi berdasarkan karakteristik wilayahnya.

### 📊 Fitur

* 📍 Visualisasi persebaran cluster
* 🎯 Penanda centroid
* 📊 Statistik otomatis setiap cluster
* 🔥 Identifikasi Zona Ramai
* ⚠️ Identifikasi Zona Sepi
* 📁 Upload dataset CSV
* 🔮 Simulasi prediksi cluster untuk lokasi baru

---

# 🖥️ Teknologi yang Digunakan

<p align="center">

<img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white">
<img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white">
<img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white">
<img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=matplotlib&logoColor=white">

</p>

### 🔧 Tools & Library

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

---

# 🚀 Menjalankan Aplikasi

## 🌐 Cara 1 — Online Deployment

Cara paling mudah adalah menggunakan aplikasi yang telah di-deploy ke Streamlit.

### 🔗 Live Demo

👉 **https://myasir-g2wern9svgrge5p3dtrwlt.streamlit.app/**

Setelah membuka aplikasi:

1. Buka link menggunakan browser.
2. Gunakan menu navigasi pada sidebar.
3. Pilih **Klasifikasi Diabetes** atau **Clustering Gerai Kopi**.
4. Gunakan dataset default atau upload dataset CSV sendiri.
5. Masukkan parameter yang diperlukan.
6. Lihat hasil analisis dan prediksi.

---

# 💻 Cara 2 — Menjalankan Secara Lokal

### 1️⃣ Clone Repository

```bash
git clone https://github.com/USERNAME/REPOSITORY.git
cd REPOSITORY
```

### 2️⃣ Buat Virtual Environment

Disarankan menggunakan virtual environment agar dependensi project lebih terisolasi.

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

Jika tersedia file `requirements.txt`:

```bash
pip install -r requirements.txt
```

Atau install library secara manual:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib
```

### 4️⃣ Jalankan Aplikasi

Pastikan file utama bernama:

```text
app.py
```

Kemudian jalankan:

```bash
streamlit run app.py
```

### 5️⃣ Buka Browser

Setelah aplikasi berjalan, buka:

```text
http://localhost:8501
```

---

# 📁 Struktur Project

Contoh struktur repository:

```text
data-mining-uas/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── diabetes.csv
│   └── coffee_shops.csv
│
└── assets/
    └── screenshots/
```

> Struktur dapat disesuaikan dengan file yang terdapat pada repository.

---

# 📊 Alur Sistem

```text
                    DATA MINING APPLICATION
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
       SUPERVISED LEARNING          UNSUPERVISED LEARNING
              │                             │
              ▼                             ▼
      KLASIFIKASI DIABETES          CLUSTERING GERAI KOPI
              │                             │
       ┌──────┼──────┐                      │
       │      │      │                      ▼
      KNN   Naive   Decision              K-Means
            Bayes    Tree                  │
       │      │      │                      │
       └──────┼──────┘                      │
              ▼                             ▼
      Accuracy / Precision          Cluster & Centroid
      Recall / F1-Score                    │
              │                            ▼
              ▼                     Zona Ramai / Sepi
        Prediksi Risiko                    │
              │                            ▼
              └──────────────┬─────────────┘
                             ▼
                    STREAMLIT WEB APP
```

---

# 🎯 Tujuan Project

Project ini dibuat untuk menerapkan konsep Data Mining dan Machine Learning secara langsung pada studi kasus nyata.

### Tujuan utama:

1. Memahami penerapan **Supervised Learning**.
2. Memahami penerapan **Unsupervised Learning**.
3. Membandingkan beberapa algoritma klasifikasi.
4. Mengevaluasi performa model menggunakan berbagai metrik.
5. Menerapkan algoritma **K-Means Clustering**.
6. Membuat visualisasi hasil analisis data.
7. Mengembangkan aplikasi Machine Learning berbasis web menggunakan Streamlit.
8. Menghasilkan aplikasi yang dapat digunakan secara interaktif.

---

# 📌 Hasil yang Diharapkan

Melalui aplikasi ini, pengguna dapat:

* Mengetahui performa beberapa algoritma klasifikasi diabetes.
* Melakukan prediksi risiko diabetes berdasarkan data pasien.
* Melihat Confusion Matrix dari model klasifikasi.
* Menganalisis persebaran gerai kopi.
* Mengidentifikasi kelompok/cluster lokasi gerai.
* Menemukan potensi **Zona Ramai** dan **Zona Sepi**.
* Melakukan simulasi cluster untuk lokasi gerai baru.

---

# ⚠️ Catatan

Prediksi diabetes pada aplikasi ini merupakan **implementasi akademik Machine Learning**, bukan alat diagnosis medis.

Hasil model tidak boleh digunakan sebagai pengganti pemeriksaan atau diagnosis tenaga kesehatan profesional.

---

# 👨‍💻 Author

**M. Yasir**

📚 Mahasiswa — Fakultas Teknik

🎓 Project UAS Data Mining (SIF304)

---

# ⭐ Live Demo

<p align="center">

<a href="https://myasir-g2wern9svgrge5p3dtrwlt.streamlit.app/">

<img src="https://img.shields.io/badge/🚀%20OPEN%20STREAMLIT%20APP-Click%20Here-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">

</a>

</p>

---

<p align="center">
  <b>Data Mining — Supervised & Unsupervised Learning</b><br>
  <i>Built with Python & Streamlit 🚀</i>
</p>
