LAPORAN UJIAN AKHIR SEMESTER (UAS)
Mata Kuliah: Data Mining (SIF304)
Tahun Ajaran: Genap 2025/2026
Dosen Pengampu: Teuku Rizky Noviandy, S.Kom., M.Kom.
1. Identitas Mahasiswa
Nama: [NAMA LENGKAP ANDA]
NIM: [NIM ANDA]
Kelas: [KELAS ANDA, misal: SIF-45-01]
2. Penjelasan Proyek
Aplikasi ini merupakan implementasi akhir dari mata kuliah Data Mining (SIF304) yang mendemonstrasikan penerapan teknik Machine Learning, baik Supervised maupun Unsupervised Learning, pada dua studi kasus nyata. Aplikasi dibangun menggunakan bahasa pemrograman Python dan framework Streamlit untuk antarmuka yang interaktif dan modern.
Bagian A: Klasifikasi Diabetes (Supervised Learning)
Tujuan: Memprediksi risiko diabetes pada pasien berdasarkan data diagnostik medis.
Dataset: Pima Indians Diabetes Dataset (8 fitur medis dan 1 label target).
Algoritma yang Digunakan:
K-Nearest Neighbors (KNN)
Naive Bayes (Gaussian)
Decision Tree
Fitur Utama:
Perbandingan performa model secara real-time menggunakan metrik Accuracy, Precision, Recall, dan F1-Score.
Visualisasi Confusion Matrix untuk melihat detail prediksi benar/salah.
Formulir interaktif untuk memprediksi risiko diabetes pada data pasien baru.
Bagian B: Clustering Gerai Kopi (Unsupervised Learning)
Tujuan: Menganalisis persebaran gerai kopi dan mendeteksi "zona sepi" (area dengan kepadatan pelanggan rendah tetapi tingkat kompetisi tinggi) untuk membantu pengambilan keputusan bisnis.
Dataset: Data sintetis (dengan opsi unggah CSV kustom) yang berisi koordinat geografis (Latitude, Longitude), Customer Density, Competition Level, dan Distance to Center.
Algoritma yang Digunakan: K-Means Clustering.
Fitur Utama:
Visualisasi sebaran klaster menggunakan scatter plot dengan penanda centroid.
Analisis statistik otomatis per klaster untuk menandai status "Zona Ramai" atau "Zona Sepi".
Simulasi prediksi klaster untuk lokasi gerai baru berdasarkan input parameter.
3. Instruksi Menjalankan Aplikasi
Terdapat dua cara untuk menjalankan aplikasi ini, yaitu melalui deployment online (paling mudah) atau menjalankannya secara lokal di komputer.
Cara 1: Mengakses Aplikasi Online (Disarankan)
Buka browser web (Chrome, Firefox, Edge, dll).
Kunjungi tautan aplikasi Streamlit yang telah di-deploy.
Gunakan menu navigasi di sidebar kiri untuk beralih antara halaman Home, Klasifikasi Diabetes, dan Clustering Gerai Kopi.
Anda dapat langsung menggunakan dataset default atau mengunggah file CSV Anda sendiri.
Cara 2: Menjalankan Secara Lokal (Local Development)
Pastikan Python (versi 3.8 atau lebih baru) sudah terinstal di komputer Anda.
Buka terminal atau command prompt, lalu buat virtual environment (opsional namun disarankan):
bash
1234
Instal semua library dependensi yang diperlukan:
bash
1
Simpan kode sumber aplikasi yang telah diberikan ke dalam file bernama app.py.
Jalankan aplikasi Streamlit melalui terminal:
bash
1
Aplikasi akan otomatis terbuka di browser Anda pada alamat: http://localhost:8501
4. Link Aplikasi Streamlit yang Aktif
Aplikasi ini telah berhasil di-deploy dan dapat diakses secara publik melalui tautan berikut:
🔗 https://myasir-g2wern9svgrge5p3dtrwlt.streamlit.app/
