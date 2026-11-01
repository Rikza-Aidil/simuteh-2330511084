# simuteh-2330511084
# 🍃 SIMUTEH: Computer Vision Pipeline & Tea Leaf Spot Contrast Analysis

Repository ini berisi rangkaian praktikum dan eksperimen **Computer Vision (P1–P5)** dalam studi kasus **SIMUTEH (Simulasi Pengolahan Citra Daun Teh)**. Proyek ini berfokus pada teknik perbaikan mutu citra (*image enhancement*), penapisan spasial (*spatial filtering*), serta analisis kuantitatif **Kontras Bercak (KB)** pada berbagai kondisi pencahayaan.

## 👨‍💻 Informasi Mahasiswa

* **Nama:** Rikza Aidil Rifqi
* **NIM:** 2330511084
* **Program Studi:** Teknik Informatika
* **Mata Kuliah:** Computer Vision

---

## 📁 Struktur Direktori

```
simuteh-2330511084/
├── data/
│   └── sintetis/
│       ├── citra/         # Citra sintetis daun teh (dengan gangguan cahaya/derau)
│       ├── bersih/        # Citra acuan mutu (ground truth)
│       ├── mask_daun/     # Mask biner area daun (255 = daun)
│       ├── mask_bercak/   # Mask biner area bercak penyakit (255 = bercak)
│       └── label.csv      # Metadata citra (kelas, kondisi cahaya, % bercak)
├── hasil/                 # Simpanan grafik visual & file evaluasi
│   ├── b1_bgr_rgb.png             # Visualisasi konversi ruang warna BGR ke RGB
│   ├── b1_pipeline.png            # Diagram alur pipeline SIMUTEH P1-P4
│   ├── b2_kb_kanal.csv            # Tabel perbandingan KB per kanal warna
│   ├── b3_histogram_5kondisi.png  # Grafik histogram 5 kondisi cahaya
│   ├── b3_perbandingan_clahe.png  # Visualisasi perbandingan CLAHE V vs BGR
│   ├── b4_tabel_penapis.csv       # Tabel pengujian penapis & PSNR
│   └── param_praproses.json       # Konfigurasi parameter praproses terpusat
├── notebook/              # Jupyter Notebook praktikum (B1-B5)
│   ├── B1_lingkungan.ipynb
│   ├── B2_citra_warna.ipynb
│   ├── B3_histogram.ipynb
│   ├── B4_konvolusi.ipynb
│   └── B5_praproses.ipynb
└── src/                   # Modul Python utama
    ├── buat_dataset_sintetis.py
    ├── utils_cv.py        # Fungsi utilitas (baca, tampil, kontras_bercak)
    └── praproses.py       # Pipeline praproses terpusat
```

---

## 🛠️ Pipeline Praproses (`praproses.py`)

Rangkaian alur pengolahan citra yang diimplementasikan pada modul `praproses.py` mengikuti tahapan berikut:

1. **Ubah Ukuran (Resizing):** Menyamakan lebar citra ke 640 piksel dengan mempertahankan *aspect ratio* (`INTER_AREA`).
2. **Koreksi Gamma:** Menyesuaikan distribusi kecerahan citra secara non-linier menggunakan *Lookup Table* (`cv2.LUT`).
3. **CLAHE Kanal V:** Mengaplikasikan *Contrast Limited Adaptive Histogram Equalization* pada kanal *Value* (HSV) untuk mengatasi pencahayaan tidak merata (seperti kondisi bayangan).
4. **Peredaman Derau (Filtering):** Menerapkan **Median Filter** ($3 \times 3$) untuk meredam derau *salt & pepper* tanpa mengaburkan batas tepi bercak.
5. **Penajaman (Sharpening):** Menggunakan *Unsharp Masking* untuk mempertegas batas objek.

---

## 📊 Ringkasan Hasil Evaluasi Kinerja (Kontras Bercak)

Pengujian kuantitatif dilakukan dengan menghitung nilai **Kontras Bercak (KB)** pada Kanal Red (R) sebelum dan sesudah praproses:

$$KB = \frac{|\mu_{\text{bercak}} - \mu_{\text{sehat}}|}{\sigma_{\text{sehat}} + 10^{-6}}$$

| Kondisi Cahaya | $KB_{\text{sebelum}}$ | $KB_{\text{sesudah}}$ | Analisis / Implikasi |
| :--- | :---: | :---: | :--- |
| **Bayangan** | 5.63 | **8.59** | Meningkat signifikan berkat perbaikan kontras lokal CLAHE pada kanal V. |
| **Gelap** | 10.13 | **15.68** | Detail bercak terangkat tajam terhadap area daun sehat. |
| **Kontras Rendah** | 3.74 | **4.90** | Batas area bercak lebih mudah dipisahkan secara visual. |
| **Normal** | 8.07 | **9.94** | Memperjelas kekontrasan warna dan tepi objek secara optimal. |
| **Terang** | 6.24 | **6.15** | Nilai stabil; kehalusan citra terjaga tanpa distorsi berlebihan. |

---

## 🚀 Cara Menjalankan Proyek

1. **Clone repository ini:**
   ```bash
   git clone https://github.com/Rikza-Aidil/simuteh-2330511084.git
   cd simuteh-2330511084
   ```

2. **Jalankan Pembangkitan Dataset Sintetis:**
   ```bash
   python src/buat_dataset_sintetis.py
   ```

3. **Menjalankan Evaluasi Praproses di Notebook:**
   Buka notebook `notebook/B5_praproses.ipynb` di Google Colab atau Jupyter environment lokal.
