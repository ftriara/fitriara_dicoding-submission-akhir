# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

## Business Understanding

Jaya Jaya Institut merupakan salah satu institusi pendidikan tinggi yang telah berdiri sejak tahun 2000 dan telah menghasilkan banyak lulusan dengan reputasi yang baik. Namun demikian, institusi ini masih menghadapi tantangan serius berupa tingginya jumlah mahasiswa yang tidak menyelesaikan studi (dropout).

Fenomena dropout ini menjadi perhatian utama karena tidak hanya berdampak pada mahasiswa secara individu, tetapi juga mempengaruhi kinerja dan keberlanjutan institusi secara keseluruhan.

### Permasalahan Bisnis

Jaya Jaya Institut menghadapi permasalahan berupa **tingginya angka mahasiswa dropout**, yang mengindikasikan adanya kendala dalam aspek akademik, finansial, maupun keterlibatan mahasiswa selama masa studi.

Kondisi ini menimbulkan berbagai dampak negatif bagi institusi, antara lain:

- **Penurunan reputasi institusi**, karena rendahnya tingkat kelulusan dapat mempengaruhi persepsi publik dan calon mahasiswa.
- **Kerugian finansial**, akibat berkurangnya pendapatan dari mahasiswa yang tidak menyelesaikan pendidikan.
- **Inefisiensi operasional**, karena sumber daya pendidikan tidak dimanfaatkan secara optimal.
- **Menurunnya daya saing institusi**, terutama dalam persaingan antar perguruan tinggi.

Selain itu, institusi saat ini belum memiliki sistem yang mampu **mengidentifikasi mahasiswa berisiko dropout secara dini**, sehingga intervensi yang dilakukan seringkali terlambat dan kurang efektif.

Apabila permasalahan ini terus berlanjut, maka dalam jangka panjang institusi berisiko mengalami:

- Penurunan jumlah mahasiswa aktif
- Penurunan tingkat kelulusan (graduation rate)
- Menurunnya kepercayaan stakeholder (orang tua, pemerintah, dan industri)

Oleh karena itu, diperlukan solusi berbasis data yang mampu:

- Mengidentifikasi faktor utama penyebab dropout
- Memprediksi risiko dropout mahasiswa berdasarkan faktor-faktornya
- Mendukung pengambilan keputusan untuk intervensi yang lebih cepat dan tepat sasaran

### Cakupan Proyek

- Melakukan eksplorasi data (EDA) untuk memahami karakteristik karyawan
- Melakukan data preprocessing (missing values, encoding, feature engineering)
- Membangun model machine learning (Random Forest & XGBoost)
- Evaluasi model menggunakan ROC-AUC dan Recall (untuk data imbalance)
- Analisis explainable AI (SHAP & Feature Importance)
- Membangun dashboard interaktif menggunakan Metabase
- Menyusun insight bisnis dan rekomendasi strategis

### Persiapan

#### Sumber data: https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/refs/heads/main/students_performance/data.csv

#### Setup environment:
##### 1. Prasyarat
```bash
python --version
pip install --upgrade pip
```

##### 2. Clone & Setup Lingkungan
```bash
# Clone repositori
git clone https://github.com/ftriara/fitriara_dicoding-submission-akhir.git
cd fitriara_dicoding-submission-akhir

# Buat virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
# atau
venv\Scripts\activate          # Windows

# Install dependensi
pip install -r requirements.txt
```

## Business Dashboard

- Dashboard dibuat menggunakan Looker Studio untuk memvisualisasikan faktor-faktor yang mempengaruhi status kelulusan mahasiswa secara interaktif.
- Link Dashboard: https://lookerstudio.google.com/reporting/23748212-cf41-446f-8ad4-ac32d4cd2ede

### Bagian-Bagian Dashboard
- Filter
    - Terdapat 4 filter yang diterapkan yakni Gender, Course, Scholarship Holder, dan International yang memungkinkan eksplorasi data secara spesifik sesuai segmen mahasiswa
- Key Performance Indicators (KPI)
    - KPI yang ditampilkan adalah jumlah mahasiswa, rasio mahasiswa lulus, rasio mahasiswa dropout, dan rasio mahasiswa aktif
- Analisis Distribusi
    Pada bagian analisis distribusi terdapat 3 visualisasi yakni:
    - Distribusi status kelulusan mahasiswa
    - Performa mahasiswa berdasarkan gender
    - Performa mahasiswa berdasarkan kehadiran kuliah (siang atau malam)
- Faktor Risiko
    Berdasarkan faktor risiko dropout mahasiswa terdapat 3 visualisasi yaitu:
    - Approval rate (rasio SKS yang disetujui)
    - Rerata SKS yang disetujui di 2 semester
    - Rerata SKS di semester 2 yang disetujui
- Tekanan Finansial
    Berdasarkan risiko finansial pada mahasiswa dropout terdapat 3 visualisasi yaitu:
    - Tekanan finansial (Financial_Risk)
    - Biaya kuliah (Tuition_fees_up_to_date)
    - Beasiswa (Scholarship_holder)
- Segmentasi Risiko
    - Segmentasi risiko dropout (Low risk, Medium risk, High risk)
    - Tingkat dropout berdasarkan segmentasi risiko
    

## Menjalankan Sistem Machine Learning
Link Prototype: https://fitriara-dicoding-submission-akhir.streamlit.app/

#### 1. Jalankan Notebook Analisis
```bash
jupyter notebook notebook.ipynb
```

#### 2. Jalankan Prototype Streamlit (Lokal)
```bash
streamlit run app.py

# Buka aplikasi http://localhost:8501
```

## Conclusion

### 📊 Hasil Analisis

#### Temuan EDA Utama
1. **Performa Akademik Kritis**: Mahasiswa dropout rata-rata hanya menyetujui 0–2 sks di semester 2
2. **Biaya Kuliah**: 80% mahasiswa yang belum bayar SPP akhirnya dropout
3. **Debtor**: Mahasiswa berstatus debtor memiliki dropout rate 60%
4. **Beasiswa Protektif**: Penerima beasiswa 40% lebih kecil kemungkinan dropout
5. **Usia Masuk**: Mahasiswa berusia >30 tahun lebih berisiko dropout

#### Performa Model
| Model | Accuracy | F1-Weighted | F1-Macro |
|-------|----------|-------------|---------|
| Random Forest (Final) | 91.60% | 91.49% | 90.97% |
| XGBoost | 90.36% | 90.26% | 89.68% |

#### Top 5 Fitur Paling Penting
1. `Approval_rate` — Rasio SKS yang disetujui (20.35%)
2. `Avg_approved` — Rerata SKS yang disetujui (13.29%)
3. `Curricular_units_2nd_sem_approved` — SKS semester 2 yang disetujui (11.48%)
4. `Curricular_units_1st_sem_approved` — SKS semester 1 yang disetujui (7.70%)
5. `Avg_grade` — Rerata nilai (5.54%)

---

### Kesimpulan Dashboard

1. **32% mahasiswa dropout** — angka yang signifikan dan memerlukan perhatian serius
2. **Mahasiswa laki-laki** memiliki dropout rate lebih tinggi dibandingan perempuan
3. **Program malam** memiliki dropout rate lebih tinggi daripada program siang
4. **Rasio SKS yang disetujui** — semakin rendah rasio SKS yang disetujui maka risiko dropout semakin tinggi
5. **Tekanan finansial** - financial risk tinggi berkorelasi dengan peningkatan dropout.
6. **Pembayaran Kuliah** adalah prediktor paling kuat: mahasiswa yang menunggak memiliki dropout rate 80%
7. **Penerima beasiswa** secara signifikan lebih jarang dropout, menunjukkan efektivitas bantuan finansial
8. **Mahasiswa berusia 25+** memiliki risiko dropout yang lebih tinggi, mengindikasikan kebutuhan layanan khusus
9. **Segmentasi risiko** menunjukkan bahwa kelompok mahasiswa dengan risiko tinggi perlu mendapatkan perhatian khusus karena memiliki probabilitas dropout yang jauh lebih besar dibandingkan kelompok lainnya

---

### Action Items / Rekomendasi

1. Implementasi Early Warning System
   - Deteksi mahasiswa berisiko sejak dini berdasarkan approval rate rendah dan tekanan finansial tinggi, lalu kirim notifikasi ke wali akademik untuk intervensi cepat.
2. Program Dukungan Finansial
   - Sediakan skema keringanan SPP, cicilan, dan beasiswa darurat untuk mahasiswa dengan kendala finansial (tidak bayar SPP atau memiliki utang).
3. Pendampingan Akademik Terarah
   - Berikan coaching dan mentoring khusus untuk mahasiswa berisiko tinggi, terutama yang memiliki performa akademik rendah di semester awal.
4. Integrasi Dashboard & Model ke Sistem Akademik
   - Gunakan dashboard dan model prediksi secara real-time dalam sistem akademik untuk monitoring dan pengambilan keputusan berbasis data.
5. Evaluasi Kebijakan & Program Studi
   - Tinjau program studi dengan dropout tinggi serta perkuat layanan pendukung (akademik dan psikologis) untuk meningkatkan retensi mahasiswa.