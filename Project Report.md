# Laporan Proyek Machine Learning - Weather Type Classification
### Oleh Sulthan Muhammad Rafif Ilham

# 🌍 Domain Proyek

## 1. Latar Belakang
Cuaca merupakan aspek fundamental yang memengaruhi berbagai sendi kehidupan, terutama dalam konteks industri dan aktivitas harian. Prediksi cuaca yang tidak akurat dapat muncul akibat kompleksitas fenomena atmosfer dan berdampak langsung pada sektor vital seperti transportasi, pertanian, dan pariwisata. Sayangnya, upaya prediksi seringkali menghadapi tantangan karena pola data yang dinamis dan kompleks, sehingga memerlukan metode analisis tingkat lanjut untuk menghasilkan prediksi yang andal (Siregar et al., 2020).

Studi oleh Dandy, Udjulawa, dan Yohannes (2023) menegaskan bahwa ketersediaan dataset cuaca historis yang besar, seperti yang disediakan oleh berbagai platform data, membuka peluang untuk penerapan metode komputasi. Inisiatif untuk mengumpulkan dan membagikan data ini memungkinkan dilakukannya analisis mendalam untuk mengukur dan memprediksi kondisi cuaca. Data dari analisis ini menunjukkan bahwa berbagai fitur meteorologi memiliki pengaruh signifikan terhadap tipe cuaca dan menyarankan perlunya intervensi berbasis data untuk prediksi yang lebih akurat.

Lebih lanjut, penerapan algoritma machine learning dapat menjadi solusi strategis untuk mengklasifikasikan tipe cuaca secara preventif dan akurat. Dengan memanfaatkan perbandingan model seperti K-Nearest Neighbor, Decision Tree, dan terutama metode ensemble learning seperti Random Forest (Siregar, 2020), dapat diperoleh prediksi yang andal untuk mengidentifikasi kondisi cuaca mendatang. Hal ini memungkinkan berbagai sektor untuk melakukan perencanaan yang lebih baik dan mitigasi risiko secara lebih efektif.

## 2. Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan
Prediksi cuaca yang tidak akurat tidak hanya berdampak pada operasional harian, tetapi juga berimplikasi pada kerugian ekonomi, gangguan pada rantai pasok, kegagalan panen, dan meningkatnya risiko keselamatan. Berdasarkan temuan dari Siregar et al. (2020), pemilihan metode prediksi yang tepat akan sangat memengaruhi keakuratan hasilnya. Oleh karena itu, penting bagi berbagai sektor untuk memiliki sistem prediksi cuaca yang andal dan akurat untuk pengambilan keputusan yang lebih baik.

Solusi teknologi berbasis machine learning memungkinkan analisis data cuaca historis secara proaktif dan prediktif untuk mengklasifikasikan kondisi cuaca mendatang. Model klasifikasi seperti K-Nearest Neighbor (Dandy et al., 2023), Decision Tree, dan metode ensemble seperti Random Forest (Siregar, 2020) telah terbukti dapat mengklasifikasikan kondisi cuaca dengan tingkat akurasi yang tinggi. Berbagai penelitian menunjukkan bahwa perbandingan antar algoritma ini dapat membantu menemukan model yang paling sesuai untuk karakteristik dataset tertentu (Siregar et al., 2020). Model-model ini dapat diintegrasikan ke dalam sistem pendukung keputusan untuk logistik, pertanian, atau sistem peringatan dini.

Implementasi sistem prediksi ini tidak hanya bersifat reaktif terhadap kondisi cuaca, tetapi juga menjadi pendekatan preventif yang efektif untuk mitigasi risiko. Dengan demikian, berbagai sektor industri tidak hanya dapat meminimalkan kerugian akibat cuaca buruk, tetapi juga meningkatkan efisiensi operasional dan ketahanan (resilience) organisasi secara keseluruhan.

## 3. Referensi
- Dandy, D., Udjulawa, D., & Yohannes, Y. (2023). Implementasi Algoritma K-Nearest Neighbor untuk Klasifikasi Cuaca. Jurnal Algoritme, 4(1), 1-12.
- Siregar, A. M., Faisal, S., Cahyana, Y., & Priyatna, B. (2020). Perbandingan Algoritme Klasifikasi Untuk Prediksi Cuaca. Jurnal Accounting Information System (AIMS), 3(1), 15-24.
- Siregar, A. M. (2020). Klasifikasi Untuk Prediksi Cuaca Menggunakan Esemble Learning. Petir, 13(2), 522607.

# 📈 Business Understanding
## 1. Problem Statements (Pernyataan Masalah)
Prediksi cuaca yang akurat merupakan aspek krusial yang berpengaruh langsung terhadap efisiensi operasional dan keselamatan di berbagai sektor. Namun, banyak organisasi masih bergantung pada prakiraan umum yang kurang memiliki akurasi spesifik, sehingga tidak mampu mengantisipasi perubahan cuaca secara optimal. Akibatnya, keterlambatan atau ketidakakuratan prediksi sering menyebabkan kerugian finansial, seperti kegagalan panen di sektor pertanian, pembatalan penerbangan di industri transportasi, hingga risiko keselamatan pada acara luar ruangan.

Pernyataan masalah utama yang diangkat dalam proyek ini adalah:

* Bagaimana **membangun model *machine learning*** yang mampu mengklasifikasikan tipe cuaca dengan akurasi tinggi dan dapat diandalkan sebagai alat bantu pengambilan keputusan?
* Bagaimana **model ini dapat digunakan** oleh berbagai sektor untuk melakukan perencanaan proaktif dan mitigasi risiko terkait cuaca secara lebih efektif?

## 2. Goals (Tujuan)
Tujuan dari proyek ini adalah:

1.  Mengembangkan model klasifikasi tipe cuaca berdasarkan data yang mencakup fitur-fitur meteorologi seperti `Temperature`, `Humidity`, `Wind Speed`, dan lain-lain.
2.  Mengidentifikasi dan mengklasifikasikan kondisi cuaca mendatang ke dalam kategori spesifik: 'Sunny', 'Rainy', 'Cloudy', dan 'Snowy'.
3.  Menyediakan model prediktif yang dapat menjadi alat bantu bagi berbagai sektor (seperti pertanian, logistik, dan pariwisata) untuk melakukan perencanaan proaktif dan mitigasi risiko.
4.  Menunjukkan efektivitas penggunaan *machine learning* untuk meningkatkan akurasi prediksi cuaca sebagai bagian dari pengambilan keputusan strategis.

## 3. Solution Statement (Pernyataan Solusi)
Untuk mencapai tujuan di atas, berikut adalah dua solusi yang dirancang secara sistematis dan terukur:

### **Solusi 1: Implementasi dan Perbandingan Beberapa Algoritma Klasifikasi (Tahap yang Telah Dilaksanakan)**

Tahap pertama berfokus pada pelatihan dan perbandingan beberapa model *supervised learning* berbasis klasifikasi untuk menentukan model dasar (*baseline*) dengan performa terbaik.

* **Algoritma yang Digunakan**:
    Tiga model klasifikasi diimplementasikan menggunakan *dataset* cuaca yang telah dipersiapkan:
    * K-Nearest Neighbor (KNN)
    * Decision Tree
    * Random Forest

* **Metrik Evaluasi**:
    Performa setiap model diukur secara kuantitatif menggunakan metrik standar:
    * *Accuracy*
    * *Precision*
    * *Recall*
    * *F1-score*

* **Kriteria Pemilihan Model**:
    Model terbaik dipilih berdasarkan **akurasi tertinggi pada data uji** serta mempertimbangkan **stabilitas model** (perbedaan performa yang kecil antara data latih dan data uji).

### **Solusi 2: Peningkatan Model dengan Hyperparameter Tuning (Rekomendasi Tahap Lanjutan)**

Setelah menentukan model dasar terbaik (yaitu, Random Forest) dari Solusi 1, proses optimasi dapat dilakukan untuk meningkatkan performa model lebih lanjut.

* **Langkah yang Dapat Dilakukan**:
    * **Hyperparameter Tuning**: Melakukan pencarian parameter terbaik untuk model Random Forest menggunakan teknik seperti `GridSearchCV`. Parameter yang dapat dioptimalkan meliputi `n_estimators` (jumlah pohon), `max_depth` (kedalaman maksimum pohon), dan lainnya.
    * ***Feature Selection***: Menganalisis `feature_importances_` dari model Random Forest yang telah dilatih untuk mengidentifikasi dan mungkin hanya menggunakan fitur-fitur yang paling berpengaruh untuk menyederhanakan model tanpa mengorbankan akurasi.

* **Metrik Evaluasi Setelah Tuning**:
    * Model yang telah dioptimalkan akan dievaluasi kembali menggunakan metrik yang sama untuk menilai apakah terdapat peningkatan performa yang signifikan.
    * Penekanan pada peningkatan **akurasi dan F1-score secara keseluruhan** untuk mendapatkan model prediksi yang paling andal dan seimbang.

# 🧠 Data Understanding
## Deskripsi Dataset
*Dataset* yang digunakan dalam proyek ini adalah [Kaggle Weather Type Classification](https://www.kaggle.com/datasets/nikhil7280/weather-type-classification) yang bersumber dari platform publik **Kaggle**. *Dataset* ini berisi **13.200 baris** data historis cuaca yang dirancang khusus untuk tugas klasifikasi.

Setiap baris dalam data merepresentasikan sebuah catatan kondisi cuaca dengan **11 atribut (fitur)** meteorologi yang berbeda, seperti `Temperature`, `Humidity`, `Wind Speed`, dan `Cloud Cover`. Data ini disusun untuk mencerminkan berbagai kondisi cuaca, di mana fitur-fitur tersebut digunakan untuk melatih model dalam mengenali dan memprediksi `Weather Type` sebagai variabel target.

## Tujuan dan Kegunaan Dataset
*Dataset* ini dirancang untuk mendukung berbagai eksperimen dalam analisis dan prediksi cuaca, di antaranya:

1.  Memprediksi tipe cuaca ('Sunny', 'Rainy', 'Cloudy', 'Snowy') berdasarkan data fitur meteorologi.
2.  Menganalisis faktor-faktor cuaca (seperti suhu, kelembapan, dan tekanan atmosfer) yang paling berpengaruh dalam menentukan tipe cuaca tertentu.
3.  Melatih dan membandingkan berbagai model klasifikasi untuk menemukan algoritma yang paling akurat dan efisien.
4.  Menjadi dasar untuk membangun aplikasi praktis seperti sistem peringatan dini cuaca atau *dashboard* perencanaan untuk sektor agrikultur dan logistik.

*Dataset* ini ideal untuk melatih dan menguji berbagai model *machine learning*, melakukan analisis korelasi fitur, dan membangun sistem klasifikasi cuaca berbasis data.

## Data yang Digunakan
Dari total **13.200 baris data** awal, proyek ini melakukan tahap pembersihan data untuk menangani nilai pencilan (*outliers*) yang dapat memengaruhi kualitas dan performa model. Proses ini dilakukan dengan menggunakan metode *Interquartile Range* (IQR) untuk mengidentifikasi dan menghapus baris data yang memiliki nilai ekstrem pada fitur-fitur numerik.

Setelah proses pembersihan data selesai, *dataset* final yang digunakan untuk tahap pemodelan adalah sebanyak **11.586 baris**. Pengurangan jumlah baris ini memastikan bahwa data yang digunakan lebih bersih dan representatif terhadap kondisi cuaca yang umum, sehingga dapat menghasilkan model yang lebih andal.

## Struktur Dataset
Berikut adalah informasi awal dari *dataset* berdasarkan fungsi `df.info()` dan `df.describe()`:

* Terdapat **11 kolom (fitur)**, yang terdiri dari 7 fitur numerik (`int64`, `float64`) dan 4 fitur kategorikal (`object`).
* **Tidak ada nilai yang hilang** di dalam *dataset* (semua 13.200 baris data terisi lengkap).
* Distribusi nilai numerik menunjukkan beberapa karakteristik penting:
    * **`Temperature`**: Memiliki rentang yang sangat lebar (dari -25 hingga 109), yang mengindikasikan variasi data dari berbagai musim atau kondisi.
    * **`Humidity` dan `Precipitation (%)`**: Menunjukkan nilai maksimum yang tidak lazim (di atas 100), menandakan adanya potensi anomali data yang perlu ditangani.
    * **`UV Index`**: Memiliki rentang nilai yang realistis (0 hingga 14).

## Deskripsi Fitur
Berikut adalah penjelasan untuk setiap fitur yang terdapat dalam dataset.

### **Fitur Kategorikal**

* **`Cloud Cover`**
    Kondisi tutupan awan di atmosfer. Nilai: `'clear'`, `'partly cloudy'`, `'cloudy'`, `'overcast'`.
* **`Season`**
    Musim saat data cuaca direkam. Nilai: `'Winter'`, `'Spring'`, `'Summer'`, `'Autumn'`.
* **`Location`**
    Jenis lokasi geografis tempat data diambil. Nilai: `'inland'`, `'mountain'`, `'coastal'`.
* **`Weather Type`**
    **(Variabel Target)**. Kategori tipe cuaca yang akan diprediksi. Nilai: `'Sunny'`, `'Cloudy'`, `'Rainy'`, `'Snowy'`.

### **Fitur Numerikal**

* **`Temperature`**
    Suhu udara yang terukur, dalam derajat.
* **`Humidity`**
    Tingkat kelembapan udara, dalam persen (%).
* **`Wind Speed`**
    Kecepatan angin yang tercatat.
* **`Precipitation (%)`**
    Persentase atau tingkat presipitasi (curah hujan).
* **`Atmospheric Pressure`**
    Tekanan udara atmosfer.
* **`UV Index`**
    Skor indeks paparan radiasi ultraviolet dari matahari.
* **`Visibility (km)`**
    Jarak pandang di udara, dalam satuan kilometer.

## Ringkasan Eksplorasi (Exploratory)
* *Dataset* awal berisi **13.200 baris** data yang lengkap.
* Semua fitur memiliki tipe data yang sesuai konteksnya: **numerik** (`float64`, `int64`) untuk pengukuran dan **kategorikal** (`object`) untuk klasifikasi.
* Tidak terdapat **nilai yang hilang (*missing values*)**, sehingga tidak diperlukan proses imputasi.
* Rata-rata **suhu (`Temperature`)** adalah **19.1°**, dengan rata-rata **kelembapan (`Humidity`)** sekitar **68.7%**.
* Teridentifikasi adanya ***outlier* yang signifikan** pada beberapa fitur numerik seperti `Temperature`, `Wind Speed`, dan `Atmospheric Pressure`, yang memerlukan penanganan khusus pada tahap persiapan data.
* Analisis korelasi menunjukkan adanya hubungan linear yang kuat, baik **positif** (antara `Temperature` dan `UV Index` sebesar +0.81) maupun **negatif** (antara `Temperature` dan `Humidity` sebesar -0.85).
* Fitur kategorikal independen (`Season`, `Location`) menunjukkan distribusi yang relatif seimbang, namun ditemukan adanya **ketidakseimbangan kelas (*class imbalance*)** pada variabel target `Weather Type`, di mana kelas 'Snowy' lebih sedikit jumlahnya.

# ⚙️ Data Preparation
Tahapan *data preparation* dilakukan untuk memastikan bahwa data dalam kondisi optimal sebelum digunakan dalam proses *modeling*. Langkah-langkah ini sangat krusial karena kualitas data input akan secara langsung memengaruhi performa dan keandalan model yang dihasilkan. Proses dilakukan secara berurutan sebagai berikut:

### **1. Encoding Variabel Kategorikal**

Langkah pertama adalah mengubah semua variabel yang bersifat kategorikal (teks) menjadi format numerik, karena algoritma *machine learning* hanya dapat memproses angka.

* **Metode Encoding yang Digunakan**
    Pada proyek ini, *encoding* dilakukan secara **manual menggunakan fungsi `map()`** pada pandas. Metode ini dipilih karena beberapa alasan:
    * **Transparansi dan Kontrol Penuh**: Memberikan kontrol langsung terhadap nilai numerik yang dipetakan ke setiap kategori.
    * **Kesederhanaan**: Efisien untuk variabel dengan jumlah kategori yang tidak terlalu banyak.
    * **Menghindari Overhead**: Tidak memerlukan *import library encoder* tambahan.

* **Implementasi Encoding**
    Setiap variabel kategorikal, termasuk fitur independen dan variabel target, diubah menjadi representasi numerik sebagai berikut:
    * **`Cloud Cover`**: `'overcast'`→0, `'cloudy'`→1, `'partly cloudy'`→2, `'clear'`→3.
    * **`Season`**: `'Winter'`→0, `'Autumn'`→1, `'Spring'`→2, `'Summer'`→3.
    * **`Location`**: `'inland'`→0, `'mountain'`→1, `'coastal'`→2.
    * **`Weather Type` (Variabel Target)**: `'Snowy'`→0, `'Rainy'`→1, `'Cloudy'`→2, `'Sunny'`→3.

### **2. Penanganan Data Tidak Seimbang (*Class Imbalance*)**

Tahap eksplorasi data (EDA) sebelumnya mengidentifikasi adanya ketidakseimbangan pada distribusi kelas variabel target `Weather Type`. Untuk mencegah model menjadi bias dan cenderung memprediksi kelas mayoritas, diterapkan sebuah teknik **penyeimbangan data (seperti *oversampling*)**. Hasil dari langkah ini adalah sebuah *dataset* di mana setiap kelas cuaca memiliki jumlah sampel yang sama persis (3.300 data), menciptakan kondisi yang ideal untuk pelatihan model yang adil.

### **3. Penanganan Outlier**

Berdasarkan deteksi *outlier* pada tahap EDA, langkah selanjutnya adalah membersihkan data dari nilai-nilai ekstrem yang dapat mengganggu proses pembelajaran model.
* **Metode**: Metode **Interquartile Range (IQR)** digunakan untuk mengidentifikasi *outlier*. Sebuah titik data dianggap *outlier* jika nilainya berada di luar rentang `Q1 - 1.5 * IQR` dan `Q3 + 1.5 * IQR`.
* **Tindakan**: Seluruh baris yang mengandung nilai *outlier* pada salah satu fitur numerik **dihapus** dari *dataset*.
* **Hasil**: Proses ini mengurangi jumlah data dari 13.200 menjadi **11.586 baris**. *Dataset* yang dihasilkan kini lebih bersih dan lebih representatif terhadap kondisi cuaca yang umum.

### **4. Standarisasi Fitur Numerikal**

Fitur-fitur numerik dalam *dataset* memiliki skala dan rentang nilai yang sangat berbeda (misalnya, `Temperature` vs. `Atmospheric Pressure`). Untuk menyamakan skala tersebut, dilakukan standarisasi menggunakan **`StandardScaler`** dari *library* scikit-learn. Proses ini mengubah setiap fitur sehingga memiliki:
* **Rata-rata (Mean) = 0**
* **Standar Deviasi (Standard Deviation) = 1**

Standarisasi sangat penting untuk memastikan setiap fitur memberikan kontribusi yang seimbang, terutama untuk algoritma yang sensitif terhadap jarak seperti KNN.

### **5. Train-Test Split**

Langkah terakhir dalam persiapan data adalah membagi *dataset* yang sudah bersih dan terstandarisasi menjadi dua bagian untuk tujuan pelatihan dan pengujian.
* **Proporsi Pembagian**: Data dibagi dengan rasio **80% untuk *training set*** dan **20% untuk *testing set***.
* **Ukuran Data Setelah Pembagian**:
    * Total data: 11.586 sampel
    * *Training set*: 9.268 sampel
    * *Testing set*: 2.318 sampel
* **Reproducibility**: Pemisahan dilakukan dengan `random_state=123` untuk memastikan bahwa pembagian data bersifat konsisten dan dapat direplikasi, sehingga hasil eksperimen menjadi valid.

Pembagian ini bertujuan untuk melatih model pada sebagian besar data dan mengevaluasi performa akhirnya pada data yang belum pernah dilihat sama sekali, demi mendapatkan gambaran objektif mengenai kemampuan generalisasi model.

# 🧮 Modeling
Tahap ini berfokus pada pembangunan dan pelatihan model *machine learning* untuk mengklasifikasikan tipe cuaca berdasarkan fitur-fitur yang telah diproses. Untuk memperoleh gambaran performa awal dan membandingkan efektivitas berbagai pendekatan, digunakan **tiga model klasifikasi populer** sebagai model dasar (*baseline*) dengan menggunakan *hyperparameter default*. Tujuan utamanya adalah mengidentifikasi model dengan performa terbaik dari ketiga kandidat tersebut.

## **Model yang Digunakan**

Berikut adalah tiga model klasifikasi yang diimplementasikan:

### **1. K-Nearest Neighbors (KNN)**

KNN mengklasifikasikan data baru berdasarkan mayoritas label dari 'k' tetangga terdekatnya dalam ruang fitur. Penggunaan model ini didukung oleh keberhasilan tahap standarisasi data, karena sebagai algoritma berbasis jarak, performa KNN sangat bergantung pada skala fitur yang seragam. Studi oleh Dandy et al. (2023) juga menunjukkan penerapan yang berhasil dari KNN untuk kasus klasifikasi cuaca.

*Hyperparameter Default:*
```python
KNeighborsClassifier(
    n_neighbors=5,       # Jumlah tetangga yang digunakan
    weights='uniform',   # Bobot: 'uniform' (sama) atau 'distance' (berdasarkan jarak)
    algorithm='auto',    # Algoritma pencarian tetangga
    p=2,                 # Parameter jarak (1=manhattan, 2=euclidean)
    metric='minkowski'   # Metrik jarak
)
```

### **2. Decision Tree (DT)**

Decision Tree membangun model prediktif dalam bentuk struktur pohon keputusan. Algoritma ini mampu menangkap hubungan non-linear antar fitur dan memberikan hasil yang mudah diinterpretasikan dalam bentuk aturan 'jika-maka'. Kemampuannya untuk memisahkan kelas target menjadikannya kandidat yang kuat, sejalan dengan penelitian yang membandingkan berbagai algoritma untuk prediksi cuaca (Siregar et al., 2020).

*Hyperparameter Default:*
```python
DecisionTreeClassifier(
    criterion='gini',     # Kriteria untuk mengukur kualitas split
    max_depth=None,       # Kedalaman maksimal pohon (None=tanpa batas)
    min_samples_split=2,  # Minimum sampel untuk melakukan split
    min_samples_leaf=1    # Minimum sampel di setiap daun (leaf node)
)
```

### **3. Random Forest (RF)**

Random Forest adalah algoritma *ensemble* yang membangun banyak Decision Tree dan menggabungkan prediksinya untuk meningkatkan akurasi serta mengurangi risiko *overfitting*. Mengingat kompleksitas data cuaca, pendekatan *ensemble* seperti ini sangat cocok untuk menangani interaksi fitur yang rumit. Penggunaan *ensemble learning* untuk prediksi cuaca juga didukung oleh penelitian Siregar (2020) yang menyoroti keunggulan metode ini.

*Hyperparameter Default:*
```python
RandomForestClassifier(
    n_estimators=100,     # Jumlah pohon dalam "hutan"
    criterion='gini',     # Kriteria untuk mengukur kualitas split
    max_depth=None,       # Kedalaman maksimal setiap pohon
    bootstrap=True        # Metode sampling dengan replacement
)
```
---
## **Alasan Penggunaan Beberapa Model**

Penggunaan tiga model yang berbeda ini bertujuan untuk:
* **Membandingkan Performa**: Melihat kinerja dari pendekatan algoritma yang berbeda (berbasis jarak, berbasis aturan tunggal, dan berbasis *ensemble*).
* **Menemukan Model Terbaik**: Mengidentifikasi model yang paling sesuai dengan karakteristik *dataset* cuaca yang telah dipersiapkan.
* **Memberikan Landasan Objektif**: Menjadi dasar yang kuat dalam proses seleksi model terbaik berdasarkan data empiris dari hasil evaluasi.

Seluruh model ini dilatih menggunakan `X_train` dan `y_train`, dan selanjutnya akan dievaluasi secara komprehensif.

# 💯 Evaluation
Tahap evaluasi bertujuan untuk menilai performa setiap model yang telah dilatih secara kuantitatif. Proses ini krusial untuk membandingkan efektivitas masing-masing algoritma dan memilih model terbaik yang memiliki kemampuan generalisasi paling andal pada data baru. Evaluasi dilakukan menggunakan metrik standar untuk kasus klasifikasi.

## **Metrik Evaluasi yang Digunakan**

* **Akurasi**: Mengukur proporsi total prediksi yang benar dari keseluruhan data.
  $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
* **Precision**: Mengukur tingkat ketepatan dari prediksi positif yang dibuat oleh model.
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
* **Recall**: Mengukur kemampuan model untuk menemukan kembali semua sampel positif yang sebenarnya ada.
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
* **F1-Score**: Rata-rata harmonik dari *Precision* dan *Recall*, memberikan skor tunggal yang menyeimbangkan kedua metrik tersebut.
  $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
* **Confusion Matrix**: Sebuah tabel yang memvisualisasikan performa model dengan merangkum jumlah prediksi yang benar (*True Positives*, *True Negatives*) dan yang salah (*False Positives*, *False Negatives*) untuk setiap kelas.

---
## **Hasil Evaluasi pada Data Latih**

Evaluasi pertama dilakukan pada *training set* untuk melihat seberapa baik model "mempelajari" data yang diberikan kepadanya.

| Model | Akurasi | Precision (Avg) | Recall (Avg) | F1-score (Avg) |
| :--- | :---: | :---: | :---: | :---: |
| **KNN** | 0.97 | 0.97 | 0.97 | 0.97 |
| **Decision Tree** | 1.00 | 1.00 | 1.00 | 1.00 |
| **Random Forest** | 1.00 | 1.00 | 1.00 | 1.00 |

**Interpretasi:**
Hasil pada data latih menunjukkan bahwa model **Decision Tree** dan **Random Forest** mencapai performa sempurna (100%). Ini adalah indikasi kuat terjadinya **overfitting**, di mana model tidak hanya belajar pola, tetapi juga "menghafal" data latih. Sebaliknya, model **KNN** menunjukkan performa sangat tinggi (97%) namun tidak sempurna, menandakan kemampuan belajar yang baik tanpa menghafal data secara ekstrem.

---
## **Evaluasi pada Data Uji**

Ini adalah evaluasi yang paling penting untuk mengukur kemampuan generalisasi model pada data yang belum pernah dilihat sebelumnya.

| Model | Akurasi | Precision (Avg) | Recall (Avg) | F1-score (Avg) |
| :--- | :---: | :---: | :---: | :---: |
| **KNN** | 0.96 | 0.96 | 0.96 | 0.96 |
| **Decision Tree** | 0.98 | 0.98 | 0.98 | 0.98 |
| **Random Forest** | 0.98 | 0.98 | 0.98 | 0.98 |

**Interpretasi:**
* **Random Forest** dan **Decision Tree**: Meskipun menunjukkan tanda *overfitting* yang kuat pada data latih, kedua model ini ternyata mampu melakukan generalisasi dengan sangat baik. Penurunan performa yang hanya sekitar 2% (dari 100% ke 98%) menunjukkan bahwa model berhasil menangkap pola fundamental yang relevan dan tidak hanya "menghafal" noise. Keduanya menjadi kandidat model terbaik.
* **KNN**: Model ini membuktikan stabilitasnya yang luar biasa. Dengan penurunan akurasi hanya 1% (dari 97% ke 96%), KNN adalah model yang paling konsisten dan dapat diandalkan, dengan risiko *overfitting* yang paling rendah di antara ketiganya.
* **Kesimpulan Perbandingan**: Secara keseluruhan, meskipun KNN sangat stabil, **Random Forest dan Decision Tree menunjukkan performa prediktif terbaik** pada data uji, yang merupakan tolok ukur utama keberhasilan sebuah model.

# 📋 Kesimpulan
Berdasarkan hasil evaluasi yang telah dilakukan pada data latih dan data uji, dapat ditarik beberapa poin kesimpulan utama mengenai performa masing-masing model:

* **Decision Tree dan Random Forest**: Kedua model ini berhasil mencapai **performa prediktif tertinggi** dengan akurasi **98%** pada data uji. Namun, keduanya juga menunjukkan tanda **overfitting** yang kuat pada data latih (di mana akurasinya mencapai 100%), yang menandakan model-model ini sangat kompleks dan mampu "menghafal" data.

* **K-Nearest Neighbors (KNN)**: Model ini terbukti menjadi yang **paling stabil dan konsisten**. Dengan selisih akurasi hanya 1% antara data latih (97%) dan data uji (96%), KNN menunjukkan kemampuan generalisasi yang sangat baik dan risiko *overfitting* yang paling rendah.

## **Rekomendasi Model**

Pemilihan model terbaik tidak hanya bergantung pada akurasi tertinggi, tetapi juga pada kebutuhan spesifik dari kasus penggunaan, seperti interpretability, stabilitas, dan keandalan. Berikut adalah rekomendasi mendalam untuk setiap model:

* **Rekomendasi Utama: Random Forest**
    **Random Forest adalah pilihan terbaik secara keseluruhan untuk proyek ini.** Model ini menawarkan kombinasi yang unggul antara akurasi prediksi tertinggi (98%) dan keandalan. Sebagai model *ensemble*, ia secara signifikan lebih **robust** dan stabil daripada satu Decision Tree tunggal. Dengan menggabungkan prediksi dari banyak pohon, Random Forest lebih tahan terhadap *noise* dan variasi kecil dalam data, menjadikannya pilihan yang paling aman dan dapat diandalkan untuk aplikasi prediksi cuaca di dunia nyata.

* **Alternatif 1: Decision Tree**
    Model Decision Tree cocok digunakan jika **prioritas utamanya adalah interpretability (kemampuan untuk ditafsirkan)**. Dengan akurasi yang setara dengan Random Forest (98%), keunggulan utamanya adalah kemampuannya untuk menyajikan alur keputusan yang jelas dan mudah dipahami dalam bentuk aturan 'jika-maka'. Ini sangat berguna jika tujuannya adalah untuk memahami faktor-faktor apa saja yang paling memengaruhi prediksi cuaca. Namun, kelemahannya adalah risiko *overfitting* dan ketidakstabilan yang lebih tinggi.

* **Alternatif 2: K-Nearest Neighbors (KNN)**
    Model KNN adalah pilihan yang sangat solid jika **stabilitas dan konsistensi menjadi prioritas utama**. Meskipun akurasinya sedikit di bawah model berbasis pohon (96%), performanya yang sangat konsisten antara data latih dan data uji membuktikan bahwa model ini sangat dapat diandalkan. KNN bisa menjadi *baseline* yang sangat kuat atau bahkan model produksi jika selisih akurasi 2% dianggap tidak terlalu signifikan untuk kasus penggunaannya.
