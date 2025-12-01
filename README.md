Berikut isi README.md yang bisa langsung kamu pakai. Tinggal simpan sebagai file `README.md` di folder `submission/`.

---

# CNN Image Classification – 5 Flower Types

Proyek ini adalah implementasi Convolutional Neural Network (CNN) untuk klasifikasi citra 5 jenis bunga menggunakan TensorFlow dan Keras. Proyek dibuat untuk memenuhi kriteria submission, termasuk penggunaan dataset ≥ 1000 gambar, pembagian train/validation/test, penggunaan model Sequential dengan Conv2D dan pooling, target akurasi ≥ 85%, serta penyimpanan model dalam format SavedModel, TFLite, dan TFJS.

## 1. Dataset

* Nama dataset: 5 Flower Types Classification Dataset
* Sumber: Kaggle
* URL: [https://www.kaggle.com/datasets/kausthubkannan/5-flower-types-classification-dataset](https://www.kaggle.com/datasets/kausthubkannan/5-flower-types-classification-dataset)
* Kelas:

  * Orchid
  * Tulip
  * Lilly
  * Sunflower
  * Lotus

Jumlah gambar (setelah download dan cek):

* Orchid: 1000 gambar
* Tulip: 1000 gambar
* Lilly: 999 gambar
* Sunflower: 1000 gambar
* Lotus: 1000 gambar

Total gambar: 4999 (≥ 1000, sesuai kriteria).

Semua gambar memiliki resolusi bervariasi, kemudian dinormalisasi (resize) ke ukuran 180×180 piksel pada saat loading dataset.

## 2. Struktur Direktori Submission

Struktur direktori akhir yang digunakan untuk submission:

```text
submission
├── tfjs_model
│   ├── model.json
│   ├── group1-shard1of8.bin
│   ├── group1-shard2of8.bin
│   ├── ...
│   └── group1-shard8of8.bin
├── tflite
│   ├── model.tflite
│   └── label.txt
├── saved_model
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── notebook.ipynb
├── README.md
└── requirements.txt
```

Struktur dataset di dalam Colab sebelum training:

```text
/content
├── dataset_raw
│   └── flower_images
│       ├── Orchid
│       ├── Tulip
│       ├── Lilly
│       ├── Sunflower
│       └── Lotus
└── dataset
    ├── train
    │   ├── Orchid
    │   ├── Tulip
    │   ├── Lilly
    │   ├── Sunflower
    │   └── Lotus
    ├── val
    │   ├── Orchid
    │   ├── Tulip
    │   ├── Lilly
    │   ├── Sunflower
    │   └── Lotus
    └── test
        ├── Orchid
        ├── Tulip
        ├── Lilly
        ├── Sunflower
        └── Lotus
```

Pembagian dataset dilakukan dengan skrip Python di dalam `notebook.ipynb` (train/val/test) dengan rasio yang dijelaskan di notebook.

## 3. Lingkungan dan Dependensi

Proyek dikerjakan menggunakan:

* Python (versi Colab saat training, ~3.12)
* TensorFlow (versi terbaru di Google Colab)
* TensorFlow Datasets & Keras
* TensorFlow Lite Converter
* TensorFlow.js Converter (tensorflowjs)
* Kaggle API (untuk download dataset)
* Library pendukung: numpy, matplotlib, pathlib, pillow (PIL), dll.

File `requirements.txt` dibuat dengan `pip freeze` dari environment Colab sehingga berisi daftar lengkap library yang dipakai pada saat notebook dijalankan.

## 4. Arsitektur Model

Model dibangun menggunakan `tf.keras.Sequential` dengan layer utama sebagai berikut:

1. Data augmentation (hanya pada training):

   * `RandomFlip("horizontal")`
   * `RandomRotation(0.1)`
   * `RandomZoom(0.1)`

2. Preprocessing:

   * `Rescaling(1./255)` untuk normalisasi piksel 0–255 menjadi 0–1.

3. Feature extractor (CNN):

   * `Conv2D(32, (3,3), activation="relu", padding="same")`
   * `MaxPooling2D(2,2)`
   * `Conv2D(64, (3,3), activation="relu", padding="same")`
   * `MaxPooling2D(2,2)`
   * `Conv2D(128, (3,3), activation="relu", padding="same")`
   * `MaxPooling2D(2,2)`

4. Classifier:

   * `Flatten()`
   * `Dense(128, activation="relu")`
   * `Dropout(0.5)`
   * `Dense(5, activation="softmax")` (5 kelas bunga)

Loss dan optimizer:

* Loss: `SparseCategoricalCrossentropy(from_logits=False)`
* Optimizer: `Adam()`
* Metrics: `accuracy`

Input shape gambar: `(180, 180, 3)`.

## 5. Pembagian Data dan Pipeline

Dataset di-load menggunakan `tf.keras.utils.image_dataset_from_directory`:

* Train set: digunakan untuk training model dan data augmentation.
* Validation set: digunakan untuk memonitor kinerja selama training.
* Test set: digunakan untuk evaluasi akhir model.

Pipeline `tf.data`:

* `cache()` untuk mempercepat loading data.
* `shuffle()` pada train set.
* `prefetch(buffer_size=tf.data.AUTOTUNE)` untuk pipeline yang lebih efisien di GPU/TPU.

## 6. Training dan Evaluasi

Training dilakukan di Google Colab dengan GPU (T4). Parameter utama (disesuaikan di notebook):

* Epoch: sekitar 10–20 (disesuaikan hingga akurasi memenuhi ≥ 85%).
* Batch size: misalnya 32.
* Callbacks: dapat berupa `EarlyStopping` dan `ModelCheckpoint` (jika diaktifkan di notebook).

Hasil utama:

* Train accuracy: ≥ 0.85
* Validation accuracy: ≥ 0.85
* Test accuracy: ≥ 0.85

Plot `accuracy` dan `loss` untuk train dan validation dibuat menggunakan `matplotlib` dan ditampilkan di notebook:

* Grafik akurasi vs epoch.
* Grafik loss vs epoch.

Grafik ini digunakan untuk mengecek indikasi overfitting/underfitting.

## 7. Penyimpanan Model (SavedModel, TFLite, TFJS)

Tiga format model yang dihasilkan:

1. SavedModel

   * Model diexport menggunakan:

     ```python
     model.export("saved_model")
     ```
   * Hasil: folder `saved_model` berisi `saved_model.pb` dan subfolder `variables/`.

2. TFLite

   * Konversi dilakukan dari SavedModel:

     ```python
     converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
     tflite_model = converter.convert()
     with open("tflite/model.tflite", "wb") as f:
         f.write(tflite_model)
     ```
   * File label kelas disimpan ke `tflite/label.txt` sesuai urutan `class_names`.

3. TFJS

   * Konversi dilakukan dengan tensorflowjs:

     ```python
     import tensorflowjs as tfjs
     tfjs.converters.convert_tf_saved_model(
         "saved_model",
         "tfjs_model"
     )
     ```
   * Hasil: `tfjs_model/model.json` dan beberapa file `group1-shardXofY.bin`. Banyaknya shard bergantung pada besar ukuran weight. Dalam kasus ini terbentuk 8 shard, yang seluruhnya wajib disertakan.

## 8. Cara Menjalankan Proyek

### 8.1. Menjalankan di Google Colab

1. Upload `notebook.ipynb` ke Google Colab.
2. Pastikan sudah mengatur Kaggle API (upload `kaggle.json` ke Colab dan set ke `~/.kaggle/kaggle.json`).
3. Jalankan cell secara berurutan:

   * Install dependensi (kaggle, tensorflowjs, dll).
   * Download dan ekstrak dataset dari Kaggle.
   * Split dataset menjadi train/val/test.
   * Load dataset dengan `image_dataset_from_directory`.
   * Bangun dan train model.
   * Plot akurasi dan loss.
   * Export SavedModel, TFLite, dan TFJS.
4. Di akhir, semua artefak akan berada di direktori sesuai struktur submission.

### 8.2. Menjalankan Inference (Contoh)

Contoh inference di notebook:

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from pathlib import Path

# Contoh path satu citra dari test set
img_path = Path("/content/dataset/test/Orchid/ec4df4d6c3.jpg")

img = image.load_img(img_path, target_size=(180, 180))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

pred = model.predict(img_array)
class_idx = np.argmax(pred[0])
class_names = train_ds.class_names  # dari dataset training
print("Predicted class:", class_names[class_idx])
print("Probabilities:", pred[0])
```

Prediksi bisa bervariasi, dan perlu dicek secara umum pada banyak sampel (bukan hanya 1 gambar) untuk menilai kualitas model.

## 9. File yang Disertakan dalam Submission

Submission .zip/.rar berisi:

* `notebook.ipynb`
  Notebook utama yang sudah dijalankan hingga selesai, dengan seluruh output terlihat.

* `saved_model/`
  Folder SavedModel hasil `model.export`.

* `tflite/model.tflite` dan `tflite/label.txt`
  Model untuk deployment di perangkat mobile/embedded dan file label kelas.

* `tfjs_model/`
  Model dalam format TensorFlow.js (file `model.json` dan semua shard bobot `.bin`).

* `requirements.txt`
  Daftar library Python beserta versinya dari environment training.

* `README.md`
  Dokumen penjelasan proyek (file ini).

## 10. Catatan

* Dataset yang digunakan berbeda dari dataset contoh kelas (bukan Rock–Paper–Scissors atau X-Ray) dan berisi lebih dari 1000 gambar, sehingga memenuhi kriteria.
* Model dibangun dengan `tf.keras.Sequential`, menggunakan Conv2D dan MaxPooling layer, sesuai permintaan.
* Akurasi training, validation, dan test telah mencapai ≥ 85%.
* Semua format model (SavedModel, TFLite, TFJS) berhasil dibuat dan disertakan dalam folder submission.

Jika file dijalankan kembali di environment baru, pastikan:

* Versi TensorFlow dan tensorflowjs kompatibel.
* Jalur folder (`dataset`, `saved_model`, `tflite`, `tfjs_model`) disesuaikan jika perlu.
