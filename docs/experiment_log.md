# Experiment Log — Phase 3

Dokumen ini mencatat eksekusi teknis pipeline fitur teks semantik. Isinya
meliputi lingkungan, konfigurasi, hasil, verifikasi, dan temuan awal. Dokumen
ini **bukan** laporan hasil evaluasi rekomendasi formal.

Tanggal pencatatan: 12 November 2026.

## 1. Ringkasan

| Item | Nilai |
|---|---|
| Dataset mentah | 443.610 baris, 29 kolom |
| Film unik setelah deduplikasi | 429.755 |
| Film dengan teks semantik | 396.254 |
| Film tanpa teks semantik | 33.501 |
| Dimensi embedding | 768 |
| Jumlah embedding | 396.254 |
| Waktu encoding | 1.671,52 detik (± 27,9 menit) |
| Peak VRAM | 393,8 MiB |
| Device | `cuda:0` |
| Batch size | 8 |

## 2. Lingkungan

| Komponen | Versi / Nilai |
|---|---|
| Python | 3.11.18 |
| Package manager | uv 0.13.7 |
| PyTorch | 2.15.0+cu134 |
| CUDA tersedia | Ya |
| sentence-transformers | 6.2.0 |
| NumPy | 2.4.6 |
| pandas | 3.1.1 |
| scikit-learn | 1.9.1 |
| GPU | NVIDIA GeForce RTX 3050, 4 GB VRAM |
| RAM | ± 15,6 GB |
| Parquet engine | pyarrow |

## 3. Eksperimen 1 — Audit Dataset

**Tujuan:** memahami struktur dan kualitas data sebelum preprocessing.

**Input:** `data/raw/TMDB  IMDB Movies Dataset.csv`

**Hasil:**

- 443.610 baris, 29 kolom.
- 429.755 ID unik.
- 3.463 grup ID duplikat, melibatkan 17.318 baris.
- Tidak ada ID kosong.
- Tidak ada baris yang identik penuh.
- `id` bertipe `int64`; metadata berupa string comma-separated, bukan JSON.

**Missing value pada kolom penelitian:**

| Kolom | Missing | Whitespace-only |
|---|---:|---:|
| `overview` | 44.973 | 64 |
| `tagline` | 349.602 | 0 |
| `keywords` | 271.970 | 0 |
| `genres` | 83.793 | 0 |
| `directors` | 10.003 | 0 |
| `writers` | 64.952 | 0 |
| `cast` | 69.641 | 0 |

**Catatan:** empat tagline berisi nilai placeholder `none`, `n/a`, atau `-`.

## 4. Eksperimen 2 — Deduplikasi

**Tujuan:** menghasilkan satu record per film dengan metadata terlengkap.

**Aturan:** skor kelengkapan tujuh kolom relevan; tie-break panjang teks
setelah whitespace dibersihkan; tie-break terakhir urutan baris.

**Hasil:**

```text
input  : (443610, 29)
output : (429755, 29)
duplicate IDs: 0
```

**Perbandingan dengan `keep="first"`:**

| Metrik | Nilai |
|---|---:|
| Film dengan baris berbeda | 2.294 |
| `cast` yang berhasil dipertahankan | 1.662 |
| `genres` yang berhasil dipertahankan | 18 |
| `overview` yang berhasil dipertahankan | 4 |
| `tagline` yang berhasil dipertahankan | 1 |
| `keywords` yang berhasil dipertahankan | 1 |

**Output:** `data/processed/movies.parquet` (128.069.380 byte).

Verifikasi round-trip dengan `pd.testing.assert_frame_equal` berhasil.

## 5. Eksperimen 3 — Pembentukan dan Audit Teks Semantik

**Tujuan:** membentuk `semantic_text` dan mengukur panjang token sebelum
encoding.

**Aturan penggabungan:**

```text
Overview: {overview}

Tagline: {tagline}

Keywords: {keywords}
```

Field kosong dilewati.

**Hasil:**

| Metrik | Nilai |
|---|---:|
| Baris master | 429.755 |
| `semantic_text` tidak kosong | 396.254 |
| `semantic_text` kosong | 33.501 |
| Punya metadata tetapi tanpa teks | 33.405 |
| Tanpa teks dan tanpa metadata | 96 |

**Output:**

- `artifacts/semantic_baseline/semantic_text.parquet` (60.001.596 byte)
- `artifacts/semantic_baseline/token_lengths.parquet` (1.721.495 byte)

**Token audit:**

| Statistik | Nilai |
|---|---:|
| Teks diukur | 396.254 |
| Minimum | 5 |
| Median | 61 |
| Persentil 90 | 140 |
| Persentil 95 | 168 |
| Persentil 99 | 215 |
| Maksimum | 782 |
| Teks > 250 token | 520 (0,13%) |
| Teks > 512 token | 16 (0,0040%) |

`max_seq_length` model: **512**.

Keputusan baseline: pertahankan batas 512. Teks yang melebihi batas akan
terpotong saat encoding.

## 6. Eksperimen 4 — Encoding SBERT

**Model:** `sentence-transformers/multi-qa-distilbert-cos-v1`

**Konfigurasi:**

```text
device       : cuda:0
batch_size   : 8
max_length   : 512
normalize    : True
dtype output : float32
```

**Hasil:**

```text
embeddings shape : (396254, 768)
embeddings dtype : float32
movie_ids shape  : (396254,)
movie_ids dtype  : int64
unique ids       : 396254
```

**Waktu dan memori:**

- Encoding: 1.671,52 detik.
- Peak GPU memory: 393,8 MiB.
- Benchmark 500 teks sebelumnya: 4,09 detik, ± 122 teks/detik.

Estimasi awal sekitar 54 menit, tetapi eksekusi aktual sekitar 27,9 menit.
Perbedaan ini menunjukkan sampel benchmark tidak sepenuhnya mewakili distribusi
panjang teks seluruh dataset.

**Output:**

| File | Ukuran |
|---|---:|
| `artifacts/semantic_baseline/embeddings.npy` | 1.217.292.416 byte |
| `artifacts/semantic_baseline/movie_ids.npy` | 3.170.160 byte |
| `artifacts/semantic_baseline/config.json` | 638 byte |

## 7. Verifikasi Artifact

**Pemeriksaan:**

| Pemeriksaan | Hasil |
|---|---|
| Shape embedding | `(396254, 768)` |
| Dtype embedding | `float32` |
| Shape movie_ids | `(396254,)` |
| Dtype movie_ids | `int64` |
| ID unik | 396.254 |
| Norm sampel | 0,99999994–1,0000001 |
| Nilai finite sampel | Ya |
| Pembacaan ulang `mmap_mode="r"` | Berhasil |

**Spot check alignment:**

| ID | Judul | Baris |
|---|---|---:|
| 27205 | Inception | 0 |
| 157336 | Interstellar | 1 |
| 155 | The Dark Knight | 2 |
| 862 | Toy Story | 73 |

## 8. Sanity Check Similarity

**Metode:** cosine similarity dihitung sebagai dot product karena embedding
sudah L2-normalized. Film query dikeluarkan dengan mengubah skornya menjadi
`-inf`.

**Contoh hasil Top-1:**

| Query | Top-1 | Similarity | Observasi |
|---|---|---:|---|
| Inception | Respect the Jux | 0,5609 | Cluster crime/heist |
| Interstellar | The Beyond | 0,6550 | Cluster science fiction/space |
| The Dark Knight | The Dark Knight Rises | 0,6963 | Franchise Batman/DC |
| Toy Story | Toy Story 3 | 0,7221 | Sekuel resmi, lalu kartun lama |

**Rentang skor empat query Inggris:** ± 0,52–0,72.

**Temuan kualitatif:**

- Interstellar menghasilkan cluster film luar angkasa yang koheren.
- The Dark Knight menghasilkan film Batman/DC.
- Inception menghasilkan film crime/heist/psychological thriller.
- Toy Story menghasilkan tiga sekuel resmi, lalu beberapa kartun lama dengan
  karakter "Woody". Ini menunjukkan embedding juga dapat tertarik pada
  kemiripan leksikal, bukan hanya tema.
- Beberapa rekomendasi memiliki `genre_overlap = 0` tetapi similarity cukup
  tinggi, misalnya Inception → American Psycho. Artinya sinyal teks tidak
  identik dengan genre.

**Output contoh:**

- `results/recommendations/semantic_top10_descriptive.csv`
- `results/recommendations/semantic_top10_non_english.csv`
- `results/semantic_text_audit.csv`

## 9. Pemeriksaan Film Non-Inggris

Dua film dengan teks berbahasa Rusia diuji:

| Query | Similarity | Komposisi `original_language` Top-10 |
|---|---|---:|
| Yegor's Grief (77994) | 0,80–0,83 | 7 `ru`, 3 `en` |
| Sokrovishche (64060) | 0,77–0,80 | 6 `ru`, 4 `en` |

Similarity lebih tinggi daripada contoh film Inggris. Namun hasil ini **tidak**
dapat diartikan sebagai kualitas rekomendasi yang lebih baik.

**Catatan metadata:** sebagian film berlabel `original_language = en` memiliki
`spoken_languages = Russian` atau kosong. Karena itu, analisis bahasa tidak
boleh hanya bergantung pada `original_language`.

**Interpretasi awal:** kemiripan tinggi pada teks Cyrillic dapat dipengaruhi
kesamaan bahasa/script. Model yang dipakai berfokus pada bahasa Inggris.
Temuan ini dicatat sebagai keterbatasan, bukan kesimpulan final.

## 10. Unit Test

Perintah:

```powershell
uv run pytest -q
```

Hasil: **32 passed**.

Cakupan:

- loader: path, folder, suffix, file kosong
- validator: missing, whitespace, duplikat ID, kolom hilang
- deduplikasi: kelengkapan, tie-break, urutan, null ID
- path: root project dan path absolut
- text preprocessing: label, field kosong, placeholder, index
- SBERT wrapper: validasi input, shape output, nilai non-finite

## 11. Catatan Teknis

- Model dimuat sekali dan dipakai ulang; tidak ada encoding berulang per
  eksperimen alpha.
- Embedding disimpan sebagai `float32` untuk menghemat ruang.
- `movie_ids.npy` wajib dibaca bersama `embeddings.npy`; tanpa file ID,
  embedding tidak dapat dipetakan ke film.
- `semantic_text` kosong tidak di-encode; film tetap berada di master dataset.
- Encoding menggunakan model pretrained tanpa fine-tuning.

## 12. Rencana Berikutnya

Belum dieksekusi:

1. Phase 4 — metadata feature dan CountVectorizer.
2. Phase 5 — cosine similarity dua pipeline dan late fusion `alpha`.
3. Phase 6 — recommendation engine Top-K.
4. Phase 7 — ground truth, Precision@K, Recall@K, MAP@K.
5. Phase 8 — eksperimen tujuh nilai `alpha`.

Phase 3 dianggap selesai setelah dokumentasi ini dan script reproduksi
tersedia.
