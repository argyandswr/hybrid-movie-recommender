# Dataset Dictionary

Dokumen ini menjelaskan struktur dataset yang digunakan pada penelitian
**Hybrid Metadata Content-Based Movie Recommendation System**.

## 1. Sumber Data

| Item | Nilai |
|---|---|
| File mentah | `data/raw/TMDB  IMDB Movies Dataset.csv` |
| Format | CSV |
| Jumlah baris | 443.610 |
| Jumlah kolom | 29 |
| Granularitas | Satu baris merepresentasikan satu record film |
| Kunci utama data mentah | Belum unik; `id` memiliki duplikat |

Dataset mentah **tidak** memiliki interaksi user-item. Tidak ada rating user,
riwayat tontonan, atau data personalisasi. Karena itu, pendekatan yang digunakan
adalah content-based item-to-item, bukan collaborative filtering.

## 2. Struktur Kolom

Deskripsi di bawah merupakan deskripsi kerja berdasarkan nama kolom dan
inspeksi data, bukan dokumentasi resmi dari penyedia dataset.

| No | Kolom | Tipe | Missing | Deskripsi kerja |
|---:|---|---|---:|---|
| 1 | `id` | int64 | 0 | ID film dari TMDB |
| 2 | `title` | string | 0 | Judul film |
| 3 | `vote_average` | float64 | 0 | Rata-rata rating vote |
| 4 | `vote_count` | int64 | 0 | Jumlah vote |
| 5 | `status` | string | 0 | Status produksi/rilis |
| 6 | `release_date` | string | 25.344 | Tanggal rilis |
| 7 | `revenue` | int64 | 0 | Pendapatan |
| 8 | `runtime` | int64 | 0 | Durasi film dalam menit |
| 9 | `adult` | bool | 0 | Penanda konten dewasa |
| 10 | `backdrop_path` | string | 256.596 | Path gambar backdrop |
| 11 | `budget` | int64 | 0 | Anggaran produksi |
| 12 | `homepage` | string | 387.845 | Situs resmi |
| 13 | `tconst` | string | 0 | ID film dari IMDb |
| 14 | `original_language` | string | 0 | Bahasa asli film |
| 15 | `original_title` | string | 0 | Judul asli |
| 16 | `overview` | string | 44.973 | Sinopsis film |
| 17 | `popularity` | float64 | 0 | Skor popularitas |
| 18 | `poster_path` | string | 79.352 | Path gambar poster |
| 19 | `tagline` | string | 349.602 | Tagline film |
| 20 | `genres` | string | 83.793 | Genre, dipisahkan koma |
| 21 | `production_companies` | string | 180.228 | Perusahaan produksi, dipisahkan koma |
| 22 | `production_countries` | string | 121.839 | Negara produksi, dipisahkan koma |
| 23 | `spoken_languages` | string | 109.749 | Bahasa yang digunakan, dipisahkan koma |
| 24 | `keywords` | string | 271.970 | Kata kunci tema, dipisahkan koma |
| 25 | `directors` | string | 10.003 | Sutradara, dipisahkan koma |
| 26 | `writers` | string | 64.952 | Penulis, dipisahkan koma |
| 27 | `averageRating` | float64 | 0 | Rating dari IMDb |
| 28 | `numVotes` | int64 | 0 | Jumlah vote IMDb |
| 29 | `cast` | string | 69.641 | Aktor, dipisahkan koma |

## 3. Kolom yang Digunakan dalam Penelitian

### 3.1 Fitur teks semantik

| Kolom | Missing | Peran |
|---|---:|---|
| `overview` | 44.973 | Sinopsis film |
| `tagline` | 349.602 | Tagline film |
| `keywords` | 271.970 | Kata kunci tema |

Ketiga kolom digabungkan menjadi satu kolom turunan bernama `semantic_text`
dengan format berlabel:

```text
Overview: {overview}

Tagline: {tagline}

Keywords: {keywords}
```

Field yang kosong dilewati. Jika ketiganya kosong, `semantic_text` bernilai
string kosong dan film tersebut tidak diikutkan dalam encoding SBERT.

### 3.2 Fitur metadata

| Kolom | Missing | Peran |
|---|---:|---|
| `genres` | 83.793 | Genre film |
| `directors` | 10.003 | Sutradara |
| `writers` | 64.952 | Penulis |
| `cast` | 69.641 | Aktor |

Keempat kolom ini belum diproses pada Phase 3. Rencananya digunakan pada
Phase 4 dengan representasi bag-of-words yang mempertahankan peran metadata
melalui prefiks seperti `genre_`, `director_`, dan `cast_`.

## 4. Format Nilai Metadata

Kolom `genres`, `keywords`, `directors`, `writers`, dan `cast` berupa string
yang dipisahkan koma, bukan JSON list.

Contoh:

```text
Action, Science Fiction, Adventure
```

Catatan penting:

- Sebagian nama orang mengandung koma, misalnya gelar atau sufiks.
- Pemecahan sederhana dengan `split(",")` berisiko memecah satu nama menjadi
  dua bagian.
- Penanganan format ini harus dilakukan secara eksplisit pada Phase 4.

## 5. Duplikasi ID

| Metrik | Nilai |
|---|---:|
| Baris dengan ID duplikat | 17.318 |
| Grup ID duplikat | 3.463 |
| ID unik | 429.755 |
| Duplikat baris penuh | 0 |

Baris dengan ID sama tidak selalu identik. Kolom yang paling sering berbeda
di dalam grup duplikat adalah:

| Kolom | Jumlah grup yang berbeda |
|---|---:|
| `cast` | 3.414 |
| `popularity` | 51 |
| `release_date` | 39 |
| `genres` | 37 |
| `overview` | 9 |
| `keywords` | 3 |

Karena itu, deduplikasi tidak cukup dilakukan dengan `keep="first"`. Aturan
yang digunakan:

> Untuk setiap `id`, pertahankan baris dengan jumlah kolom relevan terisi
> terbanyak. Jika seri, pilih teks terpanjang setelah whitespace dibersihkan.
> Jika masih seri, pilih baris yang lebih awal.

Ketujuh kolom relevan diberi bobot yang sama. Aturan ini merupakan heuristic
pemilihan record, bukan ukuran kualitas film.

## 6. Dataset Turunan

| Artefak | Jumlah baris | Keterangan |
|---|---:|---|
| `data/processed/movies.parquet` | 429.755 | Master dataset setelah deduplikasi |
| `artifacts/semantic/semantic_text.parquet` | 429.755 | ID dan `semantic_text` |
| `artifacts/semantic/token_lengths.parquet` | 396.254 | Panjang token untuk teks non-kosong |
| `artifacts/semantic/embeddings.npy` | 396.254 | Embedding SBERT 768 dimensi |
| `artifacts/semantic/movie_ids.npy` | 396.254 | ID film yang sejajar dengan baris embedding |

## 7. Ketersediaan Teks dan Metadata

| Kelompok | Jumlah |
|---|---:|
| Film unik setelah deduplikasi | 429.755 |
| Film dengan `semantic_text` tidak kosong | 396.254 |
| Film tanpa `semantic_text` | 33.501 |
| Film tanpa `semantic_text` tetapi punya metadata | 33.405 |
| Film tanpa `semantic_text` dan tanpa metadata | 96 |

Film tanpa `semantic_text` tetap disimpan di master dataset. Mereka tidak
memperoleh embedding semantik, tetapi masih berpotensi digunakan oleh
metadata pipeline pada Phase 4.

## 8. Keterbatasan Data

- 33.501 film tidak memiliki teks semantik dan tidak dapat direpresentasikan
  oleh pipeline SBERT.
- 96 film tidak memiliki teks semantik maupun metadata yang dapat digunakan.
- Sebagian nilai `original_language` tidak konsisten dengan
  `spoken_languages`.
- Dataset tidak memiliki interaksi user-item, sehingga evaluasi tidak dapat
  menggunakan Collaborative Filtering dan harus memakai proxy relevance.
- Nama orang pada metadata dapat mengandung koma, sehingga parsing metadata
  perlu penanganan khusus.
