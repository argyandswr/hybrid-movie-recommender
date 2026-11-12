# Hybrid Movie Recommender

Content-based movie recommendation menggunakan fitur teks semantik SBERT dan
metadata film. Penelitian ini membandingkan text-only, metadata-only, dan
hybrid dengan parameter `alpha`.

## Status

Phase 1–3 selesai:

- audit dataset
- deduplikasi
- pembentukan `semantic_text`
- audit panjang token
- embedding SBERT
- verifikasi artifact
- sanity check similarity

Phase 4–8 belum diimplementasikan.

## Requirement

- Python 3.11
- uv
- GPU NVIDIA opsional; encoding dapat berjalan di CPU

## Setup

```powershell
uv sync
```

## Dataset

Letakkan dataset mentah di:

```text
data/raw/TMDB  IMDB Movies Dataset.csv
```

Dataset tidak disertakan di repository karena ukurannya besar.

## Reproduksi Phase 3

### 1. Preprocessing

```powershell
uv run python scripts/01_preprocess.py
```

Menghasilkan:

```text
data/processed/movies.parquet
artifacts/semantic/semantic_text.parquet
```

### 2. Generate embeddings

```powershell
uv run python scripts/02_generate_sbert_embeddings.py
```

Proses ini meng-encode 396.254 teks dan membutuhkan sekitar 28 menit pada
RTX 3050 4 GB. Output:

```text
artifacts/semantic/embeddings.npy
artifacts/semantic/movie_ids.npy
artifacts/semantic/token_lengths.parquet
artifacts/semantic/config.json
results/semantic_text_audit.csv
```

### 3. Verifikasi artifact

```powershell
uv run python scripts/03_verify_embeddings.py
```

Tidak melakukan encoding ulang; hanya memeriksa shape, dtype, jumlah ID,
norm, dan alignment.

## Test

```powershell
uv run pytest -q
```

## Struktur

```text
src/movie_recommender/   kode pipeline
scripts/                 entry point reproducible
notebooks/               eksplorasi dan analisis
docs/                    dokumentasi metodologi
artifacts/               hasil generate (tidak semua di-commit)
results/                 tabel dan contoh hasil
data/raw/                dataset mentah (diabaikan Git)
data/processed/          dataset olahan (diabaikan Git)
```

## Dokumentasi

- `docs/dataset_dictionary.md` — struktur dan kualitas dataset
- `docs/research_design.md` — desain penelitian dan keputusan metodologis
- `docs/experiment_log.md` — catatan eksekusi Phase 3

## Keterbatasan

- Model SBERT yang dipakai berfokus pada bahasa Inggris.
- 33.501 film tidak memiliki `semantic_text`.
- Belum ada evaluasi formal Precision/Recall/MAP pada Phase 3.
- Similarity sanity check belum merupakan evaluasi rekomendasi formal.
