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
artifacts/semantic_baseline/semantic_text.parquet
```

### 2. Generate embeddings

```powershell
uv run python scripts/02_generate_sbert_embeddings.py
```

Proses ini meng-encode 396.254 teks dan membutuhkan sekitar 28 menit pada
RTX 3050 4 GB. Output:

```text
artifacts/semantic_baseline/embeddings.npy
artifacts/semantic_baseline/movie_ids.npy
artifacts/semantic_baseline/token_lengths.parquet
artifacts/semantic_baseline/config.json
results/semantic_text_audit.csv
```

### 3. Verifikasi artifact

```powershell
uv run python scripts/03_verify_embeddings.py final
uv run python scripts/03_verify_embeddings.py baseline
```

Tidak melakukan encoding ulang; hanya memeriksa shape, dtype, jumlah ID,
norm, dan alignment.

## Struktur Artifact Semantik

| Folder | Isi | Jumlah baris |
|---|---|---:|
| `artifacts/semantic_baseline/` | Baseline sebelum filter bahasa | 396.254 |
| `artifacts/semantic/` | Dataset final English-only | 388.241 |

`SEMANTIC_BASELINE_ARTIFACTS_DIR` menunjuk ke baseline, sedangkan
`SEMANTIC_ARTIFACTS_DIR` menunjuk ke artifact final. Baseline tidak pernah
ditimpa oleh proses encoding dataset final.

Encoding dataset final dijalankan dari `notebooks/04_semantic_feature.ipynb`
dengan `RUN_ENCODE = True`. Proses ini merupakan inference dengan model
pretrained yang sama, bukan training atau fine-tuning.

## Tabel Top-K Seluruh Film

### 1. Bangun pasangan top-K

```powershell
uv run python scripts/05_build_topk_pairs.py
```

Menghitung cosine similarity untuk seluruh 388.241 film dan menyimpan 10
rekomendasi teratas per film, total 3.882.410 baris. Perhitungan dilakukan per
blok karena matriks skor penuh berukuran sekitar 561 GB.

Opsi yang sering dipakai:

```powershell
uv run python scripts/05_build_topk_pairs.py --k 20 --block-size 128
uv run python scripts/05_build_topk_pairs.py --limit 500 --skip-metadata
uv run python scripts/05_build_topk_pairs.py --force
```

Tanpa `--force`, script berhenti kalau file keluaran sudah ada. Ini mencegah
penulisan ulang yang tidak disengaja.

Output:

```text
results/recommendations/semantic_all_top10_raw.csv
results/recommendations/semantic_all_top10.csv
```

### 2. Audit teks kembar

```powershell
uv run python scripts/06_audit_duplicate_text.py
```

Mengukur teks yang identik antar film dan dampaknya pada tabel top-K. Script
memuat kolom teks satu per satu supaya pemakaian RAM tetap rendah.

Output:

```text
results/semantic_duplicate_audit.csv
results/semantic_duplicate_summary.csv
results/semantic_duplicate_length_profile.csv
results/semantic_duplicate_largest_groups.csv
```

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
