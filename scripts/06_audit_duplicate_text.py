"""Audit duplicate overview, semantic_text, tagline, and keywords.

Teks yang identik menghasilkan embedding yang identik, sehingga cosine similarity
menjadi 1,0. Script ini mengukur seberapa besar kondisi tersebut muncul dan
seberapa besar pengaruhnya pada tabel top-K.

Kolom teks dimuat satu per satu, bukan sekaligus, karena kolom teks pada dataset
final berukuran besar. Pendekatan ini menjaga pemakaian RAM tetap rendah.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from movie_recommender.utils.paths import (
    DATA_PROCESSED_DIR,
    RESULTS_DIR,
)

MOVIES_PARQUET = DATA_PROCESSED_DIR / "movies_final.parquet"

DEFAULT_OUTPUT_CSV = RESULTS_DIR / "semantic_duplicate_audit.csv"
DEFAULT_SUMMARY_CSV = RESULTS_DIR / "semantic_duplicate_summary.csv"
DEFAULT_LARGEST_CSV = RESULTS_DIR / "semantic_duplicate_largest_groups.csv"
DEFAULT_LENGTH_CSV = RESULTS_DIR / "semantic_duplicate_length_profile.csv"
DEFAULT_PAIRS_CSV = RESULTS_DIR / "recommendations" / "semantic_all_top10.csv"

TEXT_COLUMNS = ("overview", "semantic_text", "tagline", "keywords")
LENGTH_THRESHOLDS = (30, 50, 80, 100, 150)
WORD_THRESHOLDS = (5, 8, 10, 15)
SAMPLE_MEMBERS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit teks kembar pada dataset final dan ukur pengaruhnya terhadap "
            "tabel top-K."
        )
    )
    parser.add_argument(
        "pairs_csv",
        nargs="?",
        type=Path,
        default=DEFAULT_PAIRS_CSV,
        help="File top-K pasangan. Dipakai untuk mengukur dampak.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path untuk tabel ringkasan per kolom.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="Path untuk ringkasan gabungan.",
    )
    parser.add_argument(
        "--largest",
        type=Path,
        default=DEFAULT_LARGEST_CSV,
        help="Path untuk daftar grup overview terbesar.",
    )
    parser.add_argument(
        "--length",
        type=Path,
        default=DEFAULT_LENGTH_CSV,
        help="Path untuk profil panjang overview.",
    )
    parser.add_argument(
        "--top-groups",
        type=int,
        default=50,
        help="Jumlah grup terbesar yang disimpan. Default 50.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250_000,
        help="Ukuran blok saat membaca file pasangan. Default 250000.",
    )
    parser.add_argument(
        "--skip-impact",
        action="store_true",
        help="Lewati pengukuran dampak ke tabel top-K.",
    )
    return parser.parse_args()


def clean(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def load_column(column: str) -> pd.Series:
    frame = pd.read_parquet(MOVIES_PARQUET, columns=[column])
    return clean(frame[column])


def audit_columns() -> pd.DataFrame:
    """Hitung pengulangan per kolom teks, satu kolom pada satu waktu."""
    rows = []

    for column in TEXT_COLUMNS:
        values = load_column(column)
        filled = values[values.ne("")]

        duplicate_mask = filled.duplicated(keep=False)
        duplicated_values = filled[duplicate_mask]
        group_sizes = duplicated_values.groupby(duplicated_values).size()

        rows.append(
            {
                "kolom": column,
                "kosong": int(values.eq("").sum()),
                "nilai_unik": int(filled.nunique()),
                "baris_dalam_grup_kembar": int(duplicate_mask.sum()),
                "jumlah_grup_kembar": int(group_sizes.size),
                "grup_terbesar": int(group_sizes.max()) if group_sizes.size else 0,
                "film_dengan_teks_kembar": int(
                    filled.map(group_sizes).fillna(0).gt(1).sum()
                ),
            }
        )

        del values, filled, duplicate_mask, duplicated_values, group_sizes

    return pd.DataFrame(rows).set_index("kolom")


def classify_groups() -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_parquet(
        MOVIES_PARQUET,
        columns=["id", "title", "overview", "release_date", "original_language"],
    )

    overview = clean(frame["overview"])
    is_duplicate = overview.duplicated(keep=False) & overview.ne("")

    groups = frame.loc[is_duplicate].assign(_overview=overview[is_duplicate])
    del frame, overview, is_duplicate

    records = []

    for text, group in groups.groupby("_overview"):
        titles = clean(group["title"]).str.casefold().unique()
        members = group.head(SAMPLE_MEMBERS)

        records.append(
            {
                "overview": text,
                "n_films": len(group),
                "n_judul_unik": len(titles),
                "panjang_karakter": len(text),
                "jumlah_kata": len(text.split()),
                "jenis": (
                    "duplikasi entri"
                    if len(titles) == 1
                    else "teks dipakai bersama"
                ),
                "contoh_id": ", ".join(str(value) for value in members["id"]),
                "contoh_judul": " | ".join(
                    str(value) for value in members["title"].tolist()
                ),
            }
        )

    group_table = pd.DataFrame(records).sort_values(
        ["n_films", "panjang_karakter"],
        ascending=[False, True],
    )
    group_table = group_table.reset_index(drop=True)

    summary = {
        "grup overview kembar": len(group_table),
        "grup dengan judul sama semua": int(
            (group_table["n_judul_unik"] == 1).sum()
        ),
        "grup dengan judul berbeda": int((group_table["n_judul_unik"] > 1).sum()),
        "grup berisi 2 film": int((group_table["n_films"] == 2).sum()),
        "grup berisi 10 film atau lebih": int((group_table["n_films"] >= 10).sum()),
        "film dalam grup judul sama": int(
            group_table.loc[group_table["n_judul_unik"] == 1, "n_films"].sum()
        ),
        "film dalam grup judul berbeda": int(
            group_table.loc[group_table["n_judul_unik"] > 1, "n_films"].sum()
        ),
        "baris overview dalam grup kembar": int(group_table["n_films"].sum()),
    }

    return group_table, pd.Series(summary, name="value")


def length_profile() -> pd.DataFrame:
    overview = load_column("overview")
    characters = overview.str.len()
    words = overview.str.split().str.len()

    rows = []

    for threshold in LENGTH_THRESHOLDS:
        rows.append(
            {
                "metrik": "panjang_karakter",
                "ambang": threshold,
                "jumlah_film": int(characters.le(threshold).sum()),
                "persen": round(float(characters.le(threshold).mean() * 100), 2),
            }
        )

    for threshold in WORD_THRESHOLDS:
        rows.append(
            {
                "metrik": "jumlah_kata",
                "ambang": threshold,
                "jumlah_film": int(words.le(threshold).sum()),
                "persen": round(float(words.le(threshold).mean() * 100), 2),
            }
        )

    profile = pd.DataFrame(rows)

    profile.loc[len(profile)] = {
        "metrik": "median_karakter",
        "ambang": np.nan,
        "jumlah_film": int(characters.median()),
        "persen": np.nan,
    }
    profile.loc[len(profile)] = {
        "metrik": "median_kata",
        "ambang": np.nan,
        "jumlah_film": int(words.median()),
        "persen": np.nan,
    }

    return profile


def measure_impact(pairs_csv: Path, chunk_size: int) -> pd.Series:
    frame = pd.read_parquet(
        MOVIES_PARQUET,
        columns=["id", "semantic_text"],
    )
    text_by_id = pd.Series(
        clean(frame["semantic_text"]).to_numpy(),
        index=frame["id"].to_numpy(),
    )
    duplicate_size = text_by_id.value_counts()
    duplicate_size_by_id = text_by_id.map(duplicate_size)
    del frame

    total_rows = 0
    rows_touching_duplicate = 0
    affected_queries: set[int] = set()
    pairs_above_threshold = 0
    identical_text_pairs = 0

    for chunk in pd.read_csv(
        pairs_csv,
        usecols=[
            "query_movie_id",
            "recommended_movie_id",
            "similarity",
        ],
        chunksize=chunk_size,
    ):
        total_rows += len(chunk)

        query_size = chunk["query_movie_id"].map(duplicate_size_by_id).fillna(1)
        recommended_size = chunk["recommended_movie_id"].map(
            duplicate_size_by_id
        ).fillna(1)
        touched = (query_size > 1) | (recommended_size > 1)

        rows_touching_duplicate += int(touched.sum())
        affected_queries.update(
            chunk.loc[touched, "query_movie_id"].unique().tolist()
        )

        similarity = chunk["similarity"].to_numpy()
        pairs_above_threshold += int((similarity > 0.9999).sum())

        query_text = chunk["query_movie_id"].map(text_by_id)
        recommended_text = chunk["recommended_movie_id"].map(text_by_id)
        identical_text_pairs += int((query_text == recommended_text).sum())

    return pd.Series(
        {
            "total_baris": total_rows,
            "baris_melibatkan_film_berteks_kembar": rows_touching_duplicate,
            "persen_baris_terdampak": (
                round(rows_touching_duplicate / total_rows * 100, 2)
                if total_rows
                else 0.0
            ),
            "film_query_terdampak": len(affected_queries),
            "pasangan_similarity_di_atas_0_9999": pairs_above_threshold,
            "pasangan_dengan_teks_identik": identical_text_pairs,
        },
        name="value",
    )


def main() -> None:
    args = parse_args()

    if args.chunk_size < 1:
        raise ValueError("--chunk-size harus bilangan positif")

    print("=== audit per kolom ===", flush=True)
    column_audit = audit_columns()
    print(column_audit.to_string())
    print()

    print("=== klasifikasi grup overview kembar ===", flush=True)
    group_table, group_summary = classify_groups()
    print(group_summary.to_string())
    print()

    print("=== profil panjang overview ===", flush=True)
    profile = length_profile()
    print(profile.to_string(index=False))
    print()

    impact = None

    if args.skip_impact:
        print("pengukuran dampak dilewati (--skip-impact)")
    elif not args.pairs_csv.exists():
        print(f"file pasangan tidak ditemukan: {args.pairs_csv}")
        print("pengukuran dampak dilewati")
    else:
        print(f"mengukur dampak dari: {args.pairs_csv}", flush=True)
        impact = measure_impact(args.pairs_csv, args.chunk_size)
        print("=== dampak pada tabel top-K ===")
        print(impact.to_string())
        print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    column_audit.to_csv(args.output)
    print("tersimpan:", args.output)

    profile.to_csv(args.length, index=False)
    print("tersimpan:", args.length)

    group_table.head(args.top_groups).assign(
        overview=lambda frame: frame["overview"].str.slice(0, 300)
    ).to_csv(args.largest, index=False, encoding="utf-8-sig")
    print("tersimpan:", args.largest)

    summary_parts = [group_summary]

    if impact is not None:
        summary_parts.append(impact)

    summary = pd.concat(summary_parts)
    summary.index.name = "metric"
    summary.to_csv(args.summary, header=["value"])
    print("tersimpan:", args.summary)


if __name__ == "__main__":
    main()
