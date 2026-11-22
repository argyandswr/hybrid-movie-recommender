"""Build top-K semantic recommendation pairs for every movie.

The full score matrix is 388241 x 388241 and cannot be materialised, so the
computation is streamed per block: score one block of query movies against all
movies, keep the K best neighbours, then move to the next block.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from movie_recommender.utils.paths import (
    DATA_PROCESSED_DIR,
    RESULTS_DIR,
    SEMANTIC_ARTIFACTS_DIR,
)

DEFAULT_OUTPUT_DIR = RESULTS_DIR / "recommendations"
RAW_FILENAME = "semantic_all_top10_raw.csv"
FINAL_FILENAME = "semantic_all_top10.csv"

MOVIE_COLUMNS = [
    "id",
    "title",
    "genres",
    "directors",
    "vote_average",
    "release_date",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hitung top-K cosine similarity untuk setiap film pada dataset final. "
            "Embedding dibaca dari artifacts/semantic/."
        )
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Jumlah tetangga terdekat per film. Default 10.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help=(
            "Jumlah film query per blok. Turunkan bila VRAM atau RAM terbatas. "
            "Default 256."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Batasi jumlah film query. Berguna untuk uji cepat.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=None,
        help="Device komputasi. Default memakai CUDA bila tersedia.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Direktori keluaran. Default results/recommendations.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=40,
        help="Cetak progres setiap N blok. Default 40.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Timpa file keluaran yang sudah ada.",
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Hanya tulis file mentah tanpa judul dan metadata.",
    )
    return parser.parse_args()


def resolve_device(requested: str | None) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diminta tetapi tidak tersedia di mesin ini.")
    if requested is not None:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_inputs(limit: int | None) -> tuple[np.ndarray, np.ndarray, int]:
    embeddings = np.load(SEMANTIC_ARTIFACTS_DIR / "embeddings.npy")
    movie_ids = np.load(SEMANTIC_ARTIFACTS_DIR / "movie_ids.npy")

    if embeddings.ndim != 2:
        raise ValueError("embeddings.npy harus berupa matriks dua dimensi")

    if embeddings.shape[0] != len(movie_ids):
        raise ValueError(
            "Jumlah baris embedding tidak sama dengan jumlah movie_ids: "
            f"{embeddings.shape[0]} vs {len(movie_ids)}"
        )

    if len(np.unique(movie_ids)) != len(movie_ids):
        raise ValueError("movie_ids mengandung duplikat")

    if limit is None:
        n_queries = len(movie_ids)
    else:
        if limit < 1:
            raise ValueError("--limit harus bilangan positif")
        n_queries = min(limit, len(movie_ids))

    return embeddings, movie_ids, n_queries


def build_pairs(
    embeddings: np.ndarray,
    movie_ids: np.ndarray,
    n_queries: int,
    k: int,
    block_size: int,
    device: str,
    output_path: Path,
    progress_every: int,
) -> int:
    matrix = torch.from_numpy(embeddings).to(device)
    total_movies = matrix.shape[0]

    start_time = time.perf_counter()
    header_written = False
    rows_written = 0
    blocks = 0

    for start in range(0, n_queries, block_size):
        stop = min(start + block_size, n_queries)
        block = matrix[start:stop]

        scores = block @ matrix.T

        # Keluarkan film query dari kandidat, supaya rank 1 bukan dirinya sendiri.
        local_rows = torch.arange(stop - start, device=device)
        global_cols = torch.arange(start, stop, device=device)
        scores[local_rows, global_cols] = float("-inf")

        top_scores, top_index = torch.topk(scores, k, dim=1)

        frame = pd.DataFrame(
            {
                "query_movie_id": np.repeat(movie_ids[start:stop], k),
                "rank": np.tile(np.arange(1, k + 1), stop - start),
                "recommended_movie_id": movie_ids[
                    top_index.cpu().numpy().ravel()
                ],
                "similarity": top_scores.cpu().numpy().ravel().astype(np.float32),
            }
        )
        frame.to_csv(
            output_path,
            mode="w" if not header_written else "a",
            header=not header_written,
            index=False,
            float_format="%.6f",
        )
        header_written = True
        rows_written += len(frame)
        blocks += 1

        del scores, top_scores, top_index, frame

        if progress_every > 0 and blocks % progress_every == 0:
            elapsed = time.perf_counter() - start_time
            print(
                f"{stop}/{n_queries} film | {elapsed:.0f} detik | "
                f"{rows_written} baris",
                flush=True,
            )

    if device == "cuda":
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - start_time
    print(f"selesai menghitung: {elapsed:.1f} detik")
    print(f"film query diproses: {n_queries} dari {total_movies}")

    return rows_written


def attach_metadata(raw_path: Path, output_path: Path) -> pd.DataFrame:
    movies = pd.read_parquet(
        DATA_PROCESSED_DIR / "movies_final.parquet",
        columns=MOVIE_COLUMNS,
    ).set_index("id")

    if not movies.index.is_unique:
        raise ValueError("movies_final.parquet mengandung ID duplikat")

    pairs = pd.read_csv(raw_path)

    title = movies["title"]
    genres = movies["genres"]
    directors = movies["directors"]
    vote_average = movies["vote_average"]
    release_date = movies["release_date"]

    result = pairs.copy()
    result["query_title"] = result["query_movie_id"].map(title)
    result["query_genres"] = result["query_movie_id"].map(genres)
    result["recommended_title"] = result["recommended_movie_id"].map(title)
    result["recommended_release_date"] = result["recommended_movie_id"].map(
        release_date
    )
    result["recommended_genres"] = result["recommended_movie_id"].map(genres)
    result["recommended_directors"] = result["recommended_movie_id"].map(
        directors
    )
    result["recommended_vote_average"] = result["recommended_movie_id"].map(
        vote_average
    )

    result = result[
        [
            "query_movie_id",
            "rank",
            "recommended_movie_id",
            "similarity",
            "query_title",
            "query_genres",
            "recommended_title",
            "recommended_release_date",
            "recommended_genres",
            "recommended_directors",
            "recommended_vote_average",
        ]
    ]

    missing = int(result["query_title"].isna().sum())
    if missing:
        print(f"peringatan: {missing} baris tanpa judul query")

    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def verify(
    raw_path: Path,
    final_path: Path | None,
    n_queries: int,
    k: int,
) -> None:
    expected = n_queries * k
    columns = ["query_movie_id", "rank", "recommended_movie_id"]

    total = 0
    queries: set[int] = set()
    ranks: set[int] = set()
    self_hits = 0

    for chunk in pd.read_csv(raw_path, usecols=columns, chunksize=500_000):
        total += len(chunk)
        queries.update(chunk["query_movie_id"].unique().tolist())
        ranks.update(chunk["rank"].unique().tolist())
        self_hits += int(
            (chunk["query_movie_id"] == chunk["recommended_movie_id"]).sum()
        )

    print()
    print("verifikasi:")
    print(f"  baris mentah        : {total}")
    print(f"  baris diharapkan    : {expected}")
    print(f"  film query unik     : {len(queries)}")
    print(f"  rank lengkap 1..{k}  : {sorted(ranks) == list(range(1, k + 1))}")
    print(f"  self recommendation : {self_hits}")

    if total != expected:
        raise ValueError(
            f"Jumlah baris tidak sesuai: {total} vs {expected}"
        )

    if self_hits != 0:
        raise ValueError(
            f"Ada {self_hits} baris yang merekomendasikan film itu sendiri"
        )

    if final_path is not None and final_path.exists():
        merged_rows = sum(
            1 for _ in open(final_path, encoding="utf-8-sig")
        ) - 1
        print(f"  baris gabungan      : {merged_rows}")

        if merged_rows != expected:
            raise ValueError(
                f"Jumlah baris gabungan tidak sesuai: {merged_rows} vs {expected}"
            )


def main() -> None:
    args = parse_args()

    if args.k < 1:
        raise ValueError("--k harus bilangan positif")

    if args.block_size < 1:
        raise ValueError("--block-size harus bilangan positif")

    raw_path = args.output_dir / RAW_FILENAME
    final_path = args.output_dir / FINAL_FILENAME

    existing = [path for path in (raw_path, final_path) if path.exists()]
    if existing and not args.force:
        print("file keluaran sudah ada:")
        for path in existing:
            size_mb = path.stat().st_size / 1024**2
            print(f"  {path} ({size_mb:.1f} MB)")
        print()
        print("gunakan --force untuk membangun ulang")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    embeddings, movie_ids, n_queries = load_inputs(args.limit)

    print(f"device           : {device}")
    print(f"film total       : {len(movie_ids)}")
    print(f"film diproses    : {n_queries}")
    print(f"k                : {args.k}")
    print(f"block size       : {args.block_size}")
    print(f"output mentah    : {raw_path}")
    print()

    if raw_path.exists():
        raw_path.unlink()

    rows = build_pairs(
        embeddings=embeddings,
        movie_ids=movie_ids,
        n_queries=n_queries,
        k=args.k,
        block_size=args.block_size,
        device=device,
        output_path=raw_path,
        progress_every=args.progress_every,
    )

    print(f"baris mentah ditulis: {rows}")
    print(f"ukuran mentah MB    : {raw_path.stat().st_size / 1024**2:.1f}")

    merged_path: Path | None = None

    if not args.skip_metadata:
        print()
        print("menggabungkan metadata...")
        attach_metadata(raw_path, final_path)
        merged_path = final_path
        print(f"baris gabungan      : {sum(1 for _ in open(final_path, encoding='utf-8-sig')) - 1}")
        print(f"ukuran gabungan MB  : {final_path.stat().st_size / 1024**2:.1f}")

    verify(raw_path, merged_path, n_queries, args.k)
    print()
    print("selesai")


if __name__ == "__main__":
    main()
