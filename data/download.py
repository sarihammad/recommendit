"""
Downloads and extracts the MovieLens 1M dataset from GroupLens.
"""
import os
import sys
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
EXPECTED_FILES = ["ratings.dat", "movies.dat", "users.dat", "README"]


def download_file(url: str, dest_path: Path) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f, tqdm(
        desc=dest_path.name,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def verify_dataset(data_dir: Path) -> bool:
    missing = [f for f in EXPECTED_FILES if not (data_dir / f).exists()]
    if missing:
        print(f"Missing files: {missing}")
        return False
    print(f"All expected files present in {data_dir}")
    for fname in EXPECTED_FILES:
        fpath = data_dir / fname
        size_mb = fpath.stat().st_size / (1024 * 1024)
        print(f"  {fname}: {size_mb:.2f} MB")
    return True


def download_movielens(output_dir: str = "data") -> Path:
    output_dir = Path(output_dir)
    data_dir = output_dir / "ml-1m"

    if data_dir.exists() and verify_dataset(data_dir):
        print("Dataset already downloaded and verified.")
        return data_dir

    zip_path = output_dir / "ml-1m.zip"
    print(f"Downloading MovieLens 1M from {MOVIELENS_URL} ...")
    download_file(MOVIELENS_URL, zip_path)

    print(f"Extracting to {output_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    zip_path.unlink()
    print("Removed zip archive.")

    if not verify_dataset(data_dir):
        raise RuntimeError("Dataset verification failed after download.")

    return data_dir


if __name__ == "__main__":
    # Allow passing a custom output directory
    out = sys.argv[1] if len(sys.argv) > 1 else "data"
    result = download_movielens(out)
    print(f"\nDataset ready at: {result.resolve()}")
