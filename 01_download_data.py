import shutil
import urllib.request

import pandas as pd

from config import (
    BASE_URL,
    CYCLES,
    FIGURES_DIR,
    MODELS_DIR,
    NHANES_CACHE,
    OPTIONAL_DATASETS,
    PROCESSED_DIR,
    RAW_DIR,
    REQUIRED_DATASETS,
    TABLES_DIR,
)


def copy_from_cache(filename, destination):
    if NHANES_CACHE is None:
        return False

    source = NHANES_CACHE / filename
    if not source.exists():
        return False

    shutil.copy2(source, destination)
    print(f"  Copied {filename} from NHANES_CACHE")
    return True


def download_file(dataset, suffix, year, destination):
    url = BASE_URL.format(year=year, filename=f"{dataset}_{suffix}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        with open(destination, "wb") as file:
            file.write(response.read())


def fetch_dataset(dataset, required):
    failed = []
    for suffix, info in CYCLES.items():
        filename = f"{dataset}_{suffix}.xpt"
        destination = RAW_DIR / filename
        if destination.exists():
            print(f"  {filename} already exists, skipping")
            continue

        if copy_from_cache(filename, destination):
            continue

        print(f"  Downloading {filename}...", end=" ", flush=True)
        try:
            download_file(dataset, suffix, info["year"], destination)
            print("OK")
        except Exception as exc:
            print(f"FAILED ({exc})")
            if required:
                failed.append(filename)
    return failed


def verify_files():
    print("\n=== Verifying raw files ===")
    total = 0
    failed = 0
    for xpt_file in sorted(RAW_DIR.glob("*.xpt")):
        try:
            df = pd.read_sas(xpt_file, format="xport", encoding="latin-1")
            print(f"  {xpt_file.name}: {len(df)} rows, {len(df.columns)} cols")
            total += 1
        except Exception as exc:
            print(f"  {xpt_file.name}: FAILED ({exc})")
            failed += 1
    print(f"\nTotal files: {total}, Failed verification: {failed}")


def main():
    for directory in [RAW_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    print("=== Fetching required NHANES files ===")
    missing_required = []
    for dataset in REQUIRED_DATASETS:
        missing_required.extend(fetch_dataset(dataset, required=True))

    print("\n=== Fetching optional NHANES files ===")
    for dataset in OPTIONAL_DATASETS:
        fetch_dataset(dataset, required=False)

    verify_files()

    if missing_required:
        missing = ", ".join(missing_required)
        raise RuntimeError(f"Missing required NHANES files: {missing}")


if __name__ == "__main__":
    main()
