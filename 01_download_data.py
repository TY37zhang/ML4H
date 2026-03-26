import shutil
import urllib.request
import pandas as pd
from config import *

def main():
    for d in [RAW_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("=== Copying cached files ===")
    for dataset in CACHED_DATASETS:
        for suffix, info in CYCLES.items():
            filename = f"{dataset}_{suffix}.xpt"
            src = NHANES_CACHE / filename
            dst = RAW_DIR / filename
            if dst.exists():
                print(f"  {filename} already exists, skipping")
                continue
            if src.exists():
                shutil.copy2(src, dst)
                print(f"  Copied {filename}")
            else:
                print(f"  WARNING: {filename} not found in cache")

    print("\n=== Downloading additional files ===")
    for dataset in DOWNLOAD_DATASETS:
        for suffix, info in CYCLES.items():
            filename = f"{dataset}_{suffix}.xpt"
            dst = RAW_DIR / filename
            if dst.exists():
                print(f"  {filename} already exists, skipping")
                continue
            url = BASE_URL.format(year=info["year"], filename=f"{dataset}_{suffix}")
            print(f"  Downloading {filename}...", end=" ", flush=True)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    with open(dst, "wb") as f:
                        f.write(resp.read())
                print("OK")
            except Exception as e:
                print(f"FAILED ({e})")

    print("\n=== Verifying all files ===")
    total = 0
    failed = 0
    for xpt_file in sorted(RAW_DIR.glob("*.xpt")):
        try:
            df = pd.read_sas(xpt_file, format="xport", encoding="latin-1")
            print(f"  {xpt_file.name}: {len(df)} rows, {len(df.columns)} cols")
            total += 1
        except Exception as e:
            print(f"  {xpt_file.name}: FAILED ({e})")
            failed += 1

    print(f"\nTotal files: {total}, Failed: {failed}")


if __name__ == "__main__":
    main()
