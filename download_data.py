"""Download + verify the Robot Identification Benchmark archives.

Downloads the two data packs from the DOI file base into data/ at the
repo root and verifies each by size and MD5 (values from DATA.md, which
were checked against the official .md5 files on 2026-10-24). Files that
already exist and pass verification are skipped, not re-downloaded.

This script does NOT extract the archives (RAR v4 needs an external
tool) and does NOT download Videos.rar (~9.6 GB, no script needs it).
After it runs, extract the raw-data archive (e.g. `unrar x` or `unar`)
and move the 12 recording_*.mat files into data/raw_data/ at the repo
root, then follow the pipeline order in README.md.

Standard library only.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://fdm-fallback.uni-kl.de/TUK/FB/MV/WSKL/0001/"

FILES = [
    # (filename, expected size in bytes, expected MD5)
    ("Robot_Identification_Benchmark_With_Raw_Data.rar",
     227_384_738, "4a9d3adbbca1c9b68577d6db63d58c58"),
    ("Robot_Identification_Benchmark_Without_Raw_Data.rar",
     12_717_003, "6fffa993f60d65f85100bad053888df2"),
]

CHUNK = 1 << 20  # 1 MiB


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verified(path, size, md5):
    if not path.exists():
        return False
    if path.stat().st_size != size:
        return False
    return md5_of(path) == md5


def download(url, dest, size):
    req = urllib.request.Request(url, headers={"User-Agent": "kr300-identification-download/1.0"})
    downloaded = 0
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if downloaded % (10 * CHUNK) < CHUNK:
                print(f"  {dest.name}: {downloaded / 1e6:.0f} / {size / 1e6:.0f} MB", flush=True)
    print(f"  {dest.name}: downloaded {downloaded} bytes")


def main():
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    ok = True
    for name, size, md5 in FILES:
        dest = data_dir / name
        if verified(dest, size, md5):
            print(f"{name}: already present and verified, skipping")
            continue
        if dest.exists():
            print(f"{name}: present but failed verification, re-downloading")
            dest.unlink()
        print(f"{name}: downloading from {BASE_URL}{name}")
        try:
            download(BASE_URL + name, dest, size)
        except Exception as exc:  # noqa: BLE001 - report and continue to next file
            print(f"{name}: download FAILED: {exc}", file=sys.stderr)
            ok = False
            continue
        if verified(dest, size, md5):
            print(f"{name}: verified OK (size + MD5)")
        else:
            print(f"{name}: verification FAILED after download", file=sys.stderr)
            ok = False
    print()
    if ok:
        print("Next steps:")
        print("  1. Extract Robot_Identification_Benchmark_With_Raw_Data.rar")
        print("     (RAR v4: `unrar x` or `unar`).")
        print("  2. Move the 12 recording_*.mat files into data/raw_data/.")
        print("  3. Follow the pipeline order in README.md.")
    else:
        print("Some files failed. Re-run this script to retry.", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
