"""Download the Hill of Towie open dataset from Zenodo into data/raw.

The file list and checksums come from the Zenodo API for the chosen record; no file
names or URLs are hard-coded. Existing files with a matching checksum are skipped.

Examples:
    python scripts/download_data.py --list
    python scripts/download_data.py --years 2022 2023 --dry-run
    python scripts/download_data.py            # metadata + all year zips (~15 GB)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# TODO: team to confirm dataset version (docs/DECISIONS.md); keep in sync with protocol.yaml.
DEFAULT_RECORD = 22662930  # Hill of Towie v2.1.0 (concept DOI 10.5281/zenodo.14870021)
API = "https://zenodo.org/api/records/{record}"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
YEAR_ZIP = re.compile(r"^(\d{4})\.zip$")
# Large optional files not needed for the project tables.
EXCLUDE_BY_DEFAULT = {"lidar_data.zip", "turbine_fastlog.zip"}
CHUNK = 1024 * 1024


def fetch_record(record: int) -> dict:
    url = API.format(record=record)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return json.load(resp)
    except (urllib.error.URLError, TimeoutError) as exc:
        sys.exit(
            f"ERROR: cannot reach Zenodo API ({url}): {exc}\n"
            "TODO: if the network is blocked, download the files manually from the Zenodo "
            "record page into data/raw/ and re-run to verify checksums."
        )


def hash_file(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def checksum_ok(path: Path, checksum: str | None) -> bool:
    if not checksum or ":" not in checksum:
        return True  # API gave no checksum; cannot verify
    algo, expected = checksum.split(":", 1)
    return hash_file(path, algo) == expected


def select_files(files: list[dict], years: list[int] | None, include_large: bool,
                 metadata_only: bool) -> list[dict]:
    chosen = []
    for f in files:
        key = f["key"]
        m = YEAR_ZIP.match(key)
        if m:
            if metadata_only or (years and int(m.group(1)) not in years):
                continue
        elif key in EXCLUDE_BY_DEFAULT and not include_large:
            continue
        chosen.append(f)
    return sorted(chosen, key=lambda f: f["key"])


def download(url: str, dest: Path, size: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as out:
        done = 0
        while chunk := resp.read(CHUNK):
            out.write(chunk)
            done += len(chunk)
            print(f"\r  {done / 1e6:,.0f} / {size / 1e6:,.0f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--record", type=int, default=DEFAULT_RECORD,
                   help=f"Zenodo record id (default {DEFAULT_RECORD})")
    p.add_argument("--out", type=Path, default=RAW_DIR, help="output directory (default data/raw)")
    p.add_argument("--years", type=int, nargs="+", help="only these year zips (e.g. 2022 2023)")
    p.add_argument("--metadata-only", action="store_true", help="skip all year zips")
    p.add_argument("--include-large", action="store_true",
                   help="also fetch lidar_data.zip and turbine_fastlog.zip (~14 GB)")
    p.add_argument("--list", action="store_true", help="list files and sizes, then exit")
    p.add_argument("--dry-run", action="store_true", help="show what would be downloaded")
    args = p.parse_args()

    rec = fetch_record(args.record)
    meta = rec["metadata"]
    print(f"Dataset : {meta['title']}")
    print(f"Version : {meta.get('version')}   Record: {rec['id']}")
    print(f"DOI     : {rec['doi']}   License: {meta.get('license', {}).get('id')}")

    files = select_files(rec["files"], args.years, args.include_large, args.metadata_only)
    total = sum(f["size"] for f in files)
    print(f"Selected: {len(files)} files, {total / 1e9:.2f} GB")
    if args.list or args.dry_run:
        for f in files:
            print(f"  {f['key']:50s} {f['size'] / 1e6:10,.1f} MB  {f.get('checksum', '')}")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = args.out / f["key"]
        present = dest.exists() and dest.stat().st_size == f["size"]
        if present and checksum_ok(dest, f.get("checksum")):
            print(f"skip   {f['key']} (present, checksum ok)")
            continue
        print(f"get    {f['key']}")
        download(f["links"]["self"], dest, f["size"])
        if not checksum_ok(dest, f.get("checksum")):
            dest.unlink()
            sys.exit(f"ERROR: checksum mismatch for {f['key']}; file removed. Re-run to retry.")
    print(f"Done. Version {meta.get('version')}, DOI {rec['doi']}")


if __name__ == "__main__":
    main()
