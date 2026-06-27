#!/usr/bin/env python3
"""Download official VoxCeleb1-B protocol files."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "voxceleb1b"
FILES = {
    "list_test_bilingual.txt": "https://mm.kaist.ac.kr/projects/voxceleb1-b/list_test_bilingual.txt",
    "vox1_lang_label.csv": "https://mm.kaist.ac.kr/projects/voxceleb1-b/vox1_lang_label.csv",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in FILES.items():
        destination = OUTPUT_DIR / filename
        if destination.exists() and destination.stat().st_size:
            print(f"Already exists: {destination}")
            continue
        print(f"Downloading {url} -> {destination}")
        urlretrieve(url, destination)
    print(f"VoxCeleb1-B protocol files are in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

