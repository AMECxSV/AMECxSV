#!/usr/bin/env python3
"""Download TidyVoiceX_ASV using Mozilla Data Collective credentials."""

from __future__ import annotations

import os
from pathlib import Path

import requests
from datacollective import DataCollective


DATASET_ID = "cmihtsewu023so207xot1iqqw"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "tidyvoice" / "TidyVoiceX_ASV"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    if load_dotenv:
        load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.environ.get("MDC_API_KEY")
    if not api_key:
        raise SystemExit(
            "MDC_API_KEY is not set. Create a Mozilla Data Collective API key, then run:\n"
            "  MDC_API_KEY='your_key' python data_prep/download_tidyvoice_asv.py"
        )

    output_dir = Path(os.environ.get("MDC_DOWNLOAD_PATH", DEFAULT_OUTPUT_DIR)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MDC_API_URL", "https://mozilladatacollective.com/api")
    client = DataCollective(api_key=api_key, download_path=str(output_dir))
    response = requests.post(
        f"{client.api_url}datasets/{DATASET_ID}/download",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    if response.status_code == 403:
        raise SystemExit(
            "Dataset terms are not accepted yet. Open the dataset page and accept the terms first."
        )
    response.raise_for_status()

    download_info = response.json()
    download_url = download_info["downloadUrl"]
    filename = download_info["filename"]
    destination = output_dir / filename

    if destination.exists() and destination.stat().st_size:
        print(f"Archive already exists: {destination}")
        return

    print(f"Downloading {filename} to {destination}")
    with requests.get(download_url, stream=True, timeout=120) as download_response:
        download_response.raise_for_status()
        with destination.open("wb") as archive:
            for chunk in download_response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    archive.write(chunk)

    print(f"Downloaded TidyVoiceX_ASV archive to {destination}")


if __name__ == "__main__":
    main()
