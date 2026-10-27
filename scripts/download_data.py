"""Download the Online Shoppers Purchasing Intention Dataset into data/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modelpulse import config  # noqa: E402
from modelpulse.data import download_dataset, file_sha256  # noqa: E402

if __name__ == "__main__":
    path = download_dataset(config.DATA_FILE, overwrite="--force" in sys.argv)
    print(f"Saved {path} (sha256 {file_sha256(path)[:16]}...)")
