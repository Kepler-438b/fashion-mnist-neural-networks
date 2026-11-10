"""Download the dataset files from the GitHub release into data/.

Usage:
    python download_data.py
"""
import urllib.error
import urllib.request
from pathlib import Path

# Release that holds the data files. Change the tag if you publish a new release.
BASE_URL = "https://github.com/Kepler-438b/fashion-mnist-neural-networks/releases/download/v1.0.0"
FILES = ["train.csv", "evaluate.csv"]

# Save next to this script, so it works no matter which folder it is run from
DATA_DIR = Path(__file__).resolve().parent / "data"


def download(filename):
    target = DATA_DIR / filename
    if target.exists():
        print(f"{filename} already exists, skipping.")
        return

    url = f"{BASE_URL}/{filename}"
    partial = target.with_suffix(target.suffix + ".part")
    print(f"Downloading {filename} ...")
    try:
        urllib.request.urlretrieve(url, partial)
    except urllib.error.HTTPError as error:
        partial.unlink(missing_ok=True)
        raise SystemExit(f"Could not download {url} (HTTP {error.code}). "
                         "Check that the release and file name exist.") from error
    except BaseException:
        partial.unlink(missing_ok=True)  # never leave a half-downloaded file behind
        raise

    partial.rename(target)
    print(f"Saved {filename} ({target.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    for name in FILES:
        download(name)
    print("All dataset files are ready in data/.")
