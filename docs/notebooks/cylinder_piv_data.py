"""Fetch Shang and Tu's cylinder-wake PIV dataset (Zenodo 20765567)."""

import hashlib
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

DATASET_DOI = "https://doi.org/10.5281/zenodo.20765567"
DATASET_URL = "https://zenodo.org/records/20765567/files/cylinder_vel.mat?download=1"
DATASET_MD5 = "4cc876439c48f7970afa477ea17d217a"
DATASET_PATH = Path(__file__).resolve().parent.parent / "datasets/cylinder_piv/cylinder_vel.mat"


def _matches_checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "md5").hexdigest() == DATASET_MD5


def fetch_cylinder_piv():
    """Return the verified local MAT-file, downloading it on a cache miss.

    The first fetch transfers about 1.1 GB. Only a complete file matching the
    published Zenodo checksum is moved into the cache.
    """
    if DATASET_PATH.is_file() and _matches_checksum(DATASET_PATH):
        return DATASET_PATH

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=DATASET_PATH.parent, prefix="cylinder_vel-", suffix=".part", delete=False
    ) as download:
        temporary = Path(download.name)
        try:
            with urlopen(DATASET_URL, timeout=60) as response:
                shutil.copyfileobj(response, download, length=1024 * 1024)
            download.close()
            if not _matches_checksum(temporary):
                raise ValueError(f"Cylinder PIV checksum mismatch. Dataset: {DATASET_DOI}")
            temporary.replace(DATASET_PATH)
        finally:
            temporary.unlink(missing_ok=True)

    return DATASET_PATH


if __name__ == "__main__":
    print(fetch_cylinder_piv())
