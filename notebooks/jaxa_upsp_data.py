"""Fetch JAXA's 4.71-degree CRM uPSP measurements for DPW-8 Test Case 3."""

import re
import shutil
from itertools import islice
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

DATASET_SOURCE = "https://cfdws.chofu.jaxa.jp/apc/dpw/upsp.html"
ARTICLE_DOI = "https://doi.org/10.1007/s00348-020-03118-y"
ARCHIVE_NAME = "upsp_AoA471.zip"
ARCHIVE_URL = (
    "https://cfdws.chofu.jaxa.jp/apc/dpw/exp_data_tc3/Unsteady-PSP/" + ARCHIVE_NAME
)
ARCHIVE_SIZE = 566_686_060
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "datasets/jaxa_upsp"
STATISTICS_MEMBER = "AoA471/AoA471_Cp-Cprms_upsp_surface.dat"
SNAPSHOTS_MEMBER = "AoA471/AoA471_time-snapshots_upsp_surface.dat"


def _is_complete(path):
    if not path.is_file() or path.stat().st_size != ARCHIVE_SIZE:
        return False

    try:
        with ZipFile(path) as archive:
            return {STATISTICS_MEMBER, SNAPSHOTS_MEMBER}.issubset(archive.namelist())
    except BadZipFile:
        return False


def fetch_jaxa_upsp(cache_dir=DEFAULT_CACHE):
    """Return the cached JAXA ZIP path, downloading about 567 MB on a cache miss.

    The default cache is local to the course and ignored by Git. The archive
    stays compressed so callers can read just the members they need.
    """
    cache_dir = Path(cache_dir)
    archive = cache_dir / ARCHIVE_NAME
    if _is_complete(archive):
        return archive

    cache_dir.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=cache_dir, suffix=".part", delete=False) as download:
        temporary = Path(download.name)
        try:
            with urlopen(ARCHIVE_URL, timeout=60) as response:
                shutil.copyfileobj(response, download, length=1024 * 1024)
            download.close()
            if not _is_complete(temporary):
                raise ValueError(f"Incomplete JAXA archive. Source: {DATASET_SOURCE}")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    return archive


def load_upper_wing_snapshots(archive_path):
    """Return XYZ coordinates in MAC units, triangles, times in s, and nodal pressure.

    Triangles are zero-based. Pressure has shape (nodes, times) and uses NaN
    for JAXA's masked values. Only the first Tecplot zone is read.
    """
    import numpy as np

    with ZipFile(archive_path) as archive, archive.open(SNAPSHOTS_MEMBER) as stream:
        stream.readline()
        times_ms = np.array(
            re.findall(rb"Cp\(t=\s*(\d+(?:\.\d+)?)ms\)", stream.readline()), dtype=float
        )
        if len(times_ms) < 2 or stream.readline().strip() != b'ZONE T="right wing upper"':
            raise ValueError("Expected the timed upper-wing zone at the start of the JAXA file.")
        nodes = int(stream.readline().split(b"=")[1])
        elements = int(stream.readline().split(b"=")[1])
        if [line.strip() for line in islice(stream, 3)] != [
            b"ZONETYPE=FETRIANGLE", b"DATAPACKING=POINT", b""
        ]:
            raise ValueError("Unexpected Tecplot layout in the JAXA upper-wing zone.")

        xyz = np.empty((nodes, 3))
        pressure = np.empty((nodes, len(times_ms)), dtype=np.float32)
        for start in range(0, nodes, 4096):
            stop = min(start + 4096, nodes)
            data = np.loadtxt(islice(stream, stop - start))
            xyz[start:stop] = data[:, :3]
            pressure[start:stop] = data[:, 3:]
        triangles = np.loadtxt(islice(stream, elements), dtype=np.int32) - 1

    pressure[pressure == -999] = np.nan
    return xyz, triangles, times_ms / 1000, pressure


if __name__ == "__main__":
    print(fetch_jaxa_upsp())
