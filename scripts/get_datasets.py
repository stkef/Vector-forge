"""Download ann-benchmarks datasets (HDF5 with train / test / neighbors).

    python scripts/get_datasets.py sift        # 1M x 128, L2      (~500 MB)
    python scripts/get_datasets.py glove       # 1.2M x 100, cosine (~460 MB)
"""
import sys
import urllib.request
from pathlib import Path

URLS = {
    "sift": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
    "glove": "http://ann-benchmarks.com/glove-100-angular.hdf5",
}

def main() -> None:
    names = sys.argv[1:] or ["sift"]
    out = Path("datasets")
    out.mkdir(exist_ok=True)
    for name in names:
        url = URLS[name]
        dest = out / url.rsplit("/", 1)[1]
        if dest.exists():
            print(f"{dest} already there")
            continue
        print(f"downloading {url}")

        def hook(blocks, size, total):
            done = blocks * size
            if total > 0:
                print(f"\r  {done / total:6.1%}  {done / 1e6:,.0f} MB", end="", flush=True)

        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp, hook)
        tmp.rename(dest)
        print(f"\n  saved {dest}")

if __name__ == "__main__":
    main()
