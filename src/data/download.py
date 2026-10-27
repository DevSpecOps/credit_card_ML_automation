import sys
import urllib.request
from pathlib import Path

MIRRORS = [
    "https://raw.githubusercontent.com/selva86/datasets/master/UCI_Credit_Card.csv",
    "https://raw.githubusercontent.com/dennislwm/UCI_Credit_Card/master/UCI_Credit_Card.csv",
]


def download(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in MIRRORS:
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"downloaded {url} -> {dest}")
            return
        except Exception as exc:
            print(f"mirror failed: {url}: {exc}")
    sys.exit("all mirrors failed")


def main() -> None:
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/UCI_Credit_Card.csv")
    download(dest)


if __name__ == "__main__":
    main()
