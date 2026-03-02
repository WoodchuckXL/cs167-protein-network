import gzip
import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

URLS = [
    "https://stringdb-downloads.org/download/protein.links.v12.0/4932.protein.links.v12.0.txt.gz",
    "https://stringdb-downloads.org/download/protein.links.detailed.v12.0/4932.protein.links.detailed.v12.0.txt.gz",
    "https://stringdb-downloads.org/download/protein.enrichment.terms.v12.0/4932.protein.enrichment.terms.v12.0.txt.gz",
]


def fetch_and_decompress(url, output_path):
    """Download a gzipped file from url, decompress it, and save to output_path."""
    print(f"Downloading {url} ...")
    gz_path = output_path + ".gz"
    urllib.request.urlretrieve(url, gz_path)

    with gzip.open(gz_path, "rb") as f_in, open(output_path, "wb") as f_out:
        f_out.write(f_in.read())

    os.remove(gz_path)
    print(f"Saved to {output_path}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for url in URLS:
        filename = os.path.basename(url).removesuffix(".gz")
        output_path = os.path.join(DATA_DIR, filename)

        if os.path.exists(output_path):
            print(f"Already exists: {filename}")
        else:
            fetch_and_decompress(url, output_path)


if __name__ == "__main__":
    main()
