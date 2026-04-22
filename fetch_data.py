import gzip
import os
import urllib.request

from combine_subscores import make_fixed_file

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

URLS = [
    # "https://stringdb-downloads.org/download/protein.links.v12.0/4932.protein.links.v12.0.txt.gz",
    # "https://stringdb-downloads.org/download/protein.links.detailed.v12.0/4932.protein.links.detailed.v12.0.txt.gz",
    # "https://stringdb-downloads.org/download/protein.enrichment.terms.v12.0/4932.protein.enrichment.terms.v12.0.txt.gz",
   "https://stringdb-downloads.org/download/protein.enrichment.terms.v12.0/9606.protein.enrichment.terms.v12.0.txt.gz",
   "https://stringdb-downloads.org/download/protein.links.full.v12.0/9606.protein.links.full.v12.0.txt.gz",
    # "https://stringdb-downloads.org/download/protein.links.full.v12.0/4932.protein.links.full.v12.0.txt.gz"
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

        if "protein.links.full.v12.0.txt" in filename:
            filenum = filename[0:4]
            fixname = filenum + ".protein.links.fixed.v12.0.txt"
            output_path_fixed = os.path.join(DATA_DIR, fixname)
            #a little confusingly, the input to this function is the file we just 
            #decompressed, so the variable is still named 'output path'
            make_fixed_file(output_path, output_path_fixed)



if __name__ == "__main__":
    main()
