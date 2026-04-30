import os
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.03/output/association_by_datasource_direct/"
OUTPUT_DIR = "data/raw/opentargets"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_parquet_links():
    """
    Fetch all parquet file links from Open Targets FTP directory.
    """
    print("Fetching file list from Open Targets...")

    response = requests.get(BASE_URL)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    links = []
    for link in soup.find_all("a"):
        href = link.get("href")

        if href and href.endswith(".parquet"):
            full_url = BASE_URL + href
            links.append(full_url)

    print(f"Found {len(links)} parquet files")
    return links


def download_file(url):
    """
    Download a single parquet file.
    """
    filename = os.path.join(OUTPUT_DIR, url.split("/")[-1])

    if os.path.exists(filename):
        print(f"Skipping (exists): {filename}")
        return

    print(f"Downloading: {filename}")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()

        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def download_all():
    links = fetch_parquet_links()

    for i, url in enumerate(links):
        print(f"\n[{i+1}/{len(links)}]")
        download_file(url)


if __name__ == "__main__":
    download_all()
    print("\nDownload completed.")