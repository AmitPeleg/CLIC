import csv
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm

from local_setting import COGVLM_DATA_DIR, TRAIN_DATA_DIR, COGVLM_CSV_FILE

# Define paths
data_dir = TRAIN_DATA_DIR
data_dir.mkdir(parents=True, exist_ok=True)
csv_url_file = TRAIN_DATA_DIR / "CLIC-CogVLM-relabelled-Laion.csv"
download_dir = COGVLM_DATA_DIR
download_dir.mkdir(parents=True, exist_ok=True)


def download_image(row):
    """Download a single image given its URL."""
    image_url = row[0]  # Assuming the URL is in the first column
    image_name = image_url.split("/")[-1]
    image_path = download_dir / image_name

    if image_path.exists():
        return str(image_path)  # Skip if already downloaded

    try:
        response = requests.get(image_url, stream=True, timeout=10)
        if response.status_code == 200:
            with image_path.open('wb') as img_file:
                for chunk in response.iter_content(1024):
                    img_file.write(chunk)
            new_row = [image_path, *row[1:]]  # Keep the rest of the row
            return new_row
    except requests.RequestException:
        print(f"Failed to download {image_url}")
        return None


def download_images_parallel(max_workers=10, max_images=None):
    """Download images in parallel using ThreadPoolExecutor."""
    with csv_url_file.open(newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(filter(None, reader))  # Filter out empty rows
        if max_images is not None:
            # Limit the number of rows to download
            rows = rows[:max_images]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(download_image, rows), total=len(rows), desc="Downloading images"))

    return results


def create_cogvlm_csv(new_rows):
    output_csv = COGVLM_CSV_FILE
    cnt = 0
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in new_rows:
            if row is not None:
                writer.writerow(row)
                cnt += 1
    print(f"CSV file created: {output_csv} with {cnt} rows.")


if __name__ == "__main__":
    new_rows = download_images_parallel(max_workers=10, max_images=None)
    create_cogvlm_csv(new_rows)
