import zipfile
import os

# Set your paths
ZIP_PATH = 'data/raw/march-machine-learning-mania-2026.zip'
EXTRACT_PATH = 'data/raw/'

def unzip_data(zip_path=ZIP_PATH, extract_path=EXTRACT_PATH):
    """Extract Kaggle competition data to our raw data directory."""
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Zip file not found at {zip_path}. "
            "Run: kaggle competitions download -c march-machine-learning-mania-2026 --path data/raw"
        )

    os.makedirs(extract_path, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    files = os.listdir(extract_path)
    print(f"Extracted {len(files)} files to {extract_path}")
    print("Sample files:", files[:5])

if __name__ == "__main__":
    unzip_data()