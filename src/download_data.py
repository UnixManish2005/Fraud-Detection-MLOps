import os
import gdown

def download_dataset():
    file_id = "1VP7TLlgacST2veOpxiDQBFkEJeHheBn3"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "data/creditcard.csv"

    os.makedirs("data", exist_ok=True)

    if not os.path.exists(output):
        print("Downloading dataset...")
        gdown.download(url, output, quiet=False)
        print("Download complete.")
    else:
        print("Dataset already exists.")

if __name__ == "__main__":
    download_dataset()