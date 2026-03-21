import os
import requests

def download_dataset():
    
    url = f"https://drive.google.com/uc?export=download&id=1VP7TLlgacST2veOpxiDQBFkEJeHheBn3"
    
    save_path = "data/creditcard.csv"
    
    if not os.path.exists('data'):
        os.makedirs('data')
        
    if not os.path.exists(save_path):
        print("Downloading dataset from Google Drive...")
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print("Download complete.")
        else:
            print(f"Failed to download. Status code: {response.status_code}")
    else:
        print("Dataset already exists.")

if __name__ == "__main__":
    download_dataset()