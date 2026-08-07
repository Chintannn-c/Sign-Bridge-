import urllib.request
import json
import os
import zipfile

os.makedirs('src/dataset/Mendeley_ISL/extracted', exist_ok=True)
zip_path = 'src/dataset/Mendeley_ISL/ISL_Dataset.zip'

url = "https://data.mendeley.com/public-files/datasets/98mzk82wbb/files/8f10ee4e-9d2a-4a6c-9a4f-56bb547d2568/file_downloaded"
req = urllib.request.Request(
    url, 
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
)

try:
    with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
        out_file.write(response.read())
    
    if zipfile.is_zipfile(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('src/dataset/Mendeley_ISL/extracted')
        print("Successfully extracted Mendeley ISL dataset zip archive!")
    else:
        print("Zip file processed.")
except Exception as e:
    print(f"Extraction result: {e}")
