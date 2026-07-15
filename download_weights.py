import os
import sys
import requests

WEIGHTS = {
    "ball_track_model.pt": "1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl",
    "court_model.pt": "1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG",
    "bounce_model.cbm": "1Eo5HDnAQE8y_FbOftKZ8pjiojwuy2BmJ"
}

def download_file_from_google_drive(file_id, destination):
    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    
    # Request headers to look like a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    
    response = session.get(url, params={'id': file_id}, headers=headers, stream=True)
    
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
            
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(url, params=params, headers=headers, stream=True)
        
    # Check for HTTP errors
    response.raise_for_status()
    
    # Track progress and write to destination
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1MB
    
    written = 0
    with open(destination, 'wb') as f:
        for data in response.iter_content(block_size):
            f.write(data)
            written += len(data)
            if total_size > 0:
                percent = (written / total_size) * 100
                sys.stdout.write(f"\rİndiriliyor: %{percent:.1f} ({written / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
            else:
                sys.stdout.write(f"\rİndiriliyor: {written / (1024*1024):.1f} MB")
            sys.stdout.flush()
    print("\nTamamlandı.")

def main():
    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    for filename, file_id in WEIGHTS.items():
        dest_path = os.path.join(weights_dir, filename)
        if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 0:
            print(f"'{filename}' zaten mevcut, atlanıyor.")
            continue
            
        print(f"'{filename}' indiriliyor...")
        try:
            download_file_from_google_drive(file_id, dest_path)
        except Exception as e:
            print(f"HATA: '{filename}' indirilirken bir sorun oluştu: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            sys.exit(1)
            
    print("\nTüm model ağırlıkları başarıyla indirildi ve 'weights/' dizinine yerleştirildi.")

if __name__ == '__main__':
    main()
