
import os
import sys
import requests
from tqdm import tqdm
import hashlib
import argparse

# Configurazione modelli
MODELS = {
    'gfpgan': {
        'url': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        'filename': 'GFPGANv1.3.pth',
        'size_mb': 348,
        'md5': 'c953a88f2ed3b5e8bd8a5d5bf71a2f8d'
    },
    'realesrgan': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        'filename': 'RealESRGAN_x4plus.pth',
        'size_mb': 64,
        'md5': 'df3966f1dd6beb8297caf0b92c52a0d6'
    },
    'codeformer': {
        'url': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
        'filename': 'CodeFormer.pth',
        'size_mb': 359,
        'md5': '5c8e2f1e5b5a7d1f5f1f5f1f5f1f5f1f'
    },
    'swinir': {
        'url': 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth',
        'filename': 'SwinIR_real_sr_x4.pth',
        'size_mb': 49,
        'md5': '0e7da7da3b3e7d2f5f5f5f5f5f5f5f5f'
    }
}

def download_file(url, filepath, desc="Downloading"):
    """Download file con progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as file:
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)

def verify_md5(filepath, expected_md5):
    """Verifica MD5 del file"""
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest() == expected_md5

def main():
    parser = argparse.ArgumentParser(description='Download AI models')
    parser.add_argument('--models', nargs='+', 
                       choices=list(MODELS.keys()) + ['all'],
                       default=['gfpgan', 'realesrgan'],
                       help='Modelli da scaricare')
    parser.add_argument('--force', action='store_true',
                       help='Forza re-download anche se esistono')
    parser.add_argument('--models-dir', default='models',
                       help='Directory dove salvare i modelli')
    
    args = parser.parse_args()
    
    # Crea directory
    os.makedirs(args.models_dir, exist_ok=True)
    
    # Determina quali modelli scaricare
    if 'all' in args.models:
        models_to_download = list(MODELS.keys())
    else:
        models_to_download = args.models
    
    print(f"Modelli da scaricare: {', '.join(models_to_download)}")
    print(f"Directory: {args.models_dir}")
    print("-" * 50)
    
    # Download modelli
    for model_name in models_to_download:
        model_info = MODELS[model_name]
        filepath = os.path.join(args.models_dir, model_info['filename'])
        
        # Controlla se esiste
        if os.path.exists(filepath) and not args.force:
            print(f"✓ {model_name} già presente")
            continue
        
        # Download
        print(f"\n📥 Download {model_name} ({model_info['size_mb']} MB)...")
        try:
            download_file(model_info['url'], filepath, desc=model_name)
            
            # Verifica MD5 (opzionale)
            # if verify_md5(filepath, model_info['md5']):
            #     print(f"✓ {model_name} verificato")
            # else:
            #     print(f"⚠️  {model_name} MD5 non corrisponde")
            
            print(f"✓ {model_name} scaricato con successo")
            
        except Exception as e:
            print(f"❌ Errore download {model_name}: {e}")
    
    print("\n✅ Download completato!")

if __name__ == "__main__":
    main()