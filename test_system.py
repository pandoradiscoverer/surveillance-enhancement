"""
Test completo del sistema
"""

import sys
import os
import platform
import shutil
import subprocess

def print_header(text):
    """Stampa header formattato"""
    print(f"\\n{'=' * 50}")
    print(f"{text:^50}")
    print('=' * 50)

def check_python():
    """Verifica versione Python"""
    version = sys.version_info
    status = "✓" if 3.9 <= version.major + version.minor/10 <= 3.11 else "✗"
    print(f"{status} Python {sys.version.split()[0]}")
    return status == "✓"

def check_cuda():
    """Verifica CUDA/GPU"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA {torch.version.cuda}") # type: ignore
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("ℹ️  CUDA non disponibile (usando CPU)")
        return True
    except ImportError:
        print("✗ PyTorch non installato")
        return False

def check_dependencies():
    """Verifica dipendenze"""
    deps = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'flask': 'Flask',
        'numpy': 'NumPy',
        'gfpgan': 'GFPGAN',
        'realesrgan': 'Real-ESRGAN',
        'basicsr': 'BasicSR'
    }
    
    all_ok = True
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} non installato")
            all_ok = False
    
    return all_ok

def check_models():
    """Verifica presenza modelli"""
    models_dir = 'models'
    expected_models = [
        'GFPGANv1.3.pth',
        'RealESRGAN_x4plus.pth'
    ]
    
    if not os.path.exists(models_dir):
        print(f"✗ Directory '{models_dir}' non trovata")
        return False
    
    all_ok = True
    for model in expected_models:
        path = os.path.join(models_dir, model)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            print(f"✓ {model} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {model} non trovato")
            all_ok = False
    
    return all_ok

def check_disk_space():
    """Verifica spazio disco"""
    stat = shutil.disk_usage(".")
    free_gb = stat.free / 1e9
    status = "✓" if free_gb > 5 else "⚠️"
    print(f"{status} Spazio libero: {free_gb:.1f} GB")
    return free_gb > 1

def run_tests():
    """Esegue tutti i test"""
    print_header("SYSTEM CHECK")
    
    # Informazioni sistema
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor()}")
    print(f"Architecture: {platform.machine()}")
    
    # Test componenti
    print_header("PYTHON")
    python_ok = check_python()
    
    print_header("GPU/CUDA")
    cuda_ok = check_cuda()
    
    print_header("DEPENDENCIES")
    deps_ok = check_dependencies()
    
    print_header("AI MODELS")
    models_ok = check_models()
    
    print_header("SYSTEM RESOURCES")
    disk_ok = check_disk_space()
    
    # Risultato finale
    all_ok = python_ok and cuda_ok and deps_ok and models_ok and disk_ok
    
    print_header("RESULT")
    if all_ok:
        print("✅ ALL CHECKS PASSED")
        print("Il sistema è pronto all'uso!")
    else:
        print("❌ SOME CHECKS FAILED")
        print("Risolvi i problemi sopra indicati")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(run_tests())