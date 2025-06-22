# Contributing to Surveillance Enhancement System

## 🛠️ Setup Ambiente Sviluppo

### Prerequisiti
- Windows 10/11
- Miniconda/Anaconda
- Git for Windows
- GPU NVIDIA (raccomandato)

### Setup
```powershell
git clone https://github.com/pandoradiscoverer/surveillance-enhancement.git
cd surveillance-enhancement
conda env create -f environment.yaml
conda activate surveillance-enhancement
```

## 📋 Tipi di Contributi Accettati

### ✅ Benvenuti
- **Bug fixes** per stabilità
- **Ottimizzazioni performance** GPU/CPU
- **Nuovi modelli AI** per volti e targhe
- **Miglioramenti UI** per operatori CCTV
- **Documentazione** tecnica
- **Test automatizzati**

### ❌ Non Accettati
- Features non correlate a videosorveglianza
- Dipendenze con licenze commerciali
- Codice che modifica file originali

## 🔄 Workflow Contributi

### 1. Issue First
Discuti proposta prima di iniziare sviluppo

### 2. Fork e Branch
```powershell
git checkout -b feature/nome-feature
git checkout -b bugfix/nome-bug
git checkout -b model/nome-modello
```

### 3. Sviluppo
```powershell
# Test GPU e CPU
python -m pytest tests/
python test_installation.py
```

### 4. Pull Request
- Titolo descrittivo
- Link alla issue correlata
- Screenshot per UI changes
- Performance benchmarks
- Test su Windows 10/11

## 🧪 Testing Requirements

### Test Obbligatori
```powershell
python test_installation.py
python tests/test_models.py
python tests/benchmark_models.py
```

### Coverage
- Minimo 80% coverage per nuovo codice
- Test GPU + CPU fallback
- Test formati file supportati

## 📝 Standard Codice

### Python Style
```python
# PEP 8 compliance + type hints
def process_image(image_path: str, model: str) -> Image.Image:
    """Process CCTV image with specified AI model.
    
    Args:
        image_path: Path to surveillance image
        model: Model name ('gfpgan', 'realesrgan', etc.)
        
    Returns:
        Enhanced PIL Image
    """
```

### Error Handling
```python
try:
    enhanced = model.enhance(image)
    success = True
except Exception as e:
    logger.error(f"Enhancement failed: {e}")
    enhanced = original_image  # Fallback sicuro
    success = False
```

## 🎨 Aggiungere Nuovi Modelli AI

### Template Modello
```python
def _setup_new_model(self):
    """Setup nuovo modello per enhancement"""
    model_path = self.models_dir / "new_model.pth"
    
    if model_path.exists():
        try:
            self.models['new_model'] = NewModel(
                model_path=str(model_path),
                device=self.device
            )
            print("✅ NewModel loaded")
        except Exception as e:
            print(f"❌ NewModel setup failed: {e}")

def enhance_with_new_model(self, image: np.ndarray) -> np.ndarray:
    """Enhancement con nuovo modello"""
    if 'new_model' not in self.models:
        raise ValueError("NewModel not available")
    
    return self.models['new_model'].enhance(image)
```

### Aggiornare ModelDownloader
```python
MODEL_URLS = {
    "new_model.pth": {
        "url": "https://github.com/author/model/releases/download/v1.0/model.pth",
        "size": 12345678,
        "sha256": "hash_del_file"
    }
}
```

## 📊 Performance Guidelines

### Ottimizzazioni GPU
```python
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
with torch.cuda.amp.autocast():
    enhanced = model.enhance(image)
```

### Benchmarking
```python
start_time = time.time()
enhanced = model.enhance(image)
processing_time = time.time() - start_time

performance_logger.log_metrics(
    model=model_name,
    image_size=f"{width}x{height}",
    processing_time=processing_time
)
```

## 🐛 Reporting Issues

### Bug Report Template
```markdown
**Environment:**
- OS: Windows 11
- GPU: RTX 3060 12GB
- Python: 3.10.x
- CUDA: 11.8

**Steps:** 
1. Upload CCTV image 4K
2. Select GFPGAN model
3. Click enhance

**Expected:** Enhanced successfully
**Actual:** GPU out of memory

**Files:** Attach log files
```

## 🚀 Release Process

### Version Numbering
- `MAJOR.MINOR.PATCH` (semantic versioning)
- Focus su stabilità per uso professionale

### Release Checklist
- [ ] Tests passing
- [ ] Performance benchmarks updated
- [ ] Windows compatibility verified
- [ ] Documentation updated

## 📞 Supporto

- **Issues**: [GitHub Issues](https://github.com/pandoradiscoverer/surveillance-enhancement/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pandoradiscoverer/surveillance-enhancement/discussions)

---

**Focus**: Enhancement di volti e targhe da telecamere di sicurezza