# Contributing to Surveillance Enhancement System

## 🚨 Importante per Sviluppatori

Questo progetto è destinato all'uso forense di polizia giudiziaria. Ogni contributo deve rispettare:
- **Tracciabilità**: Tutte le modifiche devono essere documentate
- **Integrità**: Preservazione dell'evidenza digitale
- **Affidabilità**: Codice stabile per uso legale

## 🛠️ Setup Ambiente Sviluppo

### Prerequisiti
- Windows 10/11 (ambiente target principale)
- Miniconda/Anaconda
- Git for Windows
- GPU NVIDIA (raccomandato)

### Setup
```powershell
# Clone del repository
git clone https://github.com/pandoradiscoverer/surveillance-enhancement.git
cd surveillance-enhancement

# Ambiente sviluppo
conda env create -f environment.yaml
conda activate surveillance-enhancement

# Hook pre-commit (opzionale)
pip install pre-commit
pre-commit install
```

## 📋 Tipi di Contributi Accettati

### ✅ Benvenuti
- **Bug fixes** per stabilità
- **Ottimizzazioni performance** GPU/CPU
- **Nuovi modelli AI** (ESRGAN, GFPGAN variants)
- **Miglioramenti UI** per operatori
- **Documentazione** tecnica
- **Test automatizzati**

### ❌ Non Accettati
- Modifiche che compromettono l'audit trail
- Features non forensi
- Dipendenze che richiedono licenze commerciali
- Codice che modifica file originali

## 🔄 Workflow Contributi

### 1. Issue First
```powershell
# Cerca issue esistenti o crea nuovo
# Discuti proposta prima di iniziare
```

### 2. Fork e Branch
```powershell
git checkout -b feature/nome-feature
git checkout -b bugfix/nome-bug
git checkout -b model/nome-modello
```

### 3. Sviluppo
```powershell
# Testa sempre con GPU e CPU
python -m pytest tests/
python test_installation.py

# Verifica forensic compliance
python tests/test_forensic_integrity.py
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
# Test base
python test_installation.py

# Test modelli AI
python tests/test_models.py

# Test forensic logging
python tests/test_forensic.py

# Test performance
python tests/benchmark_models.py
```

### Test Coverage
- Minimo 80% coverage per nuovo codice
- Test GPU + CPU fallback
- Test dimensioni immagini varie
- Test formati file supportati

## 📝 Standard Codice

### Python Style
```python
# PEP 8 compliance
# Type hints obbligatori
def process_image(image_path: str, model: str) -> Image.Image:
    """Process image with specified model.
    
    Args:
        image_path: Path to input image
        model: Model name ('gfpgan', 'realesrgan', etc.)
        
    Returns:
        Enhanced PIL Image
        
    Raises:
        ValueError: If model not available
    """
```

### Logging Forense
```python
# Sempre tracciare operazioni critiche
forensic_logger.start_operation(
    original_file=image_path,
    enhancement_type=enhancement_type,
    model_used=model_name,
    parameters=parameters
)
```

### Error Handling
```python
# Gestione errori robusta
try:
    enhanced = model.enhance(image)
    success = True
except Exception as e:
    logger.error(f"Enhancement failed: {e}")
    enhanced = original_image  # Fallback sicuro
    success = False
finally:
    forensic_logger.complete_operation(
        operation_id=op_id,
        success=success
    )
```

## 🎨 Aggiungere Nuovi Modelli AI

### Template Modello
```python
def _setup_new_model(self):
    """Setup nuovo modello AI"""
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
    
    # Preprocessing specifico
    processed_image = preprocess(image)
    
    # Enhancement
    enhanced = self.models['new_model'].enhance(processed_image)
    
    # Postprocessing
    return postprocess(enhanced)
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

## 🔒 Security & Forensic Compliance

### Validazione Input
```python
# Sempre validare file upload
if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
    raise ValueError("Unsupported file format")

if file_size > MAX_FILE_SIZE:
    raise ValueError("File too large")
```

### Hash Verification
```python
# Calcolare hash per ogni file processato
original_hash = compute_sha256(original_file)
enhanced_hash = compute_sha256(enhanced_file)

# Log nel database forensic
forensic_db.record_operation(
    original_hash=original_hash,
    enhanced_hash=enhanced_hash,
    timestamp=datetime.utcnow(),
    operator_id=current_user.id
)
```

## 📊 Performance Guidelines

### Ottimizzazioni GPU
```python
# Gestione memoria GPU efficiente
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Prima dell'enhancement
    
with torch.cuda.amp.autocast():  # Mixed precision
    enhanced = model.enhance(image)
```

### Benchmarking
```python
# Misurare performance per ogni modello
start_time = time.time()
enhanced = model.enhance(image)
processing_time = time.time() - start_time

# Log performance metrics
performance_logger.log_metrics(
    model=model_name,
    image_size=f"{width}x{height}",
    processing_time=processing_time,
    gpu_memory_used=torch.cuda.memory_allocated()
)
```

## 📚 Documentazione

### Aggiornare README
- Nuovi modelli supportati
- Requirements aggiornati
- Benchmark performance
- Troubleshooting specifico

### Code Documentation
```python
class NewEnhancer:
    """Enhanced image processor for forensic applications.
    
    This class implements state-of-the-art AI models for image
    enhancement specifically designed for law enforcement use.
    
    Attributes:
        device: CUDA device or 'cpu'
        models: Dictionary of loaded AI models
        forensic_logger: Logger for audit trail
        
    Example:
        >>> enhancer = NewEnhancer()
        >>> enhanced = enhancer.process_image('evidence.jpg', 'gfpgan')
        >>> enhanced.save('enhanced_evidence.jpg')
    """
```

## 🐛 Reporting Issues

### Bug Report Template
```markdown
**Environment:**
- OS: Windows 11
- GPU: RTX 3060 12GB
- Python: 3.10.x
- CUDA: 11.8

**Steps to Reproduce:**
1. Upload image size 4K
2. Select GFPGAN model
3. Click enhance

**Expected:** Image enhanced successfully
**Actual:** GPU out of memory error

**Files:** Attach log files from ./logs/
```

### Performance Issue
- Include benchmark results
- GPU memory usage
- Processing times
- Image dimensions tested

## 🚀 Release Process

### Version Numbering
- `MAJOR.MINOR.PATCH` (semantic versioning)
- `MAJOR`: Breaking changes
- `MINOR`: New models/features
- `PATCH`: Bug fixes

### Release Checklist
- [ ] All tests passing
- [ ] Performance benchmarks updated
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Models compatibility verified
- [ ] Windows 10/11 testing complete

## 📞 Supporto Sviluppatori

- **Discord**: [Server sviluppatori](https://discord.gg/surveillance-dev)
- **Issues**: [GitHub Issues](https://github.com/pandoradiscoverer/surveillance-enhancement/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pandoradiscoverer/surveillance-enhancement/discussions)

---

**Ricorda**: Ogni contributo deve mantenere l'integrità forensica del sistema.