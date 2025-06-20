# 🔍 AI-Powered Surveillance Image Enhancement System

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Flask](https://img.shields.io/badge/Flask-3.0+-black.svg)](https://flask.palletsprojects.com/)

Sistema avanzato di enhancement per immagini di videosorveglianza basato su AI, progettato specificamente per applicazioni forensi e di polizia giudiziaria. Integra i migliori modelli di deep learning per il miglioramento di volti, targhe e scene generali.

![Demo Interface](docs/images/interface_demo.png)

## 📋 Indice

- [Caratteristiche](#-caratteristiche)
- [Modelli AI Integrati](#-modelli-ai-integrati)
- [Requisiti di Sistema](#-requisiti-di-sistema)
- [Installazione](#-installazione)
- [Uso](#-uso)
- [API Documentation](#-api-documentation)
- [Esempi](#-esempi)
- [Configurazione Avanzata](#-configurazione-avanzata)
- [Note Forensi](#-note-forensi)
- [Troubleshooting](#-troubleshooting)
- [Contributi](#-contributi)
- [Licenza](#-licenza)

## ✨ Caratteristiche

- **🎯 Enhancement Specializzato**: Algoritmi ottimizzati per volti, targhe e scene generali
- **🤖 Multi-Modello**: Integra GFPGAN, Real-ESRGAN, CodeFormer e SwinIR
- **🚀 GPU Accelerato**: Supporto CUDA per elaborazione in tempo reale
- **🌐 Interfaccia Web**: UI intuitiva con drag-and-drop
- **📊 Confronto Side-by-Side**: Visualizzazione immediata dei risultati
- **💾 Export HD**: Download delle immagini migliorate in alta qualità
- **🔒 Privacy**: Elaborazione locale, nessun upload su server esterni
- **📝 Logging Forense**: Tracciabilità completa delle operazioni

## 🤖 Modelli AI Integrati

### GFPGAN v1.3
- **Specialità**: Face restoration veloce e affidabile
- **Velocità**: ~6 secondi per immagine
- **Uso ideale**: Volti frontali e di profilo
- **Paper**: [Towards Real-World Blind Face Restoration with Generative Facial Prior](https://arxiv.org/abs/2101.04061)

### Real-ESRGAN
- **Specialità**: Super-resolution generale 4x
- **Velocità**: ~8 secondi per immagine
- **Uso ideale**: Scene complete, oggetti, veicoli
- **Paper**: [Real-ESRGAN: Training Real-World Blind Super-Resolution](https://arxiv.org/abs/2107.10833)

### CodeFormer (Opzionale)
- **Specialità**: Face restoration di ultima generazione
- **Velocità**: ~10 secondi per immagine
- **Uso ideale**: Volti molto degradati
- **Paper**: [Towards Robust Blind Face Restoration with Codebook Lookup Transformer](https://arxiv.org/abs/2206.11253)

### SwinIR (Opzionale)
- **Specialità**: Transformer-based super-resolution
- **Velocità**: ~15 secondi per immagine
- **Uso ideale**: Massima qualità per targhe e testo
- **Paper**: [SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/abs/2108.10257)

## 💻 Requisiti di Sistema

### Hardware Minimo
- **CPU**: Intel i5 o AMD Ryzen 5
- **RAM**: 8GB
- **GPU**: NVIDIA GTX 1060 6GB (opzionale ma consigliato)
- **Storage**: 10GB liberi

### Hardware Consigliato
- **CPU**: Intel i7 o AMD Ryzen 7
- **RAM**: 16GB o più
- **GPU**: NVIDIA RTX 3060 o superiore
- **Storage**: 20GB liberi SSD

### Software
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 11+
- **Python**: 3.9 - 3.11
- **CUDA**: 11.8 o 12.1 (per GPU NVIDIA)

## 🚀 Installazione

### Metodo 1: Installazione Automatica (Windows)

1. **Clona il repository**
```bash
git clone https://github.com/tuousername/surveillance-enhancement.git
cd surveillance-enhancement
```

2. **Esegui lo script di installazione**
```bash
install_and_run.bat
```

Questo script automaticamente:
- Crea un ambiente virtuale Python
- Installa tutte le dipendenze
- Scarica i modelli AI
- Avvia il server

### Metodo 2: Installazione Manuale

1. **Clona il repository**
```bash
git clone https://github.com/tuousername/surveillance-enhancement.git
cd surveillance-enhancement
```

2. **Crea ambiente virtuale**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

3. **Installa PyTorch**
```bash
# Per GPU NVIDIA (consigliato)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Per CPU only
pip install torch torchvision torchaudio
```

4. **Installa altre dipendenze**
```bash
pip install -r requirements.txt
```

5. **Scarica i modelli**
```bash
python download_models.py
```

### Metodo 3: Docker

```bash
docker build -t surveillance-enhancement .
docker run -p 5000:5000 --gpus all surveillance-enhancement
```

## 📖 Uso

### Avvio del Sistema

1. **Attiva l'ambiente virtuale** (se non già attivo)
```bash
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

2. **Avvia il server**
```bash
python app.py
```

3. **Apri il browser**
Naviga su: `http://localhost:5000`

### Interfaccia Web

![Interface Guide](docs/images/interface_guide.png)

1. **Upload Immagine**: Drag & drop o click per selezionare
2. **Selezione Tipo Enhancement**:
   - `Auto`: Il sistema sceglie automaticamente
   - `Volti`: Ottimizzato per face restoration
   - `Targhe`: Ottimizzato per testo e caratteri
3. **Selezione Modello**:
   - `Migliore Disponibile`: Scelta automatica
   - `GFPGAN`: Veloce per volti
   - `Real-ESRGAN`: Versatile
   - `Combined`: Usa più modelli in sequenza
4. **Elabora**: Avvia il processing
5. **Download**: Scarica il risultato in HD

### Uso da Command Line

```python
from app import WorkingImageEnhancer
from PIL import Image

# Inizializza enhancer
enhancer = WorkingImageEnhancer()

# Processa immagine
enhanced_img = enhancer.process_image(
    'path/to/image.jpg',
    enhancement_type='face',  # 'face', 'plate', 'auto'
    model_choice='gfpgan'     # 'gfpgan', 'realesrgan', 'combined'
)

# Salva risultato
enhanced_img.save('enhanced_output.jpg', quality=95)
```

## 🔌 API Documentation

### Endpoints

#### `GET /`
Ritorna la pagina principale con l'interfaccia web.

#### `GET /status`
Verifica lo stato del sistema e i modelli disponibili.

**Response:**
```json
{
    "device": "cuda",
    "cuda_available": true,
    "models": {
        "gfpgan": true,
        "realesrgan": true,
        "codeformer": false,
        "combined": true
    },
    "models_path": "models/",
    "total_models": 2
}
```

#### `POST /upload`
Carica e processa un'immagine.

**Parameters:**
- `file`: File immagine (multipart/form-data)
- `enhancement_type`: `auto`, `face`, `plate`
- `model_choice`: `best`, `gfpgan`, `realesrgan`, `combined`

**Response:**
```json
{
    "success": true,
    "original": "data:image/jpeg;base64,/9j/4AAQ...",
    "enhanced": "data:image/jpeg;base64,/9j/4BBQ...",
    "enhanced_filename": "enhanced_20240120_123456_image.jpg",
    "original_size": "640x480",
    "enhanced_size": "2560x1920",
    "model_used": "gfpgan"
}
```

#### `GET /download/<filename>`
Scarica l'immagine elaborata.

### Python API

```python
# Esempio di integrazione
import requests

# Upload e processing
with open('suspect_face.jpg', 'rb') as f:
    files = {'file': f}
    data = {
        'enhancement_type': 'face',
        'model_choice': 'gfpgan'
    }
    response = requests.post('http://localhost:5000/upload', 
                           files=files, data=data)
    
result = response.json()
if result['success']:
    # Scarica immagine enhanced
    enhanced_url = f"http://localhost:5000/download/{result['enhanced_filename']}"
```

## 📸 Esempi

### Esempio 1: Enhancement Volto

**Input** (320x240, sfocato):
![Face Input](docs/examples/face_input.jpg)

**Output** (1280x960, dettagli ricostruiti):
![Face Output](docs/examples/face_output.jpg)

### Esempio 2: Enhancement Targa

**Input** (Targa illeggibile):
![Plate Input](docs/examples/plate_input.jpg)

**Output** (Caratteri leggibili):
![Plate Output](docs/examples/plate_output.jpg)

### Esempio 3: Scena Completa

**Input** (Bassa risoluzione):
![Scene Input](docs/examples/scene_input.jpg)

**Output** (4x Super-resolution):
![Scene Output](docs/examples/scene_output.jpg)

## ⚙️ Configurazione Avanzata

### Configurazione GPU

Per ottimizzare l'uso della GPU, modifica `config.yaml`:

```yaml
gpu:
  device_id: 0  # ID della GPU da usare
  memory_fraction: 0.8  # Frazione di memoria GPU da allocare
  allow_growth: true  # Alloca memoria dinamicamente
  
models:
  tile_size: 512  # Dimensione tile per immagini grandi
  batch_size: 1   # Numero immagini da processare insieme
```

### Configurazione Modelli

Personalizza i parametri dei modelli in `app.py`:

```python
# GFPGAN parameters
self.models['gfpgan'] = GFPGANer(
    model_path=model_path,
    upscale=2,          # Fattore di upscaling (1, 2, 4)
    arch='clean',       # Architettura ('clean', 'original')
    bg_upsampler=None,  # Background upsampler
    device=self.device
)

# Real-ESRGAN parameters
self.models['realesrgan'] = RealESRGANer(
    scale=4,            # Fattore di upscaling
    tile=0,             # 0 = no tiling, >0 = tile size
    tile_pad=10,        # Padding per tiles
    pre_pad=0,          # Pre-padding immagine
    half=True,          # Usa FP16 (più veloce, meno memoria)
)
```

### Aggiungere Nuovi Modelli

Per aggiungere un nuovo modello:

1. Installa le dipendenze del modello
2. Aggiungi il setup in `setup_models()`
3. Implementa il metodo `enhance_with_newmodel()`
4. Aggiungi la logica in `process_image()`

Esempio:
```python
def setup_newmodel(self):
    """Setup per nuovo modello"""
    try:
        from newmodel import NewModel
        self.models['newmodel'] = NewModel()
        print("✓ NewModel caricato")
    except ImportError:
        print("NewModel non disponibile")

def enhance_with_newmodel(self, img_path):
    """Enhancement con nuovo modello"""
    img = cv2.imread(img_path)
    enhanced = self.models['newmodel'].enhance(img)
    return enhanced
```

## 🔒 Note Forensi

### Validità Legale

⚠️ **IMPORTANTE**: Le immagini elaborate con AI potrebbero non essere ammissibili come prove in tribunale senza appropriate validazioni. Consultare sempre un esperto legale.

### Best Practices Forensi

1. **Conserva sempre gli originali**: Mai sovrascrivere le immagini originali
2. **Documenta il processo**: Il sistema salva automaticamente:
   - Timestamp di elaborazione
   - Modello utilizzato
   - Parametri applicati
3. **Chain of custody**: Mantieni log di chi ha processato cosa e quando
4. **Validazione**: Confronta sempre con l'originale per verificare che non ci siano artefatti

### Metadata Preservation

Il sistema preserva i metadata EXIF quando possibile:

```python
# Preserva metadata
from PIL import Image
from PIL.ExifTags import TAGS

original = Image.open(original_path)
exif_data = original.getexif()
enhanced.save(output_path, exif=exif_data)
```

### Audit Trail

Tutti i processing sono loggati in `logs/processing.log`:

```
2024-01-20 15:30:45 - INFO - Processing: suspect_face.jpg
2024-01-20 15:30:45 - INFO - Type: face, Model: gfpgan
2024-01-20 15:30:51 - INFO - Success: enhanced_20240120_153045_suspect_face.jpg
2024-01-20 15:30:51 - INFO - Original: 320x240, Enhanced: 640x480
```

## 🔧 Troubleshooting

### Problemi Comuni

#### 1. "CUDA out of memory"
**Soluzione**: Riduci la dimensione delle tiles o usa CPU
```python
# In app.py
tile=256  # Riduci da 512
```

#### 2. "Module not found"
**Soluzione**: Reinstalla le dipendenze
```bash
pip install --upgrade -r requirements.txt
```

#### 3. "Model file not found"
**Soluzione**: Scarica manualmente i modelli
```bash
python download_models.py --force
```

#### 4. Performance lenta
**Soluzioni**:
- Verifica che stai usando GPU: `nvidia-smi`
- Riduci la risoluzione di output
- Usa modelli più leggeri (GFPGAN invece di CodeFormer)

### Diagnostica

Esegui il test di sistema:
```bash
python test_system.py
```

Output atteso:
```
=== SYSTEM CHECK ===
✓ Python 3.9.16
✓ PyTorch 2.0.1+cu118
✓ CUDA Available: NVIDIA GeForce RTX 3060
✓ OpenCV 4.8.0
✓ GFPGAN Available
✓ Real-ESRGAN Available
✓ Models directory exists
✓ Free disk space: 125.3 GB
=== ALL CHECKS PASSED ===
```

### Log Dettagliati

Per debug avanzato, abilita logging verbose:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributi

Contributi sono benvenuti! Per contribuire:

1. Fork il repository
2. Crea un branch (`git checkout -b feature/AmazingFeature`)
3. Commit le modifiche (`git commit -m 'Add AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Apri una Pull Request

### Guidelines

- Segui PEP 8 per il codice Python
- Aggiungi test per nuove features
- Aggiorna la documentazione
- Mantieni retrocompatibilità

### Sviluppo

Setup ambiente di sviluppo:
```bash
pip install -r requirements-dev.txt
pre-commit install
```

Run tests:
```bash
pytest tests/
```

## 📄 Licenza

Questo progetto è distribuito sotto licenza MIT. Vedi [LICENSE](LICENSE) per dettagli.

### Licenze Modelli

- **GFPGAN**: Apache License 2.0
- **Real-ESRGAN**: BSD 3-Clause
- **CodeFormer**: S-Lab License 1.0
- **SwinIR**: Apache License 2.0

## 🙏 Ringraziamenti

- Team [TencentARC](https://github.com/TencentARC) per GFPGAN
- Team [xinntao](https://github.com/xinntao) per Real-ESRGAN
- Team [sczhou](https://github.com/sczhou) per CodeFormer
- Team [JingyunLiang](https://github.com/JingyunLiang) per SwinIR

## 📞 Supporto

- **Issues**: [GitHub Issues](https://github.com/tuousername/surveillance-enhancement/issues)
- **Discussions**: [GitHub Discussions](https://github.com/tuousername/surveillance-enhancement/discussions)
- **Email**: support@example.com

---

<p align="center">
Made with ❤️ for Law Enforcement and Forensic Analysis
</p>