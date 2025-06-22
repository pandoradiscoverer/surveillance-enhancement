# Surveillance Enhancement System
**Sistema AI per Enhancement di Immagini di Videosorveglianza - Windows**

Sistema professionale per il miglioramento di volti e targhe da immagini CCTV con modelli GFPGAN, Real-ESRGAN, CodeFormer.

## 🚀 Installazione Automatica (Raccomandato)

### Setup One-Click
1. Scarica il progetto: [Download ZIP](https://github.com/pandoradiscoverer/surveillance-enhancement/archive/main.zip)
2. Estrai in una cartella (es. `C:\surveillance-enhancement`)
3. **Doppio click** su `install_and_run.bat`

Lo script automaticamente:
- ✅ Installa Miniconda (se mancante)
- ✅ Installa Git (se mancante) 
- ✅ Configura ambiente Conda
- ✅ Scarica modelli AI (3.5GB)
- ✅ Avvia sistema con browser

**URL Accesso:** http://localhost:5000

## 🔧 Installazione Manuale

### Prerequisiti
- **Windows 10/11** (64-bit)
- **8GB RAM** (16GB raccomandati)
- **15GB storage** liberi

### Setup Manuale
```powershell
# 1. Installa Miniconda
# Download: https://docs.conda.io/en/latest/miniconda.html

# 2. Clone repository
git clone https://github.com/pandoradiscoverer/surveillance-enhancement.git
cd surveillance-enhancement

# 3. Crea ambiente
conda env create -f environment.yaml
conda activate surveillance-enhancement

# 4. Avvia
python app.py
```

## 🎯 Modelli AI Supportati

| Modello | Velocità | Specialità | Uso Ideale |
|---------|----------|------------|------------|
| **GFPGAN** | 6s | Face restoration | Volti da telecamere CCTV |
| **CodeFormer** | 10s | Face enhancement | Volti molto degradati |
| **Real-ESRGAN** | 8s | Super-resolution | Targhe automobilistiche |

## 💻 Requisiti Hardware

### GPU NVIDIA (Raccomandato)
- **RTX 4090**: ~3s per immagine 1080p
- **RTX 3080**: ~5s per immagine 1080p  
- **RTX 3060**: ~8s per immagine 1080p
- **GTX 1060**: ~15s per immagine 1080p

### CPU Mode
- **Intel i7/AMD Ryzen 7**: ~30s per immagine
- **Intel i5/AMD Ryzen 5**: ~45s per immagine

## 🔧 Configurazione

### GPU Settings
File `config.yaml`:
```yaml
gpu:
  memory_fraction: 0.8  # 80% VRAM
  device_id: 0          # Prima GPU
  fallback_to_cpu: true
```

### Server Settings
```yaml
server:
  host: "127.0.0.1"
  port: 5000
  max_file_size_mb: 100
```

## 🎨 Casi d'Uso

### Miglioramento Volti CCTV
- Enhancement volti per identificazione
- Risoluzione immagini telecamere sicurezza
- Miglioramento contrasto e dettagli

### Lettura Targhe
- Super-resolution targhe sfocate
- Enhancement caratteri per OCR
- Miglioramento visibilità notturna

## 🚨 Troubleshooting

### GPU Non Rilevata
```powershell
# Verifica driver
nvidia-smi

# Reinstalla CUDA
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Memoria Insufficiente
```yaml
# config.yaml - riduci utilizzo
gpu:
  memory_fraction: 0.6
models:
  tile_size: 256
```

### Errori Antivirus
- Aggiungi cartella alle esclusioni
- Escludi `python.exe` da scansione real-time

## 📁 Struttura Progetto
```
surveillance-enhancement/
├── install_and_run.bat     # Setup automatico
├── app.py                  # Server principale  
├── environment.yaml        # Dipendenze Conda
├── config.yaml            # Configurazione
├── templates/
│   └── index.html         # Web interface
├── models/                # Modelli AI (auto-download)
└── outputs/               # Immagini elaborate
```

## 🔄 Avvio Rapido

### Prima Volta
```batch
install_and_run.bat
```

### Avvii Successivi
```batch
REM Opzione 1: Script automatico
install_and_run.bat

REM Opzione 2: Manuale
conda activate surveillance-enhancement
python app.py
```

## 🔄 Aggiornamenti

```powershell
git pull origin main
conda env update -f environment.yaml --prune
```

## 📞 Supporto

- **Issues**: [GitHub Issues](https://github.com/pandoradiscoverer/surveillance-enhancement/issues)
- **Documentation**: Repository GitHub

---

**🎯 SISTEMA PROFESSIONALE PER ENHANCEMENT VIDEOSORVEGLIANZA**