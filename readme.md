# Surveillance Enhancement System
**Sistema AI per Enhancement di Immagini di Videosorveglianza - Windows**

Sistema professionale per il miglioramento di volti e targhe da immagini CCTV con modelli GFPGAN, Real-ESRGAN, CodeFormer.

## 🚨 REQUISITI WINDOWS

### Software Richiesto
- **Windows 10/11** (64-bit)
- **Miniconda/Anaconda** - [Download](https://docs.conda.io/en/latest/miniconda.html)
- **Git for Windows** - [Download](https://git-scm.com/download/win)
- **NVIDIA Drivers** (opzionale per GPU) - [Download](https://www.nvidia.com/drivers)

### Hardware Minimo
- CPU: Intel i5-8400 / AMD Ryzen 5 2600
- RAM: 8GB
- Storage: 15GB liberi
- GPU: NVIDIA GTX 1060 6GB (opzionale)

### Hardware Raccomandato
- CPU: Intel i7-10700K / AMD Ryzen 7 3700X
- RAM: 16GB+
- Storage: 25GB SSD
- GPU: NVIDIA RTX 3060+ (12GB VRAM)

## 🚀 Installazione Windows

### 1. Installa Anaconda/Miniconda
```powershell
# Scarica e installa Miniconda per Windows
# https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
# Riavvia PowerShell dopo installazione
```

### 2. Clona Repository
```powershell
git clone https://github.com/pandoradiscoverer/surveillance-enhancement.git
cd surveillance-enhancement
```

### 3. Setup Ambiente Conda
```powershell
# Crea ambiente da environment.yaml
conda env create -f environment.yaml

# Attiva ambiente
conda activate surveillance-enhancement
```

### 4. Avvia Sistema
```powershell
python app.py
```

**URL Accesso:** http://localhost:5000

## 🛠️ Setup Automatico Windows

Script PowerShell per installazione completa:

```powershell
# Esegui come amministratore
python setup_conda_environment.py
```

Lo script automaticamente:
- ✅ Verifica prerequisiti Windows
- ✅ Configura ambiente Conda
- ✅ Scarica modelli AI (3.5GB)
- ✅ Testa GPU NVIDIA
- ✅ Configura server Waitress
- ✅ Crea script avvio Windows

## 🌐 Server di Produzione

Il sistema usa **Waitress WSGI** per prestazioni ottimali su Windows:
- Server multi-threaded
- Gestione memoria ottimizzata
- Resilienza a crash
- Performance superiori al dev server Flask

## 🎯 Modelli AI Supportati

| Modello | Velocità | Specialità | Uso Ideale |
|---------|----------|------------|------------|
| **GFPGAN** | 6s | Face restoration | Volti da telecamere CCTV |
| **CodeFormer** | 10s | Face enhancement | Volti molto degradati |
| **Real-ESRGAN** | 8s | Super-resolution | Targhe automobilistiche |

## 🔧 Configurazione Windows

### GPU NVIDIA
Verifica CUDA:
```powershell
conda activate surveillance-enhancement
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Ottimizzazione Memoria
File `config.yaml`:
```yaml
gpu:
  memory_fraction: 0.8  # Usa 80% VRAM
  device_id: 0          # Prima GPU
models:
  tile_size: 512        # Riduci se errori memoria
```

## 📁 Struttura Directory Windows
```
surveillance-enhancement/
├── app.py                    # Server principale
├── environment.yaml          # Dipendenze Conda
├── config.yaml              # Configurazione
├── templates/
│   └── index.html           # Web interface
├── models/                  # Modelli AI (auto-download)
├── logs/                    # Log applicazione
└── outputs/                 # Immagini elaborate
```

## 🚨 Troubleshooting Windows

### GPU Non Rilevata
```powershell
# Verifica driver NVIDIA
nvidia-smi

# Reinstalla PyTorch con CUDA
conda activate surveillance-enhancement
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Errore Memoria
```yaml
# config.yaml
gpu:
  memory_fraction: 0.6
models:
  tile_size: 256
```

### Modelli Non Scaricati
```powershell
python app.py --download-models
```

### Antivirus Interferenza
- Aggiungi cartella progetto alle esclusioni
- Escludi `python.exe` dell'ambiente Conda

## 🎨 Casi d'Uso

### Miglioramento Volti CCTV
- Risoluzione immagini da telecamere di sicurezza
- Enhancement di volti per identificazione
- Miglioramento qualità per analisi

### Lettura Targhe
- Super-resolution per targhe sfocate
- Miglioramento contrasto caratteri
- Enhancement per sistemi OCR

## 🖥️ Avvio Automatico Windows

### Script Batch
```batch
@echo off
cd /d "C:\surveillance-enhancement"
conda activate surveillance-enhancement
python app.py
pause
```

### Servizio Windows (Opzionale)
```powershell
# Installa NSSM (Non-Sucking Service Manager)
# Crea servizio Windows per avvio automatico
```

## 📊 Performance Windows

### RTX 4090
- GFPGAN: ~3s per immagine 1080p
- CodeFormer: ~5s per immagine 1080p
- Real-ESRGAN: ~4s per immagine 1080p

### RTX 3060
- GFPGAN: ~8s per immagine 1080p
- CodeFormer: ~12s per immagine 1080p
- Real-ESRGAN: ~10s per immagine 1080p

## 🔄 Aggiornamenti

```powershell
conda activate surveillance-enhancement
git pull origin main
conda env update -f environment.yaml --prune
```

## 🆘 Supporto Windows

**Prerequisiti mancanti:**
- Installa Visual Studio C++ Redistributable
- Installa .NET Framework 4.8+

**Errori comuni:**
- Path troppo lungo: Abilita path lunghi Windows
- Permessi: Esegui PowerShell come amministratore
- Firewall: Consenti Python attraverso Windows Firewall

## 📞 Contatti

- **Issues**: [GitHub Issues](https://github.com/pandoradiscoverer/surveillance-enhancement/issues)
- **Supporto Tecnico**: Documentazione GitHub
- **Aggiornamenti**: Seguire repository per nuove release

---

**🎯 SISTEMA PROFESSIONALE PER ENHANCEMENT VIDEOSORVEGLIANZA**
*Ottimizzato per Windows e telecamere di sicurezza*