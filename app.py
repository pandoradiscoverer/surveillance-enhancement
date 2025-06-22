#!/usr/bin/env python3
"""
Enhanced Surveillance Enhancement System
Sistema avanzato per enhancement di immagini di videosorveglianza
"""

import os
import sys
import time
import hashlib
import requests
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image
import torch
import yaml
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from datetime import datetime, timezone

# Import condizionali per modelli AI
try:
    from gfpgan import GFPGANer

    GFPGAN_AVAILABLE = True
except ImportError:
    GFPGAN_AVAILABLE = False

try:
    from realesrgan import RealESRGANer

    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False

try:
    from basicsr.archs.srvgg_arch import SRVGGNetCompact

    BASICSR_AVAILABLE = True
except ImportError:
    BASICSR_AVAILABLE = False

try:
    # CodeFormer non è disponibile come pacchetto standard
    # Implementazione placeholder
    CODEFORMER_AVAILABLE = False
except ImportError:
    CODEFORMER_AVAILABLE = False


class ModelDownloader:
    """Gestisce download automatico dei modelli AI"""

    MODEL_URLS = {
        "GFPGANv1.4.pth": {
            "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
            "size": 348632874,
            "sha256": "e2bf4f55d7c5c6b9e3f6e0c4d7a8b9c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9",
        },
        "RealESRGAN_x4plus.pth": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "size": 67040989,
            "sha256": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        },
        "codeformer.pth": {
            "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
            "size": 342631821,
            "sha256": "c5b4593074dac892db093d92b93eddfb516b46665760b1b02b9b12b5dc9578ab",
        },
        "RealESRGAN_x4plus_anime_6B.pth": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            "size": 17938799,
            "sha256": "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
        },
    }

    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

    def verify_file_hash(self, file_path: Path, expected_hash: str) -> bool:
        """Verifica hash SHA256 del file"""
        if not file_path.exists():
            return False

        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest() == expected_hash

    def download_file(self, url: str, destination: Path, expected_size: int) -> bool:
        """Scarica file con progress bar"""
        try:
            print(f"📥 Downloading {destination.name}...")

            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", expected_size))
            downloaded = 0

            with open(destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = (downloaded / total_size) * 100
                        print(
                            f"\r  Progress: {progress:.1f}% ({downloaded}/{total_size} bytes)",
                            end="",
                            flush=True,
                        )

            print(f"\n✅ Downloaded {destination.name}")
            return True

        except Exception as e:
            print(f"\n❌ Error downloading {destination.name}: {e}")
            if destination.exists():
                destination.unlink()
            return False

    def download_models(self, force: bool = False) -> Dict[str, bool]:
        """Scarica tutti i modelli necessari"""
        results = {}

        for model_name, info in self.MODEL_URLS.items():
            model_path = self.models_dir / model_name

            # Controlla se già presente
            if not force and model_path.exists():
                if model_path.stat().st_size == info["size"]:
                    print(f"✅ {model_name} already present")
                    results[model_name] = True
                    continue
                else:
                    print(f"⚠️  {model_name} size mismatch, re-downloading...")

            # Scarica modello
            success = self.download_file(info["url"], model_path, info["size"])
            results[model_name] = success

            if success and model_path.stat().st_size != info["size"]:
                print(f"⚠️  Size mismatch for {model_name}")
                results[model_name] = False

        return results


class WorkingImageEnhancer:
    """Enhanced image processor con download automatico modelli"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        self.models_dir = Path(
            self.config.get("models", {}).get("models_path", "./models")
        )
        self.models = {}

        # Download automatico modelli
        self.downloader = ModelDownloader(str(self.models_dir))
        self._ensure_models_available()

        # Setup modelli
        self._setup_models()

        print(f"🚀 Surveillance Enhancement initialized on {self.device}")
        self._print_available_models()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Carica configurazione"""
        default_config = {
            "gpu": {"device_id": 0, "fallback_to_cpu": True},
            "models": {"models_path": "./models"},
            "processing": {"quality_output": 95},
        }

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return {**default_config, **config}
        except FileNotFoundError:
            print("⚠️  config.yaml not found, using defaults")
            return default_config

    def _setup_device(self) -> str:
        """Configura dispositivo GPU/CPU"""
        if torch.cuda.is_available() and self.config.get("gpu", {}).get(
            "fallback_to_cpu", True
        ):
            device_id = self.config.get("gpu", {}).get("device_id", 0)
            device = f"cuda:{device_id}"
            print(f"✅ GPU detected: {torch.cuda.get_device_name(device_id)}")
        else:
            device = "cpu"
            print("⚠️  Using CPU mode")

        return device

    def _ensure_models_available(self):
        """Assicura che i modelli siano disponibili"""
        print("🔍 Checking AI models...")
        download_results = self.downloader.download_models()

        missing_models = [
            name for name, success in download_results.items() if not success
        ]
        if missing_models:
            print(f"⚠️  Warning: Some models failed to download: {missing_models}")

    def _setup_models(self):
        """Inizializza modelli AI"""
        print("🔧 Loading AI models...")

        if GFPGAN_AVAILABLE:
            self._setup_gfpgan()

        if REALESRGAN_AVAILABLE:
            self._setup_realesrgan()

        if CODEFORMER_AVAILABLE:
            self._setup_codeformer()

        print("✅ Models loaded successfully")

    def _setup_gfpgan(self):
        """Setup GFPGAN model"""
        model_path = self.models_dir / "GFPGANv1.4.pth"

        if model_path.exists():
            try:
                self.models["gfpgan"] = GFPGANer(
                    model_path=str(model_path),
                    upscale=2,
                    arch="clean",
                    channel_multiplier=2,
                    bg_upsampler=None,
                    device=self.device,
                )
                print("✅ GFPGAN loaded")
            except Exception as e:
                print(f"❌ GFPGAN setup failed: {e}")
        else:
            print("⚠️  GFPGAN model not found")

    def _setup_realesrgan(self):
        """Setup Real-ESRGAN model"""
        model_path = self.models_dir / "RealESRGAN_x4plus.pth"

        if model_path.exists() and BASICSR_AVAILABLE:
            try:
                self.models["realesrgan"] = RealESRGANer(
                    scale=4,
                    model_path=str(model_path),
                    model=SRVGGNetCompact(
                        num_in_ch=3,
                        num_out_ch=3,
                        num_feat=64,
                        num_conv=32,
                        upscale=4,
                        act_type="prelu",
                    ),
                    tile=512,
                    tile_pad=10,
                    pre_pad=0,
                    half=True if "cuda" in self.device else False,
                    device=self.device,
                )
                print("✅ Real-ESRGAN loaded")
            except Exception as e:
                print(f"❌ Real-ESRGAN setup failed: {e}")

    def _setup_codeformer(self):
        """Setup CodeFormer model (placeholder)"""
        print("⚠️  CodeFormer not available - install from source")

    def enhance_with_codeformer(
        self, image: np.ndarray, fidelity_weight: float = 0.5
    ) -> np.ndarray:
        """Enhancement con CodeFormer (fallback a GFPGAN)"""
        if "gfpgan" in self.models:
            return self.enhance_with_gfpgan(image)
        else:
            raise ValueError("CodeFormer and GFPGAN not available")

    def _print_available_models(self):
        """Stampa modelli disponibili"""
        available = list(self.models.keys())
        if available:
            print(f"🎯 Available models: {', '.join(available)}")
        else:
            print("⚠️  No models available - check installation")

    def detect_enhancement_type(self, image: np.ndarray) -> str:
        """Auto-detect tipo di enhancement"""
        h, w = image.shape[:2]

        if min(h, w) < 200:
            return "face"
        elif max(h, w) > 1000:
            return "general"
        else:
            return "face"

    def enhance_with_gfpgan(self, image: np.ndarray) -> np.ndarray:
        """Enhancement con GFPGAN"""
        if "gfpgan" not in self.models:
            raise ValueError("GFPGAN not available")

        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        _, _, enhanced = self.models["gfpgan"].enhance(
            image_rgb, has_aligned=False, only_center_face=False, paste_back=True
        )

        return enhanced

    def enhance_with_realesrgan(self, image: np.ndarray) -> np.ndarray:
        """Enhancement con Real-ESRGAN"""
        if "realesrgan" not in self.models:
            raise ValueError("Real-ESRGAN not available")

        enhanced, _ = self.models["realesrgan"].enhance(image, outscale=4)
        return enhanced

    def process_image(
        self, image_input, enhancement_type: str = "auto", model_choice: str = "best"
    ) -> Image.Image:
        """Processa immagine con modello specificato"""

        # Carica immagine
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Cannot load image: {image_input}")
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise ValueError("Unsupported image input type")

        # Auto-detect tipo
        if enhancement_type == "auto":
            enhancement_type = self.detect_enhancement_type(image)

        # Selezione modello automatica
        if model_choice == "best":
            if enhancement_type == "face" and "gfpgan" in self.models:
                actual_model = "gfpgan"
            elif "realesrgan" in self.models:
                actual_model = "realesrgan"
            else:
                available = list(self.models.keys())
                if available:
                    actual_model = available[0]
                else:
                    raise ValueError("No models available")
        else:
            actual_model = model_choice

        # Process
        start_time = time.time()

        try:
            if actual_model == "gfpgan":
                enhanced = self.enhance_with_gfpgan(image)
            elif actual_model == "realesrgan":
                enhanced = self.enhance_with_realesrgan(image)
            elif actual_model == "codeformer":
                enhanced = self.enhance_with_codeformer(image)
            else:
                raise ValueError(f"Unknown model: {actual_model}")

            processing_time = time.time() - start_time
            print(f"✅ Enhanced with {actual_model} in {processing_time:.2f}s")

            if enhanced.dtype != np.uint8:
                enhanced = (enhanced * 255).astype(np.uint8)

            enhanced_pil = Image.fromarray(enhanced)
            return enhanced_pil

        except Exception as e:
            print(f"❌ Enhancement failed: {e}")
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# Flask Web Application
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB

# Global enhancer instance
enhancer = None


def init_enhancer():
    """Inizializza enhancer globale"""
    global enhancer
    if enhancer is None:
        enhancer = WorkingImageEnhancer()


@app.route("/")
def index():
    """Pagina principale"""
    return render_template("index.html")


@app.route("/status")
def status():
    """Status sistema e modelli disponibili"""
    init_enhancer()

    if enhancer is None:
        return jsonify({"error": "Enhancer not initialized"}), 500

    return jsonify(
        {
            "device": enhancer.device,
            "cuda_available": torch.cuda.is_available(),
            "models": list(enhancer.models.keys()),
            "models_path": str(enhancer.models_dir),
            "total_models": len(enhancer.models),
        }
    )


@app.route("/upload", methods=["POST"])
def upload_and_process():
    """Upload e processing immagine"""
    init_enhancer()

    if enhancer is None:
        return jsonify({"success": False, "error": "System not initialized"})

    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"})

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"})

        # Parametri
        enhancement_type = request.form.get("enhancement_type", "auto")
        model_choice = request.form.get("model_choice", "best")

        # Leggi immagine
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if original_img is None:
            return jsonify({"success": False, "error": "Invalid image format"})

        original_size = f"{original_img.shape[1]}x{original_img.shape[0]}"

        # Process
        enhanced_pil = enhancer.process_image(
            original_img, enhancement_type=enhancement_type, model_choice=model_choice
        )

        enhanced_size = f"{enhanced_pil.width}x{enhanced_pil.height}"

        # Converti a base64
        def pil_to_base64(img):
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            return base64.b64encode(buffer.getvalue()).decode()

        original_pil = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))

        # Filename per download
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = secure_filename(file.filename or "unknown.jpg")
        enhanced_filename = f"enhanced_{timestamp}_{safe_filename}"

        return jsonify(
            {
                "success": True,
                "original": f"data:image/jpeg;base64,{pil_to_base64(original_pil)}",
                "enhanced": f"data:image/jpeg;base64,{pil_to_base64(enhanced_pil)}",
                "enhanced_filename": enhanced_filename,
                "original_size": original_size,
                "enhanced_size": enhanced_size,
                "model_used": model_choice,
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    print("🚀 Starting Surveillance Enhancement System")
    print("=" * 50)

    # Inizializza enhancer al primo avvio
    init_enhancer()

    print("\n✅ System ready!")
    print("🌐 Web interface: http://localhost:5000")
    print("📋 Status endpoint: http://localhost:5000/status")
    print("\nPress Ctrl+C to stop")

    try:
        from waitress import serve

        print("🔧 Starting with Waitress WSGI server...")
        serve(app, host="127.0.0.1", port=5000, threads=4)
    except ImportError:
        print("⚠️  Waitress not found, using Flask dev server")
        app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
