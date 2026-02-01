"""
Model management module for Polyglot.

Handles model downloading, storage, and path management.
All models are stored in the project's `models/` directory.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()
MODELS_DIR = PROJECT_ROOT / "models"

# Model subdirectories
HF_CACHE_DIR = MODELS_DIR / "huggingface"
TTS_CACHE_DIR = MODELS_DIR / "tts"
FISH_CACHE_DIR = MODELS_DIR / "fish_speech"

# Model identifiers
class ModelType(Enum):
    ASR = "asr"
    TRANSLATION = "translation"
    TTS = "tts"
    TTS_FISH = "tts_fish"


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    model_id: str
    model_type: ModelType
    size_gb: float
    description: str
    downloaded: bool = False
    download_progress: float = 0.0


# Model registry
MODELS: Dict[str, ModelInfo] = {
    "asr": ModelInfo(
        name="Qwen3-ASR-1.7B",
        model_id="Qwen/Qwen3-ASR-1.7B",
        model_type=ModelType.ASR,
        size_gb=4.4,
        description="Speech recognition & language detection (52 languages)"
    ),
    "translation": ModelInfo(
        name="NLLB-200-600M",
        model_id="facebook/nllb-200-distilled-600M",
        model_type=ModelType.TRANSLATION,
        size_gb=2.3,
        description="Translation model (200 languages)"
    ),
    "tts": ModelInfo(
        name="XTTS-v2",
        model_id="tts_models/multilingual/multi-dataset/xtts_v2",
        model_type=ModelType.TTS,
        size_gb=1.9,
        description="Text-to-speech with voice cloning (17 languages)"
    ),
    "tts_fish": ModelInfo(
        name="Fish Speech 1.5",
        model_id="fishaudio/fish-speech-1.5",
        model_type=ModelType.TTS_FISH,
        size_gb=2.0,
        description="State-of-the-art TTS with emotion control (13+ languages incl. Thai)"
    ),
}


def setup_environment():
    """
    Set up environment variables for model caching.
    Must be called BEFORE importing transformers/TTS libraries.
    """
    # Create directories
    MODELS_DIR.mkdir(exist_ok=True)
    HF_CACHE_DIR.mkdir(exist_ok=True)
    TTS_CACHE_DIR.mkdir(exist_ok=True)
    FISH_CACHE_DIR.mkdir(exist_ok=True)
    
    # Set HuggingFace cache directory
    # Point directly to HF_CACHE_DIR so models load from models/huggingface/
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR)
    
    # Set TTS cache directory (Coqui TTS uses this)
    os.environ["COQUI_TOS_AGREED"] = "1"
    
    # For macOS, TTS uses different path
    if sys.platform == "darwin":
        # Override the default ~/Library/Application Support/tts
        os.environ["TTS_HOME"] = str(TTS_CACHE_DIR)
    else:
        os.environ["TTS_HOME"] = str(TTS_CACHE_DIR)


def get_hf_cache_dir() -> Path:
    """Get the HuggingFace cache directory."""
    return HF_CACHE_DIR / "hub"


def get_tts_cache_dir() -> Path:
    """Get the TTS cache directory."""
    return TTS_CACHE_DIR


def get_fish_cache_dir() -> Path:
    """Get the Fish Speech cache directory."""
    return FISH_CACHE_DIR


def is_model_downloaded(model_key: str) -> bool:
    """Check if a model is already downloaded."""
    if model_key not in MODELS:
        return False
    
    model = MODELS[model_key]
    
    if model.model_type == ModelType.ASR:
        # Check for Qwen ASR model - check both possible locations
        model_dir = HF_CACHE_DIR / f"models--{model.model_id.replace('/', '--')}"
        if not model_dir.exists():
            model_dir = HF_CACHE_DIR / "hub" / f"models--{model.model_id.replace('/', '--')}"
        return model_dir.exists() and any(model_dir.glob("**/*.safetensors"))
    
    elif model.model_type == ModelType.TRANSLATION:
        # Check for NLLB model - check both possible locations
        model_dir = HF_CACHE_DIR / f"models--{model.model_id.replace('/', '--')}"
        if not model_dir.exists():
            model_dir = HF_CACHE_DIR / "hub" / f"models--{model.model_id.replace('/', '--')}"
        return model_dir.exists() and any(model_dir.glob("**/*.safetensors"))
    
    elif model.model_type == ModelType.TTS:
        # Check for XTTS model
        model_dir = TTS_CACHE_DIR / "tts_models--multilingual--multi-dataset--xtts_v2"
        return model_dir.exists() and (model_dir / "model.pth").exists()
    
    elif model.model_type == ModelType.TTS_FISH:
        # Check for Fish Speech model - we download to a named directory
        model_dir = FISH_CACHE_DIR / model.model_id.replace("/", "--")
        return model_dir.exists() and (model_dir / "model.pth").exists()
    
    return False


def get_model_size_on_disk(model_key: str) -> float:
    """Get the actual size of a downloaded model in GB."""
    if model_key not in MODELS:
        return 0.0
    
    model = MODELS[model_key]
    
    if model.model_type in [ModelType.ASR, ModelType.TRANSLATION]:
        # Check both possible locations
        model_dir = HF_CACHE_DIR / f"models--{model.model_id.replace('/', '--')}"
        if not model_dir.exists():
            model_dir = HF_CACHE_DIR / "hub" / f"models--{model.model_id.replace('/', '--')}"
    elif model.model_type == ModelType.TTS:
        model_dir = TTS_CACHE_DIR / "tts_models--multilingual--multi-dataset--xtts_v2"
    elif model.model_type == ModelType.TTS_FISH:
        model_dir = FISH_CACHE_DIR / model.model_id.replace("/", "--")
    else:
        return 0.0
    
    if not model_dir.exists():
        return 0.0
    
    total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    return total_size / (1024 ** 3)  # Convert to GB


def get_all_models_status() -> Dict[str, ModelInfo]:
    """Get status of all models."""
    result = {}
    for key, model in MODELS.items():
        model.downloaded = is_model_downloaded(key)
        result[key] = model
    return result


def download_model(
    model_key: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    verbose: bool = False
) -> bool:
    """
    Download a specific model.
    
    Args:
        model_key: Key from MODELS dict ('asr', 'translation', 'tts')
        progress_callback: Callback function(downloaded_bytes, total_bytes)
        verbose: Enable verbose logging
    
    Returns:
        True if successful
    """
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}")
    
    model = MODELS[model_key]
    
    def log(msg: str):
        if verbose:
            print(f"[ModelManager] {msg}")
    
    log(f"Downloading {model.name}...")
    
    try:
        if model.model_type == ModelType.ASR:
            return _download_asr_model(model, progress_callback, verbose)
        elif model.model_type == ModelType.TRANSLATION:
            return _download_translation_model(model, progress_callback, verbose)
        elif model.model_type == ModelType.TTS:
            return _download_tts_model(model, progress_callback, verbose)
        elif model.model_type == ModelType.TTS_FISH:
            return _download_fish_speech_model(model, progress_callback, verbose)
    except Exception as e:
        log(f"Error downloading {model.name}: {e}")
        raise
    
    return False


def _download_asr_model(model: ModelInfo, progress_callback: Optional[Callable], verbose: bool) -> bool:
    """Download the ASR model."""
    from huggingface_hub import snapshot_download
    
    if verbose:
        print(f"[ModelManager] Downloading {model.model_id} from HuggingFace...")
    
    # Download with progress tracking
    snapshot_download(
        repo_id=model.model_id,
        cache_dir=str(HF_CACHE_DIR / "hub"),
        local_dir_use_symlinks=False,
    )
    
    if verbose:
        print(f"[ModelManager] ✓ {model.name} downloaded successfully")
    return True


def _download_translation_model(model: ModelInfo, progress_callback: Optional[Callable], verbose: bool) -> bool:
    """Download the translation model."""
    from huggingface_hub import snapshot_download
    
    if verbose:
        print(f"[ModelManager] Downloading {model.model_id} from HuggingFace...")
    
    snapshot_download(
        repo_id=model.model_id,
        cache_dir=str(HF_CACHE_DIR / "hub"),
        local_dir_use_symlinks=False,
    )
    
    if verbose:
        print(f"[ModelManager] ✓ {model.name} downloaded successfully")
    return True


def _download_tts_model(model: ModelInfo, progress_callback: Optional[Callable], verbose: bool) -> bool:
    """Download the TTS model using the TTS library."""
    if verbose:
        print("[ModelManager] Downloading XTTS-v2 model...")
        print("[ModelManager] This requires the TTS virtualenv (.venv-tts)")
    
    import subprocess
    
    # Use the TTS venv to download the model
    tts_python = PROJECT_ROOT / ".venv-tts" / "bin" / "python"
    
    if not tts_python.exists():
        raise RuntimeError(
            "TTS virtualenv not found. Please run:\n"
            "python3.10 -m venv .venv-tts && "
            ".venv-tts/bin/pip install TTS torch"
        )
    
    # Create a download script
    download_script = f'''
import os
os.environ["TTS_HOME"] = "{TTS_CACHE_DIR}"
os.environ["COQUI_TOS_AGREED"] = "1"

from TTS.api import TTS
print("Downloading XTTS-v2 model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
print("Download complete!")
'''
    
    result = subprocess.run(
        [str(tts_python), "-c", download_script],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"TTS download failed: {result.stderr}")
    
    if verbose:
        print(f"[ModelManager] ✓ {model.name} downloaded successfully")
    return True


def _download_fish_speech_model(model: ModelInfo, progress_callback: Optional[Callable], verbose: bool) -> bool:
    """Download the Fish Speech model."""
    from huggingface_hub import snapshot_download
    
    if verbose:
        print(f"[ModelManager] Downloading {model.model_id} from HuggingFace...")
        print("[ModelManager] This may take a while (~2GB)...")
    
    # Download to a named directory inside FISH_CACHE_DIR
    local_dir = FISH_CACHE_DIR / model.model_id.replace("/", "--")
    
    snapshot_download(
        repo_id=model.model_id,
        local_dir=str(local_dir),
    )
    
    if verbose:
        print(f"[ModelManager] ✓ {model.name} downloaded successfully")
    return True


def delete_model(model_key: str, verbose: bool = False) -> bool:
    """Delete a downloaded model to free disk space."""
    if model_key not in MODELS:
        return False
    
    model = MODELS[model_key]
    
    if model.model_type in [ModelType.ASR, ModelType.TRANSLATION]:
        # Check both possible locations
        model_dir = HF_CACHE_DIR / f"models--{model.model_id.replace('/', '--')}"
        if not model_dir.exists():
            model_dir = HF_CACHE_DIR / "hub" / f"models--{model.model_id.replace('/', '--')}"
    elif model.model_type == ModelType.TTS:
        model_dir = TTS_CACHE_DIR / "tts_models--multilingual--multi-dataset--xtts_v2"
    elif model.model_type == ModelType.TTS_FISH:
        model_dir = FISH_CACHE_DIR / model.model_id.replace("/", "--")
    else:
        return False
    
    if model_dir.exists():
        if verbose:
            print(f"[ModelManager] Deleting {model_dir}")
        shutil.rmtree(model_dir)
        return True
    
    return False


def clean_old_cache_locations(verbose: bool = False) -> Dict[str, bool]:
    """
    Remove models from old default cache locations.
    
    Returns dict of location -> was_cleaned
    """
    results = {}
    
    old_locations = [
        # HuggingFace default locations
        Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-ASR-1.7B",
        Path.home() / ".cache" / "huggingface" / "hub" / "models--facebook--nllb-200-distilled-600M",
        # TTS default locations
        Path.home() / ".local" / "share" / "tts",
        Path.home() / "Library" / "Application Support" / "tts",
    ]
    
    for location in old_locations:
        if location.exists():
            if verbose:
                print(f"[ModelManager] Cleaning old cache: {location}")
            try:
                shutil.rmtree(location)
                results[str(location)] = True
            except Exception as e:
                if verbose:
                    print(f"[ModelManager] Failed to clean {location}: {e}")
                results[str(location)] = False
        else:
            results[str(location)] = None  # Didn't exist
    
    return results


def get_total_models_size() -> float:
    """Get total size of all downloaded models in GB."""
    total = 0.0
    for key in MODELS:
        total += get_model_size_on_disk(key)
    return total


# Initialize environment when module is imported
setup_environment()
