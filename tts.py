"""
Text-to-Speech module using Coqui XTTS-v2.

Supports 17 languages with voice cloning from a 6-second audio clip.
Uses a separate virtualenv (.venv-tts) to avoid dependency conflicts.
"""

import os
import subprocess
import tempfile
import json
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import numpy as np
from rich.console import Console

console = Console()

# Path to TTS virtualenv Python
TTS_VENV_PYTHON = Path(__file__).parent / ".venv-tts" / "bin" / "python"
TTS_WORKER_SCRIPT = Path(__file__).parent / "tts_worker.py"

# XTTS-v2 supported languages
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Polish": "pl",
    "Turkish": "tr",
    "Russian": "ru",
    "Dutch": "nl",
    "Czech": "cs",
    "Arabic": "ar",
    "Chinese": "zh-cn",
    "Japanese": "ja",
    "Hungarian": "hu",
    "Korean": "ko",
    "Hindi": "hi",
}

# Reverse mapping
LANGUAGE_TO_CODE = {k.lower(): v for k, v in SUPPORTED_LANGUAGES.items()}
CODE_TO_LANGUAGE = {v: k for k, v in SUPPORTED_LANGUAGES.items()}


@dataclass
class TTSResult:
    """Result from TTS synthesis."""
    audio: np.ndarray
    sample_rate: int
    language: str
    output_path: Optional[str] = None


class TextToSpeech:
    """XTTS-v2 Text-to-Speech wrapper with voice cloning support.
    
    Uses subprocess to call the TTS worker in a separate virtualenv
    to avoid transformers version conflicts with qwen-asr.
    """
    
    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = "auto",
    ):
        self.model_name = model_name
        self.device = device
        self._loaded = False
        
        # Verify TTS venv exists
        if not TTS_VENV_PYTHON.exists():
            raise RuntimeError(
                f"TTS virtualenv not found at {TTS_VENV_PYTHON}. "
                "Please run: python3.10 -m venv .venv-tts && "
                ".venv-tts/bin/pip install TTS torch"
            )
    
    def load_model(self) -> None:
        """Pre-load check (actual loading happens in subprocess)."""
        if self._loaded:
            return
        
        console.print("[dim]TTS ready (XTTS-v2 via subprocess)[/dim]")
        self._loaded = True
        console.print("[green]✓ TTS engine ready[/green]")
    
    def get_language_code(self, language: str) -> str:
        """Convert language name to XTTS language code."""
        lang_lower = language.lower()
        
        # Direct code match
        if lang_lower in CODE_TO_LANGUAGE:
            return lang_lower
        
        # Name to code
        if lang_lower in LANGUAGE_TO_CODE:
            return LANGUAGE_TO_CODE[lang_lower]
        
        # Partial match
        for name, code in LANGUAGE_TO_CODE.items():
            if lang_lower in name or name in lang_lower:
                return code
        
        # Map from ASR language names to TTS codes
        asr_to_tts = {
            "mandarin": "zh-cn",
            "cantonese": "zh-cn",
            "japanese": "ja",
            "korean": "ko",
            "polish": "pl",
            "german": "de",
            "french": "fr",
            "spanish": "es",
            "italian": "it",
            "portuguese": "pt",
            "russian": "ru",
            "turkish": "tr",
            "dutch": "nl",
            "czech": "cs",
            "arabic": "ar",
            "hungarian": "hu",
            "hindi": "hi",
        }
        
        if lang_lower in asr_to_tts:
            return asr_to_tts[lang_lower]
        
        raise ValueError(f"Unsupported language: {language}. Supported: {list(SUPPORTED_LANGUAGES.keys())}")
    
    def synthesize(
        self,
        text: str,
        language: str,
        speaker_wav: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> TTSResult:
        """
        Synthesize speech from text using subprocess.
        
        Args:
            text: Text to synthesize
            language: Language name or code
            speaker_wav: Path to reference audio for voice cloning (6+ seconds)
            output_path: Optional path to save the audio file
        
        Returns:
            TTSResult with audio data and metadata
        """
        # Get language code
        lang_code = self.get_language_code(language)
        
        console.print(f"[dim]Synthesizing speech in {CODE_TO_LANGUAGE.get(lang_code, lang_code)}...[/dim]")
        
        # Generate output path if not provided
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav")
        
        # Build command
        cmd = [
            str(TTS_VENV_PYTHON),
            str(TTS_WORKER_SCRIPT),
            "--text", text,
            "--language", lang_code,
            "--output", output_path,
            "--device", self.device,
        ]
        
        if speaker_wav:
            console.print(f"[dim]Using voice clone from: {speaker_wav}[/dim]")
            cmd.extend(["--speaker-wav", speaker_wav])
        
        # Run TTS subprocess
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
            )
            
            # Print TTS info messages
            for line in result.stderr.split('\n'):
                if line.startswith('TTS_INFO:'):
                    console.print(f"[dim]{line.replace('TTS_INFO: ', '')}[/dim]")
            
            if result.returncode != 0:
                raise RuntimeError(f"TTS failed: {result.stderr}")
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("TTS timed out after 2 minutes")
        
        # Load the generated audio
        import scipy.io.wavfile as wav
        sample_rate, audio = wav.read(output_path)
        
        # Convert to float32 normalized
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        
        console.print("[green]✓ Speech synthesized[/green]")
        
        return TTSResult(
            audio=audio,
            sample_rate=sample_rate,
            language=lang_code,
            output_path=output_path,
        )
    
    def speak(
        self,
        text: str,
        language: str,
        speaker_wav: Optional[str] = None,
    ) -> None:
        """
        Synthesize and play speech immediately.
        
        Args:
            text: Text to speak
            language: Language name or code
            speaker_wav: Optional reference audio for voice cloning
        """
        import sounddevice as sd
        
        result = self.synthesize(text, language, speaker_wav)
        
        console.print("[dim]Playing audio...[/dim]")
        sd.play(result.audio, result.sample_rate)
        sd.wait()
        
        # Clean up temp file
        if result.output_path and os.path.exists(result.output_path):
            os.remove(result.output_path)


def get_supported_languages() -> list[str]:
    """Get list of supported TTS languages."""
    return list(SUPPORTED_LANGUAGES.keys())


def is_language_supported(language: str) -> bool:
    """Check if a language is supported for TTS."""
    lang_lower = language.lower()
    return (
        lang_lower in LANGUAGE_TO_CODE or
        lang_lower in CODE_TO_LANGUAGE or
        any(lang_lower in name for name in LANGUAGE_TO_CODE.keys())
    )
