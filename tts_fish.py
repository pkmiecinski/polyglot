"""
Text-to-Speech module using Fish Speech 1.5.

Fish Speech is a state-of-the-art TTS model supporting:
- 13+ languages with high quality
- Zero-shot voice cloning
- Emotion control
- No phoneme dependency (works with any script including Thai)
"""

import os
import sys
import subprocess
import tempfile
import json
import socket
import struct
import time
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import numpy as np
from rich.console import Console

console = Console()

# Path to Fish Speech virtualenv Python
FISH_VENV_PYTHON = Path(__file__).parent / ".venv-fish" / "bin" / "python"
FISH_SERVER_SCRIPT = Path(__file__).parent / "tts_fish_server.py"

# Socket path for Fish Speech server
SOCKET_PATH = "/tmp/polyglot_fish_tts.sock"

# Fish Speech supported languages (directly supported with emotion markers)
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Chinese": "zh",
    "Japanese": "ja",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Korean": "ko",
    "Arabic": "ar",
    "Russian": "ru",
    "Dutch": "nl",
    "Italian": "it",
    "Polish": "pl",
    "Portuguese": "pt",
    # These work via generalization (no phoneme dependency)
    "Thai": "th",
    "Vietnamese": "vi",
    "Indonesian": "id",
    "Hindi": "hi",
    "Turkish": "tr",
}

# Reverse mapping
LANGUAGE_TO_CODE = {k.lower(): v for k, v in SUPPORTED_LANGUAGES.items()}
CODE_TO_LANGUAGE = {v: k for k, v in SUPPORTED_LANGUAGES.items()}


def _send_message(sock, data: dict):
    """Send a JSON message with length prefix."""
    msg = json.dumps(data).encode('utf-8')
    sock.sendall(struct.pack('!I', len(msg)) + msg)


def _recv_message(sock) -> dict:
    """Receive a JSON message with length prefix."""
    raw_len = sock.recv(4)
    if not raw_len:
        return None
    msg_len = struct.unpack('!I', raw_len)[0]
    
    data = b''
    while len(data) < msg_len:
        chunk = sock.recv(min(msg_len - len(data), 4096))
        if not chunk:
            return None
        data += chunk
    
    return json.loads(data.decode('utf-8'))


def _is_server_running() -> bool:
    """Check if the Fish Speech server is running."""
    if not os.path.exists(SOCKET_PATH):
        return False
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect(SOCKET_PATH)
        _send_message(sock, {"cmd": "ping"})
        response = _recv_message(sock)
        sock.close()
        return response and response.get("status") == "ok"
    except:
        return False


@dataclass
class TTSResult:
    """Result from TTS synthesis."""
    audio: np.ndarray
    sample_rate: int
    language: str
    output_path: Optional[str] = None


def get_supported_languages() -> list:
    """Get list of supported language names."""
    return list(SUPPORTED_LANGUAGES.keys())


def is_language_supported(language: str) -> bool:
    """Check if a language is supported."""
    lang_lower = language.lower()
    return lang_lower in LANGUAGE_TO_CODE or lang_lower in CODE_TO_LANGUAGE


class FishSpeechTTS:
    """Fish Speech TTS wrapper with voice cloning support.
    
    Uses OpenAudio S1-mini model for high-quality multilingual TTS.
    """
    
    def __init__(
        self,
        model_name: str = "fishaudio/openaudio-s1-mini",
        device: str = "auto",
    ):
        self.model_name = model_name
        self.device = device
        self._server_process = None
        self._loaded = False
        
        # Verify Fish Speech venv exists
        if not FISH_VENV_PYTHON.exists():
            raise RuntimeError(
                f"Fish Speech virtualenv not found at {FISH_VENV_PYTHON}. "
                "Please run: python3.10 -m venv .venv-fish && "
                ".venv-fish/bin/pip install fish-speech torch"
            )
    
    def load_model(self) -> None:
        """Start the Fish Speech server if not already running."""
        if self._loaded:
            return
        
        # Check if server is already running
        if _is_server_running():
            console.print("[dim]Fish Speech server already running[/dim]")
            self._loaded = True
            console.print("[green]✓ Fish Speech TTS ready[/green]")
            return
        
        console.print("[dim]Starting Fish Speech server (S1-mini)...[/dim]")
        console.print("[dim]This will take ~30s on first load, then be instant.[/dim]")
        
        # Start the server in the background
        self._server_process = subprocess.Popen(
            [
                str(FISH_VENV_PYTHON),
                str(FISH_SERVER_SCRIPT),
                "--mode", "server",
                "--device", self.device,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        
        # Wait for server to be ready
        max_wait = 180  # 3 minutes max for first download
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if _is_server_running():
                self._loaded = True
                console.print("[green]✓ Fish Speech TTS ready[/green]")
                return
            
            if self._server_process.poll() is not None:
                stderr = self._server_process.stderr.read().decode() if self._server_process.stderr else ""
                raise RuntimeError(f"Fish Speech server failed to start: {stderr}")
            
            time.sleep(0.5)
        
        raise RuntimeError("Fish Speech server failed to start within timeout")
    
    def get_language_code(self, language: str) -> str:
        """Convert language name to Fish Speech language code."""
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
        
        # Default to English
        console.print(f"[yellow]Warning: Unknown language '{language}', using English[/yellow]")
        return "en"
    
    def synthesize(
        self,
        text: str,
        language: str = "en",
        speaker_wav: Optional[str] = None,
        output_path: Optional[str] = None,
        emotion: Optional[str] = None,
    ) -> TTSResult:
        """Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            language: Target language code or name
            speaker_wav: Path to reference audio for voice cloning
            output_path: Path to save output WAV file
            emotion: Optional emotion marker (e.g., "happy", "sad", "excited")
        
        Returns:
            TTSResult with audio data and metadata
        """
        if not self._loaded:
            self.load_model()
        
        lang_code = self.get_language_code(language)
        
        # Prepare output path
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav")
        
        # Connect to server
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(120.0)  # 2 min timeout for long texts
        
        try:
            sock.connect(SOCKET_PATH)
            
            request = {
                "cmd": "synthesize",
                "text": text,
                "language": lang_code,
                "speaker_wav": speaker_wav,
                "output_path": output_path,
                "emotion": emotion,
            }
            
            _send_message(sock, request)
            response = _recv_message(sock)
            
            if not response:
                raise RuntimeError("No response from Fish Speech server")
            
            if not response.get("success"):
                raise RuntimeError(f"Synthesis failed: {response.get('error', 'Unknown error')}")
            
            # Load the output audio
            import scipy.io.wavfile as wav
            sample_rate, audio = wav.read(output_path)
            
            # Convert to float32 if needed
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            
            return TTSResult(
                audio=audio,
                sample_rate=sample_rate,
                language=lang_code,
                output_path=output_path,
            )
            
        finally:
            sock.close()
    
    def shutdown_server(self):
        """Shutdown the Fish Speech server."""
        if not _is_server_running():
            return
        
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect(SOCKET_PATH)
            _send_message(sock, {"cmd": "shutdown"})
            sock.close()
        except:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API FUNCTIONS (for GUI compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

def get_supported_languages() -> list:
    """Return list of supported language names for GUI dropdown."""
    return list(SUPPORTED_LANGUAGES.keys())


def is_language_supported(language: str) -> bool:
    """Check if a language is supported."""
    lang_lower = language.lower()
    return (
        lang_lower in LANGUAGE_TO_CODE or
        language in SUPPORTED_LANGUAGES or
        language in CODE_TO_LANGUAGE
    )


# Convenience function
def get_tts_engine(device: str = "auto") -> FishSpeechTTS:
    """Get a configured Fish Speech TTS engine."""
    return FishSpeechTTS(device=device)


if __name__ == "__main__":
    # Quick test
    tts = FishSpeechTTS()
    tts.load_model()
    
    result = tts.synthesize(
        text="Hello! This is a test of the Fish Speech text-to-speech system.",
        language="English",
        output_path="test_output.wav",
    )
    
    print(f"Generated audio: {result.output_path}")
    print(f"Sample rate: {result.sample_rate}")
    print(f"Duration: {len(result.audio) / result.sample_rate:.2f}s")
