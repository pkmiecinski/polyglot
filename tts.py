"""
Text-to-Speech module using Coqui XTTS-v2.

Supports 17 languages with voice cloning from a 6-second audio clip.
Uses a persistent TTS server to avoid model reloading on each request.
"""

import os
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

# Path to TTS virtualenv Python
TTS_VENV_PYTHON = Path(__file__).parent / ".venv-tts" / "bin" / "python"
TTS_SERVER_SCRIPT = Path(__file__).parent / "tts_server.py"

# Socket path for TTS server
SOCKET_PATH = "/tmp/polyglot_tts.sock"

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
    """Check if the TTS server is running."""
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


class TextToSpeech:
    """XTTS-v2 Text-to-Speech wrapper with voice cloning support.
    
    Uses a persistent server process to keep the model loaded in memory.
    This eliminates the ~30s model loading time on each synthesis.
    """
    
    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = "auto",
    ):
        self.model_name = model_name
        self.device = device
        self._server_process = None
        self._loaded = False
        
        # Verify TTS venv exists
        if not TTS_VENV_PYTHON.exists():
            raise RuntimeError(
                f"TTS virtualenv not found at {TTS_VENV_PYTHON}. "
                "Please run: python3.10 -m venv .venv-tts && "
                ".venv-tts/bin/pip install TTS torch"
            )
    
    def load_model(self) -> None:
        """Start the TTS server if not already running."""
        if self._loaded:
            return
        
        # Check if server is already running
        if _is_server_running():
            console.print("[dim]TTS server already running[/dim]")
            self._loaded = True
            console.print("[green]✓ TTS engine ready[/green]")
            return
        
        console.print("[dim]Starting TTS server (XTTS-v2)...[/dim]")
        console.print("[dim]This will take ~30s on first load, then be instant.[/dim]")
        
        # Start the server in the background
        self._server_process = subprocess.Popen(
            [
                str(TTS_VENV_PYTHON),
                str(TTS_SERVER_SCRIPT),
                "--mode", "server",
                "--device", self.device,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # Detach from parent
        )
        
        # Wait for server to be ready (with timeout)
        max_wait = 120  # 2 minutes max
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if _is_server_running():
                self._loaded = True
                console.print("[green]✓ TTS engine ready[/green]")
                return
            
            # Check if process died
            if self._server_process.poll() is not None:
                stderr = self._server_process.stderr.read().decode() if self._server_process.stderr else ""
                raise RuntimeError(f"TTS server failed to start: {stderr}")
            
            time.sleep(0.5)
        
        raise RuntimeError("TTS server failed to start within timeout")
    
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
        Synthesize speech from text using the TTS server.
        
        Args:
            text: Text to synthesize
            language: Language name or code
            speaker_wav: Path to reference audio for voice cloning (6+ seconds)
            output_path: Optional path to save the audio file
        
        Returns:
            TTSResult with audio data and metadata
        """
        if not self._loaded:
            raise RuntimeError("TTS not loaded. Call load_model() first.")
        
        # Get language code
        lang_code = self.get_language_code(language)
        
        # Generate output path if not provided
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav")
        
        # Connect to server
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(120.0)  # 2 minute timeout
        
        try:
            sock.connect(SOCKET_PATH)
            
            request = {
                "cmd": "synthesize",
                "text": text,
                "language": lang_code,
                "output_path": output_path,
            }
            if speaker_wav:
                request["speaker_wav"] = speaker_wav
            
            _send_message(sock, request)
            response = _recv_message(sock)
            
            if not response:
                raise RuntimeError("No response from TTS server")
            
            if not response.get("success"):
                raise RuntimeError(f"TTS failed: {response.get('error', 'Unknown error')}")
            
        finally:
            sock.close()
        
        # Load the generated audio
        import scipy.io.wavfile as wav
        sample_rate, audio = wav.read(output_path)
        
        # Convert to float32 normalized
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
    
    def shutdown_server(self):
        """Shutdown the TTS server."""
        if _is_server_running():
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect(SOCKET_PATH)
                _send_message(sock, {"cmd": "shutdown"})
                sock.close()
            except:
                pass
    
    def __del__(self):
        """Don't shutdown server on object deletion - keep it running."""
        pass


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


def stop_tts_server():
    """Stop the TTS server if running."""
    if _is_server_running():
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect(SOCKET_PATH)
            _send_message(sock, {"cmd": "shutdown"})
            sock.close()
            console.print("[dim]TTS server stopped[/dim]")
        except:
            pass
