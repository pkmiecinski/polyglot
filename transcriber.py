"""
Qwen3-ASR transcription wrapper for speech-to-text and language detection.
"""

# Import model_manager first to set up cache directories BEFORE importing transformers
import model_manager

import torch
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from rich.console import Console

console = Console()


@dataclass
class TranscriptionResult:
    """Result from transcription."""
    text: str
    language: str
    confidence: Optional[float] = None


class Transcriber:
    """
    Wrapper for Qwen3-ASR model for speech transcription and language detection.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 512
    ):
        """
        Initialize the transcriber.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to use ('auto', 'cuda', 'cpu', 'mps')
            dtype: Model dtype (torch.bfloat16 recommended)
            max_new_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.model = None
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda:0"
            elif torch.backends.mps.is_available():
                self.device = "mps"
                dtype = torch.float16  # MPS doesn't support bfloat16 well
            else:
                self.device = "cpu"
                dtype = torch.float32  # CPU is slow with half precision
        else:
            self.device = device
        
        self.dtype = dtype
        console.print(f"[dim]Using device: {self.device}, dtype: {self.dtype}[/dim]")
    
    def load_model(self) -> None:
        """Load the Qwen3-ASR model."""
        console.print(f"\n[bold]Loading model: {self.model_name}[/bold]")
        console.print("[dim]This may take a while on first run (downloading model)...[/dim]")
        
        try:
            from qwen_asr import Qwen3ASRModel
            
            # Determine device_map based on device
            if self.device.startswith("cuda"):
                device_map = self.device
            elif self.device == "mps":
                device_map = "mps"
            else:
                device_map = "cpu"
            
            self.model = Qwen3ASRModel.from_pretrained(
                self.model_name,
                dtype=self.dtype,
                device_map=device_map,
                max_inference_batch_size=1,
                max_new_tokens=self.max_new_tokens,
            )
            
            console.print("[green]✓ Model loaded successfully[/green]")
            
        except ImportError:
            console.print("[red]✗ qwen-asr package not installed. Run: pip install qwen-asr[/red]")
            raise
        except Exception as e:
            console.print(f"[red]✗ Failed to load model: {e}[/red]")
            raise
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio and detect language.
        
        Args:
            audio: Audio data as numpy array (float32, mono)
            sample_rate: Audio sample rate
            language: Force specific language (None for auto-detection)
        
        Returns:
            TranscriptionResult with text and detected language
        """
        if self.model is None:
            self.load_model()
        
        console.print("\n[bold]Transcribing...[/bold]")
        
        # Ensure audio is the right format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize audio if needed
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
        
        try:
            # Pass audio as (numpy array, sample_rate) tuple
            results = self.model.transcribe(
                audio=(audio, sample_rate),
                language=language,  # None for auto-detection
            )
            
            result = results[0]
            
            return TranscriptionResult(
                text=result.text,
                language=result.language,
            )
            
        except Exception as e:
            console.print(f"[red]✗ Transcription error: {e}[/red]")
            raise
    
    def transcribe_batch(
        self,
        audio_list: List[Tuple[np.ndarray, int]],
        languages: Optional[List[str]] = None
    ) -> List[TranscriptionResult]:
        """
        Transcribe multiple audio samples.
        
        Args:
            audio_list: List of (audio_array, sample_rate) tuples
            languages: List of languages (None for auto-detection)
        
        Returns:
            List of TranscriptionResult
        """
        if self.model is None:
            self.load_model()
        
        # Format audio inputs
        audio_inputs = [(audio, sr) for audio, sr in audio_list]
        
        results = self.model.transcribe(
            audio=audio_inputs,
            language=languages,
        )
        
        return [
            TranscriptionResult(text=r.text, language=r.language)
            for r in results
        ]


def get_supported_languages() -> List[str]:
    """Get list of supported languages."""
    return [
        "Chinese", "English", "Cantonese", "Arabic", "German", "French",
        "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian",
        "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay",
        "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech",
        "Filipino", "Persian", "Greek", "Hungarian", "Macedonian", "Romanian"
    ]
