"""
Text-to-Speech module using Qwen3-TTS.

Supports 10 major languages (Chinese, English, Japanese, Korean, German,
French, Russian, Portuguese, Spanish, Italian) with excellent voice quality.

Features:
- CustomVoice: 9 premium voices with style control
- VoiceDesign: Create voices from natural language descriptions  
- VoiceClone: 3-second rapid voice cloning from audio input

Uses the official qwen-tts Python package.
"""

import os
import tempfile
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

import numpy as np
from rich.console import Console

console = Console()

# Qwen3-TTS supported languages
SUPPORTED_LANGUAGES = {
    "Chinese": "Chinese",
    "English": "English",
    "Japanese": "Japanese",
    "Korean": "Korean",
    "German": "German",
    "French": "French",
    "Russian": "Russian",
    "Portuguese": "Portuguese",
    "Spanish": "Spanish",
    "Italian": "Italian",
}

# Map from common language names/codes to Qwen3-TTS language names
LANGUAGE_ALIASES = {
    # Direct matches
    "chinese": "Chinese",
    "english": "English",
    "japanese": "Japanese",
    "korean": "Korean",
    "german": "German",
    "french": "French",
    "russian": "Russian",
    "portuguese": "Portuguese",
    "spanish": "Spanish",
    "italian": "Italian",
    # ISO codes
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
    # ASR language names
    "mandarin": "Chinese",
    "cantonese": "Chinese",
}

# Available speakers in CustomVoice model
SPEAKERS = {
    "Vivian": {"description": "Bright, slightly edgy young female voice", "native_lang": "Chinese"},
    "Serena": {"description": "Warm, gentle young female voice", "native_lang": "Chinese"},
    "Uncle_Fu": {"description": "Seasoned male voice with a low, mellow timbre", "native_lang": "Chinese"},
    "Dylan": {"description": "Youthful Beijing male voice with clear, natural timbre", "native_lang": "Chinese"},
    "Eric": {"description": "Lively Chengdu male voice with slightly husky brightness", "native_lang": "Chinese"},
    "Ryan": {"description": "Dynamic male voice with strong rhythmic drive", "native_lang": "English"},
    "Aiden": {"description": "Sunny American male voice with clear midrange", "native_lang": "English"},
    "Ono_Anna": {"description": "Playful Japanese female voice with light, nimble timbre", "native_lang": "Japanese"},
    "Sohee": {"description": "Warm Korean female voice with rich emotion", "native_lang": "Korean"},
}

# Default speakers per language (for best quality)
DEFAULT_SPEAKERS = {
    "Chinese": "Vivian",
    "English": "Ryan",
    "Japanese": "Ono_Anna",
    "Korean": "Sohee",
    "German": "Ryan",
    "French": "Ryan",
    "Russian": "Ryan",
    "Portuguese": "Ryan",
    "Spanish": "Ryan",
    "Italian": "Ryan",
}


@dataclass
class TTSResult:
    """Result from TTS synthesis."""
    audio: np.ndarray
    sample_rate: int
    language: str
    output_path: Optional[str] = None


class TextToSpeech:
    """Qwen3-TTS Text-to-Speech with voice cloning support.
    
    Uses the 0.6B CustomVoice model for efficient inference with
    high-quality output across 10 languages.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device: str = "auto",
        use_flash_attention: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.use_flash_attention = use_flash_attention
        self._model = None
        self._loaded = False
        self._voice_clone_model = None
        self._voice_clone_prompt = None
    
    def load_model(self) -> None:
        """Load the Qwen3-TTS model."""
        if self._loaded:
            return
        
        console.print(f"[dim]Loading Qwen3-TTS ({self.model_name})...[/dim]")
        
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
            
            # Determine device
            if self.device == "auto":
                if torch.cuda.is_available():
                    device_map = "cuda:0"
                elif torch.backends.mps.is_available():
                    device_map = "mps"
                else:
                    device_map = "cpu"
            else:
                device_map = self.device
            
            # Determine attention implementation
            attn_impl = "flash_attention_2" if self.use_flash_attention and device_map.startswith("cuda") else "sdpa"
            
            # Determine dtype - MPS works best with float32, CUDA with bfloat16
            if device_map == "cpu":
                model_dtype = torch.float32
            elif device_map == "mps":
                model_dtype = torch.float32  # MPS has issues with bfloat16
            else:
                model_dtype = torch.bfloat16
            
            console.print(f"[dim]Using device: {device_map}, dtype: {model_dtype}, attention: {attn_impl}[/dim]")
            
            # Load model
            self._model = Qwen3TTSModel.from_pretrained(
                self.model_name,
                device_map=device_map,
                dtype=model_dtype,
                attn_implementation=attn_impl,
            )
            
            self._loaded = True
            console.print("[green]✓ Qwen3-TTS engine ready[/green]")
            
        except ImportError as e:
            raise RuntimeError(
                f"qwen-tts package not installed. Please run: pip install qwen-tts\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen3-TTS model: {e}")
    
    def get_language(self, language: str) -> str:
        """Convert language name/code to Qwen3-TTS language name."""
        lang_lower = language.lower().strip()
        
        # Check aliases first
        if lang_lower in LANGUAGE_ALIASES:
            return LANGUAGE_ALIASES[lang_lower]
        
        # Check if it's already a valid language name
        for lang_name in SUPPORTED_LANGUAGES.keys():
            if lang_lower == lang_name.lower():
                return lang_name
        
        # Default to English for unsupported languages
        console.print(f"[yellow]Warning: Language '{language}' not supported by Qwen3-TTS, using English[/yellow]")
        return "English"
    
    def get_speaker(self, language: str, preferred_speaker: Optional[str] = None) -> str:
        """Get the best speaker for a language."""
        if preferred_speaker and preferred_speaker in SPEAKERS:
            return preferred_speaker
        
        lang = self.get_language(language)
        return DEFAULT_SPEAKERS.get(lang, "Ryan")
    
    def synthesize(
        self,
        text: str,
        language: str,
        speaker_wav: Optional[str] = None,
        output_path: Optional[str] = None,
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
    ) -> TTSResult:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            language: Language name or code
            speaker_wav: Path to reference audio for voice cloning (3+ seconds)
            output_path: Optional path to save the audio file
            speaker: Speaker name (for CustomVoice model)
            instruct: Style instruction (e.g., "Speak with excitement")
        
        Returns:
            TTSResult with audio data and metadata
        """
        if not self._loaded:
            raise RuntimeError("TTS not loaded. Call load_model() first.")
        
        import soundfile as sf
        
        # Get language
        lang = self.get_language(language)
        
        # Generate output path if not provided
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav")
        
        try:
            # Check if we should use voice cloning
            if speaker_wav and os.path.exists(speaker_wav):
                # Load base model for cloning if not already loaded
                audio = self._synthesize_with_clone(text, lang, speaker_wav, output_path)
            else:
                # Use CustomVoice synthesis
                speaker_name = self.get_speaker(lang, speaker)
                
                # Estimate max tokens needed: ~12 tokens/sec audio, ~3 words/sec speech
                # For safety, allow up to 30 seconds of audio output
                # This significantly speeds up inference by limiting generation
                estimated_words = len(text.split())
                max_audio_seconds = max(10, min(60, estimated_words * 0.5))  # 0.5 sec per word, 10-60s range
                max_tokens = int(max_audio_seconds * 12 * 1.5)  # 12Hz * 1.5x safety margin
                
                wavs, sr = self._model.generate_custom_voice(
                    text=text,
                    language=lang,
                    speaker=speaker_name,
                    instruct=instruct or "",
                    max_new_tokens=max_tokens,
                )
                
                audio = wavs[0]
                sf.write(output_path, audio, sr)
            
            # Load the generated audio for consistency
            audio_data, sample_rate = sf.read(output_path)
            
            # Convert to float32 if needed
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            return TTSResult(
                audio=audio_data,
                sample_rate=sample_rate,
                language=lang,
                output_path=output_path,
            )
            
        except Exception as e:
            raise RuntimeError(f"TTS synthesis failed: {e}")
    
    def _synthesize_with_clone(
        self,
        text: str,
        language: str,
        speaker_wav: str,
        output_path: str,
    ) -> np.ndarray:
        """Synthesize speech using voice cloning from reference audio."""
        import torch
        import soundfile as sf
        from qwen_tts import Qwen3TTSModel
        
        # Load base model for cloning (separate from CustomVoice)
        if self._voice_clone_model is None:
            console.print("[dim]Loading voice clone model...[/dim]")
            
            if self.device == "auto":
                if torch.cuda.is_available():
                    device_map = "cuda:0"
                elif torch.backends.mps.is_available():
                    device_map = "mps"
                else:
                    device_map = "cpu"
            else:
                device_map = self.device
            
            attn_impl = "flash_attention_2" if self.use_flash_attention and device_map.startswith("cuda") else "sdpa"
            
            # Determine dtype - MPS works best with float32
            if device_map == "cpu" or device_map == "mps":
                model_dtype = torch.float32
            else:
                model_dtype = torch.bfloat16
            
            self._voice_clone_model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                device_map=device_map,
                dtype=model_dtype,
                attn_implementation=attn_impl,
            )
        
        # Get transcript of reference audio (optional, improves quality)
        # For simplicity, we use x_vector_only_mode which doesn't require transcript
        wavs, sr = self._voice_clone_model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=speaker_wav,
            ref_text=None,  # Not required with x_vector_only_mode
            x_vector_only_mode=True,
        )
        
        audio = wavs[0]
        sf.write(output_path, audio, sr)
        
        return audio
    
    def speak(
        self,
        text: str,
        language: str,
        speaker_wav: Optional[str] = None,
        speaker: Optional[str] = None,
    ) -> None:
        """
        Synthesize and play speech immediately.
        
        Args:
            text: Text to speak
            language: Language name or code
            speaker_wav: Optional reference audio for voice cloning
            speaker: Speaker name (for CustomVoice model)
        """
        import sounddevice as sd
        
        result = self.synthesize(text, language, speaker_wav, speaker=speaker)
        
        console.print("[dim]Playing audio...[/dim]")
        sd.play(result.audio, result.sample_rate)
        sd.wait()
        
        # Clean up temp file
        if result.output_path and os.path.exists(result.output_path):
            os.remove(result.output_path)
    
    def get_supported_speakers(self) -> List[str]:
        """Get list of available speakers."""
        return list(SPEAKERS.keys())
    
    def get_speaker_info(self, speaker: str) -> Optional[dict]:
        """Get information about a speaker."""
        return SPEAKERS.get(speaker)


def get_supported_languages() -> list[str]:
    """Get list of supported TTS languages."""
    return list(SUPPORTED_LANGUAGES.keys())


def is_language_supported(language: str) -> bool:
    """Check if a language is supported for TTS."""
    lang_lower = language.lower().strip()
    
    # Check aliases
    if lang_lower in LANGUAGE_ALIASES:
        return True
    
    # Check language names
    return any(lang_lower == name.lower() for name in SUPPORTED_LANGUAGES.keys())
