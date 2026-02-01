#!/usr/bin/env python3
"""
Edge TTS module for high-quality multilingual text-to-speech.

Uses Microsoft Edge's neural TTS API which supports 400+ voices
across 130+ languages including Thai.

Compatible with FishSpeechTTS interface for easy swapping.
"""

import asyncio
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import edge_tts


@dataclass
class TTSResult:
    """Result from TTS synthesis (compatible with FishSpeechTTS)."""
    audio: np.ndarray
    sample_rate: int
    language: str
    output_path: Optional[str] = None


# Language to voice mapping
# Using female voices by default as they tend to be clearer
LANGUAGE_VOICES = {
    "en": "en-US-JennyNeural",       # English (US) - clear, natural
    "es": "es-ES-ElviraNeural",       # Spanish (Spain)
    "fr": "fr-FR-DeniseNeural",       # French
    "de": "de-DE-KatjaNeural",        # German
    "it": "it-IT-ElsaNeural",         # Italian
    "pt": "pt-BR-FranciscaNeural",    # Portuguese (Brazil)
    "pl": "pl-PL-ZofiaNeural",        # Polish
    "ru": "ru-RU-SvetlanaNeural",     # Russian
    "ja": "ja-JP-NanamiNeural",       # Japanese
    "ko": "ko-KR-SunHiNeural",        # Korean
    "zh": "zh-CN-XiaoxiaoNeural",     # Chinese (Simplified)
    "ar": "ar-SA-ZariyahNeural",      # Arabic
    "hi": "hi-IN-SwaraNeural",        # Hindi
    "th": "th-TH-PremwadeeNeural",    # Thai (female)
    "vi": "vi-VN-HoaiMyNeural",       # Vietnamese
    "id": "id-ID-GadisNeural",        # Indonesian
    "ms": "ms-MY-YasminNeural",       # Malay
    "nl": "nl-NL-ColetteNeural",      # Dutch
    "sv": "sv-SE-SofieNeural",        # Swedish
    "da": "da-DK-ChristelNeural",     # Danish
    "no": "nb-NO-IselinNeural",       # Norwegian
    "fi": "fi-FI-SelmaNeural",        # Finnish
    "tr": "tr-TR-EmelNeural",         # Turkish
    "uk": "uk-UA-PolinaNeural",       # Ukrainian
    "cs": "cs-CZ-VlastaNeural",       # Czech
    "el": "el-GR-AthinaNeural",       # Greek
    "he": "he-IL-HilaNeural",         # Hebrew
    "hu": "hu-HU-NoemiNeural",        # Hungarian
    "ro": "ro-RO-AlinaNeural",        # Romanian
}

# Language names to codes (for compatibility with translator)
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Polish": "pl",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh",
    "Arabic": "ar",
    "Hindi": "hi",
    "Thai": "th",
    "Vietnamese": "vi",
    "Indonesian": "id",
    "Malay": "ms",
    "Dutch": "nl",
    "Swedish": "sv",
    "Danish": "da",
    "Norwegian": "no",
    "Finnish": "fi",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Czech": "cs",
    "Greek": "el",
    "Hebrew": "he",
    "Hungarian": "hu",
    "Romanian": "ro",
}

LANGUAGE_TO_CODE = {k.lower(): v for k, v in SUPPORTED_LANGUAGES.items()}
CODE_TO_LANGUAGE = {v: k for k, v in SUPPORTED_LANGUAGES.items()}


def get_supported_languages() -> list:
    """Get list of supported language names."""
    return list(SUPPORTED_LANGUAGES.keys())


def is_language_supported(language: str) -> bool:
    """Check if a language is supported."""
    lang_lower = language.lower()
    return lang_lower in LANGUAGE_TO_CODE or lang_lower in CODE_TO_LANGUAGE or lang_lower in LANGUAGE_VOICES


class EdgeTTS:
    """Edge TTS wrapper with interface compatible with FishSpeechTTS.
    
    Uses Microsoft Edge's neural TTS API for high-quality multilingual synthesis.
    Supports 400+ voices across 130+ languages including Thai.
    """
    
    def __init__(
        self,
        device: str = "auto",  # Ignored, kept for API compatibility
        rate: str = "+0%",
        volume: str = "+0%",
    ):
        """
        Initialize Edge TTS.
        
        Args:
            device: Ignored (kept for API compatibility with FishSpeechTTS)
            rate: Speech rate adjustment (e.g., "+10%", "-10%")
            volume: Volume adjustment (e.g., "+10%", "-10%")
        """
        self.device = device  # Ignored but kept for compatibility
        self.rate = rate
        self.volume = volume
        self._loaded = False
    
    def load_model(self) -> None:
        """Mark as loaded (no model to load for Edge TTS)."""
        self._loaded = True
    
    def get_language_code(self, language: str) -> str:
        """Convert language name to language code."""
        lang_lower = language.lower()
        
        # Already a code
        if lang_lower in LANGUAGE_VOICES or len(lang_lower) == 2:
            return lang_lower
        
        # Name to code
        if lang_lower in LANGUAGE_TO_CODE:
            return LANGUAGE_TO_CODE[lang_lower]
        
        return "en"  # Default to English
    
    def get_voice_for_language(self, language: str) -> str:
        """Get the best voice for a given language."""
        lang_code = self.get_language_code(language)
        return LANGUAGE_VOICES.get(lang_code, "en-US-JennyNeural")
    
    async def _synthesize_async(
        self, 
        text: str, 
        output_path: str,
        voice: str,
    ) -> str:
        """Async synthesis implementation."""
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=self.rate,
            volume=self.volume,
        )
        await communicate.save(output_path)
        return output_path
    
    def _convert_mp3_to_wav(self, mp3_path: str, wav_path: str) -> bool:
        """Convert MP3 to WAV using ffmpeg."""
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", mp3_path, "-ar", "24000", "-ac", "1", wav_path],
                capture_output=True,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _read_audio_file(self, path: str) -> tuple:
        """Read audio file and return (audio_array, sample_rate)."""
        import wave
        
        # Edge TTS outputs MP3, we need to convert to WAV for consistency
        if path.endswith('.mp3'):
            wav_path = path.replace('.mp3', '.wav')
            if self._convert_mp3_to_wav(path, wav_path):
                path = wav_path
            else:
                # If ffmpeg not available, return empty audio
                return np.zeros(24000, dtype=np.float32), 24000
        
        try:
            with wave.open(path, 'rb') as wf:
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                audio_bytes = wf.readframes(n_frames)
                
                # Convert to numpy array
                if wf.getsampwidth() == 2:  # 16-bit
                    audio = np.frombuffer(audio_bytes, dtype=np.int16)
                    audio = audio.astype(np.float32) / 32768.0
                else:
                    audio = np.zeros(n_frames, dtype=np.float32)
                
                return audio, sample_rate
        except Exception:
            return np.zeros(24000, dtype=np.float32), 24000
    
    def synthesize(
        self,
        text: str,
        language: str = "en",
        speaker_wav: str = None,  # Ignored (Edge TTS doesn't support voice cloning)
        output_path: str = None,
        emotion: str = None,  # Ignored
    ) -> TTSResult:
        """
        Synthesize text to speech.
        
        Args:
            text: Text to synthesize
            language: Language code or name (e.g., "en", "th", "Thai")
            speaker_wav: Ignored (Edge TTS doesn't support voice cloning)
            output_path: Output file path (optional, creates temp file if not specified)
            emotion: Ignored (Edge TTS doesn't support emotion control)
            
        Returns:
            TTSResult with audio data and metadata
        """
        # Determine voice based on language
        lang_code = self.get_language_code(language)
        voice = self.get_voice_for_language(lang_code)
        
        # Determine output path (Edge TTS outputs MP3)
        mp3_path = None
        wav_path = None
        
        if output_path is None:
            mp3_path = tempfile.mktemp(suffix=".mp3")
            wav_path = mp3_path.replace('.mp3', '.wav')
        elif output_path.endswith('.wav'):
            wav_path = output_path
            mp3_path = output_path.replace('.wav', '.mp3')
        else:
            mp3_path = output_path
            wav_path = output_path.replace('.mp3', '.wav') if output_path.endswith('.mp3') else output_path + '.wav'
        
        # Run async synthesis
        asyncio.run(self._synthesize_async(
            text=text,
            output_path=mp3_path,
            voice=voice,
        ))
        
        # Convert to WAV
        final_path = wav_path
        if not self._convert_mp3_to_wav(mp3_path, wav_path):
            final_path = mp3_path
        
        # Read the audio file
        audio, sample_rate = self._read_audio_file(final_path)
        
        return TTSResult(
            audio=audio,
            sample_rate=sample_rate,
            language=lang_code,
            output_path=final_path,
        )
    
    def shutdown(self) -> None:
        """Shutdown (no-op for Edge TTS)."""
        pass
    
    @staticmethod
    async def list_voices_async():
        """List all available voices (async)."""
        voices = await edge_tts.list_voices()
        return voices
    
    @staticmethod
    def list_voices():
        """List all available voices."""
        return asyncio.run(EdgeTTS.list_voices_async())


# Alias for compatibility
FishSpeechTTS = EdgeTTS


if __name__ == "__main__":
    import sys
    
    # Test synthesis
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello! This is a test of Edge TTS."
    language = sys.argv[2] if len(sys.argv) > 2 else "en"
    
    print(f"Synthesizing: '{text}'")
    print(f"Language: {language}")
    
    tts = EdgeTTS()
    tts.load_model()
    result = tts.synthesize(text=text, language=language, output_path="test_edge_tts.wav")
    print(f"Output saved to: {result.output_path}")
    print(f"Audio shape: {result.audio.shape}, Sample rate: {result.sample_rate}")
