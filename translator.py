"""
NLLB-200 Translation module for multilingual text translation.
Optimized for edge deployment with support for 200 languages.
"""

# Import model_manager first to set up cache directories BEFORE importing transformers
import model_manager

import torch
from typing import Optional, Dict, List
from dataclasses import dataclass
from rich.console import Console

console = Console()

# Language code mapping: common name -> NLLB code
LANGUAGE_CODES: Dict[str, str] = {
    # Major European languages
    "English": "eng_Latn",
    "German": "deu_Latn",
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "Portuguese": "por_Latn",
    "Italian": "ita_Latn",
    "Dutch": "nld_Latn",
    "Polish": "pol_Latn",
    "Czech": "ces_Latn",
    "Swedish": "swe_Latn",
    "Danish": "dan_Latn",
    "Norwegian": "nob_Latn",
    "Finnish": "fin_Latn",
    "Hungarian": "hun_Latn",
    "Romanian": "ron_Latn",
    "Greek": "ell_Grek",
    "Bulgarian": "bul_Cyrl",
    "Croatian": "hrv_Latn",
    "Slovak": "slk_Latn",
    "Slovenian": "slv_Latn",
    "Lithuanian": "lit_Latn",
    "Latvian": "lvs_Latn",
    "Estonian": "est_Latn",
    "Ukrainian": "ukr_Cyrl",
    "Russian": "rus_Cyrl",
    
    # Asian languages
    "Chinese": "zho_Hans",
    "Mandarin": "zho_Hans",
    "Cantonese": "yue_Hant",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Thai": "tha_Thai",
    "Vietnamese": "vie_Latn",
    "Indonesian": "ind_Latn",
    "Malay": "zsm_Latn",
    "Hindi": "hin_Deva",
    "Bengali": "ben_Beng",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Urdu": "urd_Arab",
    "Punjabi": "pan_Guru",
    "Gujarati": "guj_Gujr",
    "Marathi": "mar_Deva",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Nepali": "npi_Deva",
    "Sinhala": "sin_Sinh",
    "Burmese": "mya_Mymr",
    "Khmer": "khm_Khmr",
    "Lao": "lao_Laoo",
    "Mongolian": "khk_Cyrl",
    "Tagalog": "tgl_Latn",
    "Filipino": "tgl_Latn",
    
    # Middle Eastern / African
    "Arabic": "arb_Arab",
    "Hebrew": "heb_Hebr",
    "Turkish": "tur_Latn",
    "Persian": "pes_Arab",
    "Farsi": "pes_Arab",
    "Swahili": "swh_Latn",
    "Amharic": "amh_Ethi",
    "Yoruba": "yor_Latn",
    "Igbo": "ibo_Latn",
    "Hausa": "hau_Latn",
    "Zulu": "zul_Latn",
    "Afrikaans": "afr_Latn",
}

# Reverse mapping for display
NLLB_TO_LANGUAGE: Dict[str, str] = {v: k for k, v in LANGUAGE_CODES.items()}


@dataclass
class TranslationResult:
    """Result from translation."""
    source_text: str
    translated_text: str
    source_language: str
    target_language: str


def get_nllb_code(language: str) -> Optional[str]:
    """
    Get NLLB language code from language name.
    
    Args:
        language: Language name (e.g., "English", "Polish", "Thai")
    
    Returns:
        NLLB code or None if not found
    """
    # Direct lookup
    if language in LANGUAGE_CODES:
        return LANGUAGE_CODES[language]
    
    # Case-insensitive lookup
    for name, code in LANGUAGE_CODES.items():
        if name.lower() == language.lower():
            return code
    
    # Check if it's already an NLLB code
    if "_" in language and len(language) == 8:
        return language
    
    return None


def get_supported_languages() -> List[str]:
    """Get list of supported language names."""
    return sorted(LANGUAGE_CODES.keys())


class Translator:
    """
    NLLB-200 based translator for multilingual translation.
    Supports 200 languages with optimized edge deployment.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: str = "auto",
        dtype: torch.dtype = torch.float16,
        max_length: int = 512
    ):
        """
        Initialize the translator.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('auto', 'cuda', 'cpu', 'mps')
            dtype: Model dtype
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Adjust dtype for device
        if self.device == "cpu":
            self.dtype = torch.float32
        elif self.device == "mps":
            self.dtype = torch.float16
        else:
            self.dtype = dtype
            
        console.print(f"[dim]Translator device: {self.device}, dtype: {self.dtype}[/dim]")
    
    def load_model(self) -> None:
        """Load the NLLB translation model."""
        console.print(f"\n[bold]Loading translation model: {self.model_name}[/bold]")
        console.print("[dim]This may take a while on first run...[/dim]")
        
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
            ).to(self.device)
            
            # Set to eval mode
            self.model.eval()
            
            console.print("[green]✓ Translation model loaded successfully[/green]")
            
        except ImportError:
            console.print("[red]✗ transformers package not installed[/red]")
            raise
        except Exception as e:
            console.print(f"[red]✗ Failed to load model: {e}[/red]")
            raise
    
    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str = "English",
        num_beams: int = 4
    ) -> TranslationResult:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            source_language: Source language name or NLLB code
            target_language: Target language name or NLLB code
            num_beams: Number of beams for beam search
        
        Returns:
            TranslationResult with translated text
        """
        if self.model is None:
            self.load_model()
        
        # Get NLLB codes
        src_code = get_nllb_code(source_language)
        tgt_code = get_nllb_code(target_language)
        
        if src_code is None:
            raise ValueError(f"Unsupported source language: {source_language}")
        if tgt_code is None:
            raise ValueError(f"Unsupported target language: {target_language}")
        
        console.print(f"[dim]Translating: {source_language} → {target_language}[/dim]")
        
        # Set source language
        self.tokenizer.src_lang = src_code
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        ).to(self.device)
        
        # Get target language token ID
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
        
        # Generate translation
        with torch.no_grad():
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=self.max_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        # Decode
        translated_text = self.tokenizer.decode(
            translated_tokens[0],
            skip_special_tokens=True
        )
        
        return TranslationResult(
            source_text=text,
            translated_text=translated_text,
            source_language=source_language,
            target_language=target_language
        )
    
    def translate_batch(
        self,
        texts: List[str],
        source_language: str,
        target_language: str = "English",
        num_beams: int = 4
    ) -> List[TranslationResult]:
        """
        Translate multiple texts.
        
        Args:
            texts: List of texts to translate
            source_language: Source language
            target_language: Target language
            num_beams: Number of beams for beam search
        
        Returns:
            List of TranslationResult
        """
        if self.model is None:
            self.load_model()
        
        src_code = get_nllb_code(source_language)
        tgt_code = get_nllb_code(target_language)
        
        if src_code is None:
            raise ValueError(f"Unsupported source language: {source_language}")
        if tgt_code is None:
            raise ValueError(f"Unsupported target language: {target_language}")
        
        self.tokenizer.src_lang = src_code
        
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
        
        with torch.no_grad():
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=self.max_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        # Decode all
        translated_texts = self.tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True
        )
        
        return [
            TranslationResult(
                source_text=src,
                translated_text=tgt,
                source_language=source_language,
                target_language=target_language
            )
            for src, tgt in zip(texts, translated_texts)
        ]
