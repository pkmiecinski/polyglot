# Text-to-Speech (TTS) Models Research for Edge AI

## Executive Summary

For the Polyglot Edge AI translation device, we need a TTS model that:
- Supports multiple languages (especially Polish and Thai)
- Has small model size for edge deployment
- Has fast inference (real-time capable)
- Produces natural-sounding speech

**🏆 Recommendation: Coqui XTTS-v2** for multilingual support, or **MeloTTS** for ultra-fast CPU inference.

---

## Models Comparison

| Model | Size | Languages | Speed | Voice Clone | License | Edge Suitable |
|-------|------|-----------|-------|-------------|---------|---------------|
| **Coqui XTTS-v2** | ~1.5GB | 17 | Medium | ✅ 6s clip | CPML | ⭐⭐⭐ |
| **MeloTTS** | ~300MB | 10 | Very Fast | ❌ | MIT | ⭐⭐⭐⭐⭐ |
| **Piper** | ~30-60MB | 30+ | Ultra Fast | ❌ | MIT | ⭐⭐⭐⭐⭐ |
| **Microsoft SpeechT5** | ~350MB | 1 (EN) | Fast | ✅ | MIT | ⭐⭐⭐ |
| **Parler-TTS Mini** | ~900MB | 1 (EN) | Medium | ❌ | Apache 2.0 | ⭐⭐ |
| **Suno Bark** | ~5GB | 13 | Slow | ✅ | MIT | ⭐ |
| **Fish Speech 1.5** | ~2GB | 13 | Medium | ✅ | CC-BY-NC-SA | ⭐⭐ |
| **OuteTTS 0.3** | 500M-1B | 6 | Medium | ✅ | CC-BY-NC-SA | ⭐⭐⭐ |
| **Qwen2.5-Omni** | ~11GB | Multi | Slow | ❌ | Apache 2.0 | ❌ |

---

## Detailed Analysis

### 1. 🥇 Coqui XTTS-v2 (Recommended for Quality + Multilingual)

**Best for:** High-quality multilingual TTS with voice cloning

```python
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
tts.tts_to_file(
    text="Hello, this is a test.",
    file_path="output.wav",
    speaker_wav="reference.wav",  # 6 second clip for voice cloning
    language="en"
)
```

**Pros:**
- 17 languages: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi
- ✅ **Polish supported!**
- Voice cloning with just 6-second audio
- Emotion and style transfer
- 24kHz output quality

**Cons:**
- ~1.5GB model size
- Requires GPU for real-time
- Coqui Public Model License (commercial restrictions)
- Thai NOT supported ❌

**Installation:**
```bash
pip install TTS
```

---

### 2. 🥈 MeloTTS (Recommended for Speed + Edge)

**Best for:** Ultra-fast CPU real-time inference

```python
from melo.api import TTS

model = TTS(language='EN', device='cpu')  # CPU is sufficient!
speaker_ids = model.hps.data.spk2id

# Multiple accents available
model.tts_to_file("Hello world!", speaker_ids['EN-US'], "output.wav", speed=1.0)
model.tts_to_file("Hello world!", speaker_ids['EN-BR'], "output_british.wav")
```

**Supported Languages:**
- English (American, British, Indian, Australian)
- Spanish, French, Chinese (mixed EN), Japanese, Korean

**Pros:**
- ⚡ CPU real-time inference (no GPU needed!)
- ~300MB model size
- MIT License (fully commercial)
- Multiple accents per language
- Very high quality

**Cons:**
- Polish NOT supported ❌
- Thai NOT supported ❌
- No voice cloning

**Installation:**
```bash
pip install melo-tts
# Or from source for latest:
pip install git+https://github.com/myshell-ai/MeloTTS.git
```

---

### 3. 🥉 Piper (Best for Embedded/Edge)

**Best for:** Ultra-lightweight embedded deployment

**Pros:**
- 🪶 Extremely small (~30-60MB per voice)
- ⚡ Ultra-fast inference
- 30+ languages with many voices
- MIT License
- ONNX format (runs anywhere)
- ✅ Polish supported with multiple voices!

**Cons:**
- One model per language (need to download multiple)
- No voice cloning
- Project archived (moved to piper1-gpl)

**Usage:**
```bash
# Download voice model
echo "Hello world" | piper --model en_US-lessac-medium.onnx --output_file out.wav
```

**Note:** Development moved to https://github.com/OHF-Voice/piper1-gpl

---

### 4. Fish Speech 1.5

**Best for:** High quality with voice cloning

**Languages (13):**
- English, Chinese, Japanese (>100k hours each)
- German, French, Spanish, Korean, Arabic, Russian (~20k hours)
- Dutch, Italian, **Polish**, Portuguese (<10k hours)

**Pros:**
- ✅ Polish supported!
- Voice cloning capability
- Trained on >1M hours of audio

**Cons:**
- ~2GB model size
- CC-BY-NC-SA license (non-commercial)
- Thai NOT supported ❌

---

### 5. OuteTTS 0.3

**Best for:** LLM-based TTS with voice control

```python
import outetts

model_config = outetts.HFModelConfig_v2(
    model_path="OuteAI/OuteTTS-0.3-500M",
    tokenizer_path="OuteAI/OuteTTS-0.3-500M"
)
interface = outetts.InterfaceHF(model_version="0.3", cfg=model_config)
speaker = interface.load_default_speaker(name="en_male_1")

gen_cfg = outetts.GenerationConfig(
    text="Hello world!",
    temperature=0.4,
    speaker=speaker,
)
output = interface.generate(config=gen_cfg)
output.save("output.wav")
```

**Variants:**
- 500M (Apache 2.0 compatible) - en, jp, ko, zh, fr, de
- 1B (CC-BY-NC-SA) - better quality

**Pros:**
- Voice cloning support
- GGUF quantized versions available
- Good quality

**Cons:**
- Polish NOT supported ❌
- Thai NOT supported ❌

---

### 6. Microsoft SpeechT5

**Best for:** Simple English TTS with speaker embeddings

```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

inputs = processor(text="Hello, my dog is cute.", return_tensors="pt")
# Load speaker embedding
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
```

**Pros:**
- MIT License
- Speaker embedding support
- Well-documented

**Cons:**
- English only
- Requires external speaker embeddings

---

### 7. Suno Bark (NOT recommended for edge)

**Best for:** Creative audio generation (music, effects, emotions)

```python
from transformers import pipeline
synthesiser = pipeline("text-to-speech", "suno/bark")
speech = synthesiser("Hello! [laughs] How are you?", forward_params={"do_sample": True})
```

**Pros:**
- Can generate music, sound effects
- Supports emotions: [laughs], [sighs], [crying]
- 13 languages

**Cons:**
- ~5GB model size ❌
- Slow inference ❌
- Not suitable for edge deployment

---

### 8. Qwen2.5-Omni (NOT for TTS-only use)

**Best for:** Full multimodal AI (not standalone TTS)

This is an 11B parameter multimodal model that can:
- Perceive text, images, audio, video
- Generate text AND speech simultaneously
- Real-time voice/video chat

**NOT recommended** for TTS-only use case due to massive size.

---

## Language Coverage Matrix

| Model | 🇬🇧 EN | 🇵🇱 PL | 🇹🇭 TH | 🇨🇳 ZH | 🇯🇵 JP | 🇰🇷 KO | 🇩🇪 DE | 🇫🇷 FR | 🇪🇸 ES |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| XTTS-v2 | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| MeloTTS | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Piper | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Fish Speech | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Bark | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| OuteTTS | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

**⚠️ Thai (TH) is NOT supported by any major open-source TTS model!**

---

## Recommendations by Use Case

### For Polyglot Edge Device (Polish + Multilingual)

**Option A: XTTS-v2** (Quality Priority)
- ✅ Polish support
- ✅ 17 languages
- ✅ Voice cloning
- ⚠️ Needs GPU for real-time
- ⚠️ ~1.5GB

**Option B: Piper** (Edge Priority)
- ✅ Polish support
- ✅ Ultra-lightweight (~50MB)
- ✅ CPU real-time
- ⚠️ No voice cloning
- ⚠️ One model per language

**Option C: Fish Speech 1.5** (Best Polish Quality)
- ✅ Polish support
- ✅ Voice cloning
- ⚠️ Non-commercial license
- ⚠️ ~2GB

### For Thai Support

Unfortunately, **no major open-source TTS supports Thai** well. Options:
1. Use cloud API (Google TTS, Azure TTS)
2. Train custom model on Thai data
3. Wait for future model releases

---

## Implementation Plan for Polyglot

### Recommended: Hybrid Approach

```python
# Use Piper for supported languages (fast, lightweight)
# Fall back to XTTS-v2 for voice cloning needs

class TTSEngine:
    def __init__(self):
        self.piper_languages = {'en', 'pl', 'de', 'fr', 'es', 'zh', 'ja', 'ko'}
        self.xtts_languages = {'en', 'pl', 'de', 'it', 'pt', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh', 'ja', 'hu', 'ko', 'hi'}
    
    def speak(self, text: str, language: str):
        if language in self.piper_languages:
            return self._piper_tts(text, language)
        elif language in self.xtts_languages:
            return self._xtts_tts(text, language)
        else:
            raise ValueError(f"Language {language} not supported")
```

### Quick Start with XTTS-v2

```bash
pip install TTS
```

```python
from TTS.api import TTS

# Initialize once
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.to("mps")  # Use Apple Silicon GPU

# Generate speech
def speak(text: str, language: str, output_path: str = "output.wav"):
    tts.tts_to_file(
        text=text,
        file_path=output_path,
        language=language
    )
    return output_path
```

---

## References

- Coqui XTTS-v2: https://huggingface.co/coqui/XTTS-v2
- MeloTTS: https://github.com/myshell-ai/MeloTTS
- Piper: https://github.com/OHF-Voice/piper1-gpl
- Fish Speech: https://github.com/fishaudio/fish-speech
- OuteTTS: https://github.com/edwko/OuteTTS
- Microsoft SpeechT5: https://huggingface.co/microsoft/speecht5_tts
- Suno Bark: https://huggingface.co/suno/bark
