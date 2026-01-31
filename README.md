# Polyglot - Edge AI Translation Device PoC

A Proof of Concept for an Edge AI translation device using:
- **Qwen3-ASR** for automatic speech recognition and language detection
- **NLLB-200** for multilingual translation (200 languages)
- **XTTS-v2** for text-to-speech with voice cloning

## Features

- **Modern GUI Application**: Beautiful PyQt6 interface with real-time logging
- **Pre-loaded Models**: All models load at startup for instant translation
- **Real-time microphone input**: Captures audio from your microphone
- **Automatic language detection**: Identifies 52 languages and dialects
- **Speech transcription**: High-quality ASR using Qwen3-ASR-1.7B
- **Multilingual translation**: Translate between 200 languages with NLLB-200
- **Text-to-Speech**: Speak translations aloud with XTTS-v2 (17 languages)
- **Voice cloning**: Clone your voice from the input recording
- **Edge-ready**: Designed for local/edge deployment

## Quick Start

### GUI Application (Recommended)

```bash
./run_gui.sh
```

Or directly:
```bash
/opt/homebrew/bin/python3.10 gui.py
```

![Polyglot GUI](docs/gui-screenshot.png)

The GUI features:
- **Model Status Panel**: Shows loading progress for ASR, Translation, and TTS
- **Timestamped Activity Log**: Real-time log with color-coded messages
- **One-Click Recording**: Press REC, speak, and get instant translation
- **Voice Cloning Toggle**: Clone your voice from the recording
- **Replay Button**: Replay the last spoken translation

### CLI Application

```bash
python main.py --continuous --translate English --speak --clone-voice
```

## Supported Languages

### ASR (Qwen3-ASR)
30 languages including Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, and more.

### Translation (NLLB-200)
200 languages including:
- **Polish** ✅ and **Thai** ✅
- All major European, Asian, Middle Eastern, and African languages

## Requirements

- Python 3.12+
- CUDA-capable GPU (recommended) or CPU
- PyTorch with CUDA support (for GPU acceleration)
- PortAudio (for microphone input)

## Installation

### 1. Install system dependencies (macOS)

```bash
brew install portaudio
```

### 2. Create virtual environment

```bash
conda create -n polyglot python=3.12 -y
conda activate polyglot
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup TTS environment (separate venv due to dependency conflicts)

```bash
python3.10 -m venv .venv-tts
.venv-tts/bin/pip install TTS 'torch<2.6' 'torchaudio<2.6' 'transformers<4.46'
```

### 5. (Optional) Install FlashAttention for faster inference

```bash
pip install -U flash-attn --no-build-isolation
```

## Usage

### Basic transcription app

```bash
python main.py
```

### With translation

```bash
# Transcribe and translate to English
python main.py --continuous --translate English

# Translate to Polish
python main.py -c -T Polish

# Translate to Thai
python main.py -c -T Thai
```

### With speech output (TTS)

```bash
# Translate and speak the result
python main.py --continuous --translate English --speak

# Use voice cloning (provide 6+ second sample)
python main.py -c -T English --speak --voice my_voice.wav

# Save spoken audio to file
python main.py -c -T English --speak --save-audio output.wav
```

### Other options

```bash
# Use smaller model for faster inference
python main.py --model Qwen/Qwen3-ASR-0.6B

# Force specific language (skip auto-detection)
python main.py --language English

# Adjust recording duration
python main.py --duration 10

# Use CPU instead of GPU
python main.py --device cpu

# Loop mode - keep recording and translating
python main.py --loop --continuous --translate English
```

## Project Structure

```
polyglot/
├── gui.py               # 🖥️ GUI application (PyQt6)
├── run_gui.sh           # GUI launcher script
├── main.py              # CLI application entry point
├── audio_capture.py     # Microphone audio capture utilities
├── transcriber.py       # Qwen3-ASR transcription wrapper
├── translator.py        # NLLB-200 translation wrapper
├── tts.py               # XTTS-v2 text-to-speech wrapper
├── tts_worker.py        # TTS subprocess worker (runs in .venv-tts)
├── .venv-tts/           # Separate venv for TTS (created during setup)
├── output/              # Auto-saved TTS output files
│   └── last_output.wav  # Last spoken translation (for replay)
├── requirements.txt     # Python dependencies
├── docs/                # Documentation
│   ├── TRANSLATION_MODELS_RESEARCH.md
│   └── TTS_MODELS_RESEARCH.md
└── README.md            # This file
```

## How It Works

1. **Audio Capture**: Records audio from the microphone at 16kHz sample rate
2. **Preprocessing**: Converts audio to the format expected by Qwen3-ASR
3. **Language Detection**: Model automatically identifies the spoken language
4. **Transcription**: Generates text transcript of the spoken audio
5. **Translation** (optional): NLLB-200 translates to your target language
6. **Speech Output** (optional): XTTS-v2 speaks the translation aloud

```
🎤 Audio → [Qwen3-ASR] → Text + Language → [NLLB-200] → Translated Text → [XTTS-v2] → 🔊 Speech
```

### GUI Pipeline

With the GUI, all models are **pre-loaded at startup**, so the pipeline runs instantly:

| Step | Time (approx) |
|------|---------------|
| Recording | Until silence detected |
| ASR (Qwen3-ASR) | ~500ms |
| Translation (NLLB-200) | ~200ms |
| TTS (XTTS-v2) | ~2-3s |
| **Total** | **~3-4s after speaking** |

## Models Used

| Component | Model | Size | Languages |
|-----------|-------|------|-----------|
| ASR | Qwen3-ASR-1.7B | ~4.7GB | 52 |
| Translation | NLLB-200-distilled-600M | ~1.2GB | 200 |
| TTS | XTTS-v2 | ~1.9GB | 17 |

### TTS Supported Languages (XTTS-v2)
English, Spanish, French, German, Italian, Portuguese, **Polish**, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Japanese, Hungarian, Korean, Hindi

**Voice cloning**: Provide a 6+ second audio sample to clone any voice!

## License

Apache 2.0 (same as Qwen3-ASR model)
CC-BY-NC for NLLB-200
CPML for XTTS-v2 (Coqui Public Model License)
