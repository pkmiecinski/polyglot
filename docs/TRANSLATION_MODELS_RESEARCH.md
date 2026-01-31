# Edge AI Translation Models Research

## Requirements
- Support for **30+ major languages** including **Polish** and **Thai**
- Suitable for **edge/embedded devices**
- Balance between model size, speed, and quality

---

## 🏆 Recommended Models Comparison

| Model | Size | Languages | Polish | Thai | Edge Suitable | License |
|-------|------|-----------|--------|------|---------------|---------|
| **NLLB-200-distilled-600M** ⭐ | ~1.2GB | 200 | ✅ | ✅ | ✅ Best | CC-BY-NC |
| **M2M100-418M** | ~1.8GB | 100 | ✅ | ✅ | ✅ Good | MIT |
| **mBART-50** | ~2.4GB | 50 | ✅ | ✅ | ⚠️ Medium | MIT |
| NLLB-200-distilled-1.3B | ~2.6GB | 200 | ✅ | ✅ | ⚠️ Medium | CC-BY-NC |
| MADLAD-400-3B | ~6GB | 400+ | ✅ | ✅ | ❌ Too large | Apache 2.0 |
| SeamlessM4T-Medium | ~4.6GB | 101 speech | ✅ | ✅ | ❌ Too large | CC-BY-NC |

---

## 🥇 Top Recommendation: NLLB-200-distilled-600M

### Why It's Best for Edge:
1. **Smallest footprint** (~600M params, ~1.2GB disk)
2. **200 languages** including all 30+ you need
3. **State-of-the-art quality** for its size (Meta AI research)
4. **Optimized for efficiency** - distilled from larger model
5. **CTranslate2 compatible** for 4x smaller + 3x faster inference

### Language Coverage (includes):
- **Polish (pol_Latn)** ✅
- **Thai (tha_Thai)** ✅
- English, German, French, Spanish, Portuguese, Italian
- Chinese (Simplified/Traditional), Japanese, Korean
- Russian, Ukrainian, Arabic, Hindi, Vietnamese
- Indonesian, Malay, Turkish, Dutch, Swedish
- And 180+ more languages

### Usage Example:
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

# Translate Polish to English
tokenizer.src_lang = "pol_Latn"
inputs = tokenizer("Cześć, jak się masz?", return_tensors="pt")
translated = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"))
print(tokenizer.decode(translated[0], skip_special_tokens=True))
# => "Hello, how are you?"

# Translate Thai to English  
tokenizer.src_lang = "tha_Thai"
inputs = tokenizer("สวัสดีครับ", return_tensors="pt")
translated = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"))
```

---

## 🚀 Edge Optimization with CTranslate2

CTranslate2 is crucial for edge deployment - it provides **4x smaller models** and **3x faster inference**.

### Conversion:
```bash
pip install ctranslate2

# Convert NLLB model for edge deployment
ct2-transformers-converter \
    --model facebook/nllb-200-distilled-600M \
    --output_dir nllb-600m-ct2 \
    --quantization int8  # Options: int8, int16, float16
```

### Optimized Sizes After Quantization:

| Model | Original | INT8 | Memory Savings |
|-------|----------|------|----------------|
| NLLB-600M | ~1.2GB | ~300MB | 75% |
| M2M100-418M | ~1.8GB | ~450MB | 75% |

### CTranslate2 Usage:
```python
import ctranslate2

translator = ctranslate2.Translator("nllb-600m-ct2", device="cpu")  # or "cuda"

# Polish to English
source = tokenizer.convert_ids_to_tokens(tokenizer.encode("Cześć!"))
target_prefix = ["eng_Latn"]
results = translator.translate_batch([source], target_prefix=[target_prefix])
translation = tokenizer.decode(tokenizer.convert_tokens_to_ids(results[0].hypotheses[0]))
```

---

## 🥈 Alternative: M2M100-418M

### Pros:
- **MIT License** (commercial friendly)
- Good quality for 100 languages
- Mature, well-tested

### Cons:
- Slightly larger than NLLB-600M
- Fewer languages (100 vs 200)
- Older architecture

### Usage:
```python
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# Polish to English
tokenizer.src_lang = "pl"
inputs = tokenizer("Cześć, jak się masz?", return_tensors="pt")
translated = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id("en"))
```

---

## 📱 Edge Deployment Considerations

### Memory Requirements:

| Device Type | Recommended Model | RAM Needed |
|-------------|-------------------|------------|
| Raspberry Pi 4 (4GB) | NLLB-600M INT8 | ~1GB |
| Jetson Nano | NLLB-600M INT8/FP16 | ~1-2GB |
| Mobile (iOS/Android) | NLLB-600M INT8 | ~500MB |
| Desktop Edge | NLLB-600M FP16 | ~2GB |

### Inference Speed (approximate):

| Device | Model | Tokens/sec |
|--------|-------|------------|
| CPU (4 cores) | NLLB-600M INT8 | ~50-100 |
| GPU (RTX 3060) | NLLB-600M FP16 | ~500-1000 |
| Apple M1 | NLLB-600M | ~200-400 |

---

## 🌐 Full Language List (30 Major + Polish & Thai)

NLLB language codes for your 30+ target languages:

| Language | NLLB Code |
|----------|-----------|
| English | eng_Latn |
| **Polish** | **pol_Latn** |
| **Thai** | **tha_Thai** |
| German | deu_Latn |
| French | fra_Latn |
| Spanish | spa_Latn |
| Portuguese | por_Latn |
| Italian | ita_Latn |
| Dutch | nld_Latn |
| Russian | rus_Cyrl |
| Ukrainian | ukr_Cyrl |
| Chinese (Simplified) | zho_Hans |
| Chinese (Traditional) | zho_Hant |
| Japanese | jpn_Jpan |
| Korean | kor_Hang |
| Arabic | arb_Arab |
| Hindi | hin_Deva |
| Vietnamese | vie_Latn |
| Indonesian | ind_Latn |
| Malay | zsm_Latn |
| Turkish | tur_Latn |
| Swedish | swe_Latn |
| Danish | dan_Latn |
| Norwegian | nob_Latn |
| Finnish | fin_Latn |
| Czech | ces_Latn |
| Hungarian | hun_Latn |
| Romanian | ron_Latn |
| Greek | ell_Grek |
| Hebrew | heb_Hebr |

---

## 🔧 Implementation Plan for Polyglot

### Phase 1: Add NLLB Translation
```
polyglot/
├── main.py              # Main app
├── audio_capture.py     # Mic input
├── transcriber.py       # Qwen3-ASR
├── translator.py        # NEW: NLLB translation
└── requirements.txt
```

### Phase 2: Optimize for Edge
1. Convert NLLB to CTranslate2 INT8
2. Implement lazy loading
3. Add translation caching
4. Support offline mode

### Phase 3: Full Pipeline
```
🎤 Audio → [Qwen3-ASR] → Text + Language → [NLLB] → Translated Text
```

---

## 📚 References

- [NLLB-200 Paper](https://arxiv.org/abs/2207.04672)
- [NLLB-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [M2M100-418M](https://huggingface.co/facebook/m2m100_418M)
- [CTranslate2](https://github.com/OpenNMT/CTranslate2)
- [NLLB Language Codes](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200)
