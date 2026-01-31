#!/usr/bin/env python3
"""
Standalone TTS script that runs in the .venv-tts virtualenv.
This is called by the main application via subprocess to avoid dependency conflicts.
"""

import argparse
import json
import sys
import os

# Default speaker reference file for XTTS-v2 (will be created on first run)
DEFAULT_SPEAKER_DIR = os.path.expanduser("~/.cache/polyglot/speakers")


def main():
    parser = argparse.ArgumentParser(description="XTTS-v2 Text-to-Speech")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--language", required=True, help="Language code (e.g., en, pl, de)")
    parser.add_argument("--output", required=True, help="Output WAV file path")
    parser.add_argument("--speaker-wav", default=None, help="Path to speaker WAV for voice cloning")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, mps, cpu")
    
    args = parser.parse_args()
    
    # Import TTS here to delay loading
    from TTS.api import TTS
    import torch
    
    # Force CPU for XTTS-v2 as MPS has issues with large conv channels
    # The HiFiGAN decoder requires > 65536 channels which MPS doesn't support
    device = "cpu"
    if args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    
    print(f"TTS_INFO: Using device: {device}", file=sys.stderr)
    
    # Load model
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    
    print(f"TTS_INFO: Model loaded", file=sys.stderr)
    
    # Get speaker reference
    speaker_wav = args.speaker_wav
    
    if not speaker_wav:
        # Use a built-in speaker from the model's samples
        # XTTS-v2 requires a speaker reference, we'll use a sample
        model_path = tts.synthesizer.tts_model.config.model_dir if hasattr(tts.synthesizer.tts_model, 'config') else None
        
        # Try to use default speaker from samples directory
        samples_dir = os.path.join(os.path.dirname(tts.synthesizer.tts_model.config.model_dir or ""), "samples")
        if os.path.exists(samples_dir):
            sample_files = [f for f in os.listdir(samples_dir) if f.endswith('.wav')]
            if sample_files:
                speaker_wav = os.path.join(samples_dir, sample_files[0])
        
        # If still no speaker, create a placeholder or use internal speaker
        if not speaker_wav:
            # XTTS supports named speakers, let's try that approach
            print(f"TTS_INFO: No speaker WAV provided, using model's default", file=sys.stderr)
    
    # Synthesize - XTTS requires speaker_wav for voice cloning
    if speaker_wav and os.path.exists(speaker_wav):
        print(f"TTS_INFO: Using voice clone from: {speaker_wav}", file=sys.stderr)
        tts.tts_to_file(
            text=args.text,
            file_path=args.output,
            speaker_wav=speaker_wav,
            language=args.language,
        )
    else:
        # Use the synthesizer directly with a default speaker embedding
        # XTTS has default speakers we can use
        wav = tts.synthesizer.tts(
            text=args.text,
            language_name=args.language,
            speaker_name="Claribel Dervla",  # Default English speaker
        )
        tts.synthesizer.save_wav(wav, args.output)
    
    print(f"TTS_INFO: Audio saved to {args.output}", file=sys.stderr)
    print(json.dumps({"success": True, "output": args.output}))


if __name__ == "__main__":
    main()
