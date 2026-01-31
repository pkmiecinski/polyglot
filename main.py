#!/usr/bin/env python3
"""
Polyglot - Edge AI Translation Device PoC

Real-time speech recognition, language detection, and translation.
Powered by Qwen3-ASR and NLLB-200.
"""

import click
import sys
import os
import gc
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from audio_capture import (
    record_audio,
    record_audio_continuous,
    list_audio_devices,
    test_microphone,
    SAMPLE_RATE
)
from transcriber import Transcriber, get_supported_languages as get_asr_languages
from translator import Translator, get_supported_languages as get_translation_languages
from tts import TextToSpeech, get_supported_languages as get_tts_languages, is_language_supported as is_tts_supported

console = Console()


def print_banner():
    """Print application banner."""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                     🌐 POLYGLOT                           ║
║           Edge AI Translation Device PoC                  ║
║                                                           ║
║  ASR: Qwen3-ASR  •  Translation: NLLB-200  •  TTS: XTTS  ║
╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def print_result(text: str, language: str, translation: str = None, target_lang: str = None):
    """Print transcription and translation result in a nice format."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Label", style="bold")
    table.add_column("Value")
    
    table.add_row("🌍 Language", f"[bold green]{language}[/bold green]")
    table.add_row("📝 Transcript", f"[white]{text}[/white]")
    
    if translation:
        table.add_row("", "")  # Spacer
        table.add_row(f"🔄 {target_lang}", f"[bold cyan]{translation}[/bold cyan]")
    
    panel = Panel(
        table,
        title="[bold]Result[/bold]",
        border_style="green"
    )
    console.print(panel)


@click.command()
@click.option(
    "--model", "-m",
    default="Qwen/Qwen3-ASR-1.7B",
    help="Model name (Qwen/Qwen3-ASR-1.7B or Qwen/Qwen3-ASR-0.6B)"
)
@click.option(
    "--device", "-d",
    default="auto",
    help="Device to use (auto, cuda, cpu, mps)"
)
@click.option(
    "--duration", "-t",
    default=5.0,
    type=float,
    help="Recording duration in seconds (for fixed mode)"
)
@click.option(
    "--language", "-l",
    default=None,
    help="Force specific language (e.g., English, Chinese)"
)
@click.option(
    "--continuous", "-c",
    is_flag=True,
    help="Use continuous recording (stops on silence)"
)
@click.option(
    "--list-devices",
    is_flag=True,
    help="List available audio devices and exit"
)
@click.option(
    "--list-languages",
    is_flag=True,
    help="List supported languages and exit"
)
@click.option(
    "--test-mic",
    is_flag=True,
    help="Test microphone and exit"
)
@click.option(
    "--loop",
    is_flag=True,
    help="Keep recording in a loop"
)
@click.option(
    "--translate", "-T",
    default=None,
    help="Translate to language (e.g., English, Polish, Thai)"
)
@click.option(
    "--translation-model",
    default="facebook/nllb-200-distilled-600M",
    help="Translation model (default: NLLB-200 600M)"
)
@click.option(
    "--speak", "-S",
    is_flag=True,
    help="Speak the output using TTS (requires --translate)"
)
@click.option(
    "--voice", "-V",
    default=None,
    help="Path to voice sample WAV file for voice cloning (6+ seconds)"
)
@click.option(
    "--save-audio",
    default=None,
    help="Save spoken audio to file path"
)
def main(
    model: str,
    device: str,
    duration: float,
    language: str,
    continuous: bool,
    list_devices: bool,
    list_languages: bool,
    test_mic: bool,
    loop: bool,
    translate: str,
    translation_model: str,
    speak: bool,
    voice: str,
    save_audio: str,
):
    """
    Polyglot - Edge AI Translation Device PoC
    
    Records audio from the microphone, detects language, transcribes speech,
    and optionally translates to another language.
    
    Examples:
    
        # Basic usage with fixed 5-second recording
        python main.py
        
        # Continuous recording (stops on silence)
        python main.py --continuous
        
        # Transcribe and translate to English
        python main.py --continuous --translate English
        
        # Translate to Polish
        python main.py -c -T Polish
        
        # Use smaller/faster model
        python main.py --model Qwen/Qwen3-ASR-0.6B
        
        # Keep recording and translating in a loop
        python main.py --loop --continuous --translate English
        
        # Translate and speak the result
        python main.py -c -T English --speak
        
        # Use voice cloning (provide 6+ second sample)
        python main.py -c -T English --speak --voice my_voice.wav
        
        # Save spoken audio to file
        python main.py -c -T English --speak --save-audio output.wav
    """
    
    # Handle info commands
    if list_devices:
        list_audio_devices()
        return
    
    if list_languages:
        console.print("\n[bold]ASR Supported Languages (Qwen3-ASR):[/bold]")
        languages = get_asr_languages()
        for i, lang in enumerate(languages, 1):
            console.print(f"  {i:2}. {lang}")
        console.print("\n[dim]Plus 22 Chinese dialects[/dim]")
        
        console.print("\n[bold]Translation Supported Languages (NLLB-200):[/bold]")
        trans_languages = get_translation_languages()
        cols = 4
        for i in range(0, len(trans_languages), cols):
            row = trans_languages[i:i+cols]
            console.print("  " + "  ".join(f"{l:<15}" for l in row))
        console.print("\n[dim]200 languages total supported[/dim]")
        
        console.print("\n[bold]TTS Supported Languages (XTTS-v2):[/bold]")
        tts_languages = get_tts_languages()
        cols = 4
        for i in range(0, len(tts_languages), cols):
            row = tts_languages[i:i+cols]
            console.print("  " + "  ".join(f"{l:<15}" for l in row))
        console.print("\n[dim]Voice cloning supported with 6+ second audio sample[/dim]")
        return
    
    if test_mic:
        test_microphone()
        return
    
    # Validate TTS options
    if speak and not translate:
        console.print("[red]Error: --speak requires --translate to specify target language[/red]")
        return
    
    if voice and not speak:
        console.print("[yellow]Warning: --voice has no effect without --speak[/yellow]")
    
    if speak and translate and not is_tts_supported(translate):
        console.print(f"[red]Error: TTS does not support language '{translate}'[/red]")
        console.print(f"[dim]Supported: {', '.join(get_tts_languages())}[/dim]")
        return
    
    # Print banner
    print_banner()
    
    # Initialize transcriber (lazy loading)
    console.print(f"[dim]ASR Model: {model}[/dim]")
    if translate:
        console.print(f"[dim]Translation Model: {translation_model}[/dim]")
        console.print(f"[dim]Target Language: {translate}[/dim]")
    if speak:
        console.print(f"[dim]TTS: XTTS-v2{' (voice clone)' if voice else ''}[/dim]")
    console.print(f"[dim]Device: {device}[/dim]")
    
    transcriber = None
    translator = None
    tts_engine = None
    
    try:
        transcriber = Transcriber(
            model_name=model,
            device=device,
        )
        
        # Pre-load ASR model
        transcriber.load_model()
        
        # Initialize translator if needed
        if translate:
            translator = Translator(
                model_name=translation_model,
                device=device,
            )
            translator.load_model()
        
        # Initialize TTS if needed
        if speak:
            tts_engine = TextToSpeech(device=device)
            tts_engine.load_model()
        while True:
            # Record audio
            if continuous:
                audio, sr = record_audio_continuous()
            else:
                audio, sr = record_audio(duration=duration)
            
            # Check if we got any meaningful audio
            if len(audio) < sr * 0.5:  # Less than 0.5 seconds
                console.print("[yellow]⚠ Audio too short, skipping...[/yellow]")
                if not loop:
                    break
                continue
            
            # Transcribe
            try:
                result = transcriber.transcribe(
                    audio=audio,
                    sample_rate=sr,
                    language=language
                )
                
                # Translate if requested
                translation_text = None
                if translator and result.text.strip():
                    try:
                        trans_result = translator.translate(
                            text=result.text,
                            source_language=result.language,
                            target_language=translate
                        )
                        translation_text = trans_result.translated_text
                    except Exception as e:
                        console.print(f"[yellow]⚠ Translation error: {e}[/yellow]")
                
                # Display result
                print_result(
                    result.text, 
                    result.language,
                    translation=translation_text,
                    target_lang=translate
                )
                
                # Speak the result if TTS is enabled
                if tts_engine and translation_text:
                    try:
                        if save_audio:
                            tts_result = tts_engine.synthesize(
                                text=translation_text,
                                language=translate,
                                speaker_wav=voice,
                                output_path=save_audio,
                            )
                            console.print(f"[green]✓ Audio saved to: {save_audio}[/green]")
                            # Also play it
                            import sounddevice as sd
                            sd.play(tts_result.audio, tts_result.sample_rate)
                            sd.wait()
                        else:
                            tts_engine.speak(
                                text=translation_text,
                                language=translate,
                                speaker_wav=voice,
                            )
                    except Exception as e:
                        console.print(f"[yellow]⚠ TTS error: {e}[/yellow]")
                
            except Exception as e:
                console.print(f"[red]Error during transcription: {e}[/red]")
            
            # Exit or continue
            if not loop:
                break
            
            console.print("\n" + "─" * 50 + "\n")
            console.print("[dim]Press Ctrl+C to stop[/dim]")
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    finally:
        # Cleanup models to release resources
        console.print("[dim]Cleaning up...[/dim]")
        if transcriber:
            del transcriber
        if translator:
            del translator
        if tts_engine:
            del tts_engine
        gc.collect()
    
    console.print("\n[bold]Goodbye! 👋[/bold]")
    
    # Force exit to kill any lingering threads from transformers/torch
    os._exit(0)


if __name__ == "__main__":
    main()
