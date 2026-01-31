"""
Audio capture utilities for microphone input.
"""

import numpy as np
import sounddevice as sd
from typing import Optional, Tuple
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Qwen3-ASR expects 16kHz audio
SAMPLE_RATE = 16000


def list_audio_devices() -> None:
    """List available audio input devices."""
    console.print("\n[bold]Available Audio Devices:[/bold]")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            marker = "→ " if i == sd.default.device[0] else "  "
            console.print(f"{marker}[{i}] {device['name']} (inputs: {device['max_input_channels']})")


def get_default_input_device() -> int:
    """Get the default input device index."""
    return sd.default.device[0]


def record_audio(
    duration: float = 5.0,
    sample_rate: int = SAMPLE_RATE,
    device: Optional[int] = None,
    channels: int = 1
) -> Tuple[np.ndarray, int]:
    """
    Record audio from the microphone.
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate in Hz (default: 16000 for Qwen3-ASR)
        device: Audio device index (None for default)
        channels: Number of audio channels (1 for mono)
    
    Returns:
        Tuple of (audio_data as numpy array, sample_rate)
    """
    if device is None:
        device = get_default_input_device()
    
    console.print(f"\n[yellow]🎤 Recording for {duration} seconds...[/yellow]")
    console.print("[dim]Speak now![/dim]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Recording...", total=None)
        
        # Record audio
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype=np.float32,
            device=device
        )
        sd.wait()  # Wait until recording is finished
        
        progress.update(task, description="Recording complete!")
    
    # Convert to mono if needed and flatten
    if channels > 1:
        audio_data = np.mean(audio_data, axis=1)
    else:
        audio_data = audio_data.flatten()
    
    console.print("[green]✓ Recording captured[/green]")
    
    return audio_data, sample_rate


def record_audio_continuous(
    chunk_duration: float = 0.1,
    sample_rate: int = SAMPLE_RATE,
    device: Optional[int] = None,
    silence_threshold: float = 0.005,
    silence_duration: float = 2.0,
    max_duration: float = 30.0,
    min_duration: float = 1.0
) -> Tuple[np.ndarray, int]:
    """
    Record audio continuously using streaming (no gaps) until silence is detected.
    
    Args:
        chunk_duration: Duration of each analysis window in seconds
        sample_rate: Sample rate in Hz
        device: Audio device index
        silence_threshold: RMS threshold below which audio is considered silence
        silence_duration: Duration of silence to stop recording
        max_duration: Maximum recording duration
        min_duration: Minimum recording duration before silence detection kicks in
    
    Returns:
        Tuple of (audio_data as numpy array, sample_rate)
    """
    import threading
    import queue
    
    if device is None:
        device = get_default_input_device()
    
    console.print("\n[yellow]🎤 Listening... (will stop after silence or max duration)[/yellow]")
    console.print("[dim]Speak now![/dim]\n")
    
    # Use a queue to collect audio data from the callback
    audio_queue = queue.Queue()
    stop_flag = threading.Event()
    
    chunk_samples = int(chunk_duration * sample_rate)
    chunks_for_silence = int(silence_duration / chunk_duration)
    min_chunks = int(min_duration / chunk_duration)
    max_samples = int(max_duration * sample_rate)
    
    all_audio = []
    silence_chunks = 0
    total_chunks = 0
    has_speech = False
    
    def audio_callback(indata, frames, time, status):
        """Callback for continuous audio stream - no gaps!"""
        if status:
            console.print(f"[dim]Audio status: {status}[/dim]")
        audio_queue.put(indata.copy())
    
    console.print("[dim]Starting stream...[/dim]")
    
    # Use InputStream for continuous, gap-free recording
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32,
        device=device,
        blocksize=chunk_samples,
        callback=audio_callback
    ):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Listening...", total=None)
            
            while not stop_flag.is_set():
                try:
                    # Get audio chunk from queue (with timeout)
                    chunk = audio_queue.get(timeout=0.5)
                    chunk = chunk.flatten()
                    all_audio.append(chunk)
                    total_chunks += 1
                    
                    # Calculate RMS for this chunk
                    rms = np.sqrt(np.mean(chunk ** 2))
                    
                    # Track if we've heard any speech
                    if rms > silence_threshold * 2:
                        has_speech = True
                    
                    # Update silence detection
                    if rms < silence_threshold:
                        silence_chunks += 1
                    else:
                        silence_chunks = 0
                    
                    # Calculate total duration
                    total_samples = sum(len(c) for c in all_audio)
                    duration = total_samples / sample_rate
                    
                    # Update progress display
                    if silence_chunks > 0:
                        progress.update(task, description=f"Recording {duration:.1f}s - silence ({silence_chunks}/{chunks_for_silence})...")
                    else:
                        progress.update(task, description=f"Recording {duration:.1f}s...")
                    
                    # Stop conditions
                    if total_samples >= max_samples:
                        progress.update(task, description=f"Stopped (max {max_duration}s reached)")
                        break
                    
                    # Only stop on silence after minimum duration AND after hearing speech
                    if (total_chunks > min_chunks and 
                        has_speech and 
                        silence_chunks >= chunks_for_silence):
                        progress.update(task, description="Stopped (silence detected)")
                        break
                        
                except queue.Empty:
                    continue
    
    # Concatenate all audio (continuous, no gaps!)
    if all_audio:
        audio_data = np.concatenate(all_audio)
    else:
        audio_data = np.array([], dtype=np.float32)
    
    # Trim trailing silence (keep a little bit)
    if silence_chunks > 0 and len(all_audio) > silence_chunks:
        trim_samples = int((silence_chunks - 2) * chunk_samples)
        if trim_samples > 0 and trim_samples < len(audio_data):
            audio_data = audio_data[:-trim_samples]
    
    duration_recorded = len(audio_data) / sample_rate
    console.print(f"[green]✓ Recorded {duration_recorded:.1f} seconds of audio[/green]")
    
    return audio_data, sample_rate


def test_microphone(duration: float = 2.0) -> bool:
    """
    Test if the microphone is working.
    
    Args:
        duration: Test duration in seconds
    
    Returns:
        True if microphone is working
    """
    try:
        console.print("[dim]Testing microphone...[/dim]")
        audio, sr = record_audio(duration=duration)
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms < 0.001:
            console.print("[yellow]⚠ Very low audio level detected. Check your microphone.[/yellow]")
            return False
        
        console.print(f"[green]✓ Microphone working (RMS level: {rms:.4f})[/green]")
        return True
    except Exception as e:
        console.print(f"[red]✗ Microphone error: {e}[/red]")
        return False
