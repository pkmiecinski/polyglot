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


class AudioRecorder:
    """
    Simple audio recorder with manual start/stop control.
    """
    
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        device: Optional[int] = None,
        chunk_duration: float = 0.1,
        max_duration: float = 60.0,
    ):
        import threading
        import queue
        
        self.sample_rate = sample_rate
        self.device = device if device is not None else get_default_input_device()
        self.chunk_duration = chunk_duration
        self.max_duration = max_duration
        
        self._audio_queue = queue.Queue()
        self._stop_flag = threading.Event()
        self._all_audio = []
        self._stream = None
        self._is_recording = False
        
    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio stream."""
        if not self._stop_flag.is_set():
            self._audio_queue.put(indata.copy())
    
    def start(self):
        """Start recording."""
        import queue
        
        self._stop_flag.clear()
        self._all_audio = []
        self._audio_queue = queue.Queue()
        self._is_recording = True
        
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            device=self.device,
            blocksize=chunk_samples,
            callback=self._audio_callback
        )
        self._stream.start()
        console.print("[yellow]🎤 Recording started...[/yellow]")
        
    def stop(self) -> Tuple[np.ndarray, int]:
        """Stop recording and return audio data."""
        self._stop_flag.set()
        self._is_recording = False
        
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        # Collect remaining audio from queue
        import queue
        while True:
            try:
                chunk = self._audio_queue.get_nowait()
                self._all_audio.append(chunk.flatten())
            except queue.Empty:
                break
        
        # Concatenate all audio
        if self._all_audio:
            audio_data = np.concatenate(self._all_audio)
        else:
            audio_data = np.array([], dtype=np.float32)
        
        duration = len(audio_data) / self.sample_rate
        console.print(f"[green]✓ Recorded {duration:.1f} seconds[/green]")
        
        return audio_data, self.sample_rate
    
    def get_duration(self) -> float:
        """Get current recording duration."""
        total_samples = sum(len(c) for c in self._all_audio)
        return total_samples / self.sample_rate
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording
    
    def collect_chunks(self) -> float:
        """Collect pending audio chunks and return current amplitude."""
        import queue
        
        rms = 0.0
        max_samples = int(self.max_duration * self.sample_rate)
        
        while True:
            try:
                chunk = self._audio_queue.get_nowait()
                chunk = chunk.flatten()
                self._all_audio.append(chunk)
                rms = np.sqrt(np.mean(chunk ** 2))
                
                # Check max duration
                total_samples = sum(len(c) for c in self._all_audio)
                if total_samples >= max_samples:
                    self._stop_flag.set()
                    break
            except queue.Empty:
                break
        
        return rms


# Global recorder instance for GUI
_recorder: Optional[AudioRecorder] = None


def start_recording(
    sample_rate: int = SAMPLE_RATE,
    device: Optional[int] = None,
    max_duration: float = 60.0,
) -> AudioRecorder:
    """Start a new recording session."""
    global _recorder
    _recorder = AudioRecorder(
        sample_rate=sample_rate,
        device=device,
        max_duration=max_duration,
    )
    _recorder.start()
    return _recorder


def stop_recording() -> Tuple[np.ndarray, int]:
    """Stop the current recording and return audio."""
    global _recorder
    if _recorder is None:
        return np.array([], dtype=np.float32), SAMPLE_RATE
    
    audio, sr = _recorder.stop()
    _recorder = None
    return audio, sr


def get_recorder() -> Optional[AudioRecorder]:
    """Get the current recorder instance."""
    return _recorder


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
    Legacy: Record audio continuously until silence is detected.
    For GUI use, prefer start_recording() and stop_recording().
    """
    recorder = AudioRecorder(
        sample_rate=sample_rate,
        device=device,
        chunk_duration=chunk_duration,
        max_duration=max_duration,
    )
    recorder.start()
    
    import time
    chunks_for_silence = int(silence_duration / chunk_duration)
    min_chunks = int(min_duration / chunk_duration)
    silence_chunks = 0
    total_chunks = 0
    has_speech = False
    
    console.print("[dim]Recording... (auto-stop on silence)[/dim]")
    
    while True:
        time.sleep(chunk_duration)
        rms = recorder.collect_chunks()
        total_chunks += 1
        
        if rms > silence_threshold * 2:
            has_speech = True
        
        if rms < silence_threshold:
            silence_chunks += 1
        else:
            silence_chunks = 0
        
        # Stop conditions
        if total_chunks > min_chunks and has_speech and silence_chunks >= chunks_for_silence:
            break
        
        if recorder.get_duration() >= max_duration:
            break
    
    return recorder.stop()


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
