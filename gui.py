#!/usr/bin/env python3
"""
Polyglot GUI - Modern UI for Edge AI Translation

A PyQt6-based interface for real-time speech translation with:
- Pre-loaded models for instant response
- Live recording with visual feedback
- Timestamped logging
- Voice cloning support
"""

import sys
import os

# Import model_manager FIRST to set up cache directories before any ML imports
import model_manager

# Fix Qt/PyTorch threading issues on macOS
os.environ["QT_MAC_WANTS_LAYER"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import gc
import time
import tempfile
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable
from enum import Enum, auto

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QComboBox, QGroupBox,
    QProgressBar, QSplitter, QFrame, QCheckBox, QSlider,
    QSpacerItem, QSizePolicy, QStyle, QStyleFactory
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation,
    QEasingCurve, QSize, QObject
)
from PyQt6.QtGui import (
    QFont, QColor, QPalette, QIcon, QPainter, QPen,
    QLinearGradient, QBrush, QAction
)

# Project imports
from audio_capture import record_audio_continuous, SAMPLE_RATE
from transcriber import Transcriber
from translator import Translator
from tts_fish import FishSpeechTTS, get_supported_languages as get_tts_languages, is_language_supported


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & STYLING
# ═══════════════════════════════════════════════════════════════════════════════

APP_NAME = "Polyglot"
APP_VERSION = "1.0.0"

COLORS = {
    'bg_dark': '#1a1a2e',
    'bg_medium': '#16213e',
    'bg_light': '#0f3460',
    'accent': '#e94560',
    'accent_light': '#ff6b8a',
    'success': '#00d9a0',
    'warning': '#ffc107',
    'error': '#ff5252',
    'text': '#ffffff',
    'text_dim': '#8892b0',
    'border': '#233554',
}

STYLESHEET = f"""
QMainWindow {{
    background-color: {COLORS['bg_dark']};
}}

QWidget {{
    color: {COLORS['text']};
    font-family: 'SF Pro Display', 'Segoe UI', 'Helvetica Neue', sans-serif;
}}

QGroupBox {{
    background-color: {COLORS['bg_medium']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    margin-top: 16px;
    padding: 16px;
    font-weight: bold;
    font-size: 13px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 12px;
    color: {COLORS['text_dim']};
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

QTextEdit {{
    background-color: {COLORS['bg_dark']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', 'Monaco', monospace;
    font-size: 12px;
    line-height: 1.5;
}}

QComboBox {{
    background-color: {COLORS['bg_light']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 8px 12px;
    min-width: 150px;
    font-size: 13px;
}}

QComboBox:hover {{
    border-color: {COLORS['accent']};
}}

QComboBox::drop-down {{
    border: none;
    padding-right: 12px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS['bg_medium']};
    border: 1px solid {COLORS['border']};
    selection-background-color: {COLORS['accent']};
}}

QProgressBar {{
    background-color: {COLORS['bg_dark']};
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
}}

QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {COLORS['accent']}, stop:1 {COLORS['accent_light']});
    border-radius: 4px;
}}

QCheckBox {{
    spacing: 8px;
    font-size: 13px;
}}

QCheckBox::indicator {{
    width: 20px;
    height: 20px;
    border-radius: 4px;
    border: 2px solid {COLORS['border']};
    background-color: {COLORS['bg_dark']};
}}

QCheckBox::indicator:checked {{
    background-color: {COLORS['accent']};
    border-color: {COLORS['accent']};
}}

QLabel {{
    font-size: 13px;
}}

QSplitter::handle {{
    background-color: {COLORS['border']};
}}

QSplitter::handle:horizontal {{
    width: 2px;
}}

QSplitter::handle:vertical {{
    height: 2px;
}}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS & DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class AppState(Enum):
    INITIALIZING = auto()
    LOADING_MODELS = auto()
    READY = auto()
    RECORDING = auto()
    PROCESSING = auto()
    SPEAKING = auto()
    ERROR = auto()


@dataclass
class TranslationJob:
    """Represents a single translation job."""
    audio: np.ndarray
    sample_rate: int
    target_language: str
    clone_voice: bool = False
    voice_sample: Optional[str] = None


@dataclass
class TranslationResult:
    """Result from the translation pipeline."""
    source_text: str
    source_language: str
    translated_text: str
    target_language: str
    audio_path: Optional[str] = None
    audio_data: Optional[np.ndarray] = None
    audio_sr: Optional[int] = None
    duration_ms: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# WORKER THREADS
# ═══════════════════════════════════════════════════════════════════════════════

class ModelLoaderThread(QThread):
    """Background thread for loading AI models."""
    
    progress = pyqtSignal(str, int)  # message, percentage
    model_loaded = pyqtSignal(str)   # model name
    finished = pyqtSignal(bool, str) # success, message
    log = pyqtSignal(str, str)       # message, level
    
    def __init__(self, device: str = "auto", verbose: bool = False):
        super().__init__()
        self.device = device
        self.verbose = verbose
        self.transcriber = None
        self.translator = None
        self.tts = None
        
    def run(self):
        try:
            # Load ASR model (Qwen3-ASR)
            self.log.emit("Initializing ASR engine (Qwen3-ASR-1.7B)...", "info")
            self.progress.emit("Loading ASR model...", 10)
            
            self.transcriber = Transcriber(
                model_name="Qwen/Qwen3-ASR-1.7B",
                device=self.device,
            )
            self.transcriber.load_model()
            self.model_loaded.emit("ASR")
            self.log.emit("✓ ASR model loaded successfully", "success")
            self.progress.emit("ASR model loaded", 33)
            
            # Load Translation model (NLLB-200)
            self.log.emit("Initializing Translation engine (NLLB-200-600M)...", "info")
            self.progress.emit("Loading translation model...", 40)
            
            self.translator = Translator(
                model_name="facebook/nllb-200-distilled-600M",
                device=self.device,
            )
            self.translator.load_model()
            self.model_loaded.emit("Translation")
            self.log.emit("✓ Translation model loaded successfully", "success")
            self.progress.emit("Translation model loaded", 66)
            
            # Load TTS server (Fish Speech S1-mini) - powerful multilingual TTS with Thai support
            self.log.emit("Starting TTS server (Fish Speech S1-mini)...", "info")
            self.log.emit("⏳ TTS model loading (~30s first time, then instant)", "info")
            self.progress.emit("Starting TTS server...", 70)
            
            self.tts = FishSpeechTTS(device=self.device)
            self.tts.load_model()  # This starts the persistent server
            self.model_loaded.emit("TTS")
            self.log.emit("✓ TTS server running (model pre-loaded)", "success")
            self.progress.emit("TTS server ready", 100)
            
            self.log.emit("All models loaded! Ready for translation.", "success")
            self.finished.emit(True, "All models loaded successfully")
            
        except Exception as e:
            self.log.emit(f"✗ Error loading models: {str(e)}", "error")
            self.finished.emit(False, str(e))


class ModelDownloadThread(QThread):
    """Background thread for downloading AI models."""
    
    progress = pyqtSignal(str, int, int)  # model_name, downloaded_mb, total_mb
    model_finished = pyqtSignal(str, bool, str)  # model_name, success, message
    finished = pyqtSignal(bool, str)  # success, message
    log = pyqtSignal(str, str)  # message, level
    
    def __init__(self, models_to_download: list = None, verbose: bool = False):
        super().__init__()
        self.models_to_download = models_to_download or ["asr", "translation", "tts_fish"]
        self.verbose = verbose
        
    def _progress_callback(self, model_name: str, downloaded: int, total: int):
        """Callback for download progress."""
        downloaded_mb = downloaded // (1024 * 1024)
        total_mb = total // (1024 * 1024) if total > 0 else 0
        self.progress.emit(model_name, downloaded_mb, total_mb)
        
        if self.verbose and total > 0:
            pct = (downloaded / total) * 100
            self.log.emit(f"  [{model_name}] {downloaded_mb}MB / {total_mb}MB ({pct:.1f}%)", "debug")
    
    def run(self):
        try:
            self.log.emit("Starting model download...", "info")
            self.log.emit(f"Models will be stored in: {model_manager.MODELS_DIR}", "info")
            
            all_success = True
            
            for model_key in self.models_to_download:
                model_info = model_manager.MODELS.get(model_key)
                if not model_info:
                    continue
                    
                self.log.emit(f"", "debug")
                self.log.emit(f"📥 Downloading {model_info.name}...", "info")
                self.log.emit(f"   Size: ~{model_info.size_gb}GB", "debug")
                
                try:
                    success = model_manager.download_model(
                        model_key,
                        progress_callback=lambda d, t, m=model_key: self._progress_callback(m, d, t)
                    )
                    
                    if success:
                        self.log.emit(f"✓ {model_info.name} downloaded successfully", "success")
                        self.model_finished.emit(model_key, True, "Downloaded")
                    else:
                        self.log.emit(f"✗ {model_info.name} download failed", "error")
                        self.model_finished.emit(model_key, False, "Download failed")
                        all_success = False
                        
                except Exception as e:
                    self.log.emit(f"✗ Error downloading {model_info.name}: {str(e)}", "error")
                    self.model_finished.emit(model_key, False, str(e))
                    all_success = False
            
            if all_success:
                self.log.emit("", "debug")
                self.log.emit("✓ All models downloaded successfully!", "success")
                self.finished.emit(True, "All models downloaded")
            else:
                self.finished.emit(False, "Some models failed to download")
                
        except Exception as e:
            self.log.emit(f"✗ Download error: {str(e)}", "error")
            self.finished.emit(False, str(e))


class RecordingThread(QThread):
    """Thread for audio recording."""
    
    started = pyqtSignal()
    progress = pyqtSignal(float, float)  # duration, amplitude
    finished = pyqtSignal(np.ndarray, int)  # audio, sample_rate
    error = pyqtSignal(str)
    log = pyqtSignal(str, str)
    
    def __init__(self):
        super().__init__()
        self._stop_requested = False
        
    def run(self):
        try:
            self.started.emit()
            self.log.emit("🎤 Recording started... speak now!", "info")
            
            # Use continuous recording with silence detection
            audio, sr = record_audio_continuous(
                silence_threshold=0.008,
                silence_duration=1.5,
                max_duration=30.0,
                min_duration=1.0,
            )
            
            duration = len(audio) / sr
            self.log.emit(f"Recording captured: {duration:.1f}s", "info")
            self.finished.emit(audio, sr)
            
        except Exception as e:
            self.log.emit(f"Recording error: {str(e)}", "error")
            self.error.emit(str(e))
    
    def stop(self):
        self._stop_requested = True


class TranslationThread(QThread):
    """Thread for running the translation pipeline."""
    
    progress = pyqtSignal(str)  # stage name
    finished = pyqtSignal(TranslationResult)
    error = pyqtSignal(str)
    log = pyqtSignal(str, str)
    
    def __init__(
        self,
        transcriber: Transcriber,
        translator: Translator,
        tts: FishSpeechTTS,
        job: TranslationJob,
    ):
        super().__init__()
        self.transcriber = transcriber
        self.translator = translator
        self.tts = tts
        self.job = job
        
    def run(self):
        start_time = time.time()
        
        try:
            # Step 1: Transcribe
            self.progress.emit("Transcribing...")
            self.log.emit("🔊 Transcribing audio...", "info")
            
            asr_start = time.time()
            asr_result = self.transcriber.transcribe(
                audio=self.job.audio,
                sample_rate=self.job.sample_rate,
            )
            asr_time = (time.time() - asr_start) * 1000
            
            self.log.emit(
                f"Detected: [{asr_result.language}] \"{asr_result.text}\" ({asr_time:.0f}ms)",
                "info"
            )
            
            if not asr_result.text.strip():
                self.log.emit("No speech detected in audio", "warning")
                self.error.emit("No speech detected")
                return
            
            # Step 2: Translate
            self.progress.emit("Translating...")
            self.log.emit(f"🌐 Translating to {self.job.target_language}...", "info")
            
            trans_start = time.time()
            trans_result = self.translator.translate(
                text=asr_result.text,
                source_language=asr_result.language,
                target_language=self.job.target_language,
            )
            trans_time = (time.time() - trans_start) * 1000
            
            self.log.emit(
                f"Translation: \"{trans_result.translated_text}\" ({trans_time:.0f}ms)",
                "info"
            )
            
            # Step 3: Synthesize speech
            self.progress.emit("Synthesizing speech...")
            self.log.emit("🔈 Generating speech...", "info")
            
            tts_start = time.time()
            
            # Prepare output path
            output_dir = Path(__file__).parent / "output"
            output_dir.mkdir(exist_ok=True)
            output_path = str(output_dir / "last_output.wav")
            
            # Determine voice sample
            voice_sample = self.job.voice_sample
            if self.job.clone_voice and voice_sample is None:
                # Create voice sample from input audio
                if len(self.job.audio) >= self.job.sample_rate * 3:
                    import scipy.io.wavfile as wav
                    voice_sample = tempfile.mktemp(suffix="_voice_clone.wav")
                    audio_int16 = (self.job.audio * 32767).astype('int16')
                    wav.write(voice_sample, self.job.sample_rate, audio_int16)
                    self.log.emit("Using your voice for cloning", "info")
            
            tts_result = self.tts.synthesize(
                text=trans_result.translated_text,
                language=self.job.target_language,
                speaker_wav=voice_sample,
                output_path=output_path,
            )
            tts_time = (time.time() - tts_start) * 1000
            
            self.log.emit(f"Speech synthesized ({tts_time:.0f}ms)", "success")
            
            # Clean up temp voice sample
            if self.job.clone_voice and voice_sample and voice_sample != self.job.voice_sample:
                try:
                    os.remove(voice_sample)
                except:
                    pass
            
            total_time = (time.time() - start_time) * 1000
            
            result = TranslationResult(
                source_text=asr_result.text,
                source_language=asr_result.language,
                translated_text=trans_result.translated_text,
                target_language=self.job.target_language,
                audio_path=output_path,
                audio_data=tts_result.audio,
                audio_sr=tts_result.sample_rate,
                duration_ms=int(total_time),
            )
            
            self.log.emit(
                f"✓ Pipeline complete in {total_time:.0f}ms",
                "success"
            )
            self.finished.emit(result)
            
        except Exception as e:
            self.log.emit(f"Translation error: {str(e)}", "error")
            self.error.emit(str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

class RecordButton(QPushButton):
    """Animated recording button with pulsing effect."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(120, 120)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._recording = False
        self._pulse_opacity = 0.0
        
        # Pulse animation
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._update_pulse)
        self._pulse_phase = 0
        
        self._update_style()
        
    def _update_style(self):
        if self._recording:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['error']};
                    border: 4px solid {COLORS['error']};
                    border-radius: 60px;
                    font-size: 14px;
                    font-weight: bold;
                    color: white;
                }}
            """)
            self.setText("■ STOP")
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['accent']};
                    border: 4px solid {COLORS['accent']};
                    border-radius: 60px;
                    font-size: 14px;
                    font-weight: bold;
                    color: white;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['accent_light']};
                    border-color: {COLORS['accent_light']};
                }}
                QPushButton:pressed {{
                    background-color: #c83050;
                }}
            """)
            self.setText("● REC")
    
    def set_recording(self, recording: bool):
        self._recording = recording
        self._update_style()
        
        if recording:
            self._pulse_timer.start(50)
        else:
            self._pulse_timer.stop()
            self._pulse_phase = 0
            
    def _update_pulse(self):
        self._pulse_phase += 0.15
        self._pulse_opacity = (1 + np.sin(self._pulse_phase)) / 2
        self.update()
        
    def paintEvent(self, event):
        super().paintEvent(event)
        
        if self._recording:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Draw pulsing ring
            color = QColor(COLORS['error'])
            color.setAlphaF(self._pulse_opacity * 0.5)
            
            pen = QPen(color)
            pen.setWidth(3)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            
            margin = int(8 + self._pulse_opacity * 8)
            painter.drawEllipse(margin, margin, 
                               self.width() - 2*margin, 
                               self.height() - 2*margin)


class LogWidget(QTextEdit):
    """Timestamped log display with color-coded messages."""
    
    LEVEL_COLORS = {
        'info': COLORS['text'],
        'success': COLORS['success'],
        'warning': COLORS['warning'],
        'error': COLORS['error'],
        'debug': COLORS['text_dim'],
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMinimumHeight(200)
        
    def log(self, message: str, level: str = "info"):
        """Add a timestamped log message."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        color = self.LEVEL_COLORS.get(level, COLORS['text'])
        
        # Format the HTML
        html = f'<span style="color: {COLORS["text_dim"]}">[{timestamp}]</span> '
        html += f'<span style="color: {color}">{message}</span>'
        
        self.append(html)
        
        # Auto-scroll to bottom
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


class StatusIndicator(QWidget):
    """Visual status indicator with icon and text."""
    
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        self._indicator = QLabel("○")
        self._indicator.setFixedWidth(20)
        self._indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self._label = QLabel(label)
        self._label.setStyleSheet(f"color: {COLORS['text_dim']};")
        
        layout.addWidget(self._indicator)
        layout.addWidget(self._label)
        layout.addStretch()
        
        self.set_status("idle")
        
    def set_status(self, status: str):
        """Set status: idle, loading, ready, error."""
        styles = {
            'idle': (f"color: {COLORS['text_dim']};", "○"),
            'loading': (f"color: {COLORS['warning']};", "◐"),
            'ready': (f"color: {COLORS['success']};", "●"),
            'error': (f"color: {COLORS['error']};", "✗"),
        }
        style, icon = styles.get(status, styles['idle'])
        self._indicator.setStyleSheet(style + " font-size: 16px;")
        self._indicator.setText(icon)


class ResultCard(QFrame):
    """Card displaying translation results."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_medium']};
                border: 1px solid {COLORS['border']};
                border-radius: 12px;
                padding: 16px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # Source section
        self._source_lang = QLabel("Source Language")
        self._source_lang.setStyleSheet(f"""
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: {COLORS['text_dim']};
        """)
        self._source_text = QLabel("—")
        self._source_text.setStyleSheet("font-size: 16px;")
        self._source_text.setWordWrap(True)
        
        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f"background-color: {COLORS['border']};")
        
        # Translation section
        self._target_lang = QLabel("Translation")
        self._target_lang.setStyleSheet(f"""
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: {COLORS['text_dim']};
        """)
        self._target_text = QLabel("—")
        self._target_text.setStyleSheet(f"""
            font-size: 18px;
            font-weight: bold;
            color: {COLORS['accent_light']};
        """)
        self._target_text.setWordWrap(True)
        
        # Duration
        self._duration = QLabel("")
        self._duration.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        
        layout.addWidget(self._source_lang)
        layout.addWidget(self._source_text)
        layout.addWidget(divider)
        layout.addWidget(self._target_lang)
        layout.addWidget(self._target_text)
        layout.addWidget(self._duration)
        
    def set_result(self, result: TranslationResult):
        """Display translation result."""
        self._source_lang.setText(f"Detected: {result.source_language}")
        self._source_text.setText(result.source_text)
        self._target_lang.setText(f"→ {result.target_language}")
        self._target_text.setText(result.translated_text)
        self._duration.setText(f"Total time: {result.duration_ms}ms")
        
    def clear(self):
        """Clear the result display."""
        self._source_lang.setText("Source Language")
        self._source_text.setText("—")
        self._target_lang.setText("Translation")
        self._target_text.setText("—")
        self._duration.setText("")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class PolyglotWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self._state = AppState.INITIALIZING
        self._transcriber = None
        self._translator = None
        self._tts = None
        self._verbose = False  # Verbose logging flag
        
        self._recording_thread = None
        self._translation_thread = None
        self._download_thread = None
        
        self._setup_ui()
        self._check_models_and_start()
        
    def _setup_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle(f"{APP_NAME} - Edge AI Translator")
        self.setMinimumSize(900, 700)
        self.resize(1000, 800)
        
        # Apply stylesheet
        self.setStyleSheet(STYLESHEET)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)
        
        # Left panel (controls)
        left_panel = self._create_left_panel()
        
        # Right panel (log)
        right_panel = self._create_right_panel()
        
        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 400])
        
        main_layout.addWidget(splitter)
        
    def _create_left_panel(self) -> QWidget:
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        
        # Header
        header = QLabel(f"🌐 {APP_NAME}")
        header.setStyleSheet(f"""
            font-size: 28px;
            font-weight: bold;
            color: {COLORS['text']};
            padding: 8px 0;
        """)
        
        subtitle = QLabel("Edge AI Translation Device")
        subtitle.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 14px;")
        
        # Model status group
        status_group = QGroupBox("MODEL STATUS")
        status_layout = QVBoxLayout(status_group)
        
        self._asr_status = StatusIndicator("ASR (Qwen3-ASR)")
        self._trans_status = StatusIndicator("Translation (NLLB-200)")
        self._tts_status = StatusIndicator("TTS (Fish Speech)")
        
        self._loading_progress = QProgressBar()
        self._loading_progress.setTextVisible(False)
        
        # Download button
        self._download_btn = QPushButton("📥 Download Models")
        self._download_btn.setToolTip("Download models to project directory (~9.2GB)")
        self._download_btn.clicked.connect(self._download_models)
        self._download_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_light']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 10px 16px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent']};
                border-color: {COLORS['accent']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_dim']};
            }}
        """)
        
        # Download progress label
        self._download_progress_label = QLabel("")
        self._download_progress_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        self._download_progress_label.setVisible(False)
        
        status_layout.addWidget(self._asr_status)
        status_layout.addWidget(self._trans_status)
        status_layout.addWidget(self._tts_status)
        status_layout.addWidget(self._loading_progress)
        status_layout.addWidget(self._download_btn)
        status_layout.addWidget(self._download_progress_label)
        
        # Settings group
        settings_group = QGroupBox("SETTINGS")
        settings_layout = QVBoxLayout(settings_group)
        
        # Target language
        lang_layout = QHBoxLayout()
        lang_label = QLabel("Target Language:")
        self._lang_combo = QComboBox()
        self._lang_combo.addItems(get_tts_languages())
        self._lang_combo.setCurrentText("English")
        lang_layout.addWidget(lang_label)
        lang_layout.addWidget(self._lang_combo)
        lang_layout.addStretch()
        
        # Voice cloning
        self._clone_checkbox = QCheckBox("Clone voice from recording")
        self._clone_checkbox.setToolTip("Speak the translation using your own voice")
        
        settings_layout.addLayout(lang_layout)
        settings_layout.addWidget(self._clone_checkbox)
        
        # Recording section
        rec_group = QGroupBox("RECORDING")
        rec_layout = QVBoxLayout(rec_group)
        rec_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self._record_btn = RecordButton()
        self._record_btn.clicked.connect(self._toggle_recording)
        self._record_btn.setEnabled(False)
        
        self._status_label = QLabel("Loading models...")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 13px;")
        
        rec_layout.addWidget(self._record_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        rec_layout.addWidget(self._status_label)
        
        # Result card
        self._result_card = ResultCard()
        
        # Replay button
        replay_layout = QHBoxLayout()
        self._replay_btn = QPushButton("🔊 Replay Last")
        self._replay_btn.setEnabled(False)
        self._replay_btn.clicked.connect(self._replay_audio)
        self._replay_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_light']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent']};
                border-color: {COLORS['accent']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_dim']};
            }}
        """)
        replay_layout.addStretch()
        replay_layout.addWidget(self._replay_btn)
        replay_layout.addStretch()
        
        # Assemble layout
        layout.addWidget(header)
        layout.addWidget(subtitle)
        layout.addWidget(status_group)
        layout.addWidget(settings_group)
        layout.addWidget(rec_group)
        layout.addWidget(self._result_card)
        layout.addLayout(replay_layout)
        layout.addStretch()
        
        return panel
        
    def _create_right_panel(self) -> QWidget:
        """Create the right log panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Header
        log_header = QLabel("📋 Activity Log")
        log_header.setStyleSheet(f"""
            font-size: 16px;
            font-weight: bold;
            color: {COLORS['text']};
            padding: 8px 0;
        """)
        
        # Log widget
        self._log = LogWidget()
        
        # Bottom controls
        controls_layout = QHBoxLayout()
        
        # Verbose checkbox
        self._verbose_checkbox = QCheckBox("Verbose logging")
        self._verbose_checkbox.setToolTip("Show detailed debug messages")
        self._verbose_checkbox.stateChanged.connect(self._toggle_verbose)
        self._verbose_checkbox.setStyleSheet(f"font-size: 12px; color: {COLORS['text_dim']};")
        
        # Clear button
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self._log.clear)
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 12px;
                color: {COLORS['text_dim']};
            }}
            QPushButton:hover {{
                border-color: {COLORS['accent']};
                color: {COLORS['accent']};
            }}
        """)
        
        controls_layout.addWidget(self._verbose_checkbox)
        controls_layout.addStretch()
        controls_layout.addWidget(clear_btn)
        
        layout.addWidget(log_header)
        layout.addWidget(self._log, stretch=1)
        layout.addLayout(controls_layout)
        
        return panel
    
    def _toggle_verbose(self, state: int):
        """Toggle verbose logging mode."""
        self._verbose = state == Qt.CheckState.Checked.value
        if self._verbose:
            self._log.log("Verbose logging enabled", "debug")
        else:
            self._log.log("Verbose logging disabled", "info")
    
    def _check_models_and_start(self):
        """Check model status and either download or load them."""
        self._state = AppState.LOADING_MODELS
        self._log.log("═" * 50, "debug")
        self._log.log(f"Starting {APP_NAME} v{APP_VERSION}", "info")
        self._log.log("═" * 50, "debug")
        
        # Check which models are downloaded
        status = model_manager.get_all_models_status()
        
        self._log.log(f"Model storage: {model_manager.MODELS_DIR}", "info")
        
        all_downloaded = True
        missing_models = []
        
        for model_key, info in status.items():
            if info.downloaded:
                self._log.log(f"✓ {info.name} - Downloaded", "success")
            else:
                self._log.log(f"○ {info.name} - Not downloaded (~{info.size_gb}GB)", "warning")
                all_downloaded = False
                missing_models.append(model_key)
        
        if all_downloaded:
            self._log.log("All models present. Starting load...", "info")
            self._download_btn.setText("✓ Models Downloaded")
            self._download_btn.setEnabled(False)
            self._start_model_loading()
        else:
            self._log.log("", "debug")
            self._log.log("⚠ Some models need to be downloaded first", "warning")
            self._log.log("Click 'Download Models' button to download (~9.2GB)", "info")
            self._status_label.setText("Models not downloaded")
            self._asr_status.set_status("idle")
            self._trans_status.set_status("idle")
            self._tts_status.set_status("idle")
            self._loading_progress.setVisible(False)
            
            # Update download button to show missing count
            self._download_btn.setText(f"📥 Download Models ({len(missing_models)} missing)")
    
    def _download_models(self):
        """Start downloading missing models."""
        # Get missing models
        status = model_manager.get_all_models_status()
        missing_models = [k for k, v in status.items() if not v.downloaded]
        
        if not missing_models:
            self._log.log("All models already downloaded!", "success")
            return
        
        self._download_btn.setEnabled(False)
        self._download_btn.setText("Downloading...")
        self._download_progress_label.setVisible(True)
        
        self._asr_status.set_status("loading")
        self._trans_status.set_status("loading")
        self._tts_status.set_status("loading")
        
        self._download_thread = ModelDownloadThread(
            models_to_download=missing_models,
            verbose=self._verbose
        )
        self._download_thread.progress.connect(self._on_download_progress)
        self._download_thread.model_finished.connect(self._on_model_download_finished)
        self._download_thread.finished.connect(self._on_download_finished)
        self._download_thread.log.connect(self._log.log)
        self._download_thread.start()
    
    def _on_download_progress(self, model_name: str, downloaded_mb: int, total_mb: int):
        """Handle download progress updates."""
        if total_mb > 0:
            pct = (downloaded_mb / total_mb) * 100
            self._download_progress_label.setText(
                f"{model_name}: {downloaded_mb}MB / {total_mb}MB ({pct:.0f}%)"
            )
            self._status_label.setText(f"Downloading {model_name}...")
    
    def _on_model_download_finished(self, model_key: str, success: bool, message: str):
        """Handle individual model download completion."""
        if model_key == "asr":
            self._asr_status.set_status("ready" if success else "error")
        elif model_key == "translation":
            self._trans_status.set_status("ready" if success else "error")
        elif model_key in ("tts", "tts_fish"):
            self._tts_status.set_status("ready" if success else "error")
    
    def _on_download_finished(self, success: bool, message: str):
        """Handle download completion."""
        self._download_progress_label.setVisible(False)
        
        if success:
            self._download_btn.setText("✓ Models Downloaded")
            self._download_btn.setEnabled(False)
            self._log.log("", "debug")
            self._log.log("Starting model load...", "info")
            self._start_model_loading()
        else:
            self._download_btn.setText("📥 Retry Download")
            self._download_btn.setEnabled(True)
            self._status_label.setText("Download failed - click to retry")
        
    def _start_model_loading(self):
        """Start loading models in background."""
        self._state = AppState.LOADING_MODELS
        
        self._asr_status.set_status("loading")
        self._trans_status.set_status("loading")
        self._tts_status.set_status("loading")
        self._loading_progress.setVisible(True)
        
        self._loader_thread = ModelLoaderThread(verbose=self._verbose)
        self._loader_thread.progress.connect(self._on_load_progress)
        self._loader_thread.model_loaded.connect(self._on_model_loaded)
        self._loader_thread.finished.connect(self._on_loading_finished)
        self._loader_thread.log.connect(self._log.log)
        self._loader_thread.start()
        
    def _on_load_progress(self, message: str, percentage: int):
        """Handle loading progress updates."""
        self._loading_progress.setValue(percentage)
        self._status_label.setText(message)
        
    def _on_model_loaded(self, model_name: str):
        """Handle individual model loaded."""
        if model_name == "ASR":
            self._asr_status.set_status("ready")
        elif model_name == "Translation":
            self._trans_status.set_status("ready")
        elif model_name == "TTS":
            self._tts_status.set_status("ready")
            
    def _on_loading_finished(self, success: bool, message: str):
        """Handle model loading completion."""
        if success:
            self._state = AppState.READY
            self._transcriber = self._loader_thread.transcriber
            self._translator = self._loader_thread.translator
            self._tts = self._loader_thread.tts
            
            self._record_btn.setEnabled(True)
            self._status_label.setText("Ready to record")
            self._loading_progress.setVisible(False)
            
            self._log.log("", "debug")
            self._log.log("🎤 Press the record button to start", "info")
        else:
            self._state = AppState.ERROR
            self._status_label.setText(f"Error: {message}")
            self._asr_status.set_status("error")
            self._trans_status.set_status("error")
            self._tts_status.set_status("error")
            
    def _toggle_recording(self):
        """Toggle recording state."""
        if self._state == AppState.RECORDING:
            # Stop recording (handled by silence detection)
            self._log.log("Waiting for silence to stop recording...", "info")
        else:
            self._start_recording()
            
    def _start_recording(self):
        """Start audio recording."""
        self._state = AppState.RECORDING
        self._record_btn.set_recording(True)
        self._status_label.setText("Recording... (stops on silence)")
        self._result_card.clear()
        
        self._recording_thread = RecordingThread()
        self._recording_thread.finished.connect(self._on_recording_finished)
        self._recording_thread.error.connect(self._on_recording_error)
        self._recording_thread.log.connect(self._log.log)
        self._recording_thread.start()
        
    def _on_recording_finished(self, audio: np.ndarray, sample_rate: int):
        """Handle recording completion."""
        self._record_btn.set_recording(False)
        self._state = AppState.PROCESSING
        self._status_label.setText("Processing...")
        
        # Create translation job
        job = TranslationJob(
            audio=audio,
            sample_rate=sample_rate,
            target_language=self._lang_combo.currentText(),
            clone_voice=self._clone_checkbox.isChecked(),
        )
        
        # Start translation pipeline
        self._translation_thread = TranslationThread(
            self._transcriber,
            self._translator,
            self._tts,
            job,
        )
        self._translation_thread.progress.connect(
            lambda msg: self._status_label.setText(msg)
        )
        self._translation_thread.finished.connect(self._on_translation_finished)
        self._translation_thread.error.connect(self._on_translation_error)
        self._translation_thread.log.connect(self._log.log)
        self._translation_thread.start()
        
    def _on_recording_error(self, error: str):
        """Handle recording error."""
        self._record_btn.set_recording(False)
        self._state = AppState.READY
        self._status_label.setText("Ready to record")
        self._log.log(f"Recording failed: {error}", "error")
        
    def _on_translation_finished(self, result: TranslationResult):
        """Handle translation completion."""
        self._state = AppState.SPEAKING
        self._result_card.set_result(result)
        self._replay_btn.setEnabled(True)
        self._last_result = result
        
        # Play the audio
        if result.audio_data is not None:
            self._status_label.setText("Playing translation...")
            self._log.log("🔊 Playing audio...", "info")
            
            import sounddevice as sd
            try:
                sd.play(result.audio_data, result.audio_sr)
                sd.wait()
            except Exception as e:
                self._log.log(f"Playback error: {e}", "error")
        
        self._state = AppState.READY
        self._status_label.setText("Ready to record")
        self._log.log("", "debug")
        self._log.log("🎤 Ready for next recording", "info")
        
    def _on_translation_error(self, error: str):
        """Handle translation error."""
        self._state = AppState.READY
        self._status_label.setText("Ready to record")
        
    def _replay_audio(self):
        """Replay the last audio output."""
        if hasattr(self, '_last_result') and self._last_result.audio_data is not None:
            import sounddevice as sd
            self._log.log("🔊 Replaying last audio...", "info")
            try:
                sd.play(self._last_result.audio_data, self._last_result.audio_sr)
            except Exception as e:
                self._log.log(f"Playback error: {e}", "error")
                
    def closeEvent(self, event):
        """Clean up on close."""
        self._log.log("Shutting down...", "info")
        
        # Stop any running threads
        if self._recording_thread and self._recording_thread.isRunning():
            self._recording_thread.terminate()
        if self._translation_thread and self._translation_thread.isRunning():
            self._translation_thread.terminate()
        if self._download_thread and self._download_thread.isRunning():
            self._download_thread.terminate()
            
        # Clean up models
        if self._transcriber:
            del self._transcriber
        if self._translator:
            del self._translator
        if self._tts:
            del self._tts
        gc.collect()
        
        event.accept()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Application entry point."""
    # High DPI support
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    
    # Use fusion style for consistent look
    app.setStyle(QStyleFactory.create('Fusion'))
    
    window = PolyglotWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
