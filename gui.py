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
from audio_capture import start_recording, stop_recording, get_recorder, SAMPLE_RATE
from transcriber import Transcriber
from translator import Translator
from tts import TextToSpeech, get_supported_languages as get_tts_languages, is_language_supported


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & STYLING
# ═══════════════════════════════════════════════════════════════════════════════

APP_NAME = "Polyglot"
APP_VERSION = "1.0.0"

COLORS = {
    'bg_dark': '#09090b',        # zinc-950
    'bg_medium': '#18181b',      # zinc-900
    'bg_light': '#27272a',       # zinc-800
    'bg_card': '#1c1c1f',        # card background
    'accent': '#ef4444',         # red-500 for recording
    'accent_light': '#f87171',   # red-400
    'primary': '#3b82f6',        # blue-500
    'primary_light': '#60a5fa',  # blue-400
    'success': '#22c55e',        # green-500
    'warning': '#eab308',        # yellow-500
    'error': '#ef4444',          # red-500
    'text': '#fafafa',           # zinc-50
    'text_muted': '#a1a1aa',     # zinc-400
    'text_dim': '#71717a',       # zinc-500
    'border': '#27272a',         # zinc-800
    'border_light': '#3f3f46',   # zinc-700
    'ring': '#3b82f6',           # focus ring
}

STYLESHEET = f"""
QMainWindow {{
    background-color: {COLORS['bg_dark']};
}}

QWidget {{
    color: {COLORS['text']};
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', system-ui, sans-serif;
    font-size: 13px;
}}

QGroupBox {{
    background-color: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    margin-top: 20px;
    padding: 16px;
    padding-top: 24px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 12px;
    color: {COLORS['text_muted']};
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.5px;
}}

QTextEdit {{
    background-color: {COLORS['bg_medium']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 12px;
    font-family: 'SF Mono', 'JetBrains Mono', 'Fira Code', ui-monospace, monospace;
    font-size: 12px;
    selection-background-color: {COLORS['primary']};
}}

QTextEdit:focus {{
    border-color: {COLORS['border_light']};
}}

QComboBox {{
    background-color: {COLORS['bg_medium']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 8px 12px;
    min-width: 160px;
}}

QComboBox:hover {{
    border-color: {COLORS['border_light']};
}}

QComboBox:focus {{
    border-color: {COLORS['primary']};
}}

QComboBox::drop-down {{
    border: none;
    padding-right: 8px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS['bg_medium']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    selection-background-color: {COLORS['bg_light']};
    outline: none;
}}

QProgressBar {{
    background-color: {COLORS['bg_medium']};
    border: none;
    border-radius: 4px;
    height: 6px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {COLORS['primary']};
    border-radius: 4px;
}}

QCheckBox {{
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid {COLORS['border_light']};
    background-color: {COLORS['bg_medium']};
}}

QCheckBox::indicator:hover {{
    border-color: {COLORS['text_dim']};
}}

QCheckBox::indicator:checked {{
    background-color: {COLORS['primary']};
    border-color: {COLORS['primary']};
}}

QLabel {{
    background: transparent;
}}

QSplitter::handle {{
    background-color: {COLORS['border']};
}}

QSplitter::handle:horizontal {{
    width: 1px;
}}

QSplitter::handle:vertical {{
    height: 1px;
}}

QScrollBar:vertical {{
    background-color: {COLORS['bg_dark']};
    width: 8px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical {{
    background-color: {COLORS['border_light']};
    border-radius: 4px;
    min-height: 20px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {COLORS['text_dim']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
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
    
    def __init__(self, device: str = "auto"):
        super().__init__()
        self.device = device
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
            
            # Load TTS server (XTTS-v2) - this takes ~30s on first load
            self.log.emit("Starting TTS server (XTTS-v2)...", "info")
            self.log.emit("⏳ TTS model loading (~30s first time, then instant)", "info")
            self.progress.emit("Starting TTS server...", 70)
            
            self.tts = TextToSpeech(device=self.device)
            self.tts.load_model()  # This starts the persistent server
            self.model_loaded.emit("TTS")
            self.log.emit("✓ TTS server running (model pre-loaded)", "success")
            self.progress.emit("TTS server ready", 100)
            
            self.log.emit("All models loaded! Ready for translation.", "success")
            self.finished.emit(True, "All models loaded successfully")
            
        except Exception as e:
            self.log.emit(f"✗ Error loading models: {str(e)}", "error")
            self.finished.emit(False, str(e))


class RecordingThread(QThread):
    """Thread for audio recording with manual stop control."""
    
    started = pyqtSignal()
    progress = pyqtSignal(float, float)  # duration, amplitude
    finished = pyqtSignal(np.ndarray, int)  # audio, sample_rate
    error = pyqtSignal(str)
    log = pyqtSignal(str, str)
    
    def __init__(self):
        super().__init__()
        self._stop_requested = False
        self._recorder = None
        
    def run(self):
        try:
            self.started.emit()
            self.log.emit("🎤 Recording... click Stop when done", "info")
            
            # Start recording
            self._recorder = start_recording(max_duration=60.0)
            
            # Poll for stop or max duration
            while not self._stop_requested and self._recorder.is_recording():
                self._recorder.collect_chunks()
                duration = self._recorder.get_duration()
                self.progress.emit(duration, 0.0)
                self.msleep(100)  # 100ms poll interval
            
            # Stop and get audio
            audio, sr = stop_recording()
            
            duration = len(audio) / sr
            self.log.emit(f"Recording stopped: {duration:.1f}s captured", "info")
            self.finished.emit(audio, sr)
            
        except Exception as e:
            self.log.emit(f"Recording error: {str(e)}", "error")
            self.error.emit(str(e))
    
    def stop(self):
        """Request recording to stop."""
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
        tts: TextToSpeech,
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
    """Modern recording button with clear Start/Stop states."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(140, 48)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._recording = False
        
        # Pulse animation for recording state
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._update_pulse)
        self._pulse_phase = 0
        self._pulse_opacity = 0.0
        
        self._update_style()
        
    def _update_style(self):
        if self._recording:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['error']};
                    border: none;
                    border-radius: 8px;
                    font-size: 14px;
                    font-weight: 600;
                    color: white;
                    padding: 0 24px;
                }}
                QPushButton:hover {{
                    background-color: #dc2626;
                }}
            """)
            self.setText("◼  Stop Recording")
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['primary']};
                    border: none;
                    border-radius: 8px;
                    font-size: 14px;
                    font-weight: 600;
                    color: white;
                    padding: 0 24px;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['primary_light']};
                }}
                QPushButton:disabled {{
                    background-color: {COLORS['bg_light']};
                    color: {COLORS['text_dim']};
                }}
            """)
            self.setText("●  Start Recording")
    
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
        
        # No extra painting needed for the cleaner button style


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
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = self.LEVEL_COLORS.get(level, COLORS['text'])
        
        # Format the HTML
        html = f'<span style="color: {COLORS["text_dim"]}; font-size: 11px;">{timestamp}</span> '
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
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(10)
        
        self._indicator = QLabel("○")
        self._indicator.setFixedWidth(16)
        self._indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self._label = QLabel(label)
        self._label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 13px;")
        
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
        self._indicator.setStyleSheet(style + " font-size: 14px;")
        self._indicator.setText(icon)


class ResultCard(QFrame):
    """Card displaying translation results."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Source section
        self._source_lang = QLabel("Detected Language")
        self._source_lang.setStyleSheet(f"""
            font-size: 11px;
            font-weight: 500;
            color: {COLORS['text_dim']};
            letter-spacing: 0.5px;
        """)
        self._source_text = QLabel("—")
        self._source_text.setStyleSheet(f"font-size: 14px; color: {COLORS['text_muted']};")
        self._source_text.setWordWrap(True)
        
        # Divider
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background-color: {COLORS['border']};")
        
        # Translation section
        self._target_lang = QLabel("Translation")
        self._target_lang.setStyleSheet(f"""
            font-size: 11px;
            font-weight: 500;
            color: {COLORS['text_dim']};
            letter-spacing: 0.5px;
        """)
        self._target_text = QLabel("—")
        self._target_text.setStyleSheet(f"""
            font-size: 16px;
            font-weight: 600;
            color: {COLORS['text']};
        """)
        self._target_text.setWordWrap(True)
        
        # Duration
        self._duration = QLabel("")
        self._duration.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        
        layout.addWidget(self._source_lang)
        layout.addWidget(self._source_text)
        layout.addSpacing(4)
        layout.addWidget(divider)
        layout.addSpacing(4)
        layout.addWidget(self._target_lang)
        layout.addWidget(self._target_text)
        layout.addWidget(self._duration)
        
    def set_result(self, result: TranslationResult):
        """Display translation result."""
        self._source_lang.setText(f"Detected: {result.source_language}")
        self._source_text.setText(result.source_text)
        self._target_lang.setText(f"→ {result.target_language}")
        self._target_text.setText(result.translated_text)
        self._duration.setText(f"Completed in {result.duration_ms}ms")
        
    def clear(self):
        """Clear the result display."""
        self._source_lang.setText("Detected Language")
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
        
        self._recording_thread = None
        self._translation_thread = None
        
        self._setup_ui()
        self._start_model_loading()
        
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
            font-size: 24px;
            font-weight: 700;
            color: {COLORS['text']};
            padding: 4px 0;
        """)
        
        subtitle = QLabel("Edge AI Translation")
        subtitle.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 13px; margin-bottom: 8px;")
        
        # Model status group
        status_group = QGroupBox("Models")
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(4)
        
        self._asr_status = StatusIndicator("ASR (Qwen3-ASR)")
        self._trans_status = StatusIndicator("Translation (NLLB-200)")
        self._tts_status = StatusIndicator("TTS (Qwen3-TTS)")
        
        self._loading_progress = QProgressBar()
        self._loading_progress.setTextVisible(False)
        self._loading_progress.setFixedHeight(4)
        
        status_layout.addWidget(self._asr_status)
        status_layout.addWidget(self._trans_status)
        status_layout.addWidget(self._tts_status)
        status_layout.addSpacing(8)
        status_layout.addWidget(self._loading_progress)
        
        # Settings group
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setSpacing(12)
        
        # Target language
        lang_label = QLabel("Target Language")
        lang_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px;")
        self._lang_combo = QComboBox()
        self._lang_combo.addItems(get_tts_languages())
        self._lang_combo.setCurrentText("English")
        
        # Voice cloning
        self._clone_checkbox = QCheckBox("Clone voice from recording")
        self._clone_checkbox.setToolTip("Speak the translation using your own voice")
        
        settings_layout.addWidget(lang_label)
        settings_layout.addWidget(self._lang_combo)
        settings_layout.addWidget(self._clone_checkbox)
        
        # Recording controls (no group box, cleaner)
        rec_widget = QWidget()
        rec_layout = QVBoxLayout(rec_widget)
        rec_layout.setContentsMargins(0, 8, 0, 0)
        rec_layout.setSpacing(12)
        
        self._record_btn = RecordButton()
        self._record_btn.clicked.connect(self._toggle_recording)
        self._record_btn.setEnabled(False)
        
        self._status_label = QLabel("Loading models...")
        self._status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px;")
        
        self._duration_label = QLabel("")
        self._duration_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        
        rec_layout.addWidget(self._record_btn)
        rec_layout.addWidget(self._status_label)
        rec_layout.addWidget(self._duration_label)
        
        # Result card
        self._result_card = ResultCard()
        
        # Replay button
        self._replay_btn = QPushButton("🔊  Replay Last")
        self._replay_btn.setEnabled(False)
        self._replay_btn.clicked.connect(self._replay_audio)
        self._replay_btn.setFixedHeight(40)
        self._replay_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 0 16px;
                font-size: 13px;
                color: {COLORS['text_muted']};
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_light']};
                border-color: {COLORS['border_light']};
                color: {COLORS['text']};
            }}
            QPushButton:disabled {{
                color: {COLORS['text_dim']};
                border-color: {COLORS['border']};
            }}
        """)
        
        # Assemble layout
        layout.addWidget(header)
        layout.addWidget(subtitle)
        layout.addWidget(status_group)
        layout.addWidget(settings_group)
        layout.addWidget(rec_widget)
        layout.addWidget(self._result_card)
        layout.addWidget(self._replay_btn)
        layout.addStretch()
        
        return panel
        
    def _create_right_panel(self) -> QWidget:
        """Create the right log panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        # Log widget (create first so we can connect to it)
        self._log = LogWidget()
        
        # Header row
        header_layout = QHBoxLayout()
        
        log_header = QLabel("Activity Log")
        log_header.setStyleSheet(f"""
            font-size: 14px;
            font-weight: 600;
            color: {COLORS['text']};
        """)
        
        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._log.clear)
        clear_btn.setFixedSize(60, 28)
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                font-size: 11px;
                color: {COLORS['text_muted']};
            }}
            QPushButton:hover {{
                border-color: {COLORS['border_light']};
                color: {COLORS['text']};
            }}
        """)
        
        header_layout.addWidget(log_header)
        header_layout.addStretch()
        header_layout.addWidget(clear_btn)
        
        layout.addLayout(header_layout)
        layout.addWidget(self._log, stretch=1)
        
        return panel
        
    def _start_model_loading(self):
        """Start loading models in background."""
        self._state = AppState.LOADING_MODELS
        self._log.log("═" * 50, "debug")
        self._log.log(f"Starting {APP_NAME} v{APP_VERSION}", "info")
        self._log.log("═" * 50, "debug")
        
        self._asr_status.set_status("loading")
        self._trans_status.set_status("loading")
        self._tts_status.set_status("loading")
        
        self._loader_thread = ModelLoaderThread()
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
            self._status_label.setText("Ready")
            self._loading_progress.setVisible(False)
            
            self._log.log("", "debug")
            self._log.log("🎤 Click Start Recording to begin", "info")
        else:
            self._state = AppState.ERROR
            self._status_label.setText(f"Error: {message}")
            self._asr_status.set_status("error")
            self._trans_status.set_status("error")
            self._tts_status.set_status("error")
            
    def _toggle_recording(self):
        """Toggle recording state."""
        if self._state == AppState.RECORDING:
            self._stop_recording()
        else:
            self._start_recording()
    
    def _stop_recording(self):
        """Stop the current recording."""
        if self._recording_thread and self._recording_thread.isRunning():
            self._log.log("Stopping recording...", "info")
            self._recording_thread.stop()
            
    def _start_recording(self):
        """Start audio recording."""
        self._state = AppState.RECORDING
        self._record_btn.set_recording(True)
        self._status_label.setText("Recording...")
        self._duration_label.setText("0.0s")
        self._result_card.clear()
        
        self._recording_thread = RecordingThread()
        self._recording_thread.progress.connect(self._on_recording_progress)
        self._recording_thread.finished.connect(self._on_recording_finished)
        self._recording_thread.error.connect(self._on_recording_error)
        self._recording_thread.log.connect(self._log.log)
        self._recording_thread.start()
    
    def _on_recording_progress(self, duration: float, amplitude: float):
        """Update recording progress display."""
        self._duration_label.setText(f"{duration:.1f}s")
        
    def _on_recording_finished(self, audio: np.ndarray, sample_rate: int):
        """Handle recording completion."""
        self._record_btn.set_recording(False)
        self._state = AppState.PROCESSING
        self._status_label.setText("Processing...")
        self._duration_label.setText("")
        
        # Check if we have audio
        if len(audio) < sample_rate * 0.5:  # Less than 0.5 seconds
            self._log.log("Recording too short, please try again", "warning")
            self._state = AppState.READY
            self._status_label.setText("Ready")
            return
        
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
        self._status_label.setText("Ready")
        self._duration_label.setText("")
        self._log.log(f"Recording failed: {error}", "error")
        
    def _on_translation_finished(self, result: TranslationResult):
        """Handle translation completion."""
        self._state = AppState.SPEAKING
        self._result_card.set_result(result)
        self._replay_btn.setEnabled(True)
        self._last_result = result
        
        # Play the audio
        if result.audio_data is not None:
            self._status_label.setText("Playing...")
            self._log.log("🔊 Playing audio...", "info")
            
            import sounddevice as sd
            try:
                sd.play(result.audio_data, result.audio_sr)
                sd.wait()
            except Exception as e:
                self._log.log(f"Playback error: {e}", "error")
        
        self._state = AppState.READY
        self._status_label.setText("Ready")
        self._log.log("", "debug")
        self._log.log("🎤 Ready for next recording", "info")
        
    def _on_translation_error(self, error: str):
        """Handle translation error."""
        self._state = AppState.READY
        self._status_label.setText("Ready")
        
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
