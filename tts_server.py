#!/usr/bin/env python3
"""
Persistent TTS server that keeps the model loaded in memory.
Communicates via Unix socket for fast synthesis requests.

This runs in the .venv-tts virtualenv to avoid dependency conflicts.
"""

import argparse
import json
import sys
import os
import socket
import struct
import tempfile
import signal
import threading
from pathlib import Path

# Set up TTS cache directory BEFORE importing TTS
PROJECT_ROOT = Path(__file__).parent.absolute()
TTS_CACHE_DIR = PROJECT_ROOT / "models" / "tts"
TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TTS_HOME"] = str(TTS_CACHE_DIR)
os.environ["COQUI_TOS_AGREED"] = "1"

import re


def split_text_into_chunks(text: str, max_chars: int = 250) -> list:
    """
    Split text into chunks suitable for XTTS-v2.
    
    XTTS-v2 works best with shorter segments (~250 chars).
    We split on sentence boundaries to maintain natural prosody.
    """
    if len(text) <= max_chars:
        return [text]
    
    # Split by sentence endings
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If single sentence is too long, split by commas/semicolons
        if len(sentence) > max_chars:
            # Split long sentences by clauses
            clause_pattern = r'(?<=[,;:])\s+'
            clauses = re.split(clause_pattern, sentence)
            
            for clause in clauses:
                clause = clause.strip()
                if not clause:
                    continue
                    
                if len(current_chunk) + len(clause) + 1 <= max_chars:
                    current_chunk = f"{current_chunk} {clause}".strip()
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    # If clause itself is too long, just use it anyway
                    current_chunk = clause
        else:
            # Normal sentence
            if len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk = f"{current_chunk} {sentence}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks if chunks else [text]


# Socket path
SOCKET_PATH = "/tmp/polyglot_tts.sock"
LOCK_FILE = "/tmp/polyglot_tts.lock"


def send_message(sock, data: dict):
    """Send a JSON message with length prefix."""
    msg = json.dumps(data).encode('utf-8')
    sock.sendall(struct.pack('!I', len(msg)) + msg)


def recv_message(sock) -> dict:
    """Receive a JSON message with length prefix."""
    # Read length prefix (4 bytes)
    raw_len = sock.recv(4)
    if not raw_len:
        return None
    msg_len = struct.unpack('!I', raw_len)[0]
    
    # Read message
    data = b''
    while len(data) < msg_len:
        chunk = sock.recv(min(msg_len - len(data), 4096))
        if not chunk:
            return None
        data += chunk
    
    return json.loads(data.decode('utf-8'))


class TTSServer:
    """Persistent TTS server with pre-loaded model."""
    
    def __init__(self, device: str = "auto"):
        self.device = device
        self.tts = None
        self.running = False
        
    def load_model(self):
        """Load the XTTS-v2 model."""
        from TTS.api import TTS
        import torch
        
        # Force CPU for XTTS-v2 (MPS has channel limit issues)
        device = "cpu"
        if self.device == "cuda" and torch.cuda.is_available():
            device = "cuda"
        
        print(f"[TTS Server] Loading XTTS-v2 on {device}...", file=sys.stderr)
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print(f"[TTS Server] Model loaded and ready!", file=sys.stderr)
        
    def synthesize(self, text: str, language: str, speaker_wav: str = None, output_path: str = None) -> dict:
        """Synthesize speech from text with automatic chunking for long texts."""
        if self.tts is None:
            return {"success": False, "error": "Model not loaded"}
        
        try:
            import numpy as np
            import scipy.io.wavfile as wavfile
            
            if output_path is None:
                output_path = tempfile.mktemp(suffix=".wav")
            
            # Split text into chunks for XTTS-v2 (max ~250 chars per chunk)
            chunks = split_text_into_chunks(text, max_chars=250)
            
            if len(chunks) == 1:
                # Single chunk - synthesize directly
                if speaker_wav and os.path.exists(speaker_wav):
                    self.tts.tts_to_file(
                        text=text,
                        file_path=output_path,
                        speaker_wav=speaker_wav,
                        language=language,
                    )
                else:
                    wav_data = self.tts.synthesizer.tts(
                        text=text,
                        language_name=language,
                        speaker_name="Claribel Dervla",
                    )
                    self.tts.synthesizer.save_wav(wav_data, output_path)
            else:
                # Multiple chunks - synthesize each and concatenate
                print(f"[TTS] Synthesizing {len(chunks)} chunks...", file=sys.stderr)
                all_audio = []
                sample_rate = None
                
                for i, chunk in enumerate(chunks):
                    print(f"[TTS] Chunk {i+1}/{len(chunks)}: {chunk[:50]}...", file=sys.stderr)
                    
                    if speaker_wav and os.path.exists(speaker_wav):
                        # Synthesize to temp file
                        temp_path = tempfile.mktemp(suffix=".wav")
                        self.tts.tts_to_file(
                            text=chunk,
                            file_path=temp_path,
                            speaker_wav=speaker_wav,
                            language=language,
                        )
                        sr, audio = wavfile.read(temp_path)
                        os.remove(temp_path)
                    else:
                        audio = self.tts.synthesizer.tts(
                            text=chunk,
                            language_name=language,
                            speaker_name="Claribel Dervla",
                        )
                        sr = self.tts.synthesizer.output_sample_rate
                    
                    if sample_rate is None:
                        sample_rate = sr
                    
                    # Convert to numpy array if needed
                    if not isinstance(audio, np.ndarray):
                        audio = np.array(audio)
                    
                    all_audio.append(audio)
                    
                    # Add small pause between chunks (0.15 seconds of silence)
                    if i < len(chunks) - 1:
                        pause = np.zeros(int(sample_rate * 0.15), dtype=np.float32)
                        all_audio.append(pause)
                
                # Concatenate all audio
                combined_audio = np.concatenate(all_audio)
                
                # Save combined audio
                if combined_audio.dtype == np.float32 or combined_audio.dtype == np.float64:
                    # Normalize and convert to int16
                    combined_audio = np.clip(combined_audio, -1.0, 1.0)
                    combined_audio = (combined_audio * 32767).astype(np.int16)
                
                wavfile.write(output_path, sample_rate, combined_audio)
                print(f"[TTS] Combined audio saved to {output_path}", file=sys.stderr)
            
            return {"success": True, "output": output_path}
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def handle_client(self, conn):
        """Handle a single client connection."""
        try:
            request = recv_message(conn)
            if request is None:
                return
            
            cmd = request.get("cmd")
            
            if cmd == "ping":
                send_message(conn, {"status": "ok", "loaded": self.tts is not None})
                
            elif cmd == "synthesize":
                result = self.synthesize(
                    text=request.get("text", ""),
                    language=request.get("language", "en"),
                    speaker_wav=request.get("speaker_wav"),
                    output_path=request.get("output_path"),
                )
                send_message(conn, result)
                
            elif cmd == "shutdown":
                send_message(conn, {"status": "shutting_down"})
                self.running = False
                
            else:
                send_message(conn, {"error": f"Unknown command: {cmd}"})
                
        except Exception as e:
            try:
                send_message(conn, {"error": str(e)})
            except:
                pass
        finally:
            conn.close()
    
    def run(self):
        """Run the server."""
        # Clean up old socket
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)
        
        # Load model first
        self.load_model()
        
        # Create Unix socket
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(SOCKET_PATH)
        server.listen(5)
        server.settimeout(1.0)  # Allow checking self.running
        
        # Write PID to lock file
        with open(LOCK_FILE, 'w') as f:
            f.write(str(os.getpid()))
        
        print(f"[TTS Server] Listening on {SOCKET_PATH}", file=sys.stderr)
        self.running = True
        
        def signal_handler(sig, frame):
            print(f"\n[TTS Server] Shutting down...", file=sys.stderr)
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            while self.running:
                try:
                    conn, addr = server.accept()
                    # Handle in thread to allow concurrent requests
                    thread = threading.Thread(target=self.handle_client, args=(conn,))
                    thread.daemon = True
                    thread.start()
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"[TTS Server] Error: {e}", file=sys.stderr)
        finally:
            server.close()
            if os.path.exists(SOCKET_PATH):
                os.remove(SOCKET_PATH)
            if os.path.exists(LOCK_FILE):
                os.remove(LOCK_FILE)
            print(f"[TTS Server] Stopped", file=sys.stderr)


class TTSClient:
    """Client to communicate with the TTS server."""
    
    @staticmethod
    def is_server_running() -> bool:
        """Check if the server is running."""
        if not os.path.exists(SOCKET_PATH):
            return False
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect(SOCKET_PATH)
            send_message(sock, {"cmd": "ping"})
            response = recv_message(sock)
            sock.close()
            return response and response.get("status") == "ok"
        except:
            return False
    
    @staticmethod
    def synthesize(text: str, language: str, speaker_wav: str = None, output_path: str = None) -> dict:
        """Send synthesis request to server."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(120.0)  # 2 minute timeout for synthesis
        
        try:
            sock.connect(SOCKET_PATH)
            
            request = {
                "cmd": "synthesize",
                "text": text,
                "language": language,
            }
            if speaker_wav:
                request["speaker_wav"] = speaker_wav
            if output_path:
                request["output_path"] = output_path
            
            send_message(sock, request)
            response = recv_message(sock)
            return response
            
        finally:
            sock.close()
    
    @staticmethod
    def shutdown():
        """Send shutdown command to server."""
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect(SOCKET_PATH)
            send_message(sock, {"cmd": "shutdown"})
            sock.close()
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description="XTTS-v2 TTS Server")
    parser.add_argument("--mode", choices=["server", "client", "status", "stop"], default="server",
                        help="Mode: server (run daemon), client (synthesize), status, stop")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, mps, cpu")
    
    # Client mode arguments
    parser.add_argument("--text", help="Text to synthesize (client mode)")
    parser.add_argument("--language", default="en", help="Language code")
    parser.add_argument("--speaker-wav", help="Speaker WAV for voice cloning")
    parser.add_argument("--output", help="Output WAV path")
    
    args = parser.parse_args()
    
    if args.mode == "server":
        server = TTSServer(device=args.device)
        server.run()
        
    elif args.mode == "status":
        if TTSClient.is_server_running():
            print("TTS Server is running")
            sys.exit(0)
        else:
            print("TTS Server is not running")
            sys.exit(1)
            
    elif args.mode == "stop":
        if TTSClient.is_server_running():
            TTSClient.shutdown()
            print("Shutdown signal sent")
        else:
            print("Server not running")
            
    elif args.mode == "client":
        if not args.text:
            print("Error: --text required for client mode", file=sys.stderr)
            sys.exit(1)
        
        if not TTSClient.is_server_running():
            print("Error: TTS server not running. Start with: tts_server.py --mode server", file=sys.stderr)
            sys.exit(1)
        
        result = TTSClient.synthesize(
            text=args.text,
            language=args.language,
            speaker_wav=args.speaker_wav,
            output_path=args.output,
        )
        print(json.dumps(result))


if __name__ == "__main__":
    main()
