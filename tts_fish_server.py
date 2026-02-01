#!/usr/bin/env python3
"""
Persistent Fish Speech TTS server that keeps the model loaded in memory.
Communicates via Unix socket for fast synthesis requests.

Uses Fish Speech 1.5 for high-quality multilingual TTS.
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

# Set up model cache directory BEFORE importing fish_speech
PROJECT_ROOT = Path(__file__).parent.absolute()
MODELS_DIR = PROJECT_ROOT / "models" / "fish_speech"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(MODELS_DIR)

# Socket path
SOCKET_PATH = "/tmp/polyglot_fish_tts.sock"
LOCK_FILE = "/tmp/polyglot_fish_tts.lock"


def send_message(conn, data: dict):
    """Send a JSON message with length prefix."""
    msg = json.dumps(data).encode('utf-8')
    conn.sendall(struct.pack('!I', len(msg)) + msg)


def recv_message(conn) -> dict:
    """Receive a JSON message with length prefix."""
    raw_len = conn.recv(4)
    if not raw_len:
        return None
    msg_len = struct.unpack('!I', raw_len)[0]
    
    data = b''
    while len(data) < msg_len:
        chunk = conn.recv(min(msg_len - len(data), 4096))
        if not chunk:
            return None
        data += chunk
    
    return json.loads(data.decode('utf-8'))


class FishSpeechServer:
    """Persistent Fish Speech server."""
    
    def __init__(self, device: str = "auto"):
        self.device = device
        self.model = None
        self.running = False
        
    def load_model(self):
        """Load the Fish Speech model."""
        import torch
        from fish_speech.inference import load_model, TTSInference
        
        # Determine device
        if self.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "cpu"  # MPS has issues with some ops, use CPU
            else:
                device = "cpu"
        else:
            device = self.device
        
        print(f"[Fish Speech Server] Loading model on {device}...", file=sys.stderr)
        
        # Load Fish Speech 1.5
        self.model = TTSInference(
            model_name="fishaudio/fish-speech-1.5",
            device=device,
        )
        
        print(f"[Fish Speech Server] Model loaded and ready!", file=sys.stderr)
    
    def synthesize(
        self, 
        text: str, 
        language: str = "en",
        speaker_wav: str = None, 
        output_path: str = None,
        emotion: str = None,
    ) -> dict:
        """Synthesize speech from text."""
        if self.model is None:
            return {"success": False, "error": "Model not loaded"}
        
        try:
            import torchaudio
            
            if output_path is None:
                output_path = tempfile.mktemp(suffix=".wav")
            
            # Add emotion marker if specified
            if emotion:
                text = f"({emotion}) {text}"
            
            # Synthesize
            if speaker_wav and os.path.exists(speaker_wav):
                # Voice cloning mode
                audio = self.model.synthesize(
                    text=text,
                    reference_audio=speaker_wav,
                )
            else:
                # Default voice mode
                audio = self.model.synthesize(
                    text=text,
                )
            
            # Save to file
            torchaudio.save(
                output_path,
                audio.unsqueeze(0),
                self.model.sample_rate,
            )
            
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
                send_message(conn, {"status": "ok", "loaded": self.model is not None})
                
            elif cmd == "synthesize":
                result = self.synthesize(
                    text=request.get("text", ""),
                    language=request.get("language", "en"),
                    speaker_wav=request.get("speaker_wav"),
                    output_path=request.get("output_path"),
                    emotion=request.get("emotion"),
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
        server.settimeout(1.0)
        
        # Write PID to lock file
        with open(LOCK_FILE, 'w') as f:
            f.write(str(os.getpid()))
        
        print(f"[Fish Speech Server] Listening on {SOCKET_PATH}", file=sys.stderr)
        self.running = True
        
        def signal_handler(sig, frame):
            print(f"\n[Fish Speech Server] Shutting down...", file=sys.stderr)
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            while self.running:
                try:
                    conn, addr = server.accept()
                    thread = threading.Thread(target=self.handle_client, args=(conn,))
                    thread.daemon = True
                    thread.start()
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"[Fish Speech Server] Error: {e}", file=sys.stderr)
        finally:
            server.close()
            if os.path.exists(SOCKET_PATH):
                os.remove(SOCKET_PATH)
            if os.path.exists(LOCK_FILE):
                os.remove(LOCK_FILE)
            print("[Fish Speech Server] Shutdown complete", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Fish Speech TTS Server")
    parser.add_argument("--mode", choices=["server", "synthesize"], default="server")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda")
    parser.add_argument("--text", default="Hello, world!", help="Text for synthesis mode")
    parser.add_argument("--output", default="output.wav", help="Output path for synthesis")
    parser.add_argument("--speaker", help="Reference audio for voice cloning")
    
    args = parser.parse_args()
    
    if args.mode == "server":
        server = FishSpeechServer(device=args.device)
        server.run()
    else:
        # Direct synthesis mode
        server = FishSpeechServer(device=args.device)
        server.load_model()
        result = server.synthesize(
            text=args.text,
            speaker_wav=args.speaker,
            output_path=args.output,
        )
        if result["success"]:
            print(f"Saved to: {result['output']}")
        else:
            print(f"Error: {result['error']}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
