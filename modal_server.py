import os
import time
import tempfile
import modal
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = modal.App("africa-whisper-server")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12",
        "faster-whisper==1.2.1",
        "fastapi==0.115.0",
        "uvicorn==0.32.0",
        "python-multipart==0.0.12",
        "huggingface_hub==0.26.2",
        "pydantic>=2.0.0",
    )
    .env({
        "LD_LIBRARY_PATH": "/usr/local/lib/python3.10/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib"
    })
    .run_commands(
        "python -c 'from huggingface_hub import snapshot_download; snapshot_download(\"AfriSpeech/africa-whisper-ct2\")'"
    )
)


@app.cls(
    gpu="T4",
    image=image,
    scaledown_window=60,
    timeout=180,
)
class AfricaWhisperServer:
    @modal.enter()
    def load_model(self):
        from faster_whisper import WhisperModel
        print("Loading CTranslate2 WhisperModel on CUDA...")
        self.model = WhisperModel(
            "AfriSpeech/africa-whisper-ct2",
            device="cuda",
            compute_type="float16",
        )
        print("Model loaded into GPU VRAM successfully!")

    @modal.asgi_app()
    def web(self):
        web_app = FastAPI(title="Africa-Whisper ASR Server", version="1.0.0")

        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @web_app.get("/")
        async def root():
            return {
                "status": "online",
                "service": "Africa-Whisper ASR",
                "model": "AfriSpeech/africa-whisper-ct2",
                "languages": 314,
                "framework": "faster-whisper (CTranslate2)",
            }

        @web_app.post("/transcribe")
        async def transcribe(audio: UploadFile = File(...)):
            t0 = time.time()
            contents = await audio.read()
            if not contents:
                raise HTTPException(status_code=400, detail="Empty audio file provided.")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name

            try:
                # Transcribe with English conditioning for universal Latin orthography
                segments, info = self.model.transcribe(
                    tmp_path,
                    language="en",
                    task="transcribe",
                    without_timestamps=True,
                    beam_size=5,
                )
                text = " ".join(s.text.strip() for s in segments).strip()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            latency = round(time.time() - t0, 3)
            return {
                "transcription": text,
                "duration": round(info.duration, 2),
                "latency_seconds": latency,
                "sample_rate": 16000,
            }

        return web_app
