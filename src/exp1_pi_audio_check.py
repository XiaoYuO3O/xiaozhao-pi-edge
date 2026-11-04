import math
import os
import struct
import subprocess
import sys
import wave

try:
    from src.config import ON_PI, AUDIO_TMP_DIR
except ImportError:
    from config import ON_PI, AUDIO_TMP_DIR


def check_audio_env() -> dict:
    """List ALSA capture devices on Pi; return mock status on Mac."""
    if not ON_PI:
        return {"on_pi": False, "status": "mock", "devices": "MOCK: no ALSA on this host"}
    try:
        res = subprocess.run(["arecord", "-l"], capture_output=True, text=True, timeout=10)
        return {
            "on_pi": True,
            "status": "ok" if res.returncode == 0 else "error",
            "devices": (res.stdout or res.stderr).strip(),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {"on_pi": True, "status": "error", "devices": str(e)}


def generate_test_wav(duration_sec: int = 3, sample_rate: int = 16000) -> dict:
    """Write a 440Hz mono 16-bit sine wave to audio_tmp/exp1_test_16k.wav."""
    os.makedirs(AUDIO_TMP_DIR, exist_ok=True)
    path = os.path.join(AUDIO_TMP_DIR, "exp1_test_16k.wav")
    n = int(duration_sec * sample_rate)
    frames = b"".join(
        struct.pack("<h", int(32767 * 0.5 * math.sin(2 * math.pi * 440 * i / sample_rate)))
        for i in range(n)
    )
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(frames)
    return {
        "file_path": path,
        "sample_rate": sample_rate,
        "channels": 1,
        "file_size_bytes": os.path.getsize(path),
        "on_pi": ON_PI,
    }


if __name__ == "__main__":
    info = generate_test_wav(3, 16000)
    if "--quiet-test" in sys.argv:
        assert os.path.exists(info["file_path"]), "wav file missing"
        assert info["file_size_bytes"] > 90000, f"wav too small: {info['file_size_bytes']}"
        print("EXP1_AUDIO_OK")
        sys.exit(0)
    print(check_audio_env())
    print(info)
