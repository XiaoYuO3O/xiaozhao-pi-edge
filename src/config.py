import os
import platform

def is_running_on_pi() -> bool:
    if os.getenv("MOCK_PI") == "1":
        return False
    return platform.system() == "Linux" and platform.machine() in ("aarch64", "armv7l")

ON_PI = is_running_on_pi()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_TMP_DIR = os.path.join(BASE_DIR, "audio_tmp")

# Mode A audio capture: ALSA PCM passed to `arecord -D` (e.g. "bluealsa:DEV=XX:XX:XX:XX:XX:XX,PROFILE=sco"
# for a BlueZ/BlueALSA headset, or "pipewire"/"pulse" when the headset is the default PipeWire source).
BT_AUDIO_DEVICE = os.getenv("XIAOZHAO_BT_DEVICE", "default")
# XiaoZhao Whisper STT endpoint (multipart WAV upload). Empty -> local faster_whisper -> demo sentences.
XIAOZHAO_WHISPER_API_URL = os.getenv("XIAOZHAO_WHISPER_API_URL", "")
XIAOZHAO_WHISPER_API_TIMEOUT = float(os.getenv("XIAOZHAO_WHISPER_API_TIMEOUT", "30"))
# Backend base URL for POST /api/iot/v1/utterances and /api/transcripts/{id}/confirm
XIAOZHAO_BACKEND_URL = os.getenv("XIAOZHAO_BACKEND_URL", "http://127.0.0.1:8000")
