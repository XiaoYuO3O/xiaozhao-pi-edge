import os
import platform

def is_running_on_pi() -> bool:
    if os.getenv("MOCK_PI") == "1":
        return False
    return platform.system() == "Linux" and platform.machine() in ("aarch64", "armv7l")

ON_PI = is_running_on_pi()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_TMP_DIR = os.path.join(BASE_DIR, "audio_tmp")
