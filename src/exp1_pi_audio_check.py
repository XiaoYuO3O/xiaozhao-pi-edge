import array
import math
import os
import struct
import subprocess
import sys
import wave

try:
    from src.config import ON_PI, AUDIO_TMP_DIR, BT_AUDIO_DEVICE
except ImportError:
    from config import ON_PI, AUDIO_TMP_DIR, BT_AUDIO_DEVICE


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


def compute_wav_acoustics(wav_path: str, start_sec: float = 0.0, end_sec: float = None) -> dict:
    """Real rms_energy (int16 RMS amplitude) and zcr_pitch (zero-crossing rate 0..1, pitch proxy) of a 16-bit
    PCM WAV, optionally over [start_sec, end_sec). Same scales as mpc_core MINTREC_HIGH_RMS / MINTREC_HIGH_ZCR.
    Stdlib only; multi-channel input is read from channel 0. An empty span returns zeros (= unknown)."""
    with wave.open(wav_path, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{wav_path}: expected 16-bit PCM, got {8 * w.getsampwidth()}-bit")
        rate, ch, total = w.getframerate(), w.getnchannels(), w.getnframes()
        first = min(total, max(0, int(start_sec * rate)))
        last = total if end_sec is None else min(total, max(first, int(end_sec * rate)))
        w.setpos(first)
        pcm = array.array("h")
        pcm.frombytes(w.readframes(last - first))
    if sys.byteorder == "big":
        pcm.byteswap()
    x = pcm[::ch] if ch > 1 else pcm
    n = len(x)
    if n == 0:
        return {"rms_energy": 0.0, "zcr_pitch": 0.0, "samples": 0, "sample_rate": rate}
    rms = math.sqrt(sum(v * v for v in x) / n)
    crossings = sum(1 for a, b in zip(x, x[1:]) if (a >= 0) != (b >= 0))
    return {"rms_energy": round(rms, 1), "zcr_pitch": round(crossings / max(1, n - 1), 4),
            "samples": n, "sample_rate": rate}


def record_audio(duration_sec: int = 3, sample_rate: int = 16000, device: str = None) -> dict:
    """Record via `arecord -D <device>` on the Pi (BT_AUDIO_DEVICE / XIAOZHAO_BT_DEVICE by default); else fall
    back to a synthetic test WAV. A named (non-"default") device such as a BlueALSA/PipeWire headset need not
    appear as a card in `arecord -l`, so the card check only gates the "default" device."""
    device = device or BT_AUDIO_DEVICE
    if ON_PI:
        env = check_audio_env()
        if env["status"] == "ok" and (device != "default" or "card" in env["devices"]):
            os.makedirs(AUDIO_TMP_DIR, exist_ok=True)
            out_path = os.path.join(AUDIO_TMP_DIR, "exp1_record_16k.wav")
            try:
                res = subprocess.run(
                    ["arecord", "-D", device, "-d", str(duration_sec), "-f", "S16_LE", "-r", str(sample_rate),
                     "-c", "1", out_path],
                    capture_output=True, text=True, timeout=duration_sec + 10,
                )
                if res.returncode == 0 and os.path.exists(out_path):
                    return {
                        "file_path": out_path,
                        "sample_rate": sample_rate,
                        "channels": 1,
                        "file_size_bytes": os.path.getsize(out_path),
                        "on_pi": True,
                        "source": "arecord",
                        "device": device,
                        **{k: v for k, v in compute_wav_acoustics(out_path).items()
                           if k in ("rms_energy", "zcr_pitch")},
                    }
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
    info = generate_test_wav(duration_sec, sample_rate)
    info["source"] = "synthetic_fallback"
    info["device"] = device
    return info


if __name__ == "__main__":
    info = generate_test_wav(3, 16000)
    if "--quiet-test" in sys.argv:
        assert os.path.exists(info["file_path"]), "wav file missing"
        assert info["file_size_bytes"] > 90000, f"wav too small: {info['file_size_bytes']}"
        # 0.5-amplitude 440 Hz sine: RMS = 0.5 * 32767 / sqrt(2) ~ 11585, ZCR = 2 * 440 / 16000 = 0.055
        ac = compute_wav_acoustics(info["file_path"])
        assert abs(ac["rms_energy"] - 11585) < 30 and abs(ac["zcr_pitch"] - 0.055) < 0.002, ac
        half = compute_wav_acoustics(info["file_path"], 1.0, 1.5)
        assert half["samples"] == 8000 and abs(half["zcr_pitch"] - 0.055) < 0.002, half
        assert compute_wav_acoustics(info["file_path"], 99.0)["rms_energy"] == 0.0   # span past EOF
        assert record_audio(1, device="bluealsa:TEST")["device"] == "bluealsa:TEST"
        print("EXP1_AUDIO_OK")
        sys.exit(0)
    print(check_audio_env())
    print(info)
