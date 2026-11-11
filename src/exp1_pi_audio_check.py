import array
import math
import os
import struct
import subprocess
import sys
import wave

try:
    from src.config import ON_PI, AUDIO_TMP_DIR, BT_AUDIO_DEVICE
    from src import mpc_core as _core
except ImportError:
    from config import ON_PI, AUDIO_TMP_DIR, BT_AUDIO_DEVICE
    import mpc_core as _core


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


def read_wav_span(wav_path: str, start_sec: float = 0.0, end_sec: float = None) -> tuple:
    """Bounded read of channel 0 of a 16-bit PCM WAV over [start_sec, end_sec) -> (int16 array, sample_rate,
    first_frame). Never loads more than the span; a span past EOF comes back empty."""
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
    return (pcm[::ch] if ch > 1 else pcm), rate, first


def compute_wav_acoustics(wav_path: str, start_sec: float = 0.0, end_sec: float = None) -> dict:
    """Real rms_energy (int16 RMS amplitude) and zcr_pitch (zero-crossing rate 0..1, pitch proxy) of a 16-bit
    PCM WAV, optionally over [start_sec, end_sec). Same scales as mpc_core MINTREC_HIGH_RMS / MINTREC_HIGH_ZCR.
    Stdlib only; multi-channel input is read from channel 0. An empty span returns zeros (= unknown)."""
    x, rate, _ = read_wav_span(wav_path, start_sec, end_sec)
    n = len(x)
    if n == 0:
        return {"rms_energy": 0.0, "zcr_pitch": 0.0, "samples": 0, "sample_rate": rate}
    rms = math.sqrt(sum(v * v for v in x) / n)
    crossings = sum(1 for a, b in zip(x, x[1:]) if (a >= 0) != (b >= 0))
    return {"rms_energy": round(rms, 1), "zcr_pitch": round(crossings / max(1, n - 1), 4),
            "samples": n, "sample_rate": rate}


def extract_turn_voiceprint(wav_path: str, start_sec: float = 0.0, end_sec: float = None, text: str = "") -> dict:
    """v7.7 turn voiceprint from a 16-bit WAV (channel 0): bounded read of the turn span [start_sec, end_sec), then
    mpc_core.extract_turn_voiceprint -> 0.25 s boundary collar, [0.50, 0.90] window around the 75% anchor, > 6 s
    turns split into Must-Linked sub-windows, 25 ms / 10 ms frames (13 cepstra + band / centroid / flatness / ZCR /
    cepstral F0), SE channel rescaling + attentive statistics pooling -> 24-d L2-normalised x_i, near_field_score,
    and anchor-window rms_energy / zcr_pitch. Stdlib only (wave / array / math): no ML packages on the Pi. A span
    with no samples falls back to the JSON-only proxy (source proxy_text)."""
    pcm, rate, first = read_wav_span(wav_path, start_sec, end_sec)
    end = end_sec if end_sec is not None else (first + len(pcm)) / float(rate)
    return _core.extract_turn_voiceprint(pcm if len(pcm) else None, rate, start_sec, end,
                                         offset_sec=first / float(rate), text=text)


def _write_pcm_wav(path: str, samples: list, sample_rate: int = 16000) -> None:
    pcm = array.array("h", samples)
    if sys.byteorder == "big":
        pcm.byteswap()
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def _voiceprint_self_test() -> None:
    """(a) 75%-anchor + collar + Must-Link split through the WAV path on distinct synthetic voices (low pitch /
    bright vs high pitch / dark timbre); the WAV wrapper must equal the in-memory core extractor."""
    lo, hi = _core.TEST_VOICE_LOW, _core.TEST_VOICE_HIGH
    # 0-2.5 lo | 2.5-5 hi | 5-8: 1 s hi bleed then lo | 8-15 hi (7 s -> 2 Must-Linked sub-windows)
    pcm = (_core.synth_test_voice(lo, 2.5, 21) + _core.synth_test_voice(hi, 2.5, 22) + _core.synth_test_voice(hi, 1.0, 23)
           + _core.synth_test_voice(lo, 2.0, 24) + _core.synth_test_voice(hi, 7.0, 25))
    os.makedirs(AUDIO_TMP_DIR, exist_ok=True)
    path = os.path.join(AUDIO_TMP_DIR, f"exp1_voiceprint_selftest_{os.getpid()}.wav")
    try:
        _write_pcm_wav(path, pcm)
        a, b, c, d = (extract_turn_voiceprint(path, s, e) for s, e in ((0.0, 2.5), (2.5, 5.0), (5.0, 8.0), (8.0, 15.0)))
        cos = _core._vp_cos
        assert a["source"] == _core.VP_SOURCE_WAV and len(a["voiceprint"]) == _core.VP_DIM, a["source"]
        assert a["voiceprint"] == _core.extract_turn_voiceprint(pcm, 16000, 0.0, 2.5)["voiceprint"], "wrapper == core"
        assert c["voiceprint"] == _core.extract_turn_voiceprint(pcm, 16000, 5.0, 8.0)["voiceprint"], "offset handling"
        assert cos(a["voiceprint"], c["voiceprint"]) > 0.9 > 0.5 > cos(a["voiceprint"], b["voiceprint"])
        assert c["collar_applied"] and c["windows"] == [[6.5, 7.5, 7.125]], c["windows"]   # skips the 1 s bleed
        assert len(d["sub_voiceprints"]) == 2 and all(cos(s, b["voiceprint"]) > 0.9 for s in d["sub_voiceprints"])
        assert 100 < a["f0_hz"] < 125 and 205 < b["f0_hz"] < 250 and 0.0 <= a["near_field_score"] <= 1.0
        assert a["rms_energy"] > 1000 and 0.0 < a["zcr_pitch"] < b["zcr_pitch"], (a["zcr_pitch"], b["zcr_pitch"])
        assert extract_turn_voiceprint(path, 99.0, 100.0)["source"] == _core.VP_SOURCE_PROXY_TEXT   # past EOF
    finally:
        if os.path.exists(path):
            os.remove(path)


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
        _voiceprint_self_test()
        assert not [f for f in os.listdir(AUDIO_TMP_DIR) if f.startswith("exp1_voiceprint_selftest")], "temp WAV left"
        print("EXP1_AUDIO_OK")
        sys.exit(0)
    print(check_audio_env())
    print(info)
