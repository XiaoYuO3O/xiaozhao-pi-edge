#!/usr/bin/env python3
"""Targeted hotspot re-transcription: re-run ONLY the Top-K golden 60 s hotspot windows with a larger
faster-whisper model and splice the cleaner segments back into transcripts_out/<stem>_raw_stt.json.

    python3 src/retranscribe_hotspots.py --stems interview1 interview2 --model small
    python3 src/retranscribe_hotspots.py --stems interview3_full --until-converged   # max 3 passes

- Slices [hotspot.start - pad, hotspot.end + pad] (60 s + 2*pad, default 80 s) from audio_tmp/<stem>_16k.wav.
- Old segments whose midpoint falls inside the window are replaced by the new ones (same midpoint rule).
- The original cache is backed up once to <stem>_raw_stt.base.json; refined windows are recorded in
  `refined_windows` and every segment carries a `"model"` tag, so re-runs skip windows whose 60 s span is
  already fully covered (interval union) by that model.
- --until-converged: re-run the analyzer -> refine uncovered Top-K windows -> repeat (<= MAX_PASSES) until
  the current Top-K hotspots are 100% transcribed by --model.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from src import interview_analyzer as ia
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import interview_analyzer as ia
AUDIO_TMP, OUT_DIR, WHISPER_INITIAL_PROMPT = ia.AUDIO_TMP, ia.OUT_DIR, ia.WHISPER_INITIAL_PROMPT
mmss, normalize_asr_text = ia.mmss, ia.normalize_asr_text

PAD_S = 10.0
MAX_PASSES = 3
AUDIO_EXTS = (".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac", ".mp4")


def slice_wav(wav_path: Path, start: float, end: float, out_path: Path) -> Tuple[float, float]:
    """Copy [start, end) of a PCM WAV via seek (never loads the full file). Returns the clipped range."""
    with wave.open(str(wav_path), "rb") as src:
        rate, n = src.getframerate(), src.getnframes()
        a, b = max(0, int(start * rate)), min(n, int(end * rate))
        src.setpos(a)
        frames = src.readframes(max(0, b - a))
        with wave.open(str(out_path), "wb") as dst:
            dst.setparams(src.getparams())
            dst.writeframes(frames)
    return a / rate, b / rate


def transcribe_slice(model, path: Path, offset: float, model_size: str) -> List[dict]:
    segs, _ = model.transcribe(str(path), language="zh", vad_filter=True, beam_size=5,
                               condition_on_previous_text=False,  # stops 對啦x50-style loops propagating
                               initial_prompt=WHISPER_INITIAL_PROMPT)
    return [{"start": round(s.start + offset, 2), "end": round(s.end + offset, 2),
             "text": normalize_asr_text(s.text.strip()), "refined": True,
             "model": model_size}
            for s in segs if s.text.strip()]


def splice(segments: List[dict], new: List[dict], ws: float, we: float) -> List[dict]:
    mid = lambda s: (float(s["start"]) + float(s["end"])) / 2  # noqa: E731
    kept = [s for s in segments if not (ws <= mid(s) < we)]
    return sorted(kept + [s for s in new if ws <= mid(s) < we], key=lambda s: float(s["start"]))


def tag_models(data: dict) -> None:
    """Backfill per-segment `model` tags: refined segments came from a refined window's model, rest = base."""
    base = data.get("model", "base")
    refined_model = next((d["model"] for d in data.get("refined_windows", [])), "small")
    for s in data["segments"]:
        s.setdefault("model", refined_model if s.get("refined") else base)


def covered(done: List[dict], model_size: str, ws: float, we: float) -> bool:
    """True if the union of `model_size` refined windows covers [ws, we]."""
    spans = sorted((d["start"], d["end"]) for d in done if d["model"] == model_size)
    reach = ws
    for a, b in spans:
        if a > reach:
            break
        reach = max(reach, b)
        if reach >= we:
            return True
    return reach >= we


def window_models(data: dict, ws: float, we: float) -> Dict[str, int]:
    """Count raw segments (midpoint rule) per model tag inside [ws, we)."""
    counts: Dict[str, int] = {}
    for s in data["segments"]:
        if ws <= (float(s["start"]) + float(s["end"])) / 2 < we:
            counts[s.get("model", "?")] = counts.get(s.get("model", "?"), 0) + 1
    return counts


def fully_refined(data: dict, model_size: str, ws: float, we: float) -> bool:
    """Window is done only if the refined-window ledger covers it AND every segment in it is tagged
    `model_size` (the ledger alone can be stale, e.g. metadata inherited without the spliced segments)."""
    models = window_models(data, ws, we)
    return covered(data.get("refined_windows", []), model_size, ws, we) and set(models) <= {model_size}


def refine_stem(stem: str, model, model_size: str, top_k: int, pad: float,
                hotspots: Optional[List[dict]] = None) -> List[dict]:
    cache = OUT_DIR / f"{stem}_raw_stt.json"
    wav = AUDIO_TMP / f"{stem}_16k.wav"
    data = json.loads(cache.read_text(encoding="utf-8"))
    backup = OUT_DIR / f"{stem}_raw_stt.base.json"
    if not backup.exists():
        backup.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    done = data.setdefault("refined_windows", [])
    tag_models(data)
    if hotspots is None:
        hotspots = json.loads((OUT_DIR / f"{stem}_mpc_report.json").read_text(encoding="utf-8"))["golden_hotspots"]
    log = []
    for h in hotspots[:top_k]:
        if fully_refined(data, model_size, float(h["start"]), float(h["end"])):
            log.append({"window": f"{h['start_mmss']}-{h['end_mmss']}", "status": "already refined"})
            continue
        ws, we = float(h["start"]) - pad, float(h["end"]) + pad
        clip = AUDIO_TMP / f"{stem}_hs{h['rank']}_{int(ws)}-{int(we)}.wav"
        ws, we = slice_wav(wav, ws, we, clip)
        t0 = time.time()
        new = transcribe_slice(model, clip, ws, model_size)
        old_n = sum(1 for s in data["segments"] if ws <= (float(s["start"]) + float(s["end"])) / 2 < we)
        data["segments"] = splice(data["segments"], new, ws, we)
        done.append({"start": round(ws, 2), "end": round(we, 2), "model": model_size, "hotspot_rank": h["rank"]})
        log.append({"window": f"{mmss(ws)}-{mmss(we)}", "old_segments": old_n, "new_segments": len(new),
                    "sec": round(time.time() - t0, 1)})
        clip.unlink(missing_ok=True)
    cache.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return log


def find_audio(stem: str, audio_dir: Path) -> Path:
    for ext in AUDIO_EXTS:
        if (audio_dir / f"{stem}{ext}").exists():
            return audio_dir / f"{stem}{ext}"
    raise FileNotFoundError(f"no audio for {stem} in {audio_dir}")


def current_hotspots(stem: str, audio_dir: Path) -> List[dict]:
    """Re-run the analyzer on the (spliced) cache and return its fresh hotspot ranking."""
    with contextlib.redirect_stdout(io.StringIO()):
        rc = ia.analyze_file(str(find_audio(stem, audio_dir)), "base", 0)
    if rc != 0:
        raise RuntimeError(f"analyzer failed for {stem} (rc={rc})")
    return json.loads((OUT_DIR / f"{stem}_mpc_report.json").read_text(encoding="utf-8"))["golden_hotspots"]


def converge_stem(stem: str, get_model, model_size: str, top_k: int, pad: float, audio_dir: Path) -> bool:
    """Analyzer -> refine uncovered Top-K -> repeat until Top-K are fully `model_size` (<= MAX_PASSES)."""
    for n in range(1, MAX_PASSES + 2):  # last iteration is verification only
        hs = current_hotspots(stem, audio_dir)[:top_k]
        data = json.loads((OUT_DIR / f"{stem}_raw_stt.json").read_text(encoding="utf-8"))
        tag_models(data)
        todo = [h for h in hs if not fully_refined(data, model_size, float(h["start"]), float(h["end"]))]
        print(f"[{stem}] pass {n}: Top{top_k}=" + ", ".join(f"{h['start_mmss']}" for h in hs)
              + f" | uncovered={[h['start_mmss'] for h in todo]}")
        if not todo:
            print(f"[{stem}] CONVERGED after {n - 1} refinement pass(es)")
            return True
        if n > MAX_PASSES:
            break
        for row in refine_stem(stem, get_model(), model_size, top_k, pad, hotspots=todo):
            print(f"[{stem}]   {row}")
    print(f"[{stem}] NOT converged within {MAX_PASSES} passes")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stems", nargs="+", default=["interview1", "interview2"])
    ap.add_argument("--model", default="small")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--pad", type=float, default=PAD_S, help="seconds added on each side of the 60 s window")
    ap.add_argument("--until-converged", action="store_true",
                    help=f"loop analyzer+refine (max {MAX_PASSES} passes) until Top-K are fully --model")
    ap.add_argument("--audio-dir", default="audio_raw")
    args = ap.parse_args()
    cache: Dict[str, object] = {}

    def get_model():
        if "m" not in cache:
            from faster_whisper import WhisperModel  # optional dep
            cache["m"] = WhisperModel(args.model, device="cpu", compute_type="int8")
        return cache["m"]

    ok = True
    for stem in args.stems:
        if args.until_converged:
            ok &= converge_stem(stem, get_model, args.model, args.top_k, args.pad, Path(args.audio_dir))
        else:
            for row in refine_stem(stem, get_model(), args.model, args.top_k, args.pad):
                print(f"[{stem}] {row}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
