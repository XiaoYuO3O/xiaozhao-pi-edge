#!/usr/bin/env python3
"""Multi-recording batch analyzer with 15-min auto-chunking & cross-file deduplication."""
import argparse, inspect, json, pathlib, re, subprocess, sys, time, wave
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import interview_analyzer as ia
import mpc_core
from mpc_core import XiaoZhaoMPCEngine, UtteranceTurn, classify_mintrec_taxonomy

def make_utt(turn_id: int, speaker_id: str, text: str, timestamp: float, role: str = "unknown", addressee: str = "unknown", thread_id: str = "T_MAIN", **extra) -> UtteranceTurn:
    """extra: optional UtteranceTurn fields (v7.7 end_time / rms_energy / zcr_pitch / voiceprint ...)."""
    sig = inspect.signature(UtteranceTurn)
    kwargs = {k: v for k, v in extra.items() if k in sig.parameters}
    for k in sig.parameters:
        if k == "turn_id": kwargs[k] = turn_id
        elif k == "speaker_id": kwargs[k] = speaker_id
        elif k == "text": kwargs[k] = text
        elif k == "timestamp": kwargs[k] = timestamp
        elif k == "inferred_role": kwargs[k] = role
        elif k == "addressee": kwargs[k] = addressee
        elif k == "thread_id": kwargs[k] = thread_id
    return UtteranceTurn(**kwargs)

def run_engine_session(engine: XiaoZhaoMPCEngine, session_id: str, utts: list,
                       role_priors: dict = None) -> dict:
    if hasattr(mpc_core, "DialogueSessionState"):
        cls = mpc_core.DialogueSessionState
        sig = inspect.signature(cls)
        kwargs = {}
        for k, param in sig.parameters.items():
            if k == "session_id":
                kwargs[k] = session_id
            elif k == "case_id":
                kwargs[k] = f"case_{session_id}"
            elif k == "turns":
                kwargs[k] = list(utts)
            elif param.default is inspect.Parameter.empty:
                kwargs[k] = ""
        sess = cls(**kwargs)
        if not getattr(sess, "turns", None):
            sess.turns = list(utts)
    else:
        class _DummySession:
            pass
        sess = _DummySession()
        sess.session_id = session_id
        sess.turns = list(utts)
    if role_priors and "role_priors" in inspect.signature(engine.analyze_session).parameters:
        return engine.analyze_session(sess, role_priors=role_priors)
    return engine.analyze_session(sess)

def convert_to_16k_wav(audio_path: pathlib.Path, wav_path: pathlib.Path):
    if wav_path.exists() and wav_path.stat().st_size > 1000:
        return
    print(f"[{audio_path.stem}] Converting to 16kHz mono WAV (native AudioToolbox / PyAV)...", flush=True)
    for fn_name in ("ensure_wav_16k", "to_16k_wav", "convert_to_wav", "_to_wav"):
        fn = getattr(ia, fn_name, None)
        if callable(fn):
            try:
                res = fn(str(audio_path))
                if res and pathlib.Path(res).exists():
                    return
            except Exception:
                pass
    try:
        subprocess.run([
            "/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
            str(audio_path), str(wav_path)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if wav_path.exists() and wav_path.stat().st_size > 1000:
            return
    except Exception:
        pass
    from faster_whisper.audio import decode_audio
    import numpy as np
    audio_np = decode_audio(str(audio_path), sampling_rate=16000)
    pcm16 = (np.clip(audio_np, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm16.tobytes())

def char_ngrams(text: str, n: int = 3) -> set:
    clean = re.sub(r"[\s，,。！？!?:：；;、]", "", text or "")
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i:i+n] for i in range(len(clean) - n + 1)}

def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))

def turn_quality_score(t: dict, engine: XiaoZhaoMPCEngine) -> float:
    txt = ia.normalize_asr_text(t.get("text", ""))
    intents = engine.detect_domain_intents(txt)
    mint = classify_mintrec_taxonomy(txt)
    score = min(len(txt), 80) * 0.1
    score += len(intents) * 5.0
    if mint.get("fine", "UNK") != "UNK":
        score += 2.0
    return score

def fmt_mmss(sec: float) -> str:
    sec = max(0.0, float(sec))
    return f"{int(sec//60):02d}:{int(sec%60):02d}"

def chunked_pre_transcribe_if_needed(audio_path: pathlib.Path, model_size: str = "base", chunk_Minutes: int = 15):
    out_dir = pathlib.Path("transcripts_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / f"{audio_path.stem}_raw_stt.json"
    if cache_file.exists():
        return

    tmp_dir = pathlib.Path("audio_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    wav_path = tmp_dir / f"{audio_path.stem}_16k.wav"
    convert_to_16k_wav(audio_path, wav_path)

    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        total_frames = wf.getnframes()
        total_sec = total_frames / float(sr)

    chunk_sec = chunk_Minutes * 60
    if total_sec <= chunk_sec * 1.2:
        return

    partial_file = out_dir / f"{audio_path.stem}_raw_stt.partial.json"
    all_segments = []
    start_offset = 0.0
    if partial_file.exists():
        pdata = json.loads(partial_file.read_text(encoding="utf-8"))
        all_segments = pdata.get("segments", [])
        start_offset = float(pdata.get("next_offset_sec", 0.0))
        print(f"[{audio_path.stem}] Resuming from checkpoint at {fmt_mmss(start_offset)} ({len(all_segments)} segments loaded)...", flush=True)

    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    prompt = "繁體中文台灣長照機構訪談：嘉安、照服員、居服員、居督、督導、主任、知識庫、法規、長照基本法、小照、個案、家屬、BA24、SC項目、跌倒、緊急鈕。"

    total_chunks = int((total_sec + chunk_sec - 1) // chunk_sec)
    cur_chunk = int(start_offset // chunk_sec) + 1

    while start_offset < total_sec:
        end_offset = min(total_sec, start_offset + chunk_sec)
        chunk_wav = tmp_dir / f"{audio_path.stem}_chunk_tmp.wav"
        t0 = time.time()
        print(f"[{audio_path.stem}] Transcribing chunk {cur_chunk}/{total_chunks} ({fmt_mmss(start_offset)} - {fmt_mmss(end_offset)})...", flush=True)

        with wave.open(str(wav_path), "rb") as rf:
            rf.setpos(int(start_offset * sr))
            frames = rf.readframes(int((end_offset - start_offset) * sr))
            with wave.open(str(chunk_wav), "wb") as wf_out:
                wf_out.setnchannels(1)
                wf_out.setsampwidth(2)
                wf_out.setframerate(sr)
                wf_out.writeframes(frames)

        segs, _ = model.transcribe(str(chunk_wav), language="zh", vad_filter=True, beam_size=1, best_of=1, initial_prompt=prompt)
        chunk_count = 0
        for s in segs:
            all_segments.append({
                "start": round(float(s.start) + start_offset, 2),
                "end": round(float(s.end) + start_offset, 2),
                "text": s.text.strip()
            })
            chunk_count += 1

        chunk_wav.unlink(missing_ok=True)
        start_offset = end_offset
        partial_file.write_text(json.dumps({"next_offset_sec": start_offset, "segments": all_segments}, ensure_ascii=False), encoding="utf-8")
        print(f"  -> Chunk {cur_chunk}/{total_chunks} done in {time.time()-t0:.1f}s (+{chunk_count} segs, total={len(all_segments)})", flush=True)
        cur_chunk += 1

    ref_cache = out_dir / "interview1_raw_stt.json"
    if ref_cache.exists():
        ref_obj = json.loads(ref_cache.read_text(encoding="utf-8"))
        if isinstance(ref_obj, dict):
            out_obj = dict(ref_obj)
            out_obj["segments"] = all_segments
            out_obj["duration_sec"] = total_sec
        else:
            out_obj = all_segments
    else:
        out_obj = all_segments

    cache_file.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    partial_file.unlink(missing_ok=True)
    print(f"[{audio_path.stem}] Full transcription cached -> {cache_file.name} ({len(all_segments)} segments)!", flush=True)

SAMPLES_PATH = pathlib.Path("backend_handover_package/real_interviews_extracted_samples.json")


def _hotspot_digest(h: dict) -> dict:
    """Compact MPC verdict of one hotspot (works for pre-upgrade samples that lack A/B/C fields)."""
    mpc = h.get("mpc_analysis") or h.get("mpc_result") or {}
    fusion = (mpc.get("dst_slots") or {}).get("cross_turn_fusion") or mpc.get("cross_turn_fusion") or {}
    return {"route": mpc.get("route_target"), "intents": mpc.get("detected_intents"),
            "roles": mpc.get("speaker_roles"),
            "transitions": [f"{x['type']}({x['from_turn']}->{x['to_turn']})" for x in mpc.get("mintrec_transitions") or []],
            "issues": [i["label"] for i in fusion.get("issue_triggers", [])],
            "resolutions": [r["label"] for r in fusion.get("proposed_resolutions", [])],
            "consensus": fusion.get("consensus_state")}


def snapshot_hotspots(path: pathlib.Path = SAMPLES_PATH) -> dict:
    """(source, rank) -> (window, digest) of the currently exported hotspots, taken before a rebuild."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {(iv.get("source"), idx): (h.get("window_mmss") or h.get("window"), _hotspot_digest(h))
            for iv in data.get("interviews", []) for idx, h in enumerate(iv.get("hotspots", []), 1)}


def _print_digest(tag: str, window: str, d: dict) -> None:
    print(f"    {tag} [{window}] route={d['route']} intents={d['intents']}")
    print(f"           roles={d['roles'] or '—'}")
    print(f"           transitions={d['transitions'] or '—'}")
    print(f"           issues={d['issues'] or '—'} | resolutions={d['resolutions'] or '—'} | consensus={d['consensus'] or '—'}")


def run_batch(audio_dir: str, model_size: str = "base", reuse_reports: bool = False) -> int:
    before = snapshot_hotspots()
    adir = pathlib.Path(audio_dir)
    out_dir = pathlib.Path("transcripts_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_files = sorted([
        p for p in adir.iterdir()
        if p.is_file() and p.suffix.lower() in (".mp3", ".m4a", ".wav") and not p.name.startswith(".")
    ])
    skipped = [af.stem for af in audio_files if not ia.is_handover_stem(af.stem)]
    audio_files = [af for af in audio_files if ia.is_handover_stem(af.stem)]
    # --reuse-reports also keeps recordings whose audio is absent (their hotspots fall back to proxy voiceprints)
    report_only = sorted(rp.name[: -len("_mpc_report.json")] for rp in out_dir.glob("*_mpc_report.json")
                         if rp.name[: -len("_mpc_report.json")] not in {af.stem for af in audio_files}
                         and ia.is_handover_stem(rp.name[: -len("_mpc_report.json")])) \
        if reuse_reports else []
    for stem in skipped:
        print(f"[{stem}] lab-meeting recording: excluded from the Jia-An handover corpus "
              f"(its *_mpc_report.json stays a separate report)")
    if not audio_files and not report_only:
        print(f"[WARN] No audio files found in {adir}")
        return 1

    print(f"=== Batch Scanning {len(audio_files)} Audio File(s) in {adir} ===")
    for stem in report_only:
        print(f"[{stem}] audio absent from {adir}: cached report only (voiceprints fall back to proxies)")
    for af in audio_files:
        chunked_pre_transcribe_if_needed(af, model_size=model_size, chunk_Minutes=15)
        rp = out_dir / f"{af.stem}_mpc_report.json"
        if reuse_reports and rp.exists():
            print(f"[{af.stem}] Using cached MPC report -> {rp.name}")
        else:  # clean rebuild from the cached *_raw_stt.json (no re-transcription when the cache is complete)
            rp.unlink(missing_ok=True)
            if ia.analyze_file(str(af), model_size, 0) != 0:
                print(f"[ERR] [{af.stem}] MPC report rebuild failed")
                return 2

    engine = XiaoZhaoMPCEngine()
    reports = []
    for stem in sorted({af.stem for af in audio_files} | set(report_only)):
        rp = out_dir / f"{stem}_mpc_report.json"
        if rp.exists():
            reports.append((stem, json.loads(rp.read_text(encoding="utf-8"))))

    seen_turns = []
    total_raw_turns = 0
    dup_turns_removed = 0
    quality_upgrades = 0

    rec_roles, rec_role_sources, rec_graph_roles = {}, {}, {}
    for stem, rep in reports:
        # recording-level roles first (whole recording), reused as priors for every hotspot window;
        # always re-derived (cached report roles predate engine upgrades such as GIFT graph inference)
        rec_role_sources[stem], ev = {}, {}
        rec_roles[stem] = ia.recording_speaker_roles(rep.get("merged_turns", []), rec_role_sources[stem], ev)
        rec_graph_roles[stem] = {sp: e["label"] for sp, e in ev.items()}
        gd = rep.get("general_disentangle", {})
        turns = gd.get("turns", rep.get("merged_turns", []))
        for t in turns:
            t["text"] = ia.normalize_asr_text(t.get("text", ""))
            sp = t.get("speaker_id") or t.get("speaker") or "speaker_1"
            t["speaker_id"] = sp
            t["speaker"] = sp
            if t.get("thread_id") == "T_BACKCHANNEL_NOISE":
                continue
            total_raw_turns += 1
            ng = char_ngrams(t["text"], 3)
            if len(ng) < 3:
                continue
            q = turn_quality_score(t, engine)
            matched = False
            for idx, prev in enumerate(seen_turns):
                sim = jaccard(ng, prev["ngrams"])
                if sim >= 0.35 and prev["stem"] != stem:
                    matched = True
                    dup_turns_removed += 1
                    if q > prev["quality"]:
                        quality_upgrades += 1
                        t["upgraded_from"] = f"{prev['stem']}@{fmt_mmss(prev['turn'].get('start', 0))}"
                        seen_turns[idx] = {"stem": stem, "turn": t, "ngrams": ng, "quality": q}
                    else:
                        t["is_duplicate"] = True
                        t["duplicate_of"] = f"{prev['stem']}@{fmt_mmss(prev['turn'].get('start', 0))}"
                    break
            if not matched:
                seen_turns.append({"stem": stem, "turn": t, "ngrams": ng, "quality": q})

    subprocess.run([sys.executable, "src/interview_analyzer.py", "--export-samples"], check=True)
    sample_path = SAMPLES_PATH
    sdata = json.loads(sample_path.read_text(encoding="utf-8"))

    dedup_hotspots = []
    for iv in sdata.get("interviews", []):
        src_name = iv.get("source") or iv.get("stem") or "unknown"
        iv["source"] = src_name
        if src_name in rec_roles:
            iv["recording_speaker_roles"] = rec_roles[src_name]
            iv["recording_speaker_role_sources"] = rec_role_sources[src_name]
        kept_for_iv = []
        for idx, h in enumerate(iv.get("hotspots", []), 1):
            turns = h.get("turns", [])
            st = float(h.get("start_sec", h.get("start", turns[0].get("start", 0) if turns else 0)))
            en = float(h.get("end_sec", h.get("end", turns[-1].get("end", st + 60) if turns else st + 60)))
            w_str = h.get("window_mmss") or h.get("window") or f"{fmt_mmss(st)}-{fmt_mmss(en)}"
            h["window_mmss"] = w_str
            h["window"] = w_str

            utts = []
            combined_txt = []
            for i, t in enumerate(turns, 1):
                t["text"] = ia.normalize_asr_text(t.get("text", ""))
                sp = t.get("speaker_id") or t.get("speaker") or f"speaker_{(i % 3) + 1}"
                t["speaker_id"] = sp
                t["speaker"] = sp
                t["addressee"] = t.get("addressee") or "ALL"
                t["thread_id"] = t.get("thread_id") or "T_TOPIC_1"
                t["greedy_thread_id"] = t.get("greedy_thread_id") or t["thread_id"]
                combined_txt.append(t["text"])
                utts.append(make_utt(
                    turn_id=i, speaker_id=sp, text=t["text"],
                    timestamp=float(t.get("start", st + i * 2.0)),
                    addressee=t["addressee"], thread_id=t["greedy_thread_id"],   # Bi-CL refines in-session
                    **ia.voiceprint_turn_kwargs(t)                               # v7.7 voiceprint + acoustics
                ))

            analysis = run_engine_session(engine, f"{src_name}_h{idx}", utts, rec_roles.get(src_name))
            h["mpc_analysis"] = analysis
            h["route_target"] = analysis.get("route_target")
            h["detected_intents"] = analysis.get("detected_intents")

            w_ng = char_ngrams("".join(combined_txt), 3)
            is_dup_win = False
            for prev_w in dedup_hotspots:
                if prev_w["source"] != src_name and jaccard(w_ng, prev_w["ngrams"]) >= 0.25:
                    is_dup_win = True
                    h["cross_recording_duplicate_of"] = f"{prev_w['source']}[{prev_w['window_mmss']}]"
                    break
            if not is_dup_win:
                dedup_hotspots.append({"source": src_name, "window_mmss": w_str, "ngrams": w_ng})
            kept_for_iv.append(h)
        iv["hotspots"] = kept_for_iv

    sdata["corpus_dedup_summary"] = {
        "total_recordings": len(reports),
        "recording_stems": [s for s, _ in reports],
        "total_substantive_turns_raw": total_raw_turns,
        "cross_file_duplicate_turns_removed": dup_turns_removed,
        "quality_upgraded_turns": quality_upgrades,
        "unique_net_substantive_turns": len(seen_turns),
        "unique_golden_hotspots": len(dedup_hotspots),
        "recording_speaker_roles": rec_roles,
        "recording_speaker_role_sources": rec_role_sources,
        "gift_graph_inferred_roles": rec_graph_roles,
    }
    sample_path.write_text(json.dumps(sdata, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "corpus_merged_report.json").write_text(
        json.dumps(sdata["corpus_dedup_summary"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n================ CORPUS DEDUPLICATION SUMMARY ================")
    for k, v in sdata["corpus_dedup_summary"].items():
        print(f"  {k:36s}: {v}")

    for iv in sdata.get("interviews", []):
        print(f"\n================ {iv['source']} ================")
        for idx, h in enumerate(iv.get("hotspots", []), 1):
            mpc = h.get("mpc_analysis", {})
            dup_tag = f" [DUP OF {h['cross_recording_duplicate_of']}]" if h.get("cross_recording_duplicate_of") else ""
            print(f"\n--- Hotspot #{idx} [{h['window_mmss']}]{dup_tag} | Route: {mpc.get('route_target')} | Intents: {mpc.get('detected_intents')} ---")
            fusion = (mpc.get("dst_slots") or {}).get("cross_turn_fusion") or {}
            roles = mpc.get("speaker_roles") or {}
            w2w = {d["turn_id"]: d.get("who_to_whom_role") for d in mpc.get("turns_detail", [])}
            print(f"  speaker_roles: {roles} (sources: {mpc.get('speaker_role_sources')})")
            for x in mpc.get("mintrec_transitions") or []:
                print(f"  transition: {x['type']} [{x['thread_id']}] t{x['from_turn']} {x['from_speaker']}/{x['from_role']}"
                      f" ({x['from_fine']}) -> t{x['to_turn']} {x['to_speaker']}/{x['to_role']} ({x['to_fine']})"
                      f" priority={x['priority']}")
            if not mpc.get("mintrec_transitions"):
                print("  transition: —")
            print(f"  cross_turn_fusion: consensus={fusion.get('consensus_state')} threads={fusion.get('dominant_threads')}"
                  f" roles={fusion.get('participating_roles')}")
            for i in fusion.get("issue_triggers", []):
                print(f"    issue: {i['label']} raised_by {i['raised_by']}/{i['role']} @t{i['turn_id']} ('{i['evidence']}')")
            for r in fusion.get("proposed_resolutions", []):
                print(f"    resolution: {r['label']} proposed_by {r['proposed_by']}/{r['role']} @t{r['turn_id']} ('{r['evidence']}')")
            for k, t in enumerate(h.get("turns", [])[:8], 1):
                print(f"  {t['speaker_id']} [{w2w.get(k, '?')}] ({t['thread_id']}): {t['text']}")

    print("\n================ GOLDEN HOTSPOTS: BEFORE vs AFTER MPC UPGRADE ================")
    for iv in sdata.get("interviews", []):
        for idx, h in enumerate(iv.get("hotspots", []), 1):
            print(f"\n  {iv['source']} #{idx}")
            if (iv["source"], idx) in before:
                _print_digest("BEFORE", *before[(iv["source"], idx)])
            else:
                print("    BEFORE (no previous export)")
            _print_digest("AFTER ", h["window_mmss"], _hotspot_digest(h))
    print_voiceprint_table(sdata)
    return 0


def _fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def print_voiceprint_table(sdata: dict) -> None:
    """v7.7 table: per golden hotspot, real-audio voiceprint + E2CP constrained-diarization metrics."""
    print("\n================ v7.7 GOLDEN HOTSPOTS: ACOUSTIC-LEXICAL VOICEPRINT + E2CP DIARIZATION ================")
    print(f"{'hotspot':22s} {'audio':6s} {'vp/turns':8s} {'ML':>3s} {'CL':>3s} {'intra raw->E2CP (delta)':25s}"
          f" {'inter raw->E2CP':17s} {'K/diar':6s} {'consist':7s} near_field_primary")
    tot = {"ml": 0, "cl": 0, "d_intra": [], "d_inter": []}
    for iv in sdata.get("interviews", []):
        for idx, h in enumerate(iv.get("hotspots", []), 1):
            mpc = h.get("mpc_analysis") or h.get("mpc_result") or {}
            v = mpc.get("voiceprint_diarization_summary") or {}
            if not v:
                print(f"{iv['source'][:12]:12s} #{idx} {h.get('window_mmss', ''):8s} (no v7.7 summary)")
                continue
            ri, ei = v.get("mean_intra_speaker_affinity_raw"), v.get("mean_intra_speaker_affinity_e2cp")
            ro, eo = v.get("mean_inter_speaker_affinity_raw"), v.get("mean_inter_speaker_affinity_e2cp")
            if ri is not None and ei is not None:
                tot["d_intra"].append(ei - ri)
            if ro is not None and eo is not None:
                tot["d_inter"].append(eo - ro)
            tot["ml"] += v.get("must_link_constraints", 0)
            tot["cl"] += v.get("cannot_link_constraints", 0)
            mode = "wav" if v.get("voiceprint_mode") == "wav_75pct_anchor" else "proxy"
            n = v.get("turns_with_voiceprint", 0) + v.get("turns_with_proxy_voiceprint", 0)
            d = f"({ei - ri:+.4f})" if ri is not None and ei is not None else ""
            print(f"{iv['source'][:12]:12s} #{idx} {h.get('window_mmss', '')[:5]:5s}  {mode:6s} "
                  f"{v.get('turns_with_voiceprint', 0):>3d}/{n:<4d} {v.get('must_link_constraints', 0):>3d} "
                  f"{v.get('cannot_link_constraints', 0):>3d} {_fmt(ri, 4)}->{_fmt(ei, 4)} {d:10s} "
                  f"{_fmt(ro, 4)}->{_fmt(eo, 4)}  {v.get('estimated_speakers', 0)}/{v.get('diarizer_speakers', 0):<4d} "
                  f"{_fmt(v.get('diarization_consistency_score')):7s} {v.get('near_field_primary_speaker')}")
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0   # noqa: E731
    print(f"  totals: must_link={tot['ml']} cannot_link={tot['cl']} | mean intra-speaker delta (E2CP - raw) = "
          f"{mean(tot['d_intra']):+.4f} | mean inter-speaker delta = {mean(tot['d_inter']):+.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", default="audio_raw")
    ap.add_argument("--model", default="base")
    ap.add_argument("--reuse-reports", action="store_true",
                    help="skip the clean *_mpc_report.json rebuild and reuse existing reports")
    ap.add_argument("--voiceprint-table", action="store_true",
                    help="only print the v7.7 voiceprint table of the current exported samples")
    args = ap.parse_args()
    if args.voiceprint_table:
        print_voiceprint_table(json.loads(SAMPLES_PATH.read_text(encoding="utf-8")))
        sys.exit(0)
    sys.exit(run_batch(args.audio_dir, args.model, args.reuse_reports))
