import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
import uuid

try:
    from src.mpc_core import DialogueSessionState, UtteranceTurn, XiaoZhaoMPCEngine, load_from_whisper_json
    from src.iot_api_client import XiaoZhaoIoTClient
    from src.exp1_pi_audio_check import compute_wav_acoustics, record_audio
    from src.config import XIAOZHAO_WHISPER_API_TIMEOUT, XIAOZHAO_WHISPER_API_URL
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mpc_core import DialogueSessionState, UtteranceTurn, XiaoZhaoMPCEngine, load_from_whisper_json
    from iot_api_client import XiaoZhaoIoTClient
    from exp1_pi_audio_check import compute_wav_acoustics, record_audio
    from config import XIAOZHAO_WHISPER_API_TIMEOUT, XIAOZHAO_WHISPER_API_URL

_SPEAKER_RE = re.compile(r"^speaker_\d+$")


def auto_infer_role_from_text(text: str, idx: int) -> tuple:
    """Heuristic (speaker_id, role_hint) when raw STT lacks speaker roles."""
    if "小照" in text or "回報" in text:
        return "speaker_3", "caregiver"
    if "居服員" in text or "麻煩" in text or "我們想" in text:
        return "speaker_2", "family"
    return "speaker_1", "elder"


def _fallback_segments() -> list:
    return [
        {"start": 10.0, "text": "居服員小姐，待會麻煩順便幫我買兩盒普拿疼止痛藥好嗎？"},
        {"start": 45.0, "text": "今天中午的排骨便當菜色很鹹欸，電視八點檔又在重播。"},
        {"start": 520.0, "text": "小照，家屬剛剛說的那樣可以嗎？"},
    ]


def transcribe_via_xiaozhao_whisper_api(wav_path: str, api_url: str,
                                        timeout: float = XIAOZHAO_WHISPER_API_TIMEOUT) -> list:
    """Upload the WAV slice to the XiaoZhao Whisper API (multipart/form-data: file=<wav>, language=zh,
    response_format=verbose_json) and return [{start, end, text, speaker?}]. Accepts either
    {"segments": [{start, end, text, speaker?}, ...]} or {"text": "..."} (one segment spanning the clip).
    Raises OSError / ValueError on network, HTTP or format errors so the caller can fall back."""
    boundary = uuid.uuid4().hex
    with open(wav_path, "rb") as f:
        wav = f.read()
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
             for k, v in (("language", "zh"), ("response_format", "verbose_json"))]
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                 f'filename="{os.path.basename(wav_path)}"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
                 + wav + b"\r\n")
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(api_url, data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:          # HTTPError / URLError are OSErrors
        data = json.loads(resp.read().decode("utf-8"))
    if isinstance(data.get("segments"), list):
        segs = [{"start": float(x.get("start", 0.0)), "end": x.get("end"), "text": str(x.get("text", "")),
                 **({"speaker": x["speaker"]} if x.get("speaker") else {})} for x in data["segments"]]
    elif isinstance(data.get("text"), str):
        segs = [{"start": 0.0, "end": None, "text": data["text"]}]
    else:
        raise ValueError(f"unexpected Whisper API response keys: {sorted(data)}")
    return [x for x in segs if x["text"].strip()]


def attach_acoustics(wav_path: str, segments: list) -> list:
    """Per-segment rms_energy / zcr_pitch from the segment's own [start, end) span of the local WAV."""
    for i, seg in enumerate(segments):
        end = seg.get("end")
        if end is None:   # open-ended: up to the next segment's start (or EOF)
            end = segments[i + 1]["start"] if i + 1 < len(segments) else None
        ac = compute_wav_acoustics(wav_path, float(seg.get("start", 0.0)), None if end is None else float(end))
        seg["rms_energy"], seg["zcr_pitch"] = ac["rms_energy"], ac["zcr_pitch"]
    return segments


def _local_whisper_segments(wav_path: str) -> list:
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("base", compute_type="int8")
        segs, _ = model.transcribe(wav_path, language="zh")
        return [{"start": s.start, "end": s.end, "text": s.text} for s in segs]
    except ImportError:
        pass
    try:
        import whisper
        res = whisper.load_model("base").transcribe(wav_path, language="zh")
        return [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in res.get("segments", [])]
    except ImportError:
        return []


def stt_segments(wav_path: str, api_url: str = None) -> tuple:
    """(segments, source): XiaoZhao Whisper API -> local faster_whisper / whisper -> demo sentences.
    Real-audio segments carry rms_energy / zcr_pitch; demo sentences do not (0 = unknown downstream)."""
    api_url = XIAOZHAO_WHISPER_API_URL if api_url is None else api_url
    if api_url:
        try:
            segs = transcribe_via_xiaozhao_whisper_api(wav_path, api_url)
            if segs:
                return attach_acoustics(wav_path, segs), "xiaozhao_whisper_api"
        except (OSError, ValueError) as e:
            print(f"[run_live_audio] Whisper API unavailable ({e}); falling back to local STT", file=sys.stderr)
    segs = _local_whisper_segments(wav_path)
    if segs:
        return attach_acoustics(wav_path, segs), "local_whisper"
    return _fallback_segments(), "demo_fallback"


def transcribe_wav_to_session(wav_path: str, session_id: str = "S_LIVE_AUDIO", case_id: str = "JIA_AN_LIVE",
                              api_url: str = None) -> DialogueSessionState:
    segments, source = stt_segments(wav_path, api_url)
    turns = []
    for idx, seg in enumerate(segments, start=1):
        text = seg["text"].strip()
        speaker, role = auto_infer_role_from_text(text, idx)
        if _SPEAKER_RE.match(str(seg.get("speaker", ""))):   # diarized speaker from the API wins
            speaker = seg["speaker"]
        turns.append(UtteranceTurn(idx, float(seg["start"]), speaker, role, text,
                                   rms_energy=float(seg.get("rms_energy", 0.0)),
                                   zcr_pitch=float(seg.get("zcr_pitch", 0.0))))
    session = DialogueSessionState(session_id=session_id, case_id=case_id, turns=turns)
    session.stt_source = source
    return session


def run_pipeline(session: DialogueSessionState, simulate_server_ok: bool = False) -> dict:
    """Analyse + POST to $XIAOZHAO_BACKEND_URL (queued offline on failure); simulate only in self-tests."""
    out = XiaoZhaoIoTClient().run_e2e_pipeline(session, simulate_server_ok=simulate_server_ok)
    out["selected_turn_indices"] = out["result"]["payload"]["selected_turn_indices"]
    return out


def _summary(out: dict) -> str:
    return f"sync_status={out['sync_status']} selected_turn_indices={out['selected_turn_indices']}"


def _api_self_test() -> None:
    """Mock XiaoZhao Whisper API on localhost: upload -> segments -> per-segment acoustics -> MPC; then offline."""
    import socket
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = {}

    class _Mock(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen.update(ctype=self.headers["Content-Type"], has_wav=b"RIFF" in body and b'name="file"' in body)
            out = {"segments": [{"start": 0.0, "end": 0.5, "text": "小照，阿嬤今天血壓有點高", "speaker": "speaker_2"},
                                {"start": 0.5, "end": 1.0, "text": "好，我等一下再量一次"}]}
            data = json.dumps(out, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), _Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        wav = record_audio(duration_sec=1)["file_path"]
        url = f"http://127.0.0.1:{srv.server_port}/v1/stt"
        sess = transcribe_wav_to_session(wav, api_url=url)
        assert sess.stt_source == "xiaozhao_whisper_api" and len(sess.turns) == 2, sess
        assert seen["has_wav"] and seen["ctype"].startswith("multipart/form-data"), seen
        assert sess.turns[0].speaker_id == "speaker_2" and sess.turns[1].speaker_id == "speaker_1"
        for t in sess.turns:   # synthetic 440 Hz sine: RMS ~ 11585, ZCR ~ 0.055
            assert abs(t.rms_energy - 11585) < 50 and abs(t.zcr_pitch - 0.055) < 0.003, t
        assert run_pipeline(sess, simulate_server_ok=True)["sync_status"] == "SYNCED_TO_BACKEND"
    finally:
        srv.shutdown()
        srv.server_close()
    with socket.socket() as sk:                     # a port nothing listens on -> offline fallback
        sk.bind(("127.0.0.1", 0))
        dead = f"http://127.0.0.1:{sk.getsockname()[1]}/v1/stt"
    global _local_whisper_segments
    real_local = _local_whisper_segments
    _local_whisper_segments = lambda _wav: [{"start": 0.0, "end": 1.0, "text": "本地辨識"}]  # noqa: E731
    try:   # routing only: skip loading a real model (the main quiet test already exercises it)
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()) as err:
            segs, src = stt_segments(wav, dead)
            assert src == "local_whisper" and segs[0]["rms_energy"] > 0 and "falling back" in err.getvalue()
            assert stt_segments(wav, "")[1] == "local_whisper"
            _local_whisper_segments = lambda _wav: []  # noqa: E731
            assert stt_segments(wav, dead)[1] == "demo_fallback"
    finally:
        _local_whisper_segments = real_local


def main() -> int:
    ap = argparse.ArgumentParser(description="XiaoZhao live audio -> STT -> MPC -> IoT pipeline")
    ap.add_argument("--record", type=int, metavar="SEC", help="record SEC seconds then run pipeline")
    ap.add_argument("--wav", help="existing WAV path")
    ap.add_argument("--whisper-json", help="Whisper JSON output path")
    ap.add_argument("--whisper-api-url", default=None,
                    help="XiaoZhao Whisper API URL (default: $XIAOZHAO_WHISPER_API_URL; empty = local STT)")
    ap.add_argument("--quiet-test", action="store_true")
    args = ap.parse_args()

    if args.quiet_test:
        wav = record_audio(duration_sec=1)["file_path"]
        out = run_pipeline(transcribe_wav_to_session(wav), simulate_server_ok=True)
        assert out["sync_status"] == "SYNCED_TO_BACKEND", out
        assert out["selected_turn_indices"], out
        assert auto_infer_role_from_text("小照你好", 1) == ("speaker_3", "caregiver")
        assert auto_infer_role_from_text("麻煩你", 1) == ("speaker_2", "family")
        assert auto_infer_role_from_text("好熱", 1) == ("speaker_1", "elder")
        _api_self_test()
        print("LIVE_AUDIO_PIPELINE_OK")
        return 0
    if args.record:
        session = transcribe_wav_to_session(record_audio(duration_sec=args.record)["file_path"],
                                            api_url=args.whisper_api_url)
    elif args.wav:
        session = transcribe_wav_to_session(args.wav, api_url=args.whisper_api_url)
    elif args.whisper_json:
        session = load_from_whisper_json(args.whisper_json)
    else:
        ap.print_help()
        return 2
    print(_summary(run_pipeline(session)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
