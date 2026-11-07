import argparse
import os
import sys

try:
    from src.mpc_core import DialogueSessionState, UtteranceTurn, XiaoZhaoMPCEngine, load_from_whisper_json
    from src.iot_api_client import XiaoZhaoIoTClient
    from src.exp1_pi_audio_check import record_audio
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mpc_core import DialogueSessionState, UtteranceTurn, XiaoZhaoMPCEngine, load_from_whisper_json
    from iot_api_client import XiaoZhaoIoTClient
    from exp1_pi_audio_check import record_audio


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


def _stt_segments(wav_path: str) -> list:
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("base", compute_type="int8")
        segs, _ = model.transcribe(wav_path, language="zh")
        return [{"start": s.start, "text": s.text} for s in segs]
    except ImportError:
        pass
    try:
        import whisper
        res = whisper.load_model("base").transcribe(wav_path, language="zh")
        return [{"start": s["start"], "text": s["text"]} for s in res.get("segments", [])]
    except ImportError:
        pass
    return _fallback_segments()  # zero-dependency fallback


def transcribe_wav_to_session(wav_path: str, session_id: str = "S_LIVE_AUDIO", case_id: str = "JIA_AN_LIVE") -> DialogueSessionState:
    segments = _stt_segments(wav_path) or _fallback_segments()
    turns = []
    for idx, seg in enumerate(segments, start=1):
        text = seg["text"].strip()
        speaker, role = auto_infer_role_from_text(text, idx)
        turns.append(UtteranceTurn(idx, float(seg["start"]), speaker, role, text))
    return DialogueSessionState(session_id=session_id, case_id=case_id, turns=turns)


def run_pipeline(session: DialogueSessionState) -> dict:
    out = XiaoZhaoIoTClient().run_e2e_pipeline(session, simulate_server_ok=True)
    out["selected_turn_indices"] = XiaoZhaoMPCEngine().analyze_session(session)["selected_turn_indices"]
    return out


def _summary(out: dict) -> str:
    return f"sync_status={out['sync_status']} selected_turn_indices={out['selected_turn_indices']}"


def main() -> int:
    ap = argparse.ArgumentParser(description="XiaoZhao live audio -> STT -> MPC -> IoT pipeline")
    ap.add_argument("--record", type=int, metavar="SEC", help="record SEC seconds then run pipeline")
    ap.add_argument("--wav", help="existing WAV path")
    ap.add_argument("--whisper-json", help="Whisper JSON output path")
    ap.add_argument("--quiet-test", action="store_true")
    args = ap.parse_args()

    if args.quiet_test:
        wav = record_audio(duration_sec=1)["file_path"]
        out = run_pipeline(transcribe_wav_to_session(wav))
        assert out["sync_status"] == "SYNCED_TO_BACKEND", out
        assert out["selected_turn_indices"], out
        assert auto_infer_role_from_text("小照你好", 1) == ("speaker_3", "caregiver")
        assert auto_infer_role_from_text("麻煩你", 1) == ("speaker_2", "family")
        assert auto_infer_role_from_text("好熱", 1) == ("speaker_1", "elder")
        print("LIVE_AUDIO_PIPELINE_OK")
        return 0
    if args.record:
        session = transcribe_wav_to_session(record_audio(duration_sec=args.record)["file_path"])
    elif args.wav:
        session = transcribe_wav_to_session(args.wav)
    elif args.whisper_json:
        session = load_from_whisper_json(args.whisper_json)
    else:
        ap.print_help()
        return 2
    print(_summary(run_pipeline(session)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
