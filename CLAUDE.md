# Project: XiaoZhao Raspberry Pi 5 Edge Module (v7.6)

## Tech Stack & Constraints
- OS: macOS (Dev/Mock) & Raspberry Pi OS 64-bit (Edge), Python 3.10+
- Role: Multi-party audio capture, addressee recognition, dialogue disentanglement (TD/MF + Bi-CL), MIntRec intent transitions, GIFT 4-edge graph + GSN reply/speaker flow, cross-turn slot fusion, and IoT API payload generation for LINE LIFF confirmation.
- Privacy Rule: NO pre-stored family voiceprint DB. Use speaker_1/2/3 diarization + context role inference.
- Auth Rule: Hit `/api/iot/v1/utterances` and `/api/transcripts/{id}/confirm`.

## Token-Saving Rules for Claude (STRICT)
1. DO NOT read whole files unless necessary; use `grep` or line-range reads.
2. DO NOT output full file contents in chat explanations; apply diffs directly.
3. Run `python3 tests/harness.py --full` to verify changes (`--pi` = Pi runtime checks only).
4. Keep explanations under 3 bullet points.

## File Groups
**(A) Pi runtime, Mode A** (synced by `scripts/sync_to_pi.sh`; verified by `tests/harness.py --pi`)
- `src/config.py`: Mac (`MOCK_PI=1`) vs Pi (`ON_PI=True`) detection + paths.
- `src/mpc_core.py`: Addressee + TD/MF/Bi-CL disentanglement + MIntRec + GIFT + GSN + DST fusion + Whisper JSON loader.
- `src/exp1_pi_audio_check.py`: `arecord` capture (synthetic WAV fallback off the Pi).
- `src/edge_state_machine.py`: Pi state machine + offline queue (idempotent by `client_event_id`).
- `src/iot_api_client.py`: HTTP client for the two backend endpoints.
- `src/run_live_audio.py`: record -> local whisper (fallback: demo sentences) -> MPC -> IoT.

**(B) Offline interview corpus (Mac only)**
- `src/interview_analyzer.py`, `src/batch_corpus_builder.py`, `src/retranscribe_hotspots.py`.
- Data (gitignored): `audio_raw/*.mp3`, `transcripts_out/*_raw_stt.json` (authoritative; hours to regenerate), `*_raw_stt.base.json` (pre-retranscription backups), `*_mpc_report.json`. `audio_tmp/*_16k.wav` is regenerated from the mp3s on demand.

**(C) Backend handover** (`backend_handover_package/` + `.zip`)
- `mpc_core.py` (copy of `src/`), `BACKEND_INTEGRATION.md`, `demo_real_clip.json`, `mpc_5_scenarios_handover.json` (written by `src/eval_and_handover.py`), `real_interviews_extracted_samples.json` (written by the corpus pipeline).

**(D) Tests & tooling**
- `tests/harness.py` (`--full` on Mac, `--pi` on the Pi), `src/eval_and_handover.py`, `scripts/sync_to_pi.sh`.
