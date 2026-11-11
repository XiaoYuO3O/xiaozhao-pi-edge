# Project: XiaoZhao Raspberry Pi 5 Edge Module (v7.7)

## Tech Stack & Constraints
- OS: macOS (Dev/Mock) & Raspberry Pi OS 64-bit (Edge), Python 3.10+
- Role: Multi-party audio capture, addressee recognition, dialogue disentanglement (TD/MF + Bi-CL), MIntRec intent transitions, GIFT 4-edge graph + GSN reply/speaker flow, acoustic-lexical turn voiceprints + E2CP constrained diarization, cross-turn slot fusion, and IoT API payload generation for LINE LIFF confirmation.
- Privacy Rule: NO pre-stored family voiceprint DB. Use speaker_1/2/3 diarization + context role inference. v7.7 turn voiceprints are session-local clustering evidence only (no enrollment, no identity matching) and never go into the `analyze_session()` payload.
- Auth Rule: Hit `/api/iot/v1/utterances` and `/api/transcripts/{id}/confirm`.
- Pi env vars (`src/config.py`): `XIAOZHAO_BT_DEVICE` (ALSA device for `arecord -D`, default `default`), `XIAOZHAO_WHISPER_API_URL` (empty = local faster_whisper, then demo sentences), `XIAOZHAO_WHISPER_API_TIMEOUT`, `XIAOZHAO_BACKEND_URL` (default `http://127.0.0.1:8000`).

## Token-Saving Rules for Claude (STRICT)
1. DO NOT read whole files unless necessary; use `grep` or line-range reads.
2. DO NOT output full file contents in chat explanations; apply diffs directly.
3. Run `python3 tests/harness.py --full` to verify changes (`--pi` = Pi runtime checks only).
4. Keep explanations under 3 bullet points.

## File Groups
**(A) Pi runtime, Mode A** (synced by `scripts/sync_to_pi.sh`; verified by `tests/harness.py --pi`)
- `src/config.py`: Mac (`MOCK_PI=1`) vs Pi (`ON_PI=True`) detection, paths, BT device, Whisper API URL.
- `src/mpc_core.py`: Addressee + TD/MF/Bi-CL disentanglement + MIntRec + GIFT + GSN + DST fusion + Whisper JSON loader + Upgrade G (`extract_turn_voiceprint` on PCM or JSON-only proxy, `constrained_voiceprint_diarize`: lexical Must/Cannot-Link + E2CP α=0.4 → same-speaker confidence weighting Bi-CL continuity and GSN `U`). Stdlib only.
- `src/exp1_pi_audio_check.py`: `arecord -D <BT device>` capture + `compute_wav_acoustics` (rms_energy / zcr_pitch) + `extract_turn_voiceprint(wav, start, end)` (bounded WAV read → 0.25 s collar, 75%-anchor window, >6 s Must-Link split, 24-d voiceprint, `near_field_score`).
- `src/edge_state_machine.py`: Pi state machine; POSTs each MPC payload, queues on network/5xx failure, flush re-POSTs and removes successes; 4xx goes to the dead-letter `offline_queue.rejected.jsonl` (idempotent by `client_event_id`).
- `src/iot_api_client.py`: HTTP client for the two backend endpoints.
- `src/run_live_audio.py`: record -> XiaoZhao Whisper API (fallback: local whisper -> demo) -> per-segment acoustics + voiceprints -> MPC -> IoT.

**(B) Offline interview corpus (Mac only)**
- `src/interview_analyzer.py`, `src/batch_corpus_builder.py`, `src/retranscribe_hotspots.py`.
- Hotspot voiceprints: `--export-samples` slices only each golden hotspot from `audio_raw/<stem>.mp3` into a temp WAV (`ffmpeg -ss/-t`, else PyAV) that is deleted right after extraction; absent audio → proxy vectors. `python3 src/batch_corpus_builder.py --reuse-reports` rebuilds the samples (cached reports without audio are kept); `--voiceprint-table` prints the v7.7 table.
- Data (gitignored): `audio_raw/*.mp3`, `transcripts_out/*_raw_stt.json` (authoritative; hours to regenerate), `*_raw_stt.base.json` (pre-retranscription backups), `*_mpc_report.json`. `audio_tmp/*_16k.wav` is regenerated from the mp3s on demand and deleted by `analyze_file` when that run created it.

**(C) Backend handover** (`backend_handover_package/` + `.zip`)
- `mpc_core.py` (copy of `src/`), `BACKEND_INTEGRATION.md`, `mpc_5_scenarios_handover.json` (written by `src/eval_and_handover.py`), `real_interviews_extracted_samples.json` + `demo_real_clip.json` (both written by the corpus pipeline's `--export-samples`).

**(D) Tests & tooling**
- `tests/harness.py` (`--full` on Mac, `--pi` on the Pi), `src/eval_and_handover.py`, `scripts/sync_to_pi.sh`.
