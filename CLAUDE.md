cat << 'EOF' > CLAUDE.md
# Project: XiaoZhao Raspberry Pi 5 Edge Module (v7.2)

## Tech Stack & Constraints
- OS: macOS (Dev/Mock) & Raspberry Pi OS 64-bit (Edge), Python 3.10+
- Role: Multi-party audio capture, addressee recognition, dialogue disentanglement (TD/MF), and IoT API payload generation for LINE LIFF confirmation.
- Privacy Rule: NO pre-stored family voiceprint DB. Use speaker_1/2/3 diarization + context role inference.
- Auth Rule: Hit `/api/iot/v1/utterances` and `/api/transcripts/{id}/confirm`.

## Token-Saving Rules for Claude (STRICT)
1. DO NOT read whole files unless necessary; use `grep` or line-range reads.
2. DO NOT output full file contents in chat explanations; apply diffs directly.
3. Run `python3 tests/harness.py --full` to verify changes.
4. Keep explanations under 3 bullet points.

## Core Files
- `src/config.py`: Auto-detects Mac (`MOCK_PI=1`) vs Raspberry Pi (`ON_PI=True`).
- `src/mpc_core.py`: Addressee + Disentanglement + DST state machine + Whisper JSON loader.
- `src/eval_and_handover.py`: Ablation study benchmark + Offline buffer state machine + JSON export.
- `tests/harness.py`: Automated verification harness (quiet mode).
