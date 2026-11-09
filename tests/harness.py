"""Verification harness. Run from the repo root.

    python3 tests/harness.py --full   # Mac: all 8 checks (Pi runtime + corpus / handover export)
    python3 tests/harness.py --pi     # Raspberry Pi: syntax + the 5 Mode A runtime checks only
"""
import glob
import subprocess
import sys

# Mode A runtime (exactly what scripts/sync_to_pi.sh copies to the Pi)
PI_RUNTIME = ["src/__init__.py", "src/config.py", "src/mpc_core.py", "src/exp1_pi_audio_check.py",
              "src/edge_state_machine.py", "src/iot_api_client.py", "src/run_live_audio.py"]
PI_CHECKS = [
    ("MPC Core Engine (5 Scenarios)", "python3 src/mpc_core.py --quiet-test"),
    ("Exp1 Audio Check", "python3 src/exp1_pi_audio_check.py --quiet-test"),
    ("Edge State Machine & Offline Queue", "python3 src/edge_state_machine.py --quiet-test"),
    ("End-to-End IoT API Client", "python3 src/iot_api_client.py --quiet-test"),
    ("Live Audio & WAV Bridge", "python3 src/run_live_audio.py --quiet-test"),
]
# Mac-only: write backend_handover_package/ and need the offline interview corpus pipeline
MAC_CHECKS = [
    ("Eval & Handover Export", "python3 src/eval_and_handover.py --quiet-test"),
    ("Unedited Interview Analyzer", "python3 src/interview_analyzer.py --quiet-test"),
]


def run_check(name, cmd):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"[PASS] {name}")
        return True
    print(f"[FAIL] {name}")
    err_lines = (res.stderr or res.stdout).strip().splitlines()[-8:]
    print("\n".join(err_lines))
    return False


if __name__ == "__main__":
    pi_mode = "--pi" in sys.argv
    files = PI_RUNTIME if pi_mode else sorted(glob.glob("src/*.py"))
    ok = run_check("Syntax Check" + (" (Pi runtime)" if pi_mode else ""), "python3 -m py_compile " + " ".join(files))
    if pi_mode:
        checks = PI_CHECKS
    elif "--full" in sys.argv:
        checks = PI_CHECKS[:1] + MAC_CHECKS[:1] + PI_CHECKS[1:] + MAC_CHECKS[1:]   # original --full order
    else:
        checks = []
    for name, cmd in checks:
        ok &= run_check(name, cmd)
    sys.exit(0 if ok else 1)
