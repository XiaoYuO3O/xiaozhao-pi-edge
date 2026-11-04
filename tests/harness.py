import subprocess
import sys

def run_check(name, cmd):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"[PASS] {name}")
        return True
    else:
        print(f"[FAIL] {name}")
        err_lines = (res.stderr or res.stdout).strip().splitlines()[-8:]
        print("\n".join(err_lines))
        return False

if __name__ == "__main__":
    ok = True
    ok &= run_check("Syntax Check", "python3 -m py_compile src/*.py")
    if "--full" in sys.argv:
        ok &= run_check("MPC Core Engine (5 Scenarios)", "python3 src/mpc_core.py --quiet-test")
        ok &= run_check("Eval & Handover Export", "python3 src/eval_and_handover.py --quiet-test")
    sys.exit(0 if ok else 1)
