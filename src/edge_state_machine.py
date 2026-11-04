import json
import os
import sys
from typing import Dict, List, Set

try:
    from src.mpc_core import XiaoZhaoMPCEngine, build_5_benchmark_scenarios
    from src.config import AUDIO_TMP_DIR
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mpc_core import XiaoZhaoMPCEngine, build_5_benchmark_scenarios
    from config import AUDIO_TMP_DIR


class EdgeRuntimeManager:
    def __init__(self, queue_path: str = None):
        os.makedirs(AUDIO_TMP_DIR, exist_ok=True)
        self.queue_path = queue_path or os.path.join(AUDIO_TMP_DIR, "offline_queue.jsonl")
        self.state_history: List[str] = ["1_Idle_Standby"]
        self.synced_ids: Set[str] = set()

    def _go(self, state: str) -> None:
        self.state_history.append(state)

    def _read_queue(self) -> List[Dict]:
        if not os.path.exists(self.queue_path):
            return []
        with open(self.queue_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def process_session(self, session, network_online: bool = True) -> Dict:
        self._go("2_VAD_Recording")
        self._go("3_Edge_Preprocess")
        payload = XiaoZhaoMPCEngine().analyze_session(session)
        eid = payload["client_event_id"]
        emergency = payload["safety_level"] == "CRITICAL_P0"
        if emergency:
            self._go("0E_Emergency_Broadcast")

        if not network_online:
            self._go("3E_Offline_Buffered")
            queued = {p["client_event_id"] for p in self._read_queue()}
            if eid not in queued and eid not in self.synced_ids:
                with open(self.queue_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            return {"status": "BUFFERED_OFFLINE", "client_event_id": eid, "emergency": emergency}

        self._go("4_Cloud_Sync")
        self._go("5_Wait_LINE_Confirm")
        self.synced_ids.add(eid)
        return {"status": "SYNCED_TO_BACKEND", "client_event_id": eid, "emergency": emergency}

    def flush_offline_queue(self) -> Dict:
        flushed = deduped = 0
        for item in self._read_queue():
            eid = item["client_event_id"]
            if eid in self.synced_ids:
                deduped += 1
            else:
                self.synced_ids.add(eid)
                flushed += 1
        open(self.queue_path, "w").close()
        return {"flushed": flushed, "deduplicated": deduped}


def _quiet_test() -> None:
    scens = build_5_benchmark_scenarios()
    mgr = EdgeRuntimeManager(os.path.join(AUDIO_TMP_DIR, "offline_queue_test.jsonl"))
    open(mgr.queue_path, "w").close()

    r = mgr.process_session(scens[0], network_online=True)
    assert r["status"] == "SYNCED_TO_BACKEND", r
    assert mgr.state_history[-3:] == ["3_Edge_Preprocess", "4_Cloud_Sync", "5_Wait_LINE_Confirm"], mgr.state_history

    n = len(mgr.state_history)
    r = mgr.process_session(scens[3], network_online=True)
    assert r["emergency"] and "0E_Emergency_Broadcast" in mgr.state_history[n:], mgr.state_history

    n = len(mgr.state_history)
    r1 = mgr.process_session(scens[1], network_online=False)
    r2 = mgr.process_session(scens[1], network_online=False)
    assert r1["status"] == r2["status"] == "BUFFERED_OFFLINE"
    assert "3E_Offline_Buffered" in mgr.state_history[n:]
    assert len(mgr._read_queue()) == 1, "duplicate client_event_id must not be queued twice"

    out = mgr.flush_offline_queue()
    assert out["flushed"] == 1 and out["deduplicated"] == 0, out
    assert os.path.getsize(mgr.queue_path) == 0, "queue not cleared"
    assert r1["client_event_id"] in mgr.synced_ids

    # idempotent re-flush: item re-buffered by a fresh manager that already synced it elsewhere
    mgr.synced_ids.add(r1["client_event_id"])
    with open(mgr.queue_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"client_event_id": r1["client_event_id"]}) + "\n")
    out = mgr.flush_offline_queue()
    assert out == {"flushed": 0, "deduplicated": 1}, out
    os.remove(mgr.queue_path)
    print("EDGE_STATE_MACHINE_OK")


if __name__ == "__main__":
    if "--quiet-test" in sys.argv:
        _quiet_test()
        sys.exit(0)
    print("Usage: python3 src/edge_state_machine.py --quiet-test")
