import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Set

try:
    from src.mpc_core import XiaoZhaoMPCEngine, build_5_benchmark_scenarios
    from src.config import AUDIO_TMP_DIR, XIAOZHAO_BACKEND_URL
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mpc_core import XiaoZhaoMPCEngine, build_5_benchmark_scenarios
    from config import AUDIO_TMP_DIR, XIAOZHAO_BACKEND_URL

# sender(payload) -> {"ok": bool, "http_status"?: int, "error"?: str}  (iot_api_client.post_json shape)
Sender = Callable[[Dict], Dict]

STATUS_SYNCED = "SYNCED_TO_BACKEND"
STATUS_QUEUED = "OFFLINE_QUEUED"
STATUS_REJECTED = "REJECTED_BY_BACKEND"
RETRYABLE_HTTP = (408, 425, 429)      # plus every 5xx and every network error (no http_status)
ALREADY_SYNCED_HTTP = (409,)          # backend dedupes by client_event_id -> it already has this payload


def upload_outcome(resp: Dict) -> str:
    """Classify a sender response: 'ok' | 'retry' (queue it) | 'rejected' (4xx: retrying will not help)."""
    code = resp.get("http_status")
    if resp.get("ok") or code in ALREADY_SYNCED_HTTP:
        return "ok"
    if code is None or code >= 500 or code in RETRYABLE_HTTP:
        return "retry"
    return "rejected"


class EdgeRuntimeManager:
    """Pi Mode A state machine. process_session() analyses the session, POSTs the MPC payload to
    /api/iot/v1/utterances, and on a network / 5xx failure appends it to the JSONL offline queue (deduped by
    client_event_id). flush_offline_queue() re-POSTs queued payloads and removes only the ones that succeed."""

    def __init__(self, queue_path: str = None, sender: Optional[Sender] = None, base_url: str = None):
        os.makedirs(AUDIO_TMP_DIR, exist_ok=True)
        self.queue_path = queue_path or os.path.join(AUDIO_TMP_DIR, "offline_queue.jsonl")
        # dead-letter file for 4xx-rejected payloads: out of the retry loop, kept for inspection / replay
        self.rejected_path = (self.queue_path[:-len(".jsonl")] if self.queue_path.endswith(".jsonl")
                              else self.queue_path) + ".rejected.jsonl"
        self.state_history: List[str] = ["1_Idle_Standby"]
        self.synced_ids: Set[str] = set()
        self.base_url = (base_url or XIAOZHAO_BACKEND_URL).rstrip("/")
        self._sender = sender

    def _go(self, state: str) -> None:
        self.state_history.append(state)

    def _send(self, payload: Dict, sender: Optional[Sender] = None) -> Dict:
        if sender or self._sender:
            return (sender or self._sender)(payload)
        try:   # lazy: iot_api_client imports this module
            from src.iot_api_client import UTTERANCES_PATH, post_json
        except ImportError:
            from iot_api_client import UTTERANCES_PATH, post_json
        return post_json(self.base_url, UTTERANCES_PATH, payload)

    def _read_queue(self) -> List[Dict]:
        if not os.path.exists(self.queue_path):
            return []
        with open(self.queue_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _write_queue(self, items: List[Dict]) -> None:
        tmp = self.queue_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in items)
        os.replace(tmp, self.queue_path)   # atomic: a crash mid-flush never truncates the queue

    def _enqueue(self, payload: Dict) -> bool:
        eid = payload["client_event_id"]
        if eid in self.synced_ids or eid in {p.get("client_event_id") for p in self._read_queue()}:
            return False
        with open(self.queue_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return True

    def _dead_letter(self, payload: Dict, resp: Dict, source: str) -> None:
        rec = {"client_event_id": payload.get("client_event_id"), "http_status": resp.get("http_status"),
               "error": resp.get("error"), "source": source,
               "rejected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "payload": payload}
        with open(self.rejected_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def read_rejected(self) -> List[Dict]:
        if not os.path.exists(self.rejected_path):
            return []
        with open(self.rejected_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def process_session(self, session, network_online: bool = True, sender: Optional[Sender] = None) -> Dict:
        """network_online=False skips the upload attempt (known offline). On a successful upload the offline
        queue is flushed too, since connectivity is evidently back."""
        self._go("2_VAD_Recording")
        self._go("3_Edge_Preprocess")
        payload = XiaoZhaoMPCEngine().analyze_session(session)
        eid = payload["client_event_id"]
        emergency = payload["safety_level"] == "CRITICAL_P0"
        if emergency:
            self._go("0E_Emergency_Broadcast")
        base = {"client_event_id": eid, "emergency": emergency, "payload": payload}

        if eid in self.synced_ids:
            self._go("5_Wait_LINE_Confirm")
            return {**base, "status": STATUS_SYNCED, "upload": {"ok": True, "deduplicated": True}}
        upload = self._send(payload, sender) if network_online else {"ok": False, "error": "network_offline"}
        outcome = upload_outcome(upload)
        if outcome == "retry":
            self._go("3E_Offline_Buffered")
            queued = self._enqueue(payload)
            return {**base, "status": STATUS_QUEUED, "upload": upload, "newly_queued": queued}
        if outcome == "rejected":
            self._go("3E_Upload_Rejected")
            self._dead_letter(payload, upload, "live")
            return {**base, "status": STATUS_REJECTED, "upload": upload}

        self._go("4_Cloud_Sync")
        self.synced_ids.add(eid)
        flush = self.flush_offline_queue(sender) if self._read_queue() else None
        self._go("5_Wait_LINE_Confirm")
        return {**base, "status": STATUS_SYNCED, "upload": upload, "queue_flush": flush}

    def flush_offline_queue(self, sender: Optional[Sender] = None) -> Dict:
        """POST every queued payload in order; remove the successes (and ids already synced). 4xx-rejected items
        move to the dead-letter file (rejected_path) so they are not re-sent on every flush. Stops at the first
        network/5xx failure: that item and the rest stay queued, untried."""
        flushed = deduped = rejected = 0
        keep: List[Dict] = []
        items = self._read_queue()
        for idx, item in enumerate(items):
            eid = item.get("client_event_id")
            if eid in self.synced_ids:
                deduped += 1
                continue
            resp = self._send(item, sender)
            outcome = upload_outcome(resp)
            if outcome == "ok":
                self.synced_ids.add(eid)
                flushed += 1
            elif outcome == "rejected":
                rejected += 1
                self._dead_letter(item, resp, "flush")   # written before the queue rewrite: never lost
            else:
                keep.extend(items[idx:])
                break
        self._write_queue(keep)
        return {"flushed": flushed, "deduplicated": deduped, "rejected": rejected, "remaining": len(keep)}


def _quiet_test() -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    scens = build_5_benchmark_scenarios()
    received: List[Dict] = []
    mode = {"status": 200}

    class _Backend(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            status = mode["script"].pop(0) if mode.get("script") else mode["status"]
            if status == 200:
                received.append({"path": self.path, "ctype": self.headers["Content-Type"],
                                 "body": json.loads(body.decode("utf-8"))})
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), _Backend)
    port = srv.server_port
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.makedirs(AUDIO_TMP_DIR, exist_ok=True)
    qp = os.path.join(AUDIO_TMP_DIR, "offline_queue_test.jsonl")
    open(qp, "w").close()
    try:
        mgr = EdgeRuntimeManager(qp, base_url=f"http://127.0.0.1:{port}")

        # (a) process_session genuinely POSTs the full MPC JSON payload
        r = mgr.process_session(scens[0])
        assert r["status"] == STATUS_SYNCED and r["upload"]["http_status"] == 200, r
        assert len(received) == 1 and received[0]["path"] == "/api/iot/v1/utterances", received
        assert received[0]["ctype"] == "application/json"
        got = received[0]["body"]
        assert got == r["payload"] and got["client_event_id"] == r["client_event_id"], "payload mismatch"
        assert {"dst_slots", "gsn_flow_summary", "turns_detail", "safety_level"} <= set(got), sorted(got)
        assert mgr.state_history[-3:] == ["3_Edge_Preprocess", "4_Cloud_Sync", "5_Wait_LINE_Confirm"], \
            mgr.state_history
        n = len(mgr.state_history)
        r = mgr.process_session(scens[3])
        assert r["emergency"] and "0E_Emergency_Broadcast" in mgr.state_history[n:] and len(received) == 2

        # server down (real connection refused): queued, deduped by client_event_id, nothing received
        srv.shutdown()
        srv.server_close()
        n = len(mgr.state_history)
        r1 = mgr.process_session(scens[1])
        r2 = mgr.process_session(scens[1])
        r3 = mgr.process_session(scens[2], network_online=False)
        assert r1["status"] == r2["status"] == r3["status"] == STATUS_QUEUED, (r1["status"], r2["upload"])
        assert "http_status" not in r1["upload"] and r1["newly_queued"] and not r2["newly_queued"], r1["upload"]
        assert "3E_Offline_Buffered" in mgr.state_history[n:] and len(received) == 2
        queued = mgr._read_queue()
        assert [p["client_event_id"] for p in queued] == [r1["client_event_id"], r3["client_event_id"]]
        assert queued[0] == r1["payload"], "the full payload must be queued"
        out = mgr.flush_offline_queue()   # still down: nothing removed
        assert out == {"flushed": 0, "deduplicated": 0, "rejected": 0, "remaining": 2}, out
        assert len(mgr._read_queue()) == 2

        # (b) server back up on the same port: a 503 keeps items queued, then 200 POSTs and clears them
        srv = HTTPServer(("127.0.0.1", port), _Backend)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        mode["status"] = 503
        assert mgr.flush_offline_queue()["remaining"] == 2 and len(received) == 2
        mode["status"] = 200
        out = mgr.flush_offline_queue()
        assert out == {"flushed": 2, "deduplicated": 0, "rejected": 0, "remaining": 0}, out
        assert [x["body"] for x in received[2:]] == queued, "queued payloads must be POSTed verbatim, in order"
        assert os.path.getsize(qp) == 0, "queue not cleared"

        # a successful live upload also drains whatever was queued meanwhile
        Session = type(scens[0])
        assert mgr.process_session(scens[4], network_online=False)["status"] == STATUS_QUEUED
        r = mgr.process_session(Session("S_NEW", "C", turns=list(scens[0].turns)))
        assert r["status"] == STATUS_SYNCED and r["queue_flush"]["flushed"] == 1 and os.path.getsize(qp) == 0, r

        # 4xx rejection: never queued; the live payload goes straight to the dead-letter file
        mode["status"] = 400
        n_recv = len(received)
        r = mgr.process_session(Session("S_BAD", "C", turns=list(scens[0].turns)))
        assert r["status"] == STATUS_REJECTED and os.path.getsize(qp) == 0, r
        dl = mgr.read_rejected()
        assert len(dl) == 1 and dl[0]["source"] == "live" and dl[0]["http_status"] == 400, dl
        assert dl[0]["payload"] == r["payload"] and dl[0]["client_event_id"] == r["client_event_id"]
        # a queued item rejected during flush moves out of the active queue; the good item behind it is sent
        mode["status"] = 200
        bad = {"client_event_id": "E_BAD", "turns_detail": []}
        mgr._write_queue([bad])
        r_ok = mgr.process_session(Session("S_OK", "C", turns=list(scens[0].turns)), network_online=False)
        mode["script"] = [422, 200]                  # real HTTP: first POST rejected, second accepted
        out = mgr.flush_offline_queue()
        assert not mode["script"] and received[-1]["body"]["client_event_id"] == r_ok["client_event_id"]
        assert out == {"flushed": 1, "deduplicated": 0, "rejected": 1, "remaining": 0}, out
        assert os.path.getsize(qp) == 0 and r_ok["client_event_id"] in mgr.synced_ids
        dl = mgr.read_rejected()
        assert [(d["client_event_id"], d["source"], d["http_status"]) for d in dl[1:]] == [("E_BAD", "flush", 422)]
        assert dl[1]["payload"] == bad, dl[1]
        assert mgr.flush_offline_queue() == {"flushed": 0, "deduplicated": 0, "rejected": 0, "remaining": 0}
        assert len(mgr.read_rejected()) == 2, "a re-flush must not re-send or re-dead-letter"
        n_recv = len(received)
        # 409 = backend already has this client_event_id -> treated as synced and removed
        mode["status"] = 409
        mgr._write_queue([{"client_event_id": "E_DUP", "turns_detail": []}])
        assert mgr.flush_offline_queue()["flushed"] == 1 and os.path.getsize(qp) == 0
        assert len(received) == n_recv   # mock records 200s only

        # an id already synced locally is deduplicated without another POST
        mgr._write_queue([{"client_event_id": r1["client_event_id"]}])
        mode["status"] = 200
        out = mgr.flush_offline_queue()
        assert out == {"flushed": 0, "deduplicated": 1, "rejected": 0, "remaining": 0} and len(received) == n_recv
    finally:
        srv.shutdown()
        srv.server_close()
        for path in (qp, qp[:-len(".jsonl")] + ".rejected.jsonl"):
            if os.path.exists(path):
                os.remove(path)
    print("EDGE_STATE_MACHINE_OK")


if __name__ == "__main__":
    if "--quiet-test" in sys.argv:
        _quiet_test()
        sys.exit(0)
    print("Usage: python3 src/edge_state_machine.py --quiet-test")
