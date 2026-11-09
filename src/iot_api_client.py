import json
import os
import sys
import urllib.error
import urllib.request

try:
    from src.edge_state_machine import STATUS_QUEUED, STATUS_SYNCED, EdgeRuntimeManager
    from src.mpc_core import build_5_benchmark_scenarios
    from src.exp1_pi_audio_check import record_audio
    from src.config import AUDIO_TMP_DIR, XIAOZHAO_BACKEND_URL
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from edge_state_machine import STATUS_QUEUED, STATUS_SYNCED, EdgeRuntimeManager
    from mpc_core import build_5_benchmark_scenarios
    from exp1_pi_audio_check import record_audio
    from config import AUDIO_TMP_DIR, XIAOZHAO_BACKEND_URL


UTTERANCES_PATH = "/api/iot/v1/utterances"


def post_json(base_url: str, path: str, body: dict, timeout: float = 2.0) -> dict:
    """POST JSON -> {"ok", "http_status", "endpoint"} or {"ok": False, "error"} when unreachable."""
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"ok": 200 <= resp.status < 300, "http_status": resp.status, "endpoint": path}
    except urllib.error.HTTPError as e:
        return {"ok": False, "http_status": e.code, "endpoint": path, "error": str(e)}
    except (urllib.error.URLError, OSError) as e:
        return {"ok": False, "endpoint": path, "error": str(e)}


class XiaoZhaoIoTClient:
    def __init__(self, base_url: str = None, timeout: float = 2.0, queue_path: str = None):
        self.base_url = (base_url or XIAOZHAO_BACKEND_URL).rstrip("/")
        self.timeout = timeout
        self.runtime = EdgeRuntimeManager(queue_path, sender=self.send_utterance_payload, base_url=self.base_url)

    def _post(self, path: str, body: dict) -> dict:
        return post_json(self.base_url, path, body, self.timeout)

    def send_utterance_payload(self, payload: dict, simulate_server_ok: bool = False) -> dict:
        if simulate_server_ok:
            return {"ok": True, "http_status": 200, "endpoint": UTTERANCES_PATH}
        return self._post(UTTERANCES_PATH, payload)

    def confirm_transcript(self, transcript_id: str, confirmed_role_map: dict, simulate_server_ok: bool = False) -> dict:
        path = f"/api/transcripts/{transcript_id}/confirm"
        if simulate_server_ok:
            return {"ok": True, "http_status": 200, "endpoint": path}
        return self._post(path, {"confirmed_role_map": confirmed_role_map})

    def run_e2e_pipeline(self, session, simulate_server_ok: bool = False) -> dict:
        """Record -> analyse -> POST the MPC payload (queued offline on network/5xx failure).
        simulate_server_ok=True replaces the HTTP upload with a canned 200 (tests / demos only)."""
        audio = record_audio(duration_sec=1)
        sender = (lambda p: self.send_utterance_payload(p, simulate_server_ok=True)) if simulate_server_ok else None
        result = self.runtime.process_session(session, sender=sender)
        return {
            "audio": audio,
            "network_online": "http_status" in result.get("upload", {}),
            "upload": result.get("upload"),
            "state_history": list(self.runtime.state_history),
            "sync_status": result["status"],
            "result": result,
        }

    def flush_offline_queue(self) -> dict:
        return self.runtime.flush_offline_queue()


def _quiet_test() -> None:
    import socket
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    hits = []

    class _Backend(BaseHTTPRequestHandler):
        def do_POST(self):
            hits.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *a):
            pass

    with socket.socket() as sk:                      # a port nothing listens on yet
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    qp = os.path.join(AUDIO_TMP_DIR, "iot_client_test_queue.jsonl")
    srv = None
    try:
        os.makedirs(AUDIO_TMP_DIR, exist_ok=True)
        open(qp, "w").close()
        scen = build_5_benchmark_scenarios()[0]
        r = XiaoZhaoIoTClient(queue_path=qp).run_e2e_pipeline(scen, simulate_server_ok=True)
        assert r["sync_status"] == STATUS_SYNCED, r

        client = XiaoZhaoIoTClient(base_url=url, queue_path=qp)
        r = client.run_e2e_pipeline(scen)                       # backend down -> queued, no probe
        assert r["sync_status"] == STATUS_QUEUED and r["network_online"] is False, r
        srv = HTTPServer(("127.0.0.1", port), _Backend)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        out = client.flush_offline_queue()                      # backend up -> POSTed and cleared
        assert out["flushed"] == 1 and out["remaining"] == 0 and os.path.getsize(qp) == 0, out
        assert hits == [(UTTERANCES_PATH, r["result"]["payload"])], hits   # exactly one real POST, no {} probe

        c = client.confirm_transcript("T_001", {"speaker_1": "elder"})
        assert c["ok"] and c["endpoint"] == "/api/transcripts/T_001/confirm", c
        assert hits[-1] == ("/api/transcripts/T_001/confirm", {"confirmed_role_map": {"speaker_1": "elder"}})
    finally:
        if srv:
            srv.shutdown()
            srv.server_close()
        if os.path.exists(qp):
            os.remove(qp)
    print("IOT_API_CLIENT_OK")


if __name__ == "__main__":
    if "--quiet-test" in sys.argv:
        _quiet_test()
        sys.exit(0)
    print("Usage: python3 src/iot_api_client.py --quiet-test")
