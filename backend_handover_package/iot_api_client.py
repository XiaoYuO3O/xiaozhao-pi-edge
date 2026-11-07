import json
import os
import sys
import urllib.error
import urllib.request

try:
    from src.edge_state_machine import EdgeRuntimeManager
    from src.mpc_core import build_5_benchmark_scenarios
    from src.exp1_pi_audio_check import record_audio
    from src.config import AUDIO_TMP_DIR
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from edge_state_machine import EdgeRuntimeManager
    from mpc_core import build_5_benchmark_scenarios
    from exp1_pi_audio_check import record_audio
    from config import AUDIO_TMP_DIR

UTTERANCES_PATH = "/api/iot/v1/utterances"


class XiaoZhaoIoTClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 2.0, queue_path: str = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.runtime = EdgeRuntimeManager(queue_path)

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return {"ok": 200 <= resp.status < 300, "http_status": resp.status, "endpoint": path}
        except urllib.error.HTTPError as e:
            return {"ok": False, "http_status": e.code, "endpoint": path, "error": str(e)}
        except (urllib.error.URLError, OSError) as e:
            return {"ok": False, "error": str(e)}

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
        audio = record_audio(duration_sec=1)
        probe = self.send_utterance_payload({}, simulate_server_ok=simulate_server_ok)
        # Any HTTP response (even 4xx on the empty probe) means the server is reachable.
        network_online = bool(probe.get("ok") or "http_status" in probe)
        result = self.runtime.process_session(session, network_online=network_online)
        return {
            "audio": audio,
            "network_online": network_online,
            "probe": probe,
            "state_history": list(self.runtime.state_history),
            "sync_status": result["status"],
            "result": result,
        }


def _quiet_test() -> None:
    qp = os.path.join(AUDIO_TMP_DIR, "iot_client_test_queue.jsonl")
    try:
        open(qp, "w").close()
        scen = build_5_benchmark_scenarios()[0]

        ok_client = XiaoZhaoIoTClient(queue_path=qp)
        r = ok_client.run_e2e_pipeline(scen, simulate_server_ok=True)
        assert r["sync_status"] == "SYNCED_TO_BACKEND", r

        down = XiaoZhaoIoTClient(base_url="http://127.0.0.1:59999", queue_path=qp)
        r = down.run_e2e_pipeline(scen, simulate_server_ok=False)
        assert r["sync_status"] == "BUFFERED_OFFLINE", r
        assert r["network_online"] is False, r

        c = ok_client.confirm_transcript("T_001", {"speaker_1": "elder"}, simulate_server_ok=True)
        assert c["ok"] and c["endpoint"] == "/api/transcripts/T_001/confirm", c
    finally:
        if os.path.exists(qp):
            os.remove(qp)
    print("IOT_API_CLIENT_OK")


if __name__ == "__main__":
    if "--quiet-test" in sys.argv:
        _quiet_test()
        sys.exit(0)
    print("Usage: python3 src/iot_api_client.py --quiet-test")
