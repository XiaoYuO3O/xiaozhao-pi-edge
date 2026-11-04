import json
import sys
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    from src.config import ON_PI
except ImportError:
    from config import ON_PI

@dataclass
class UtteranceTurn:
    turn_id: int
    timestamp: float
    speaker_id: str          # 免建聲紋庫標記: speaker_1, speaker_2, speaker_3[cite: 2]
    inferred_role: str       # caregiver (照服員) / family (家屬) / elder (長輩)[cite: 2]
    text: str
    addressee: str = "unknown"
    thread_id: str = "T0"

@dataclass
class DialogueSessionState:
    session_id: str
    case_id: str
    pi_state: str = "1_Idle_Standby"
    turns: List[UtteranceTurn] = field(default_factory=list)
    active_threads: Dict[str, List[int]] = field(default_factory=dict)
    dst_slots: Dict[str, str] = field(default_factory=dict)

class XiaoZhaoMPCEngine:
    """小照 × 樹莓派 5 多人多輪意圖偵測核心 (Addressee + Disentanglement + OOD + P0)"""

    P0_EMERGENCY_KW = ("跌倒", "昏倒", "流血", "暴力", "打人", "自殺", "沒呼吸", "猝逝", "車禍")
    P1_SAFETY_KW = ("摸大腿", "性騷擾", "吃豆腐", "私密處", "洗廁所", "刷馬桶")
    MED_BA16_KW = ("買藥", "成藥", "普拿疼", "止痛藥", "代購", "順便買")
    RESPITE_GA_SC_KW = ("喘息", "短照", "GA09", "SC09", "臨時托顧")
    CHITCHAT_NOISE_KW = ("便當", "排骨", "雞腿", "午餐", "八點檔", "電視劇", "天氣", "重播")
    OOD_CRISIS_KW = ("咬人狗", "野狗", "蜜蜂", "淹水", "土石流", "虎頭蜂")

    def infer_role_and_addressee(self, turn: UtteranceTurn, history: List[UtteranceTurn]) -> Tuple[str, str]:
        text = turn.text
        role = turn.inferred_role

        # 1. 若明確點名小照[cite: 1]
        if "小照" in text:
            return role, "system_xiaozhao"

        # 2. 家屬發言的受話者判定 (解決老師指出的圖3第一輪問題)[cite: 1]
        if role == "family":
            if any(k in text for k in ("幫我買", "順便幫", "可以嗎", "麻煩你", "刷馬桶", "洗廁所", "喘息", "短照")):
                return role, "caregiver"
            return role, "elder"

        # 3. 長輩發言的受話者判定
        if role == "elder":
            if any(k in text for k in ("幫我", "痛", "跌倒", "不舒服", "洗澡")):
                return role, "caregiver"
            return role, "family"

        # 4. 照服員未喚醒小照時的受話者回溯 (Ubuntu 3分鐘時間窗原則)[cite: 1]
        if role == "caregiver":
            for prev in reversed(history):
                if turn.timestamp - prev.timestamp <= 180 and prev.inferred_role != "caregiver":
                    return role, prev.inferred_role
            return role, "elder"

        return role, "unknown"

    def disentangle_threads(self, session: DialogueSessionState, current_turn: UtteranceTurn) -> Tuple[str, List[int]]:
        """多話題交錯解纏結 (TD/MF)：剔除午餐便當/八點檔等跨話題雜訊[cite: 1]"""
        text = current_turn.text

        # 若為閒聊噪音，直接歸入噪音串 T_CHITCHAT
        if any(k in text for k in self.CHITCHAT_NOISE_KW) and "小照" not in text:
            tid = "T_CHITCHAT"
            session.active_threads.setdefault(tid, []).append(current_turn.turn_id)
            return tid, [current_turn.turn_id]

        # 若向小照提問且含指代詞（「剛剛說的那樣」、「那個可以嗎」）[cite: 1]
        is_referential = any(k in text for k in ("剛剛", "那樣", "那個", "這樣", "前面", "他講的"))
        selected_indices = []

        for prev in session.turns:
            if prev.turn_id >= current_turn.turn_id or prev.thread_id == "T_CHITCHAT":
                continue
            # 10分鐘 (600秒) 內，且屬於任務線索或受話者為照服員的請求[cite: 1]
            if current_turn.timestamp - prev.timestamp <= 600:
                if is_referential and prev.addressee == "caregiver":
                    selected_indices.append(prev.turn_id)
                elif any(k in prev.text for k in self.MED_BA16_KW + self.P1_SAFETY_KW + self.RESPITE_GA_SC_KW + self.OOD_CRISIS_KW):
                    selected_indices.append(prev.turn_id)

        selected_indices.append(current_turn.turn_id)
        tid = "T_TASK_MAIN"
        session.active_threads[tid] = selected_indices
        return tid, selected_indices

    def analyze_session(self, session: DialogueSessionState) -> Dict:
        for t in session.turns:
            t.inferred_role, t.addressee = self.infer_role_and_addressee(t, session.turns[:t.turn_id - 1])
            t.thread_id, selected_idx = self.disentangle_threads(session, t)

        last_turn = session.turns[-1]
        _, selected_idx = self.disentangle_threads(session, last_turn)
        context_turns = [t for t in session.turns if t.turn_id in selected_idx]
        merged_text = " ".join(t.text for t in context_turns)

        intents = []
        route = "RAG_STANDARD_SOP"
        safety_level = "NORMAL"

        if any(k in merged_text for k in self.P0_EMERGENCY_KW):
            intents.append("P0_EMERGENCY_ALERT")
            route = "P0_INSTANT_SUPERVISOR_BROADCAST"
            safety_level = "CRITICAL_P0"
            session.pi_state = "0E_Emergency_Broadcast"
        else:
            if any(k in merged_text for k in self.MED_BA16_KW):
                intents.append("BA16_MEDICINE_PURCHASE_INQUIRY")
                session.dst_slots["service_code"] = "BA16_代購限制(不可代購成藥)"
                session.dst_slots["requester_to_target"] = "family -> caregiver"
            if any(k in merged_text for k in self.P1_SAFETY_KW):
                intents.append("WORKPLACE_HARASSMENT_REPORT")
                safety_level = "HIGH_P1"
            if "馬桶" in merged_text or "廁所" in merged_text:
                intents.append("EXTRA_OUT_OF_SCOPE_CLEANING")
            if any(k in merged_text for k in self.RESPITE_GA_SC_KW):
                intents.append("GA09_SC09_RESPITE_FORM")
                session.dst_slots["form_type"] = "GA09/SC09_喘息短照紀錄表"
                route = "FORM_GA09_SC09_ENTRY"
            if any(k in merged_text for k in self.OOD_CRISIS_KW):
                intents.append("OOS_ATYPICAL_FIELD_HAZARD")
                route = "OOD_SAFE_FALLBACK_AND_EXPERIENCE_KB"
                safety_level = "HIGH_P1"

            session.pi_state = "5_Wait_LINE_Confirm"

        if not intents:
            intents.append("GENERAL_CARE_LOG")

        return {
            "client_event_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{session.session_id}_{last_turn.turn_id}")),
            "session_id": session.session_id,
            "case_id": session.case_id,
            "run_env": "RaspberryPi5_ARM64" if ON_PI else "Mac_Mock_Env",
            "pi_state": session.pi_state,
            "safety_level": safety_level,
            "route_target": route,
            "detected_intents": intents,
            "selected_turn_indices": selected_idx,
            "filtered_out_turns": [t.turn_id for t in session.turns if t.turn_id not in selected_idx],
            "dst_slots": session.dst_slots,
            "turns_detail": [
                {
                    "turn_id": t.turn_id,
                    "speaker_id": t.speaker_id,
                    "role_pre_filled": t.inferred_role,
                    "addressee": t.addressee,
                    "thread_id": t.thread_id,
                    "text": t.text,
                }
                for t in session.turns
            ],
        }

def load_from_whisper_json(json_path: str, session_id: str = "S_WHISPER", case_id: str = "CASE_01") -> DialogueSessionState:
    path = Path(json_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    turns = []
    for idx, seg in enumerate(raw.get("segments", []), start=1):
        turns.append(
            UtteranceTurn(
                turn_id=idx,
                timestamp=float(seg.get("start", idx * 10.0)),
                speaker_id=seg.get("speaker", f"speaker_{(idx % 3) + 1}"),
                inferred_role=seg.get("role_hint", "caregiver"),
                text=seg.get("text", "").strip(),
            )
        )
    return DialogueSessionState(session_id=session_id, case_id=case_id, turns=turns)

def build_5_benchmark_scenarios() -> List[DialogueSessionState]:
    return [
        DialogueSessionState(
            session_id="SCEN_1_BA16_DISENTANGLE",
            case_id="JIA_AN_001",
            turns=[
                UtteranceTurn(1, 10.0, "speaker_2", "family", "居服員小姐，待會麻煩順便幫我買兩盒普拿疼止痛藥好嗎？"),
                UtteranceTurn(2, 45.0, "speaker_1", "elder", "今天中午的排骨便當菜色很鹹欸，電視八點檔又在重播。"),
                UtteranceTurn(3, 520.0, "speaker_3", "caregiver", "小照，家屬剛剛說的那樣可以嗎？"),
            ],
        ),
        DialogueSessionState(
            session_id="SCEN_2_MULTI_INTENT_HARASSMENT",
            case_id="JIA_AN_002",
            turns=[
                UtteranceTurn(1, 15.0, "speaker_1", "elder", "來，幫我洗私密處，順便摸一下大腿沒關係吧。"),
                UtteranceTurn(2, 30.0, "speaker_2", "family", "洗完澡順便幫我們把外面那間廁所跟馬桶刷一刷喔。"),
                UtteranceTurn(3, 60.0, "speaker_3", "caregiver", "小照，剛剛長輩摸大腿性騷擾，而且家屬又要求順便刷馬桶，我要回報！"),
            ],
        ),
        DialogueSessionState(
            session_id="SCEN_3_OOD_MOUNTAIN_DOGS",
            case_id="JIA_AN_003",
            turns=[
                UtteranceTurn(1, 5.0, "speaker_3", "caregiver", "小照，案家在山區結果外面有四隻咬人狗圍住，而且路有點淹水我進不去怎麼辦？"),
            ],
        ),
        DialogueSessionState(
            session_id="SCEN_4_P0_EMERGENCY_FALL",
            case_id="JIA_AN_004",
            turns=[
                UtteranceTurn(1, 2.0, "speaker_3", "caregiver", "小照救命！長輩在浴室跌倒而且昏倒流血了！"),
            ],
        ),
        DialogueSessionState(
            session_id="SCEN_5_GA09_SC09_RESPITE",
            case_id="JIA_AN_005",
            turns=[
                UtteranceTurn(1, 12.0, "speaker_2", "family", "我們下禮拜想申請 GA09 喘息服務跟 SC09 短照，麻煩幫我們填紀錄表。"),
                UtteranceTurn(2, 40.0, "speaker_1", "elder", "今天午餐的雞腿便當冷掉了不好吃。"),
                UtteranceTurn(3, 180.0, "speaker_3", "caregiver", "小照，家屬剛剛說的那個喘息短照要開哪張表單？"),
            ],
        ),
    ]

if __name__ == "__main__":
    engine = XiaoZhaoMPCEngine()
    scenarios = build_5_benchmark_scenarios()
    results = [engine.analyze_session(s) for s in scenarios]

    # 護欄極簡驗證斷言
    assert results[0]["selected_turn_indices"] == [1, 3], "SCEN 1 應剔除第 2 輪排骨便當閒聊"
    assert results[0]["turns_detail"][0]["addressee"] == "caregiver", "SCEN 1 家屬應對照服員說話"
    assert len(results[1]["detected_intents"]) >= 2, "SCEN 2 應偵測出多意圖"
    assert results[3]["pi_state"] == "0E_Emergency_Broadcast", "SCEN 4 應觸發 P0 緊急通道"
    assert results[4]["selected_turn_indices"] == [1, 3], "SCEN 5 應剔除第 2 輪雞腿便當閒聊"

    if "--quiet-test" in sys.argv:
        print(f"MPC_CORE_OK (env={'PI' if ON_PI else 'MAC_MOCK'}, scenarios={len(results)})")
        sys.exit(0)

    for r in results:
        print("=" * 72)
        print(f"[{r['session_id']}] 環境: {r['run_env']} | 狀態機: {r['pi_state']}")
        print(f"  偵測意圖: {r['detected_intents']} | 路由: {r['route_target']}")
        print(f"  保留對話輪次 (selected_idx): {r['selected_turn_indices']} | 剔除雜訊輪次: {r['filtered_out_turns']}")
        print(f"  跨輪槽位 (DST Slots): {r['dst_slots']}")
        for td in r["turns_detail"]:
            print(f"    - Turn {td['turn_id']} [{td['speaker_id']}/{td['role_pre_filled']} -> {td['addressee']}] ({td['thread_id']}): {td['text']}")
