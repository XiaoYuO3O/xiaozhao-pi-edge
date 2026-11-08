import json
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

try:
    from src.config import ON_PI
except ImportError:
    try:
        from config import ON_PI
    except ImportError:  # standalone copy (backend_handover_package): inline the Pi auto-detect
        import os as _os
        import platform as _platform
        ON_PI = (_os.getenv("MOCK_PI") != "1" and _platform.system() == "Linux"
                 and _platform.machine() in ("aarch64", "armv7l"))

@dataclass
class UtteranceTurn:
    turn_id: int
    timestamp: float
    speaker_id: str          # 免建聲紋庫標記: speaker_1, speaker_2, speaker_3[cite: 2]
    inferred_role: str       # caregiver (照服員) / family (家屬) / elder (長輩)[cite: 2]
    text: str
    addressee: str = "unknown"
    thread_id: str = "T0"
    rms_energy: float = 0.0  # int16 RMS amplitude of the segment (0 = unknown)
    zcr_pitch: float = 0.0   # zero-crossing rate 0..1, pitch proxy (0 = unknown)


# ------------------------------------------------- MIntRec (ACM MM '22) taxonomy
# Bi-modal (text + audio) coarse-to-fine intent labels, adapted to zh-TW home-care speech.
MINTREC_HIGH_RMS = 3000.0   # int16 RMS; loud/aroused speech
MINTREC_HIGH_ZCR = 0.12     # high zero-crossing rate ~ raised pitch / tense voice
# Pure agreement / backchannel tokens (shared with the interview analyzer's noise filter).
AGREEMENT_TOKENS_RE = r"(?:對啦|對啊|對呀|對阿|好啦|好的|是的|可以|我懂|我知道|沒錯|ok|OK|Ok|對|好|嗯|喔|哦|噢|是)"
_AGREE_SEP = r"[\s,，、。.!！~～…]"
_AGREE_ONLY_RE = re.compile(rf"^{_AGREE_SEP}*(?:{AGREEMENT_TOKENS_RE}{_AGREE_SEP}*)+$")


def is_agreement_only(text: str) -> bool:
    """True iff the text is nothing but agreement/backchannel tokens (any length, e.g. 對啦x50)."""
    return bool(_AGREE_ONLY_RE.match((text or "").strip()))


_MINTREC_RULES: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    # priority order: safety first, then goal-directed acts, then emotions/attitudes
    ("Achieve_Goals", "Ask_for_help", ("跌倒", "流血", "救命", "幫幫我", "報警", "急救", "沒有呼吸", "去世",
                                       "過世", "猝逝", "到院前死亡", "怎麼辦", "怎麼處理", "會慌")),
    ("Achieve_Goals", "Prevent", ("不要", "不行", "別摸", "住手", "千萬", "先不要")),
    ("Express_Emotions_Attitudes", "Criticize", ("怎麼搞的", "太誇張")),
    ("Achieve_Goals", "Leave", ("今天就到", "就今天", "謝謝大家", "時間太久", "先這樣")),
    ("Express_Emotions_Attitudes", "Care", ("還好嗎", "有沒有怎樣", "關心", "心理層面", "宗教信仰", "安撫鎮定",
                                            "辛苦")),
    ("Express_Emotions_Attitudes", "Thank", ("謝謝", "感謝")),
    ("Express_Emotions_Attitudes", "Praise", ("很棒", "太棒", "很成功", "很厲害", "非常厲害", "黃金的資料",
                                              "很好的方式", "寫那麼好")),
    ("Express_Emotions_Attitudes", "Comfort", ("沒關係", "不用擔心", "別擔心")),
    ("Achieve_Goals", "Arrange", ("麻煩", "順便幫我", "請你", "去買", "刷一刷", "設計成", "回去", "下次",
                                  "我們再", "再來就是", "先設計", "開發一個", "做一個", "要有一個", "我們要",
                                  "我們先", "先做", "去修改", "要修正", "請大家", "跟主任要", "加一個")),
    ("Achieve_Goals", "Advise", ("建議", "可以再把", "應該", "希望我們", "要先", "可以再", "如果是你", "怎麼做",
                                  "希望", "盡量", "安撫", "我覺得", "可以用", "可以讓", "要考慮", "要切換",
                                  "我們可以把")),
    # explicit problem statements outrank the topical nouns of Inform (real-interview hotspots)
    ("Express_Emotions_Attitudes", "Complain", ("很緊張", "沒辦法第一", "沒辦法解決", "沒有教導", "暴力",
                                                 "性騷", "被打", "很困難", "回答錯誤", "答案不正確",
                                                 "回答不完整", "不安全")),
    ("Achieve_Goals", "Introduce", ("我是", "介紹一下", "先介紹", "我叫")),
    ("Achieve_Goals", "Inform", ("納進來", "規定是", "二十幾個", "法規面", "知識庫", "遇過", "案例", "以前",
                                 "之前", "我們有", "系統", "法規", "規定", "項目", "後端", "介面", "功能",
                                 "督導", "居督", "主任", "照服員", "居服員", "個案", "案家", "家屬", "權限",
                                 "帳號", "經驗", "紀錄", "長照", "醫院", "提示語", "麥克風", "收音", "按鈕",
                                 "按鍵", "緊急", "現在是", "目前", "就是說", "比如說", "譬如說", "例如")),
    ("Express_Emotions_Attitudes", "Complain", ("困擾", "很鹹", "重播", "天馬行空", "沒辦法", "很髒",
                                                 "濕掉", "很慘", "狗", "蜜蜂", "鹹", "念你", "很臭", "很難",
                                                 "太多了", "已經夠滿")),
    ("Express_Emotions_Attitudes", "Oppose", ("不用", "不現場服務", "不是", "反對")),
    ("Express_Emotions_Attitudes", "Agree", ("對,對", "對，對", "對對", "嗯嗯", "我懂", "沒錯")),
)
_FINE_TO_COARSE = {fine: coarse for coarse, fine, _ in _MINTREC_RULES}
_MINTREC_SHORT_AGREE = ("好", "可以")      # only counted as Agree in short utterances (avoid "不好吃")
_MINTREC_DISTRESS_MARK = ("!", "！", "痛", "快", "啊")
_MINTREC_NEGATIVE_MARK = ("不", "沒", "怎麼", "又")


def classify_mintrec_taxonomy(text: str, rms_energy: float = 0.0, zcr_pitch: float = 0.0) -> dict:
    """Rule-based MIntRec coarse->fine intent. modality_cue in {text, audio, text+audio, none}."""
    t = (text or "").replace(" ", "")
    aroused = rms_energy >= MINTREC_HIGH_RMS and zcr_pitch >= MINTREC_HIGH_ZCR
    if t and is_agreement_only(t):
        return {"coarse": "Express_Emotions_Attitudes", "fine": "Agree", "modality_cue": "text"}
    for coarse, fine, kws in _MINTREC_RULES:
        if any(k in t for k in kws):
            return {"coarse": coarse, "fine": fine, "modality_cue": "text+audio" if aroused else "text"}
    if aroused and any(k in t for k in _MINTREC_DISTRESS_MARK):
        return {"coarse": "Achieve_Goals", "fine": "Ask_for_help", "modality_cue": "audio"}
    if aroused and any(k in t for k in _MINTREC_NEGATIVE_MARK):
        return {"coarse": "Express_Emotions_Attitudes", "fine": "Criticize", "modality_cue": "audio"}
    core = t.strip("，。,.!！?？～~ ")
    if (core and len(core) <= 6 and any(k in core for k in _MINTREC_SHORT_AGREE)
            and not any(k in core for k in _MINTREC_NEGATIVE_MARK)):
        return {"coarse": "Express_Emotions_Attitudes", "fine": "Agree", "modality_cue": "text"}
    return {"coarse": "UNK", "fine": "UNK", "modality_cue": "none"}


def summarize_mintrec(labels: List[dict]) -> Dict[str, Dict[str, int]]:
    coarse: Dict[str, int] = {}
    fine: Dict[str, int] = {}
    for lb in labels:
        coarse[lb["coarse"]] = coarse.get(lb["coarse"], 0) + 1
        fine[lb["fine"]] = fine.get(lb["fine"], 0) + 1
    return {"coarse_counts": coarse, "fine_counts": fine}


@dataclass
class DialogueSessionState:
    session_id: str
    case_id: str
    pi_state: str = "1_Idle_Standby"
    turns: List[UtteranceTurn] = field(default_factory=list)
    active_threads: Dict[str, List[int]] = field(default_factory=dict)
    dst_slots: Dict[str, Any] = field(default_factory=dict)
    role_priors: Dict[str, str] = field(default_factory=dict)  # recording-level speaker -> role fallback


# ------------------------------------------- Upgrade A: session-level speaker roles
HOMECARE_ROLES = ("caregiver", "family", "elder")
ROLE_DIRECTOR = "director_主任"
ROLE_SUPERVISOR = "supervisor_居督"
ROLE_RESEARCHER = "researcher_主持團隊"
ROLE_CAREGIVER_LTC = "caregiver_照服員"
# (role, cues) in tie-break priority order
EXTENDED_ROLE_CUES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (ROLE_DIRECTOR, ("主任跟督導", "量身打造")),
    (ROLE_SUPERVISOR, ("居服員的問題", "在外騎車", "外面騎車", "也在騎車", "我們督導", "安撫", "事後再回", "派案")),
    (ROLE_RESEARCHER, ("我們後端", "學生們", "知識庫系統", "做個小結", "幫大家裝綁定碼", "裝綁定碼", "做實驗")),
    (ROLE_CAREGIVER_LTC, ("褲子整個都濕掉", "沒有辦法換衣服", "沒辦法換衣服", "我的夥伴", "去案家", "遇到狗")),
)
# previous speaker hands the floor to the director -> the NEXT speaker is the director
DIRECTOR_HANDOFF_CUES = ("請主任分配", "主任分配", "請主任", "主任您", "主任你", "讓主任來")
DIRECTOR_HANDOFF_WEIGHT = 2
EXTENDED_SESSION_MIN_HITS = 2   # meeting/interview mode needs >= this many extended-cue hits session-wide
# vocative address to a role (who the speaker is talking TO, not about)
VOCATIVE_ROLE_CUES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (ROLE_DIRECTOR, ("請主任", "主任你", "主任您", "主任，", "主任,")),
    (ROLE_SUPERVISOR, ("請督導", "請居督", "督導你", "居督你", "督導們", "督導，", "居督，")),
    (ROLE_CAREGIVER_LTC, ("請照服員", "照服員你", "居服員小姐", "照服員，", "居服員，")),
)
ADDRESSEE_ADJACENCY_SEC = 180.0
NOISE_THREADS = ("T_BACKCHANNEL_NOISE", "T_CHITCHAT")
_SPEAKER_ID_RE = re.compile(r"^speaker_\d+$")


def _norm(text: str) -> str:
    return (text or "").replace(" ", "")


def infer_session_speaker_roles(turns: List[UtteranceTurn],
                                prefilled: Optional[Dict[int, str]] = None,
                                priors: Optional[Dict[str, str]] = None,
                                sources_out: Optional[Dict[str, str]] = None,
                                graph_evidence_out: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Session-level speaker -> role map from multi-turn cue evidence (no voiceprint DB).

    Home-care pre-filled roles (caregiver/family/elder) stay authoritative unless the session is in
    meeting/interview mode (>= EXTENDED_SESSION_MIN_HITS extended cues) AND that speaker has own cues.
    `prefilled` maps turn_id -> role hint (defaults to each turn's inferred_role). `priors` (e.g. roles
    inferred over the whole recording) fill speakers the session alone cannot resolve. `sources_out`, if
    given, receives speaker -> session_cues | prefilled | recording_prior | graph_inferred | no_evidence.
    GIFT: in meeting mode, a no_evidence speaker is resolved from its reply-to / replied-by exchanges with
    role-resolved participants (gift_infer_roles); `graph_evidence_out` receives that evidence."""
    priors = priors or {}
    src: Dict[str, str] = {} if sources_out is None else sources_out
    prefilled = prefilled or {t.turn_id: t.inferred_role for t in turns}
    scores: Dict[str, Dict[str, int]] = {}
    hint_votes: Dict[str, Dict[str, int]] = {}
    order: List[str] = []
    prev: Optional[UtteranceTurn] = None
    for t in turns:
        sp = t.speaker_id
        if sp not in scores:
            scores[sp], hint_votes[sp] = {}, {}
            order.append(sp)
        txt = _norm(t.text)
        for role, cues in EXTENDED_ROLE_CUES:
            hits = sum(1 for c in cues if c in txt)
            if hits:
                scores[sp][role] = scores[sp].get(role, 0) + hits
        if prev is not None and prev.speaker_id != sp and any(c in _norm(prev.text) for c in DIRECTOR_HANDOFF_CUES):
            scores[sp][ROLE_DIRECTOR] = scores[sp].get(ROLE_DIRECTOR, 0) + DIRECTOR_HANDOFF_WEIGHT
        hint = prefilled.get(t.turn_id, "unknown")
        hint_votes[sp][hint] = hint_votes[sp].get(hint, 0) + 1
        prev = t
    # the 2-hit bar only protects pre-filled home-care hints; raw diarized sessions (all unknown) need 1 cue
    has_homecare_hint = any(h in HOMECARE_ROLES for v in hint_votes.values() for h in v)
    meeting_mode = (sum(v for s in scores.values() for v in s.values())
                    >= (EXTENDED_SESSION_MIN_HITS if has_homecare_hint else 1))
    priority = {role: i for i, (role, _) in enumerate(EXTENDED_ROLE_CUES)}
    roles: Dict[str, str] = {}
    for sp in order:
        hint = max(hint_votes[sp].items(), key=lambda kv: kv[1])[0]
        if meeting_mode and scores[sp]:
            roles[sp], src[sp] = min(scores[sp].items(), key=lambda kv: (-kv[1], priority[kv[0]]))[0], "session_cues"
        elif hint and hint != "unknown":
            roles[sp], src[sp] = hint, "prefilled"
        elif priors.get(sp, "unknown") != "unknown":
            roles[sp], src[sp] = priors[sp], "recording_prior"
        else:
            roles[sp], src[sp] = "unknown", "no_evidence"
    unresolved = [sp for sp in order if src[sp] == "no_evidence"]
    if meeting_mode and unresolved and len(unresolved) < len(order):
        for sp, (role, ev) in gift_infer_roles(turns, build_gift_graph(turns), roles, unresolved).items():
            roles[sp], src[sp] = role, GRAPH_INFERRED_SOURCE
            if graph_evidence_out is not None:
                graph_evidence_out[sp] = ev
    return roles


def resolve_addressee_role(turn: UtteranceTurn, history: List[UtteranceTurn], speaker_roles: Dict[str, str],
                           engine_addressee: str = "unknown",
                           hinted_addressee: str = "unknown") -> Tuple[str, str]:
    """-> (addressee_role, addressee_speaker). Precedence: engine home-care verdict > vocative >
    upstream disentangler speaker id / ALL > adjacency (previous other speaker within 180 s)."""
    if engine_addressee not in ("unknown", "", None):
        sp = next((p.speaker_id for p in reversed(history)
                   if speaker_roles.get(p.speaker_id) == engine_addressee and p.speaker_id != turn.speaker_id), "")
        return engine_addressee, sp or ("system_xiaozhao" if engine_addressee == "system_xiaozhao" else "")
    txt = _norm(turn.text)
    for role, cues in VOCATIVE_ROLE_CUES:
        if any(c in txt for c in cues) or txt.startswith(role.split("_", 1)[1]):
            sp = next((s for s, r in speaker_roles.items() if r == role and s != turn.speaker_id), "")
            if sp or role != speaker_roles.get(turn.speaker_id):
                return role, sp
    if hinted_addressee and _SPEAKER_ID_RE.match(hinted_addressee) and hinted_addressee != turn.speaker_id:
        return speaker_roles.get(hinted_addressee, "unknown"), hinted_addressee
    if hinted_addressee == "ALL":
        return "ALL", "ALL"
    for p in reversed(history):
        if p.speaker_id != turn.speaker_id and turn.timestamp - p.timestamp <= ADDRESSEE_ADJACENCY_SEC:
            return speaker_roles.get(p.speaker_id, "unknown"), p.speaker_id
    return "unknown", ""


# ------------------------------------- Upgrade B: cross-speaker MIntRec transition chains
CRISIS_CHAIN_CUES = ("緊急", "按鈕", "按鍵", "緊急鈕", "對講機", "救命", "報警", "急救", "跌倒", "流血",
                     "昏倒", "沒有呼吸")
# a Prevent source only opens a crisis chain when it itself carries a hazard/safety cue
# (keeps meeting-moderation "不要發散" from escalating)
CRISIS_SOURCE_HAZARD_CUES = CRISIS_CHAIN_CUES + ("住手", "別摸", "危險", "暴力", "性騷", "摸大腿", "狗", "蜜蜂")
# (type, source fines, target fines, needs crisis cue, priority, staging weight)
MINTREC_TRANSITION_RULES: Tuple[Tuple[str, Tuple[str, ...], Tuple[str, ...], bool, str, int], ...] = (
    ("CRISIS_ESCALATION_CHAIN", ("Prevent", "Ask_for_help", "Complain"), ("Arrange", "Inform"), True,
     "CRITICAL_ESCALATION", 3),
    ("PROBLEM_TO_GUIDANCE", ("Complain", "Ask_for_help"), ("Advise", "Comfort", "Care"), False, "NORMAL", 1),
    ("REQUEST_TO_COMPLIANCE_OR_HESITATION", ("Arrange",), ("Agree", "UNK"), False, "HIGH_RAG_INTERVENTION", 2),
    ("REQUEST_TO_REFUSAL", ("Arrange",), ("Oppose", "Prevent"), False, "HIGH_NEGOTIATION", 2),
    ("REPORT_TO_ACTION_PLAN", ("Inform",), ("Arrange", "Advise"), False, "NORMAL", 1),
)
TRANSITION_WEIGHT = {r[0]: r[5] for r in MINTREC_TRANSITION_RULES}


def match_mintrec_transition(src_block: List[Tuple[int, str, str]], cur_fine: str,
                             cur_text: str) -> Optional[Tuple[tuple, int, str]]:
    """src_block: [(turn_id, fine, text)] of the previous speaker's consecutive turns in this thread.
    -> (rule, from_turn_id, from_fine) for the first matching rule, else None."""
    cur_t = _norm(cur_text)
    for rule in MINTREC_TRANSITION_RULES:
        rtype, srcs, tgts, needs_cue, _, _ = rule
        if cur_fine not in tgts:
            continue
        for tid, fine, text in reversed(src_block):
            if fine not in srcs:
                continue
            src_t = _norm(text)
            if needs_cue:
                if not any(c in src_t + cur_t for c in CRISIS_CHAIN_CUES):
                    continue
                if fine == "Prevent" and not any(c in src_t for c in CRISIS_SOURCE_HAZARD_CUES):
                    continue
            return rule, tid, fine
    return None


TRANSITION_LOOKAHEAD = 3   # a source turn may be answered by any of the next 3 substantive turns in its thread
# hazards that make a crisis chain *live*; mechanism words (緊急/按鈕/對講機) alone describe protocol design
CRISIS_LIVE_HAZARD_CUES = ("救命", "跌倒", "流血", "昏倒", "沒有呼吸", "沒呼吸", "暴力", "打人", "被打", "性騷",
                           "摸大腿", "住手", "別摸", "猝逝", "去世", "過世", "報警", "急救", "到院前死亡")
_LIVE_CHECKER = None


def crisis_is_live(*texts: str) -> bool:
    """A hazard cue occurs outside an example / meta-reference clause (例如…, 像剛我們說的…)."""
    global _LIVE_CHECKER
    if _LIVE_CHECKER is None:
        _LIVE_CHECKER = XiaoZhaoMPCEngine()
    return any(_LIVE_CHECKER._has_live_kw(t, CRISIS_LIVE_HAZARD_CUES) for t in texts)


def build_transition(rule: tuple, thread_id: str, src: Tuple[int, str, str, str], tgt: Tuple[int, str, str, str],
                     speaker_roles: Optional[Dict[str, str]] = None, lag: int = 1) -> Dict:
    """src/tgt = (turn_id, speaker_id, fine, text)."""
    roles = speaker_roles or {}
    rtype = rule[0]
    live = rtype == "CRISIS_ESCALATION_CHAIN" and crisis_is_live(src[3], tgt[3])
    priority = rule[4] if rtype != "CRISIS_ESCALATION_CHAIN" or live else "PROTOCOL_DESIGN_DISCUSSION"
    return {"type": rtype, "thread_id": thread_id, "from_turn": src[0], "to_turn": tgt[0],
            "from_speaker": src[1], "to_speaker": tgt[1],
            "from_role": roles.get(src[1], "unknown"), "to_role": roles.get(tgt[1], "unknown"),
            "from_fine": src[2], "to_fine": tgt[2], "lag": lag, "priority": priority, "live_hazard": live,
            "rag_intervention": rtype == "REQUEST_TO_COMPLIANCE_OR_HESITATION"}


def detect_mintrec_transitions(turns: List[UtteranceTurn], labels: Optional[List[dict]] = None,
                               thread_of: Optional[Dict[int, str]] = None,
                               speaker_roles: Optional[Dict[str, str]] = None) -> List[Dict]:
    """Cross-speaker fine-grained intent transition pairs within the same thread.

    Noise threads (T_BACKCHANNEL_NOISE / T_CHITCHAT) are skipped. Each substantive source turn i is matched
    against the next TRANSITION_LOOKAHEAD substantive turns of its thread (j in i+1..i+3) from a different
    speaker; the first match wins, so interleaved speakers (A: problem, B: aside, C: guidance) still chain."""
    labels = labels or [classify_mintrec_taxonomy(t.text, t.rms_energy, t.zcr_pitch) for t in turns]
    thread_of = thread_of or {t.turn_id: t.thread_id for t in turns}
    per_thread: Dict[str, List[Tuple[int, str, str, str]]] = {}
    for t, lb in zip(turns, labels):
        tid = thread_of.get(t.turn_id, t.thread_id)
        if tid not in NOISE_THREADS:
            per_thread.setdefault(tid, []).append((t.turn_id, t.speaker_id, lb["fine"], t.text))
    out: List[Dict] = []
    for tid, seq in per_thread.items():
        for i, src in enumerate(seq):
            for j in range(i + 1, min(len(seq), i + 1 + TRANSITION_LOOKAHEAD)):
                tgt = seq[j]
                if tgt[1] == src[1]:
                    continue
                m = match_mintrec_transition([(src[0], src[2], src[3])], tgt[2], tgt[3])
                if m:
                    out.append(build_transition(m[0], tid, src, tgt, speaker_roles, j - i))
                    break
    return sorted(out, key=lambda x: (x["to_turn"], x["from_turn"]))


# --------------------------------------- Upgrade C: cross-turn slot & context fusion
ISSUE_TRIGGER_CATALOG: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("暴力性騷", ("暴力性騷", "性騷", "摸大腿", "被摸", "吃豆腐")),
    ("排泄物/褲子濕掉", ("褲子整個都濕掉", "濕掉", "排泄物", "大便", "黃金很臭", "整個尿")),
    ("山區野狗/蜜蜂", ("野狗", "遇到狗", "咬人狗", "四隻狗", "四條狗", "被狗", "蜜蜂", "虎頭蜂")),
    ("家屬猝逝/去世", ("猝逝", "去世", "過世", "到院前死亡", "沒有呼吸")),
    ("刑法傷害/案主暴力", ("刑法", "傷害罪", "案主暴力", "被打", "打人")),
    ("代購藥品/超範圍打掃", ("代購", "買藥", "普拿疼", "成藥", "刷馬桶", "洗廁所", "刷一刷")),
    ("居督與照服員綁定", ("綁居督", "居督綁", "綁居服", "個案綁", "特定的督導", "綁定居督", "居督管哪一個",
                         "哪一個居督管")),
    # mined from the real-interview golden hotspots (iv1 16:15 / 19:00, iv2 45:45, iv3 17:00 / 09:30)
    ("培訓未涵蓋的現場狀況", ("沒有教導", "培訓過程中")),
    ("居服員緊張/在外騎車無法即時解決", ("很緊張", "在騎車", "在外騎車", "沒辦法第一", "無法即時")),
    ("照服員回報與緊急事件處理", ("緊急事件", "第一時間非常緊急", "第一時間")),
    ("聲紋收集困難/家屬多變", ("聲紋可能會沒有", "家屬都是會變", "家屬很多", "收集生活這個個人覺得很困難")),
    ("小照回答錯誤/不完整", ("回答錯誤", "答案不正確", "回答不完整", "沒有依據", "找不到資料", "沒有回答")),
)
RESOLUTION_CATALOG: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("對講機功能直連居督手機", ("對講機", "緊急對講", "直連", "居督的手機", "督導的手機")),
    ("文字切片建置AI知識庫", ("切片", "AI知識庫", "建知識庫", "知識庫系統", "進知識庫")),
    ("放進多元項目累積經驗", ("多元項目", "累積下來", "累積經驗", "多元選項", "多元問題的選項")),
    ("納入刑法法條量身打造", ("刑法法條", "刑法丟進去", "量身打造")),
    ("照服員端緊急鈕+控制APP", ("緊急鈕", "緊急按鈕", "緊急控制", "控制的APP", "APP")),
    ("綁定碼角色分配", ("綁定碼", "角色分配", "分配角色")),
    ("語氣安撫居服員", ("語氣", "安撫")),
    ("案件綁居督/介面綁定設定", ("加一個介面", "做綁定", "給我綁")),
    ("後台文字標註分角色(免聲紋庫)", ("寫字在後台", "後台文字", "A或B自己分", "三個人就不分")),
    ("回答有問題一鍵回饋機制", ("反饋", "回答有問題", "可以打字", "可以填也可以不要填")),
    ("主動式每日關懷推播", ("主動每天", "每天八部", "時間到他就")),
)
DOMINANT_THREAD_SHARE = 0.5   # threads with >= 50% of the top thread's substantive chars are dominant


def _catalog_hits(turns: List[UtteranceTurn], catalog, roles: Dict[str, str], who_key: str,
                  thread_of: Dict[int, str], min_turn: int = 0) -> List[Dict]:
    found: Dict[str, Dict] = {}
    for t in turns:
        if t.turn_id < min_turn:
            continue
        txt = _norm(t.text)
        for label, cues in catalog:
            cue = next((c for c in cues if c in txt), None)
            if cue is None:
                continue
            if label in found:
                found[label]["mentions"] += 1
                if t.speaker_id not in found[label]["contributors"]:
                    found[label]["contributors"].append(t.speaker_id)
                continue
            found[label] = {"label": label, who_key: t.speaker_id, "role": roles.get(t.speaker_id, "unknown"),
                            "turn_id": t.turn_id, "thread_id": thread_of.get(t.turn_id, t.thread_id),
                            "evidence": cue, "mentions": 1, "contributors": [t.speaker_id]}
    return sorted(found.values(), key=lambda d: d["turn_id"])


def aggregate_cross_turn_slots(turns: List[UtteranceTurn], speaker_roles: Dict[str, str],
                               thread_of: Optional[Dict[int, str]] = None,
                               transitions: Optional[List[Dict]] = None,
                               safety_level: str = "NORMAL") -> Dict[str, Any]:
    """Fuse multi-speaker contributions in the dominant substantive thread(s) into one DST slot."""
    thread_of = thread_of or {t.turn_id: t.thread_id for t in turns}
    transitions = transitions or []
    chars: Dict[str, int] = {}
    for t in turns:
        tid = thread_of.get(t.turn_id, t.thread_id)
        if tid not in NOISE_THREADS and not is_agreement_only(t.text):
            chars[tid] = chars.get(tid, 0) + len(_norm(t.text))
    top = max(chars.values(), default=0)
    dominant = sorted((tid for tid, c in chars.items() if top and c >= DOMINANT_THREAD_SHARE * top),
                      key=lambda tid: -chars[tid])
    core = [t for t in turns if thread_of.get(t.turn_id, t.thread_id) in dominant and not is_agreement_only(t.text)]
    issues = _catalog_hits(core, ISSUE_TRIGGER_CATALOG, speaker_roles, "raised_by", thread_of)
    first_issue = issues[0]["turn_id"] if issues else 0   # resolutions must follow the first raised issue
    resolutions = _catalog_hits(core, RESOLUTION_CATALOG, speaker_roles, "proposed_by", thread_of, first_issue)
    roles_seen: List[str] = []
    for t in core:
        r = speaker_roles.get(t.speaker_id, "unknown")
        if r not in roles_seen:
            roles_seen.append(r)
    dom_tr = [x for x in transitions if x["thread_id"] in dominant]
    speakers = {t.speaker_id for t in core}
    if safety_level == "CRITICAL_P0" or any(x["type"] == "CRISIS_ESCALATION_CHAIN" and x.get("live_hazard", True)
                                            for x in dom_tr):
        state = "IMMEDIATE_ESCALATION"
    elif resolutions and len(speakers) >= 2 and not (dom_tr and dom_tr[-1]["type"] == "REQUEST_TO_REFUSAL"):
        state = "CONSENSUS_ACTIONABLE"
    else:
        state = "OPEN_DISCUSSION"
    return {"dominant_threads": dominant, "issue_triggers": issues, "proposed_resolutions": resolutions,
            "participating_roles": roles_seen, "participating_speakers": sorted(speakers),
            "transition_types": [x["type"] for x in dom_tr], "consensus_state": state}


# ------------------------- Upgrade D: GIFT 4-edge graph-induced topology (Gu et al., USTC & iFLYTEK)
GIFT_EDGE_TYPES = ("reply-self", "reply-to", "replied-by", "indirect-reply")
GIFT_EDGE_WEIGHTS = {"reply-self": 1.0, "reply-to": 0.92, "replied-by": 0.88, "indirect-reply": 0.45}
GIFT_WEAK_INTENTS = ("UNK", "Inform")
# replied-by child fine intent -> what the parent (weak) turn effectively was
# (an opposed weak turn was a proposal/request -> unlocks REQUEST_TO_REFUSAL)
GIFT_CHILD_AMEND = {"Advise": "Ask_for_help", "Comfort": "Ask_for_help", "Arrange": "Inform", "Oppose": "Arrange"}
GIFT_HELP_SEEKING = ("Ask_for_help", "Complain")
# graph-induced role inference: topic cues exchanged on reply-to / replied-by edges (meeting mode only)
GRAPH_ROLE_TOPIC_CUES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (ROLE_SUPERVISOR, ("居督", "督導", "綁", "派案", "居服員", "照服員", "居服", "個案")),
    (ROLE_CAREGIVER_LTC, ("去案家", "我的個案", "換尿布", "洗澡", "服務的時候", "我的夥伴")),
    (ROLE_DIRECTOR, ("主任", "機構", "公文", "審核")),
    (ROLE_RESEARCHER, ("系統", "介面", "知識庫", "小照", "後台", "APP")),
)
GRAPH_ROLE_MIN_EDGES = 3      # reply-to + replied-by edges exchanged with role-resolved partners
GRAPH_ROLE_MIN_SCORE = 2.0    # phi-weighted topic-cue hits
GRAPH_INFERRED_SOURCE = "graph_inferred"


def build_gift_graph(turns: List[UtteranceTurn], thread_of: Optional[Dict[int, str]] = None,
                     addressee_speaker: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
    """GIFT 4-edge graph over substantive turns (agreement-only / noise-thread turns are not nodes).

    reply-to parent of u_i: the most recent earlier turn of u_i's addressee speaker in the same thread, else
    the discourse latch (the nearest earlier turn of another speaker in the same thread, skipping u_i's own
    continuation turns, i.e. A-B-A adjacency) within ADDRESSEE_ADJACENCY_SEC. replied-by is its transpose,
    i == j is reply-self, every other pair is indirect-reply. The N x N matrix stays implicit
    (gift_edge / gift_edge_matrix) so 800-turn recordings cost O(N)."""
    thread_of = thread_of or {}
    addressee_speaker = addressee_speaker or {}
    nodes = [t for t in turns if thread_of.get(t.turn_id, t.thread_id) not in NOISE_THREADS
             and not is_agreement_only(t.text)]
    reply_to: Dict[int, int] = {}
    replied_by: Dict[int, List[int]] = {t.turn_id: [] for t in nodes}
    by_thread: Dict[str, List[UtteranceTurn]] = {}
    for t in nodes:
        prior = by_thread.setdefault(thread_of.get(t.turn_id, t.thread_id), [])
        addr = addressee_speaker.get(t.turn_id) or t.addressee
        parent = None
        if addr and _SPEAKER_ID_RE.match(addr) and addr != t.speaker_id:
            parent = next((p for p in reversed(prior) if p.speaker_id == addr), None)
        if parent is None:
            latch = next((p for p in reversed(prior) if p.speaker_id != t.speaker_id), None)
            if latch is not None and t.timestamp - latch.timestamp <= ADDRESSEE_ADJACENCY_SEC:
                parent = latch
        if parent is not None:
            reply_to[t.turn_id] = parent.turn_id
            replied_by[parent.turn_id].append(t.turn_id)
        prior.append(t)
    return {"node_ids": [t.turn_id for t in nodes], "speaker_of": {t.turn_id: t.speaker_id for t in nodes},
            "reply_to": reply_to, "replied_by": replied_by}


def gift_edge(graph: Dict[str, Any], i: int, j: int) -> str:
    """e_{i,j} for node turn ids i, j."""
    if i == j:
        return "reply-self"
    if graph["reply_to"].get(i) == j:
        return "reply-to"
    if graph["reply_to"].get(j) == i:
        return "replied-by"
    return "indirect-reply"


def gift_edge_matrix(graph: Dict[str, Any]) -> List[List[str]]:
    """Explicit pairwise e_{i,j} matrix in node order (only for small sessions / inspection)."""
    ids = graph["node_ids"]
    return [[gift_edge(graph, i, j) for j in ids] for i in ids]


def gift_edge_counts(graph: Dict[str, Any]) -> Dict[str, int]:
    n, r = len(graph["node_ids"]), len(graph["reply_to"])
    return {"reply-to": r, "replied-by": r, "reply-self": n, "indirect-reply": n * n - n - 2 * r}


def _gift_has_sop_cue(text: str) -> bool:
    global _LIVE_CHECKER
    if _LIVE_CHECKER is None:
        _LIVE_CHECKER = XiaoZhaoMPCEngine()
    t = _norm(text)
    return any(c in t for c in CRISIS_SOURCE_HAZARD_CUES) or bool(_LIVE_CHECKER.detect_domain_intents(t))


def gift_amend_intents(turns: List[UtteranceTurn], labels: List[dict],
                       graph: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """GIFT replied-by posterior amendment: a UNK / weak-Inform turn whose replied-by child gives
    Advise/Comfort (-> it was a help request), Arrange (-> it was a report), Oppose (-> it was a
    proposal, Arrange) or a hazard/SOP cue
    (-> help request) is re-read; a weak reply to a help request / complaint that carries an SOP cue
    is re-read as Advise. -> {turn_id: {from_fine, gift_fine, via, evidence_turn}}."""
    fine = {t.turn_id: lb["fine"] for t, lb in zip(turns, labels)}
    text = {t.turn_id: t.text for t in turns}
    out: Dict[int, Dict[str, Any]] = {}
    for tid in graph["node_ids"]:
        cur = fine[tid]
        if cur not in GIFT_WEAK_INTENTS:
            continue
        for c in graph["replied_by"][tid]:
            new = GIFT_CHILD_AMEND.get(fine[c]) or ("Ask_for_help" if _gift_has_sop_cue(text[c]) else None)
            if new and new != cur:
                out[tid] = {"from_fine": cur, "gift_fine": new, "via": "replied-by", "evidence_turn": c}
                break
        else:
            p = graph["reply_to"].get(tid)
            if p is not None and fine[p] in GIFT_HELP_SEEKING and _gift_has_sop_cue(text[tid]):
                out[tid] = {"from_fine": cur, "gift_fine": "Advise", "via": "reply-to", "evidence_turn": p}
    return out


def gift_boost_transitions(turns: List[UtteranceTurn], labels: List[dict], graph: Dict[str, Any],
                           amendments: Dict[int, Dict[str, Any]], transitions: List[Dict],
                           thread_of: Optional[Dict[int, str]] = None,
                           speaker_roles: Optional[Dict[str, str]] = None) -> List[Dict]:
    """Tag transitions lying on a GIFT reply-to edge (gift_phi, gift_weight = phi * staging weight) and add
    the graph-induced chains unlocked by amended intents. Returns the merged, sorted transition list."""
    thread_of = thread_of or {t.turn_id: t.thread_id for t in turns}
    by_id = {t.turn_id: t for t in turns}
    fine = {t.turn_id: lb["fine"] for t, lb in zip(turns, labels)}
    fine.update({tid: a["gift_fine"] for tid, a in amendments.items()})
    phi = GIFT_EDGE_WEIGHTS["reply-to"]
    seen = {(x["from_turn"], x["to_turn"]) for x in transitions}
    for x in transitions:
        on_edge = graph["reply_to"].get(x["to_turn"]) == x["from_turn"]
        x["gift_edge"] = "reply-to" if on_edge else "indirect-reply"
        x["gift_phi"] = phi if on_edge else GIFT_EDGE_WEIGHTS["indirect-reply"]
        x["gift_weight"] = round(x["gift_phi"] * TRANSITION_WEIGHT[x["type"]], 3)
    added: List[Dict] = []
    for child, parent in graph["reply_to"].items():
        if (parent, child) in seen or not (parent in amendments or child in amendments):
            continue
        src, tgt = by_id[parent], by_id[child]
        if src.speaker_id == tgt.speaker_id:
            continue
        m = match_mintrec_transition([(parent, fine[parent], src.text)], fine[child], tgt.text)
        if not m:
            continue
        tr = build_transition(m[0], thread_of.get(child, tgt.thread_id),
                              (parent, src.speaker_id, fine[parent], src.text),
                              (child, tgt.speaker_id, fine[child], tgt.text), speaker_roles, 1)
        tr.update({"gift_edge": "reply-to", "gift_phi": phi, "graph_induced": True,
                   "gift_weight": round(phi * TRANSITION_WEIGHT[tr["type"]] * 1.5, 3)})
        added.append(tr)
    return sorted(transitions + added, key=lambda x: (x["to_turn"], x["from_turn"]))


def gift_infer_roles(turns: List[UtteranceTurn], graph: Dict[str, Any], roles: Dict[str, str],
                     unresolved: List[str]) -> Dict[str, Tuple[str, Dict[str, Any]]]:
    """Graph-induced role for speakers without lexical self-cues: count reply-to / replied-by edges exchanged
    with role-resolved meeting partners, then score the extended roles NOT held by those partners by
    topic cues on the speaker's own edge turns (phi reply-self) and the partners' edge turns (phi reply-to /
    replied-by). -> {speaker: (role, evidence)} for speakers clearing both thresholds."""
    text = {t.turn_id: _norm(t.text) for t in turns}
    sp_of = graph["speaker_of"]
    ext_roles = {r for r, _ in EXTENDED_ROLE_CUES}
    out: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for sp in unresolved:
        partners: Dict[str, int] = {}
        scores: Dict[str, float] = {}
        n_to = n_by = 0
        for child, parent in graph["reply_to"].items():
            if sp_of[child] == sp and roles.get(sp_of[parent]) in ext_roles:
                mine, other, kind = child, parent, "reply-to"
                n_to += 1
            elif sp_of[parent] == sp and roles.get(sp_of[child]) in ext_roles:
                mine, other, kind = parent, child, "replied-by"
                n_by += 1
            else:
                continue
            partners[sp_of[other]] = partners.get(sp_of[other], 0) + 1
            for role, cues in GRAPH_ROLE_TOPIC_CUES:
                hits = (GIFT_EDGE_WEIGHTS["reply-self"] * sum(c in text[mine] for c in cues)
                        + GIFT_EDGE_WEIGHTS[kind] * sum(c in text[other] for c in cues))
                if hits:
                    scores[role] = scores.get(role, 0.0) + hits
        held = {roles[p] for p in partners}
        cands = {r: s for r, s in scores.items() if r not in held}
        if n_to + n_by < GRAPH_ROLE_MIN_EDGES or not cands:
            continue
        role, score = max(cands.items(), key=lambda kv: kv[1])
        if score >= GRAPH_ROLE_MIN_SCORE:
            out[sp] = (role, {"label": f"{role} (graph_inferred)", "reply_to_edges": n_to,
                              "replied_by_edges": n_by,
                              "partners": {p: roles[p] for p in sorted(partners)},
                              "topic_scores": {r: round(s, 2) for r, s in sorted(scores.items())}})
    return out


# ------------- Upgrade E: Bi-CL bi-level session-prototype disentanglement (Zhang et al., Tsinghua/NUS/SMU)
BICL_DOMAIN_TOKENS = ("長照", "照服員", "居服員", "居服", "知識庫", "居督", "督導", "法規", "小照", "個案",
                      "家屬", "機構", "主任", "案家", "緊急", "按鍵", "介面", "聲紋", "綁定", "AI", "APP",
                      "回答", "多元", "經驗", "刑法", "暴力", "實驗", "學生", "研究")
BICL_MENTION_CUES = ("主任", "督導", "居督", "照服員", "居服員", "家屬", "老師", "學生", "小照", "長輩", "個案")
BICL_STOP_CHARS = set("的了是我你他她它在有和也都就這那嗯對啊喔呢吧嗎呀哦哈很要會不說個們一然後因為所以可以")
BICL_W_OWN, BICL_W_CONTEXT, BICL_W_DOMAIN = 1.0, 0.5, 2.0   # u_i n-grams, (u_{i-1}, u_{i+1}) n-grams, domain
BICL_W_SPEAKER, BICL_W_MENTION = 0.6, 1.0                   # multi-hot speaker s(i) and mention vector m_i
BICL_SPEAKER_CONT_BONUS = 0.04  # s(i) == s(j) local continuity: neighbour in the same session pulls u_i
BICL_TURNS_PER_SESSION = 6      # K <= max(2, floor(N_substantive / 6))
BICL_MERGE_SIM = 0.45           # prototypes this close are one session
BICL_SWITCH_MARGIN = 0.03       # E-step hysteresis against prototype churn
BICL_EM_ITERS = 8
BICL_MICRO_MAX = 2              # sessions with <= 2 turns are micro-threads
BICL_MICRO_OWN_MIN = 0.05       # E-step moves need own-content overlap with the target (or a sandwich)


def _bicl_tokens(text: str) -> set:
    t = re.sub(r"[\s，。,.!！?？、~～…:：;；\"'()（）]", "", text or "")
    grams = {t[i:i + 2] for i in range(len(t) - 1) if not (set(t[i:i + 2]) & BICL_STOP_CHARS)}
    return grams


def _bicl_vectors(nodes: List[UtteranceTurn], keyword_fn=None,
                  own_only: bool = False) -> List[Dict[str, float]]:
    """Utterance-level v_i: bidirectional context window (u_{i-1}, u_i, u_{i+1}) n-grams + domain tokens,
    concatenated with Bi-CL's multi-hot speaker s(i) and mention m_i features. own_only -> content of u_i
    alone (n-grams, domain, mentions; no context, no speaker)."""
    kw = keyword_fn or _bicl_tokens
    grams = [kw(t.text) for t in nodes]
    out: List[Dict[str, float]] = []
    for i, t in enumerate(nodes):
        v: Dict[str, float] = {}
        if own_only:
            for g in grams[i]:
                v["g:" + g] = BICL_W_OWN
            txt = _norm(t.text)
            v.update({"d:" + d: BICL_W_DOMAIN for d in BICL_DOMAIN_TOKENS if d in txt})
            v.update({"m:" + m: BICL_W_MENTION for m in BICL_MENTION_CUES if m in txt})
            out.append(v)
            continue
        for j, w in ((i - 1, BICL_W_CONTEXT), (i, BICL_W_OWN), (i + 1, BICL_W_CONTEXT)):
            if 0 <= j < len(nodes):
                for g in grams[j]:
                    v["g:" + g] = max(v.get("g:" + g, 0.0), w)
                txt = _norm(nodes[j].text)
                for d in BICL_DOMAIN_TOKENS:
                    if d in txt:
                        v["d:" + d] = max(v.get("d:" + d, 0.0), BICL_W_DOMAIN * (w / BICL_W_OWN))
        v["s:" + t.speaker_id] = BICL_W_SPEAKER
        for m in BICL_MENTION_CUES:
            if m in _norm(t.text):
                v["m:" + m] = BICL_W_MENTION
        out.append(v)
    return out


def _vadd(acc: Dict[str, float], v: Dict[str, float], sign: float = 1.0) -> None:
    for k, x in v.items():
        acc[k] = acc.get(k, 0.0) + sign * x


def _cos(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(x * b.get(k, 0.0) for k, x in a.items())
    na = sum(x * x for x in a.values()) ** 0.5
    nb = sum(x * x for x in b.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def bi_cl_disentangle_threads(turns: List[UtteranceTurn], thread_of: Optional[Dict[int, str]] = None,
                              keyword_fn=None) -> Tuple[Dict[int, str], Dict[str, Any]]:
    """Refine greedy preceding-only threads with Bi-CL's two levels: utterance vectors v_i vs. session
    prototypes p_k = mean(v_j, j in s_k), unsupervised EM (E: leave-one-out nearest prototype + local
    same-speaker continuity, M: recompute p_k) with K-estimation (merge prototypes while
    K > max(2, N_substantive // 6) or cos(p_a, p_b) >= BICL_MERGE_SIM). Micro-threads have no prototype
    besides themselves, so their turns flock to the dominant session. Noise turns keep their thread.
    -> (refined thread_of, bi_cl_disentangle_summary). Merged sessions keep the larger thread's id."""
    thread_of = dict(thread_of or {t.turn_id: t.thread_id for t in turns})
    nodes = [t for t in turns if thread_of.get(t.turn_id, t.thread_id) not in NOISE_THREADS
             and not is_agreement_only(t.text)]
    raw = [thread_of.get(t.turn_id, t.thread_id) for t in nodes]
    n = len(nodes)
    k_max = max(2, n // BICL_TURNS_PER_SESSION)
    summary: Dict[str, Any] = {"substantive_turns": n, "k_upper_bound": k_max,
                               "raw_greedy_threads": len(set(raw)), "raw_thread_ids": sorted(set(raw))}
    if n == 0:
        summary.update({"estimated_k": 0, "refined_prototype_threads": 0, "refined_thread_ids": [],
                        "prototype_cohesion_score": 0.0, "raw_cohesion_score": 0.0, "reassigned_turns": [],
                        "em_iterations": 0})
        return thread_of, summary
    vecs = _bicl_vectors(nodes, keyword_fn)
    own = _bicl_vectors(nodes, keyword_fn, own_only=True)
    first_seen = {}
    for i, lab in enumerate(raw):
        first_seen.setdefault(lab, i)
    assign = list(raw)

    def prototypes() -> Dict[str, Tuple[Dict[str, float], int]]:
        sums: Dict[str, Dict[str, float]] = {}
        cnt: Dict[str, int] = {}
        for v, lab in zip(vecs, assign):
            _vadd(sums.setdefault(lab, {}), v)
            cnt[lab] = cnt.get(lab, 0) + 1
        return {lab: ({k: x / cnt[lab] for k, x in s.items()}, cnt[lab]) for lab, s in sums.items()}

    def cohesion() -> float:
        """Mean leave-one-out cos(v_i, p_k(i) without v_i); a singleton session scores 0, so
        fragmentation into micro-threads is not rewarded."""
        sums: Dict[str, Dict[str, float]] = {}
        cnt: Dict[str, int] = {}
        for v, lab in zip(vecs, assign):
            _vadd(sums.setdefault(lab, {}), v)
            cnt[lab] = cnt.get(lab, 0) + 1
        tot = 0.0
        for v, lab in zip(vecs, assign):
            if cnt[lab] > 1:
                rest = dict(sums[lab])
                _vadd(rest, v, -1.0)
                tot += _cos(v, rest)
        return round(tot / n, 3)

    def continuity(i: int, lab: str) -> float:
        return sum(BICL_SPEAKER_CONT_BONUS for j in (i - 1, i + 1)
                   if 0 <= j < n and nodes[j].speaker_id == nodes[i].speaker_id and assign[j] == lab)

    raw_cohesion = cohesion()

    def k_step() -> bool:
        """Merge the closest prototype pair while K is over budget or two prototypes coincide."""
        changed = False
        while True:
            protos = prototypes()
            labs = sorted(protos, key=lambda l: (-protos[l][1], first_seen[l]))
            if len(labs) < 2:
                return changed
            pairs = [(a, b) for i, a in enumerate(labs) for b in labs[i + 1:]]
            if len(labs) > k_max:            # over budget: merge the closest contextual prototypes
                best = max(((_cos(protos[a][0], protos[b][0]), a, b) for a, b in pairs), key=lambda x: x[0])
            else:                            # coincidence test on own content (context leaks across interleaving)
                own_p: Dict[str, Dict[str, float]] = {}
                for j, lab in enumerate(assign):
                    _vadd(own_p.setdefault(lab, {}), own[j])
                best = max(((_cos(own_p[a], own_p[b]), a, b) for a, b in pairs), key=lambda x: x[0])
                if best[0] < BICL_MERGE_SIM:
                    return changed
            keep, drop = best[1], best[2]   # labs is size-ordered, so `keep` is the larger session
            assign[:] = [keep if lab == drop else lab for lab in assign]
            changed = True

    def may_move(i: int, lab: str, own_sums: Dict[str, Dict[str, float]]) -> bool:
        """Within the K budget u_i only moves to session `lab` when it is a singleton session sandwiched inside
        it (micro-thread fragmentation; a 2-turn interleaved topic is not) or shares own content with it:
        context n-grams alone are boundary leakage."""
        if (assign.count(assign[i]) == 1 and 0 < i < n - 1
                and assign[i - 1] == lab and assign[i + 1] == lab):
            return True
        return _cos(own[i], own_sums.get(lab, {})) >= BICL_MICRO_OWN_MIN

    # K-estimation over budget: each turn of a <= BICL_MICRO_MAX session joins its nearest established p_k
    protos = prototypes()
    established = [lab for lab, (_, c) in protos.items() if c > BICL_MICRO_MAX]
    if established and len(protos) > k_max:
        for i, lab in enumerate(list(assign)):
            if protos[lab][1] <= BICL_MICRO_MAX:
                assign[i] = max(established, key=lambda l: (_cos(vecs[i], protos[l][0]) + continuity(i, l),
                                                            protos[l][1]))
    iters = 0
    for iters in range(1, BICL_EM_ITERS + 1):
        changed = k_step()
        sums: Dict[str, Dict[str, float]] = {}
        cnt: Dict[str, int] = {}
        for v, lab in zip(vecs, assign):
            _vadd(sums.setdefault(lab, {}), v)
            cnt[lab] = cnt.get(lab, 0) + 1
        own_sums: Dict[str, Dict[str, float]] = {}
        for j, lab in enumerate(assign):
            _vadd(own_sums.setdefault(lab, {}), own[j])
        new_assign = list(assign)
        for i, (v, cur) in enumerate(zip(vecs, assign)):          # E-step (leave-one-out prototypes)
            scores = {}
            for lab, s in sums.items():
                c = cnt[lab] - (1 if lab == cur else 0)
                if c <= 0:
                    scores[lab] = 0.0
                    continue
                proto = dict(s)
                if lab == cur:
                    _vadd(proto, v, -1.0)
                scores[lab] = _cos(v, proto) + continuity(i, lab)   # + s(i) == s(j) local continuity
            best = max(scores, key=lambda l: (scores[l], cnt[l]))
            if best != cur and scores[best] - scores[cur] >= BICL_SWITCH_MARGIN and may_move(i, best, own_sums):
                new_assign[i] = best
        if new_assign != assign:                                    # M-step happens in prototypes()
            assign[:] = new_assign
            changed = True
        if not changed:
            break
    k_step()
    refined = dict(thread_of)
    reassigned = []
    for t, old, new in zip(nodes, raw, assign):
        refined[t.turn_id] = new
        if old != new:
            reassigned.append({"turn_id": t.turn_id, "from": old, "to": new})
    summary.update({"estimated_k": len(set(assign)), "refined_prototype_threads": len(set(assign)),
                    "refined_thread_ids": sorted(set(assign)), "prototype_cohesion_score": cohesion(),
                    "raw_cohesion_score": raw_cohesion, "reassigned_turns": reassigned, "em_iterations": iters})
    return refined, summary


class XiaoZhaoMPCEngine:
    """小照 × 樹莓派 5 多人多輪意圖偵測核心 (Addressee + Disentanglement + OOD + P0)"""

    P0_EMERGENCY_KW = ("跌倒", "昏倒", "流血", "暴力", "打人", "自殺", "沒呼吸", "猝逝", "車禍",
                       "報警", "急救", "沒有呼吸", "去世", "蜜蜂螫", "整個尿", "褲子整個都濕掉",
                       "過世", "到院前死亡", "心臟停")
    P1_SAFETY_KW = ("摸大腿", "性騷擾", "性騷", "被摸", "吃豆腐", "私密處", "洗廁所", "刷馬桶")
    MED_BA16_KW = ("買藥", "成藥", "普拿疼", "止痛藥", "代購", "順便買")
    RESPITE_GA_SC_KW = ("喘息", "短照", "GA09", "SC09", "臨時托顧")
    CHITCHAT_NOISE_KW = ("便當", "排骨", "雞腿", "午餐", "八點檔", "電視劇", "天氣", "重播")
    OOD_CRISIS_KW = ("咬人狗", "野狗", "蜜蜂", "淹水", "土石流", "虎頭蜂",
                    "報警", "急救", "沒有呼吸", "去世", "蜜蜂螫", "整個尿", "褲子整個都濕掉",
                    "遇到狗", "四條狗", "四隻狗", "被狗", "黃金很臭")
    OOD_HAZARD_KW = OOD_CRISIS_KW  # alias used by telemetry tooling
    LEGAL_REG_KW = ("法規", "長照基本法")
    JIA_AN_RULE_KW = ("嘉安", "內規", "本身的規定")
    SUPERVISOR_FB_KW = ("多元問題", "多元項目", "多元需求", "多元選項", "天馬行空", "不在BA24", "知識庫", "督導")
    # long-term-care intents mined from the 32-min + 52-min real interviews
    EMERGENCY_BUTTON_KW = ("緊急鈕", "緊急按鈕", "緊急鍵", "緊急對講", "對講機", "緊急通報", "緊急控制",
                           "跳紅鈴", "緊急事件")
    DEATH_AFTERCARE_KW = ("宗教信仰", "後事", "安撫鎮定", "會慌", "很慌", "公立醫院", "私人醫院", "死亡證明",
                          "到院前")
    CASE_MGMT_KW = ("指揮結構", "個案的管理", "管理介面", "收個案", "收了哪個個案", "督導管", "管理督導",
                    "權限", "帳號", "主任端", "督導端", "居督端", "派案", "指派", "控管")
    SERVICE_CODE_KW = ("服務項目", "SC項目", "SC的項目", "BA項目", "多元項目", "代購", "補充服務")
    SERVICE_CODE_RE = re.compile(r"(?<![A-Za-z])(?:BA|GA|SC)\s?\d{0,2}(?![A-Za-z])")
    EXPERIENCE_KB_KW = ("歷史經驗", "歷史的經驗", "以往", "經驗庫", "累積", "紀錄下來", "案例", "RAG",
                        "學起來", "訓練AI", "初稿", "SOP", "經驗")
    PRIVACY_KW = ("隱藏", "收音", "麥克風", "錄音", "眼鏡", "真實的姓名", "個資", "隱私")
    # route precedence (highest first); P0 short-circuits everything
    ROUTE_BY_INTENT = (
        ("OOS_ATYPICAL_FIELD_HAZARD", "OOD_SAFE_FALLBACK_AND_EXPERIENCE_KB"),
        ("GA09_SC09_RESPITE_FORM", "FORM_GA09_SC09_ENTRY"),
        ("EMERGENCY_BUTTON_ESCALATION", "SUPERVISOR_URGENT_CALL_CHANNEL"),
        ("DEATH_AT_HOME_AFTERCARE_SOP", "RAG_EXPERIENCE_KB_AFTERCARE"),
        ("CASE_ASSIGNMENT_HIERARCHY", "CASE_MGMT_ADMIN_QUEUE"),
        ("SERVICE_CODE_QUERY", "RAG_SERVICE_CODE_KB"),
        ("SUPERVISOR_DIVERSE_FEEDBACK", "SUPERVISOR_REVIEW_QUEUE"),
        ("EXPERIENCE_KB_CONTRIBUTION", "EXPERIENCE_KB_INGEST"),
        ("JIA_AN_INTERNAL_RULE", "RAG_JIA_AN_INTERNAL_RULE_KB"),
        ("LEGAL_REGULATION_QUERY", "RAG_LEGAL_REGULATION_KB"),
        ("PRIVACY_CONSENT_RECORDING", "PRIVACY_REVIEW_QUEUE"),
    )
    P1_INTENTS = ("WORKPLACE_HARASSMENT_REPORT", "OOS_ATYPICAL_FIELD_HAZARD")
    # turns carrying these stay in the routed context window (emergency-button design talk included, so a
    # meta-referenced P0 like 像剛我們說的暴力 resolves to the button/walkie-talkie intent instead of vanishing)
    TASK_CLUE_KW = MED_BA16_KW + P1_SAFETY_KW + RESPITE_GA_SC_KW + OOD_CRISIS_KW + EMERGENCY_BUTTON_KW

    # bare 剛剛說/剛才說 are deliberately excluded: 「家屬剛剛說阿公跌倒了」 reports a live event
    HYPOTHETICAL_MARKERS = ("例如", "譬如", "比如", "假如", "假設",
                            "像剛剛", "像剛才", "像剛", "剛我們說", "我們剛剛說", "我們剛才說")
    HYPOTHETICAL_SCOPE_CHARS = 16   # a marker covers the next N chars (merged STT turns lack punctuation)
    _CLAUSE_BREAK_RE = re.compile(r"[。！!？?；;，,]|可是|但是|結果|剛剛|現在")  # ends the example scope

    def _is_hypothetical_hit(self, t: str, pos: int) -> bool:
        """P0 keyword at `pos` is an illustrative example (例如說家屬去世) rather than a live event."""
        window = t[max(0, pos - self.HYPOTHETICAL_SCOPE_CHARS):pos]
        for m in self.HYPOTHETICAL_MARKERS:
            i = window.rfind(m)
            if i >= 0 and not self._CLAUSE_BREAK_RE.search(window[i + len(m):]):  # marker may contain 剛剛
                return True
        return False

    def has_live_p0(self, text: str) -> bool:
        """True iff some P0 keyword occurrence is NOT inside a 例如/譬如 example clause."""
        return self._has_live_kw(text, self.P0_EMERGENCY_KW)

    def _has_live_kw(self, text: str, kws: Tuple[str, ...]) -> bool:
        t = (text or "").replace(" ", "")
        for k in kws:
            pos = t.find(k)
            while pos >= 0:
                if not self._is_hypothetical_hit(t, pos):
                    return True
                pos = t.find(k, pos + 1)
        return False

    def detect_domain_intents(self, text: str) -> List[str]:
        """Ordered domain intents for a text span (O(len(text))); a live P0 short-circuits."""
        t = (text or "").replace(" ", "")
        if self.has_live_p0(t):
            return ["P0_EMERGENCY_ALERT"]
        intents = []
        if any(k in t for k in self.MED_BA16_KW):
            intents.append("BA16_MEDICINE_PURCHASE_INQUIRY")
        if any(k in t for k in self.P1_SAFETY_KW):
            intents.append("WORKPLACE_HARASSMENT_REPORT")
        if "馬桶" in t or "廁所" in t:
            intents.append("EXTRA_OUT_OF_SCOPE_CLEANING")
        if any(k in t for k in self.RESPITE_GA_SC_KW):
            intents.append("GA09_SC09_RESPITE_FORM")
        if self._has_live_kw(t, self.OOD_CRISIS_KW):
            intents.append("OOS_ATYPICAL_FIELD_HAZARD")
        if any(k in t for k in self.LEGAL_REG_KW):
            intents.append("LEGAL_REGULATION_QUERY")
        if any(k in t for k in self.JIA_AN_RULE_KW):
            intents.append("JIA_AN_INTERNAL_RULE")
        if any(k in t for k in self.SUPERVISOR_FB_KW):
            intents.append("SUPERVISOR_DIVERSE_FEEDBACK")
        if any(k in t for k in self.EMERGENCY_BUTTON_KW):
            intents.append("EMERGENCY_BUTTON_ESCALATION")
        if any(k in t for k in self.DEATH_AFTERCARE_KW):
            intents.append("DEATH_AT_HOME_AFTERCARE_SOP")
        if any(k in t for k in self.CASE_MGMT_KW):
            intents.append("CASE_ASSIGNMENT_HIERARCHY")
        if ("GA09_SC09_RESPITE_FORM" not in intents and "BA16_MEDICINE_PURCHASE_INQUIRY" not in intents
                and (any(k in t for k in self.SERVICE_CODE_KW) or self.SERVICE_CODE_RE.search(t))):
            intents.append("SERVICE_CODE_QUERY")
        if any(k in t for k in self.EXPERIENCE_KB_KW):
            intents.append("EXPERIENCE_KB_CONTRIBUTION")
        if any(k in t for k in self.PRIVACY_KW):
            intents.append("PRIVACY_CONSENT_RECORDING")
        return intents

    def resolve_route(self, intents: List[str]) -> Tuple[str, str, str]:
        """-> (route_target, safety_level, pi_state)"""
        if "P0_EMERGENCY_ALERT" in intents:
            return "P0_INSTANT_SUPERVISOR_BROADCAST", "CRITICAL_P0", "0E_Emergency_Broadcast"
        route = next((r for i, r in self.ROUTE_BY_INTENT if i in intents), "RAG_STANDARD_SOP")
        safety = "HIGH_P1" if any(i in intents for i in self.P1_INTENTS) else "NORMAL"
        return route, safety, "5_Wait_LINE_Confirm"

    @staticmethod
    def dst_slots_for(intents: List[str]) -> Dict[str, str]:
        slots: Dict[str, str] = {}
        if "BA16_MEDICINE_PURCHASE_INQUIRY" in intents:
            slots["service_code"] = "BA16_代購限制(不可代購成藥)"
            slots["requester_to_target"] = "family -> caregiver"
        if "GA09_SC09_RESPITE_FORM" in intents:
            slots["form_type"] = "GA09/SC09_喘息短照紀錄表"
        return slots

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
                elif any(k in prev.text for k in self.TASK_CLUE_KW):
                    selected_indices.append(prev.turn_id)

        selected_indices.append(current_turn.turn_id)
        tid = "T_TASK_MAIN"
        session.active_threads[tid] = selected_indices
        return tid, selected_indices

    def analyze_session(self, session: DialogueSessionState, role_priors: Optional[Dict[str, str]] = None) -> Dict:
        # upstream hints (e.g. general_mpc_disentangle T_TOPIC_n / speaker_N addressee) before overwrite
        pre_role = {t.turn_id: t.inferred_role for t in session.turns}
        pre_addr = {t.turn_id: t.addressee for t in session.turns}
        pre_thread = {t.turn_id: t.thread_id for t in session.turns}
        for t in session.turns:
            t.inferred_role, t.addressee = self.infer_role_and_addressee(t, session.turns[:t.turn_id - 1])
            t.thread_id, selected_idx = self.disentangle_threads(session, t)
        # transitions / fusion run on upstream topic threads when given, else on the engine's TD/MF threads
        thread_of = {t.turn_id: (t.thread_id if t.thread_id == "T_CHITCHAT" or pre_thread[t.turn_id] in ("T0", "")
                                 else pre_thread[t.turn_id]) for t in session.turns}
        # Upgrade E: Bi-CL session-prototype refinement of the (greedy) upstream threads
        thread_of, bi_cl_summary = bi_cl_disentangle_threads(session.turns, thread_of)

        # Upgrade A: session-level roles + who->whom
        role_sources: Dict[str, str] = {}
        graph_role_evidence: Dict[str, Any] = {}
        speaker_roles = infer_session_speaker_roles(session.turns, pre_role,
                                                    role_priors or getattr(session, "role_priors", None),
                                                    role_sources, graph_role_evidence)
        role_detail: Dict[int, Tuple[str, str, str]] = {}
        for i, t in enumerate(session.turns):
            role = speaker_roles.get(t.speaker_id, t.inferred_role)
            engine_addr = t.addressee if role in HOMECARE_ROLES or t.addressee == "system_xiaozhao" else "unknown"
            addr_role, addr_sp = resolve_addressee_role(t, session.turns[:i], speaker_roles, engine_addr,
                                                        pre_addr[t.turn_id])
            t.inferred_role = role
            if t.addressee == "unknown" and addr_role != "unknown":
                t.addressee = addr_role
            role_detail[t.turn_id] = (pre_role[t.turn_id], addr_sp, f"{role} -> {addr_role}")

        last_turn = session.turns[-1]
        _, selected_idx = self.disentangle_threads(session, last_turn)
        context_turns = [t for t in session.turns if t.turn_id in selected_idx]
        merged_text = " ".join(t.text for t in context_turns)

        intents = self.detect_domain_intents(merged_text)
        route, safety_level, session.pi_state = self.resolve_route(intents)
        session.dst_slots.update(self.dst_slots_for(intents))
        if not intents:
            intents.append("GENERAL_CARE_LOG")

        mintrec_labels = [classify_mintrec_taxonomy(t.text, t.rms_energy, t.zcr_pitch) for t in session.turns]
        mintrec_summary = summarize_mintrec(mintrec_labels)
        mintrec_summary["per_turn"] = [{"turn_id": t.turn_id, **lb} for t, lb in zip(session.turns, mintrec_labels)]

        # Upgrade D: GIFT 4-edge graph + replied-by posterior amendment (feeds B + C)
        gift = build_gift_graph(session.turns, thread_of,
                                {tid: d[1] for tid, d in role_detail.items() if _SPEAKER_ID_RE.match(d[1] or "")})
        amendments = gift_amend_intents(session.turns, mintrec_labels, gift)

        # Upgrade B + C
        transitions = detect_mintrec_transitions(session.turns, mintrec_labels, thread_of, speaker_roles)
        transitions = gift_boost_transitions(session.turns, mintrec_labels, gift, amendments, transitions,
                                             thread_of, speaker_roles)
        session.dst_slots["cross_turn_fusion"] = aggregate_cross_turn_slots(
            session.turns, speaker_roles, thread_of, transitions, safety_level)
        fine_of = {t.turn_id: lb["fine"] for t, lb in zip(session.turns, mintrec_labels)}
        fusion = session.dst_slots["cross_turn_fusion"]
        anchor = (fusion["issue_triggers"][0]["turn_id"] if fusion["issue_triggers"] else
                  next((tid for tid in reversed(gift["node_ids"])), last_turn.turn_id))
        role_clarified = [t.turn_id for t in session.turns if role_sources.get(t.speaker_id) == GRAPH_INFERRED_SOURCE]
        gift_summary = {
            "edge_counts": gift_edge_counts(gift),
            "edge_weights_phi": dict(GIFT_EDGE_WEIGHTS),
            "replied_by_amendments": len(set(amendments) | set(role_clarified)),
            "intent_amendments": [{"turn_id": tid, **a} for tid, a in sorted(amendments.items())],
            "graph_inferred_roles": {sp: ev["label"] for sp, ev in graph_role_evidence.items()},
            "graph_role_evidence": graph_role_evidence,
            "graph_induced_transitions": sum(1 for x in transitions if x.get("graph_induced")),
            "anchor_turn_topology": {
                "turn_id": anchor,
                "basis": "issue_trigger" if fusion["issue_triggers"] else "session_trigger",
                "reply_to_turn_id": gift["reply_to"].get(anchor),
                "replied_by_turn_ids": list(gift["replied_by"].get(anchor, [])),
            },
        }

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
            "mintrec_summary": mintrec_summary,
            "speaker_roles": speaker_roles,
            "speaker_role_sources": role_sources,
            "mintrec_transitions": transitions,
            "gift_graph_summary": gift_summary,
            "bi_cl_disentangle_summary": bi_cl_summary,
            "turns_detail": [
                {
                    "turn_id": t.turn_id,
                    "speaker_id": t.speaker_id,
                    "role_pre_filled": role_detail[t.turn_id][0],
                    "inferred_role": t.inferred_role,
                    "addressee": t.addressee,
                    "addressee_speaker": role_detail[t.turn_id][1],
                    "who_to_whom_role": role_detail[t.turn_id][2],
                    "thread_id": t.thread_id,
                    "source_thread_id": thread_of[t.turn_id],
                    "mintrec_fine": fine_of[t.turn_id],
                    "gift_fine": amendments.get(t.turn_id, {}).get("gift_fine", fine_of[t.turn_id]),
                    "gift_reply_to": gift["reply_to"].get(t.turn_id),
                    "text": t.text,
                }
                for t in session.turns
            ],
        }

class AmbientIntentPreFetcher:
    """Passive listener that stages backend packets *before* the caregiver asks 小照.

    Staging rule per turn:
      - T_CHITCHAT noise is always ignored; MIntRec Agree/UNK is ignored unless it carries a domain intent.
      - Any domain intent stages a packet keyed by that intent.
      - MIntRec Ask_for_help / Prevent stage on their own (safety); Arrange / Complain / Advise
        stage only when coupled with a care-domain cue.
    Packets are keyed in a dict so firing is O(#keys) — never a rescan of the turn history.
    """

    GOAL_SAFETY_FINE = ("Ask_for_help", "Prevent")
    GOAL_DOMAIN_FINE = ("Arrange", "Complain", "Advise", "Care")
    NOISE_FINE = ("Agree", "UNK", "Thank", "Leave")
    DOMAIN_CUES = ("長輩", "阿公", "阿嬤", "照服", "居服", "服務", "洗澡", "藥", "表單", "個案", "案家",
                   "督導", "居督", "主任", "家屬", "時數", "照顧", "照護", "復健", "長照", "醫院", "緊急",
                   "知識庫", "BA", "GA", "SC")
    REFERENTIAL = ("剛剛", "那樣", "那個", "這樣", "前面", "他講的")
    SAFETY_RANK = {"NORMAL": 0, "HIGH_P1": 1, "CRITICAL_P0": 2}

    def __init__(self, engine: Optional[XiaoZhaoMPCEngine] = None, session_id: str = "S_AMBIENT",
                 case_id: str = "CASE_AMBIENT", ttl_sec: float = 600.0):
        self.engine = engine or XiaoZhaoMPCEngine()
        self.session_id = session_id
        self.case_id = case_id
        self.ttl_sec = ttl_sec
        self.staged_packets: Dict[str, dict] = {}
        self.auto_fired: List[dict] = []
        self.ignored_count = 0
        self.ingested_count = 0
        self.domain_intent_counts: Dict[str, int] = {}
        self.mintrec_labels: List[dict] = []
        self._fire_seq = 0
        self.transitions: List[dict] = []
        self._thread_window: Dict[str, List[Tuple[Tuple[int, str, str, str], bool]]] = {}

    BASE_STAGING_PRIORITY = 1
    CHAIN_STAGE_TYPES = ("CRISIS_ESCALATION_CHAIN", "REQUEST_TO_COMPLIANCE_OR_HESITATION")

    def _track_transition(self, turn: UtteranceTurn, fine: str, is_noise: bool) -> List[dict]:
        """Upgrade B (streaming): the new turn may answer any not-yet-answered source among the last
        TRANSITION_LOOKAHEAD substantive turns of its thread (same semantics as detect_mintrec_transitions)."""
        if is_noise:
            return []
        tid = turn.thread_id
        window = self._thread_window.setdefault(tid, [])
        cur = (turn.turn_id, turn.speaker_id, fine, turn.text)
        found: List[dict] = []
        for k, entry in enumerate(window):
            src, answered = entry
            if answered or src[1] == turn.speaker_id:
                continue
            m = match_mintrec_transition([(src[0], src[2], src[3])], fine, turn.text)
            if m:
                found.append(build_transition(m[0], tid, src, cur, lag=len(window) - k))
                window[k] = (src, True)
        window.append((cur, False))
        del window[:-TRANSITION_LOOKAHEAD]
        self.transitions.extend(found)
        return found

    def _boost_for_transition(self, tr: dict, turn: UtteranceTurn) -> List[str]:
        """Raise staging priority of packets holding either end of a completed chain. A high-priority chain
        whose source was never staged (no domain cue) is staged on its own as MINTREC_CHAIN_<TYPE>."""
        w = TRANSITION_WEIGHT[tr["type"]]
        hit = [k for k, p in self.staged_packets.items()
               if tr["from_turn"] in p["turn_ids"] or tr["to_turn"] in p["turn_ids"]]
        if not hit and tr["type"] in self.CHAIN_STAGE_TYPES:
            key = f"MINTREC_CHAIN_{tr['type']}"
            src = next((e[0] for e in self._thread_window.get(tr["thread_id"], []) if e[0][0] == tr["from_turn"]),
                       None)
            pkt = self.staged_packets.setdefault(key, {
                "intent_key": key, "stage_status": "STAGED_READY", "turn_ids": [], "texts": [], "mintrec": [],
                "speakers": [], "first_ts": turn.timestamp, "last_ts": turn.timestamp,
                "staging_priority": self.BASE_STAGING_PRIORITY, "transition_chains": []})
            for t_id, sp, fine, text in ((tr["from_turn"], tr["from_speaker"], tr["from_fine"], src[3] if src else ""),
                                         (tr["to_turn"], tr["to_speaker"], tr["to_fine"], turn.text)):
                if t_id not in pkt["turn_ids"]:
                    pkt["turn_ids"].append(t_id)
                    pkt["texts"].append(text)
                    pkt["mintrec"].append({"turn_id": t_id, "coarse": _FINE_TO_COARSE.get(fine, "UNK"), "fine": fine,
                                          "modality_cue": "text"})
                if sp not in pkt["speakers"]:
                    pkt["speakers"].append(sp)
            pkt["last_ts"] = turn.timestamp
            hit = [key]
        for k in hit:
            p = self.staged_packets[k]
            p["staging_priority"] = p.get("staging_priority", self.BASE_STAGING_PRIORITY) + w
            p.setdefault("transition_chains", []).append(tr)
        return hit

    def _has_domain_cue(self, text: str) -> bool:
        return any(k in text for k in self.DOMAIN_CUES)

    def ingest_ambient_turn(self, turn: UtteranceTurn) -> dict:
        self.ingested_count += 1
        text = turn.text or ""
        mt = classify_mintrec_taxonomy(text, turn.rms_energy, turn.zcr_pitch)
        self.mintrec_labels.append(mt)
        topic_noise = ((any(k in text for k in self.engine.CHITCHAT_NOISE_KW) and "小照" not in text)
                       or turn.thread_id in NOISE_THREADS)
        is_chitchat = topic_noise or is_agreement_only(text)
        # agreement-only turns still close a chain (Arrange -> Agree); topic noise never does
        trs = self._track_transition(turn, mt["fine"], topic_noise)
        intents = [] if is_chitchat else self.engine.detect_domain_intents(text)
        for i in intents:
            self.domain_intent_counts[i] = self.domain_intent_counts.get(i, 0) + 1

        def _done(res: dict) -> dict:
            if trs:
                res["mintrec_transition"] = trs[0]
                res["mintrec_transitions"] = trs
                res["boosted_keys"] = [k for tr in trs for k in self._boost_for_transition(tr, turn)]
            return res

        if is_chitchat or (mt["fine"] in self.NOISE_FINE and not intents):
            self.ignored_count += 1
            return _done({"turn_id": turn.turn_id, "status": "IGNORED_NOISE", "mintrec": mt, "domain_intents": []})

        keys = list(intents)
        if not keys and (mt["fine"] in self.GOAL_SAFETY_FINE
                         or (mt["fine"] in self.GOAL_DOMAIN_FINE and self._has_domain_cue(text))):
            keys = [f"MINTREC_{mt['fine'].upper()}"]
        if not keys:
            return _done({"turn_id": turn.turn_id, "status": "OBSERVED", "mintrec": mt, "domain_intents": []})

        status = "AUTO_FIRED_P0" if "P0_EMERGENCY_ALERT" in intents else "STAGED_READY"
        for key in keys:
            pkt = self.staged_packets.get(key)
            if pkt is None:
                pkt = self.staged_packets[key] = {
                    "intent_key": key, "stage_status": status, "turn_ids": [], "texts": [],
                    "mintrec": [], "speakers": [], "first_ts": turn.timestamp, "last_ts": turn.timestamp,
                    "staging_priority": self.BASE_STAGING_PRIORITY, "transition_chains": [],
                }
            pkt["turn_ids"].append(turn.turn_id)
            pkt["texts"].append(text)
            pkt["mintrec"].append({"turn_id": turn.turn_id, **mt})
            if turn.speaker_id not in pkt["speakers"]:
                pkt["speakers"].append(turn.speaker_id)
            pkt["last_ts"] = turn.timestamp
            if status == "AUTO_FIRED_P0":
                pkt["stage_status"] = status
        if status == "AUTO_FIRED_P0":
            # P0 never waits for a query: emit immediately (packet stays visible in staged_packets)
            self.auto_fired.append(self._build_payload(keys, turn.timestamp, "", "P0_AUTO_FIRE", None, pop=False))
        return _done({"turn_id": turn.turn_id, "status": status, "mintrec": mt, "domain_intents": intents,
                      "staged_keys": keys})

    def _build_payload(self, keys: List[str], ts: float, query_text: str, mode: str, t0: Optional[float],
                       pop: bool = True) -> dict:
        pkts = [self.staged_packets.pop(k) if pop else self.staged_packets[k] for k in keys]
        turn_ids = sorted({tid for p in pkts for tid in p["turn_ids"]})
        intents: List[str] = []
        for k in keys:
            if not k.startswith("MINTREC_") and k not in intents:
                intents.append(k)
        intents += [i for i in self.engine.detect_domain_intents(query_text) if i not in intents]
        route, safety, pi_state = self.engine.resolve_route(intents)
        mintrec = [m for p in pkts for m in p["mintrec"]]
        chains = [c for p in pkts for c in p.get("transition_chains", [])]
        if ((any(m["fine"] == "Ask_for_help" for m in mintrec)
             or any(c["type"] == "CRISIS_ESCALATION_CHAIN" and c["live_hazard"] for c in chains))
                and self.SAFETY_RANK[safety] < 1):
            safety = "HIGH_P1"
        self._fire_seq += 1
        payload = {
            "client_event_id": str(uuid.uuid5(uuid.NAMESPACE_DNS,
                                              f"{self.session_id}_{mode}_{self._fire_seq}_{'_'.join(map(str, turn_ids))}")),
            "session_id": self.session_id,
            "case_id": self.case_id,
            "run_env": "RaspberryPi5_ARM64" if ON_PI else "Mac_Mock_Env",
            "pi_state": pi_state,
            "safety_level": safety,
            "route_target": route,
            "detected_intents": intents or ["GENERAL_CARE_LOG"],
            "selected_turn_indices": turn_ids,
            "dst_slots": self.engine.dst_slots_for(intents),
            "mintrec_summary": {**summarize_mintrec(mintrec), "per_turn": mintrec},
            "staged_keys_fired": keys,
            "staging_priority": max((p.get("staging_priority", self.BASE_STAGING_PRIORITY) for p in pkts), default=0),
            "mintrec_transitions": chains,
            "rag_intervention_required": any(c["rag_intervention"] for c in chains),
            "context_texts": [tx for p in pkts for tx in p["texts"]],
            "query_text": query_text,
            "query_timestamp": ts,
            "dispatch_mode": mode,
        }
        payload["dispatch_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3) if t0 is not None else 0.0
        return payload

    def fire_on_caregiver_query(self, query_text: str, timestamp: float) -> dict:
        """Pop + merge staged packets matching the query (by intent key), or all live ones for
        referential/unspecific queries. Cost: O(len(query) + #staged keys)."""
        t0 = time.perf_counter()
        # drop stale packets (bounded by number of keys, not history length)
        for k in [k for k, p in self.staged_packets.items() if timestamp - p["last_ts"] > self.ttl_sec]:
            del self.staged_packets[k]
        q_intents = self.engine.detect_domain_intents(query_text)
        keys = [k for k in q_intents if k in self.staged_packets]
        if not keys or any(r in query_text for r in self.REFERENTIAL):
            # matched keys first, then the rest by transition-boosted staging priority
            rest = sorted((k for k in self.staged_packets if k not in keys),
                          key=lambda k: -self.staged_packets[k].get("staging_priority", self.BASE_STAGING_PRIORITY))
            keys += rest
        mode = "PREFETCH_INSTANT_FIRE" if keys else "PREFETCH_MISS_LIVE_ONLY"
        return self._build_payload(keys, timestamp, query_text, mode, t0)

    def stats(self) -> dict:
        return {"ingested": self.ingested_count, "ignored_noise": self.ignored_count,
                "staged_count": len(self.staged_packets), "auto_fired_p0": len(self.auto_fired),
                "staged_keys": list(self.staged_packets.keys()),
                "transition_chains": len(self.transitions),
                "staging_priorities": {k: p.get("staging_priority", self.BASE_STAGING_PRIORITY)
                                       for k, p in self.staged_packets.items()}}


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

def _gift_bicl_unit_tests() -> None:
    """v7.5 unit tests: Bi-CL thread consolidation + GIFT 4-edge topology / replied-by amendment / roles."""
    U = UtteranceTurn
    # --- Bi-CL: micro-threads inside a dominant session flock to its prototype (iv2 45:45 shape)
    frag = [U(1, 0, "speaker_2", "unknown", "不是現在學生用的是每個月只有200", thread_id="T_TOPIC_11"),
            U(2, 6, "speaker_2", "unknown", "現在就做實驗", thread_id="T_TOPIC_12"),
            U(3, 8, "speaker_3", "unknown", "那如果說後續真的緊急按鍵大的時候就不需要", thread_id="T_TOPIC_10"),
            U(4, 15, "speaker_2", "unknown", "我覺得研究一下緊急按鍵說", thread_id="T_TOPIC_10"),
            U(5, 21, "speaker_3", "unknown", "怎麼讓緊急事件第一時間通知居督", thread_id="T_TOPIC_10"),
            U(6, 24, "speaker_3", "unknown", "免費了25年他發揮最大的效益這個", thread_id="T_TOPIC_13"),
            U(7, 47, "speaker_3", "unknown", "如果說這個緊急按鍵是這個", thread_id="T_TOPIC_10"),
            U(8, 51, "speaker_3", "unknown", "方式是開發一個主動每天推播的緊急關懷", thread_id="T_TOPIC_10")]
    refined, bs = bi_cl_disentangle_threads(frag)
    assert bs["raw_greedy_threads"] == 4 and bs["estimated_k"] == 1 <= bs["k_upper_bound"], bs
    assert set(refined.values()) == {"T_TOPIC_10"}, refined
    assert bs["prototype_cohesion_score"] > bs["raw_cohesion_score"], bs
    # --- Bi-CL: genuinely interleaved topics survive (no lexical overlap, within the K budget)
    inter = [U(1, 0, "speaker_1", "unknown", "今天下午的復健課程幾點開始呢", thread_id="T_A"),
             U(2, 3, "speaker_2", "unknown", "復健課程是下午三點開始喔", thread_id="T_A"),
             U(3, 6, "speaker_3", "unknown", "那晚餐大家想要吃什麼", thread_id="T_B"),
             U(4, 9, "speaker_1", "unknown", "復健課程三點開始我會準時到", thread_id="T_A"),
             U(5, 13, "speaker_3", "unknown", "晚餐想吃稀飯和青菜就好", thread_id="T_B")]
    refined, bs = bi_cl_disentangle_threads(inter)
    assert bs["estimated_k"] == 2 and refined[3] == refined[5] == "T_B", (refined, bs)
    s = DialogueSessionState("S_BICL", "C")
    s.turns = [U(t.turn_id, t.timestamp, t.speaker_id, "unknown", t.text, thread_id=t.thread_id) for t in frag]
    r = XiaoZhaoMPCEngine().analyze_session(s)
    bsum = r["bi_cl_disentangle_summary"]
    assert (bsum["raw_greedy_threads"], bsum["refined_prototype_threads"]) == (4, 1), bsum
    assert {d["source_thread_id"] for d in r["turns_detail"]} == {"T_TOPIC_10"}, r["turns_detail"]

    # --- GIFT: 4-edge topology, phi, explicit matrix
    g_turns = [U(1, 0, "speaker_1", "unknown", "我昨天去那邊遇到一個阿伯一直拉我"),
               U(2, 5, "speaker_2", "unknown", "我建議你先安撫他然後回報居督"),
               U(3, 9, "speaker_1", "unknown", "那個狀況我不知道"),
               U(4, 14, "speaker_2", "unknown", "下次我們先做一個流程"),
               U(5, 20, "speaker_1", "unknown", "對"),                               # backchannel: not a node
               U(6, 25, "speaker_3", "unknown", "那我們每週都開會檢討"),
               U(7, 30, "speaker_1", "unknown", "不用啦不是這樣")]
    g = build_gift_graph(g_turns)
    assert g["reply_to"] == {2: 1, 3: 2, 4: 3, 6: 4, 7: 6} and g["replied_by"][1] == [2], g
    n = len(g["node_ids"])
    assert gift_edge_counts(g) == {"reply-to": 5, "replied-by": 5, "reply-self": n,
                                   "indirect-reply": n * n - n - 10}, gift_edge_counts(g)
    mat = gift_edge_matrix(g)
    assert mat[0][0] == "reply-self" and mat[1][0] == "reply-to" and mat[0][1] == "replied-by"
    assert mat[0][2] == "indirect-reply" and {e for row in mat for e in row} == set(GIFT_EDGE_TYPES)
    # --- GIFT: replied-by posterior amendment (Advise / Arrange / Oppose children)
    labels = [classify_mintrec_taxonomy(t.text) for t in g_turns]
    am = gift_amend_intents(g_turns, labels, g)
    assert am[1]["gift_fine"] == "Ask_for_help" and am[1]["via"] == "replied-by" and am[1]["evidence_turn"] == 2
    assert am[3]["gift_fine"] == "Inform" and am[3]["evidence_turn"] == 4, am
    assert am[6]["gift_fine"] == "Arrange" and am[6]["evidence_turn"] == 7, am      # opposed -> was a proposal
    s = DialogueSessionState("S_GIFT", "C")
    s.turns = g_turns
    r = XiaoZhaoMPCEngine().analyze_session(s)
    gs = r["gift_graph_summary"]
    assert gs["edge_weights_phi"] == GIFT_EDGE_WEIGHTS and gs["replied_by_amendments"] == 3, gs
    assert gs["edge_counts"]["reply-to"] == 5 and gs["graph_induced_transitions"] >= 2, gs
    assert {"PROBLEM_TO_GUIDANCE", "REPORT_TO_ACTION_PLAN", "REQUEST_TO_REFUSAL"} <= {
        x["type"] for x in r["mintrec_transitions"] if x.get("graph_induced")}, r["mintrec_transitions"]
    # --- GIFT: graph-induced role for a cue-less third participant (iv3 speaker_3 shape)
    mt = []
    for k in range(4):
        b = 3 * k * 10
        mt += [U(3 * k + 1, b, "speaker_1", "unknown", "我們後端的知識庫系統可以幫大家裝綁定碼"),
               U(3 * k + 2, b + 5, "speaker_3", "unknown", "那這個居督可以綁哪一個居服員的個案"),
               U(3 * k + 3, b + 9, "speaker_2", "unknown", "主任跟督導會量身打造綁定的規則")]
    src: Dict[str, str] = {}
    ev: Dict[str, Any] = {}
    roles = infer_session_speaker_roles(mt, sources_out=src, graph_evidence_out=ev)
    assert roles == {"speaker_1": ROLE_RESEARCHER, "speaker_3": ROLE_SUPERVISOR, "speaker_2": ROLE_DIRECTOR}, roles
    assert src["speaker_3"] == GRAPH_INFERRED_SOURCE and ev["speaker_3"]["label"].endswith("(graph_inferred)"), ev


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
    assert all("mintrec_summary" in r for r in results), "analyze_session 應輸出 mintrec_summary"

    # MIntRec bi-modal taxonomy self-test
    assert classify_mintrec_taxonomy("住手！不要摸我")["fine"] == "Prevent"
    assert classify_mintrec_taxonomy("麻煩順便幫我買藥")["fine"] == "Arrange"
    assert classify_mintrec_taxonomy("對，對")["fine"] == "Agree"
    assert classify_mintrec_taxonomy("阿嬤你還好嗎？")["coarse"] == "Express_Emotions_Attitudes"
    assert classify_mintrec_taxonomy("今天出太陽")["fine"] == "UNK"
    _crit = classify_mintrec_taxonomy("你又沒弄好", rms_energy=6000.0, zcr_pitch=0.3)
    assert _crit["fine"] == "Criticize" and _crit["modality_cue"] == "audio", _crit
    assert classify_mintrec_taxonomy("你又沒弄好")["fine"] == "UNK", "無聲學線索不應判 Criticize"
    _d = engine.detect_domain_intents("依長照基本法跟嘉安內規，這個多元問題不在 BA24 裡")
    assert {"LEGAL_REGULATION_QUERY", "JIA_AN_INTERNAL_RULE", "SUPERVISOR_DIVERSE_FEEDBACK"} <= set(_d), _d

    # agreement-only + real-interview LTC intents (interview1 32m / interview2 52m telemetry)
    assert is_agreement_only("對啦" * 50) and is_agreement_only("對，對，好") and is_agreement_only("我懂 我懂")
    assert not is_agreement_only("對啦他一定就是照樣") and not is_agreement_only("是嗎？")
    assert classify_mintrec_taxonomy("對啦對啦對啦對啦對啦對啦")["fine"] == "Agree"
    assert classify_mintrec_taxonomy("謝謝大家，今天就到這裡")["fine"] == "Leave"
    assert classify_mintrec_taxonomy("所以我們回去設計一下個案把督導")["fine"] == "Arrange"
    assert "EMERGENCY_BUTTON_ESCALATION" in engine.detect_domain_intents("照服員端要有緊急鈕的功能")
    assert "CASE_ASSIGNMENT_HIERARCHY" in engine.detect_domain_intents("每一個個案一定是特定的督導，主任端權限最高")
    assert "DEATH_AT_HOME_AFTERCARE_SOP" in engine.detect_domain_intents("先安撫鎮定，問有沒有什麼宗教信仰")
    assert engine.detect_domain_intents("你去案家看到他已經過世了") == ["P0_EMERGENCY_ALERT"]
    _kb = "知識庫系統只要問到關鍵字例如說家屬中風或家屬去世他會把例如說家屬跟去世這兩個字組合起來"
    assert engine.detect_domain_intents(_kb) == ["SUPERVISOR_DIVERSE_FEEDBACK"], engine.detect_domain_intents(_kb)
    assert engine.detect_domain_intents("譬如說長輩跌倒，督導要怎麼處理") != ["P0_EMERGENCY_ALERT"]
    assert engine.detect_domain_intents("例如說知識庫。阿嬤跌倒了快來") == ["P0_EMERGENCY_ALERT"], "clause break ends example"
    assert engine.detect_domain_intents("例如說家屬去世，可是剛剛阿公真的跌倒了") == ["P0_EMERGENCY_ALERT"]
    assert "SERVICE_CODE_QUERY" in engine.detect_domain_intents("GA跟SC的項目我們還沒有拿到")
    assert "WORKPLACE_HARASSMENT_REPORT" in engine.detect_domain_intents("你是有看到被摸大腿")
    assert engine.resolve_route(["SUPERVISOR_DIVERSE_FEEDBACK", "EMERGENCY_BUTTON_ESCALATION"])[0] == \
        "SUPERVISOR_URGENT_CALL_CHANNEL"

    # AmbientIntentPreFetcher self-test: stage during chat, instant fire on caregiver query
    pf = AmbientIntentPreFetcher(engine, session_id="SCEN_AMBIENT", case_id="JIA_AN_006")
    _r1 = pf.ingest_ambient_turn(UtteranceTurn(1, 10.0, "speaker_2", "family", "居服員小姐，待會麻煩順便幫我買兩盒普拿疼止痛藥好嗎？"))
    _r2 = pf.ingest_ambient_turn(UtteranceTurn(2, 45.0, "speaker_1", "elder", "今天中午的排骨便當菜色很鹹欸，電視八點檔又在重播。"))
    _r3 = pf.ingest_ambient_turn(UtteranceTurn(3, 60.0, "speaker_2", "family", "嗯嗯"))
    assert _r1["status"] == "STAGED_READY" and _r1["mintrec"]["fine"] == "Arrange", _r1
    assert _r2["status"] == "IGNORED_NOISE" and _r3["status"] == "IGNORED_NOISE", (_r2, _r3)
    _r3b = pf.ingest_ambient_turn(UtteranceTurn(3, 61.0, "speaker_3", "family", "對啦對啦對啦對啦對啦對啦對啦對啦"))
    assert _r3b["status"] == "IGNORED_NOISE", _r3b
    assert "BA16_MEDICINE_PURCHASE_INQUIRY" in pf.staged_packets
    _fired = pf.fire_on_caregiver_query("小照，家屬剛剛說的那樣可以嗎？", 520.0)
    assert _fired["dispatch_mode"] == "PREFETCH_INSTANT_FIRE", _fired
    assert _fired["selected_turn_indices"] == [1] and "BA16_MEDICINE_PURCHASE_INQUIRY" in _fired["detected_intents"]
    assert _fired["dispatch_latency_ms"] >= 0.0 and not pf.staged_packets, "staged 封包應被 pop"
    _p0 = pf.ingest_ambient_turn(UtteranceTurn(4, 600.0, "speaker_1", "elder", "救命！我跌倒了好痛"))
    assert _p0["status"] == "AUTO_FIRED_P0" and pf.staged_packets["P0_EMERGENCY_ALERT"]["stage_status"] == "AUTO_FIRED_P0"
    assert pf.auto_fired and pf.auto_fired[-1]["safety_level"] == "CRITICAL_P0"

    # ---- Upgrade A: session-level speaker roles (home-care benchmarks unchanged)
    assert results[0]["speaker_roles"] == {"speaker_2": "family", "speaker_1": "elder", "speaker_3": "caregiver"}
    assert results[0]["turns_detail"][0]["who_to_whom_role"] == "family -> caregiver", results[0]["turns_detail"][0]
    assert results[0]["turns_detail"][2]["who_to_whom_role"] == "caregiver -> system_xiaozhao"
    assert results[1]["turns_detail"][0]["who_to_whom_role"] == "elder -> caregiver"
    assert all(set(r["speaker_roles"].values()) <= set(HOMECARE_ROLES) for r in results), "benchmarks stay home-care"
    _meet = DialogueSessionState("SCEN_MEETING_ROLES", "REAL_MEETING", turns=[
        UtteranceTurn(1, 0.0, "speaker_1", "unknown", "居服員的問題是他會很緊張，我們督導在外騎車也要先安撫他"),
        UtteranceTurn(2, 8.0, "speaker_2", "unknown", "那天我去案家遇到狗，褲子整個都濕掉，沒有辦法換衣服"),
        UtteranceTurn(3, 15.0, "speaker_3", "unknown", "我們後端的知識庫系統讓學生們做實驗，那我先做個小結，請主任分配"),
        UtteranceTurn(4, 22.0, "speaker_4", "unknown", "那就照這樣分工"),
        UtteranceTurn(5, 30.0, "speaker_1", "unknown", "督導這邊事後再回報"),
    ])
    _mr = engine.analyze_session(_meet)
    assert _mr["speaker_roles"] == {"speaker_1": ROLE_SUPERVISOR, "speaker_2": ROLE_CAREGIVER_LTC,
                                    "speaker_3": ROLE_RESEARCHER, "speaker_4": ROLE_DIRECTOR}, _mr["speaker_roles"]
    _td = {d["turn_id"]: d for d in _mr["turns_detail"]}
    assert _td[2]["who_to_whom_role"] == f"{ROLE_CAREGIVER_LTC} -> {ROLE_SUPERVISOR}", _td[2]
    assert _td[3]["who_to_whom_role"] == f"{ROLE_RESEARCHER} -> {ROLE_DIRECTOR}", _td[3]   # vocative 請主任
    assert _td[3]["addressee_speaker"] == "speaker_4" and _td[4]["inferred_role"] == ROLE_DIRECTOR
    # pre-filled home-care role survives when only one stray extended cue shows up (not meeting mode)
    assert infer_session_speaker_roles([UtteranceTurn(1, 0.0, "speaker_1", "caregiver", "我去案家了")]) == \
        {"speaker_1": "caregiver"}

    # ---- Upgrade B: cross-speaker MIntRec transition chains (same thread, noise skipped)
    _bt = [UtteranceTurn(1, 0.0, "speaker_1", "unknown", "我好困擾，那邊的狗很多", thread_id="T_A"),
           UtteranceTurn(2, 3.0, "speaker_3", "unknown", "今天便當很鹹", thread_id="T_CHITCHAT"),
           UtteranceTurn(3, 5.0, "speaker_2", "unknown", "建議你繞路走", thread_id="T_A"),
           UtteranceTurn(4, 7.0, "speaker_2", "unknown", "麻煩你明天早點過去", thread_id="T_A"),
           UtteranceTurn(5, 8.0, "speaker_3", "unknown", "對啦", thread_id="T_BACKCHANNEL_NOISE"),
           UtteranceTurn(6, 9.0, "speaker_1", "unknown", "好", thread_id="T_A"),
           UtteranceTurn(7, 20.0, "speaker_2", "unknown", "請你再去一次", thread_id="T_B"),
           UtteranceTurn(8, 22.0, "speaker_1", "unknown", "目前那邊路況", thread_id="T_C"),
           UtteranceTurn(9, 24.0, "speaker_1", "unknown", "不用，我不去", thread_id="T_B"),
           UtteranceTurn(10, 26.0, "speaker_2", "unknown", "我們先設計一個流程", thread_id="T_C"),
           UtteranceTurn(11, 40.0, "speaker_1", "unknown", "救命，他跌倒了怎麼辦", thread_id="T_D"),
           UtteranceTurn(12, 41.0, "speaker_2", "unknown", "我馬上按緊急按鈕通知", thread_id="T_D"),
           UtteranceTurn(13, 50.0, "speaker_1", "unknown", "不要讓細節發散", thread_id="T_E"),
           UtteranceTurn(14, 51.0, "speaker_2", "unknown", "照服員端要有緊急鈕的功能", thread_id="T_E")]
    _tr = detect_mintrec_transitions(_bt)
    _pairs = [(x["type"], x["from_turn"], x["to_turn"]) for x in _tr]
    assert _pairs == [("PROBLEM_TO_GUIDANCE", 1, 3), ("REQUEST_TO_COMPLIANCE_OR_HESITATION", 4, 6),
                      ("REQUEST_TO_REFUSAL", 7, 9), ("REPORT_TO_ACTION_PLAN", 8, 10),
                      ("CRISIS_ESCALATION_CHAIN", 11, 12)], _pairs   # 13->14: moderation Prevent never escalates
    assert _tr[1]["rag_intervention"] and _tr[1]["priority"] == "HIGH_RAG_INTERVENTION"
    assert not detect_mintrec_transitions([_bt[0], UtteranceTurn(2, 1.0, "speaker_1", "unknown", "建議你繞路走",
                                                                 thread_id="T_A")]), "same speaker != transition"
    assert "mintrec_transitions" in _mr and all("mintrec_transitions" in r for r in results)
    # pre-fetcher: Arrange -> (other speaker) Agree completes a chain and boosts the staged BA16 packet
    pf2 = AmbientIntentPreFetcher(engine, session_id="SCEN_CHAIN", case_id="JIA_AN_007")
    pf2.ingest_ambient_turn(UtteranceTurn(1, 10.0, "speaker_2", "family", "麻煩順便幫我買普拿疼"))
    _ack = pf2.ingest_ambient_turn(UtteranceTurn(2, 12.0, "speaker_3", "caregiver", "好"))
    assert _ack["status"] == "IGNORED_NOISE" and _ack["mintrec_transition"]["type"] == \
        "REQUEST_TO_COMPLIANCE_OR_HESITATION", _ack
    assert pf2.staged_packets["BA16_MEDICINE_PURCHASE_INQUIRY"]["staging_priority"] == 3
    pf2.ingest_ambient_turn(UtteranceTurn(3, 14.0, "speaker_1", "elder", "我好困擾，那邊的狗很多"))
    _f2 = pf2.fire_on_caregiver_query("小照，剛剛那樣可以嗎？", 30.0)
    assert _f2["staged_keys_fired"][0] == "BA16_MEDICINE_PURCHASE_INQUIRY" and _f2["rag_intervention_required"], _f2
    assert _f2["selected_turn_indices"][0] == 1 and 2 not in _f2["selected_turn_indices"]
    pf3 = AmbientIntentPreFetcher(engine)   # chain without domain cue stages its own packet
    pf3.ingest_ambient_turn(UtteranceTurn(1, 0.0, "speaker_1", "unknown", "請你等一下過來"))
    pf3.ingest_ambient_turn(UtteranceTurn(2, 1.0, "speaker_2", "unknown", "嗯"))
    assert "MINTREC_CHAIN_REQUEST_TO_COMPLIANCE_OR_HESITATION" in pf3.staged_packets, pf3.stats()

    # ---- Upgrade C: cross-turn slot & context fusion
    _ct = [UtteranceTurn(1, 0.0, "speaker_1", "unknown", "我去案家遇到狗，旁邊還有蜜蜂", thread_id="T_TOPIC_1"),
           UtteranceTurn(2, 5.0, "speaker_2", "unknown", "那要做一個對講機的功能直接打到居督的手機", thread_id="T_TOPIC_1"),
           UtteranceTurn(3, 9.0, "speaker_3", "unknown", "排骨便當", thread_id="T_TOPIC_2"),
           UtteranceTurn(4, 12.0, "speaker_1", "unknown", "對", thread_id="T_BACKCHANNEL_NOISE")]
    _roles = {"speaker_1": ROLE_CAREGIVER_LTC, "speaker_2": ROLE_SUPERVISOR, "speaker_3": "unknown"}
    _fu = aggregate_cross_turn_slots(_ct, _roles, transitions=detect_mintrec_transitions(_ct, speaker_roles=_roles))
    assert _fu["dominant_threads"] == ["T_TOPIC_1"], _fu
    assert [(i["label"], i["raised_by"], i["role"]) for i in _fu["issue_triggers"]] == \
        [("山區野狗/蜜蜂", "speaker_1", ROLE_CAREGIVER_LTC)], _fu["issue_triggers"]
    assert [(r["label"], r["proposed_by"]) for r in _fu["proposed_resolutions"]] == \
        [("對講機功能直連居督手機", "speaker_2")], _fu["proposed_resolutions"]
    assert _fu["participating_roles"] == [ROLE_CAREGIVER_LTC, ROLE_SUPERVISOR]
    assert _fu["consensus_state"] == "CONSENSUS_ACTIONABLE", _fu
    assert aggregate_cross_turn_slots(_ct[:1], _roles)["consensus_state"] == "OPEN_DISCUSSION"
    _crisis = aggregate_cross_turn_slots(_bt[10:12], {}, transitions=detect_mintrec_transitions(_bt[10:12]))
    assert _crisis["consensus_state"] == "IMMEDIATE_ESCALATION", _crisis
    _ref = [UtteranceTurn(1, 0.0, "speaker_1", "unknown", "案家要我代購成藥", thread_id="T1"),
            UtteranceTurn(2, 1.0, "speaker_2", "unknown", "請你用APP回報", thread_id="T1"),
            UtteranceTurn(3, 2.0, "speaker_1", "unknown", "不用啦", thread_id="T1")]
    assert aggregate_cross_turn_slots(_ref, {}, transitions=detect_mintrec_transitions(_ref))["consensus_state"] \
        == "OPEN_DISCUSSION", "trailing refusal blocks consensus"
    assert "cross_turn_fusion" in _mr["dst_slots"] and "cross_turn_fusion" in results[0]["dst_slots"]
    json.dumps(_mr, ensure_ascii=False)   # payload stays JSON-serialisable

    # ---- v2: meta-reference P0 guard (iv1 19:00), live reported speech still fires
    _meta = engine.detect_domain_intents("像剛我們說的暴力性騷，類似要做一個緊急對講機的功能")
    assert "P0_EMERGENCY_ALERT" not in _meta and "EMERGENCY_BUTTON_ESCALATION" in _meta, _meta
    assert engine.resolve_route(_meta)[0] == "SUPERVISOR_URGENT_CALL_CHANNEL"
    assert engine.detect_domain_intents("小照，家屬剛剛說阿公跌倒了") == ["P0_EMERGENCY_ALERT"], "reported live event"
    assert engine.detect_domain_intents("像剛剛說的那樣，阿公跌倒了快來") == ["P0_EMERGENCY_ALERT"], "clause break"
    # ---- v2: transition lookahead (<= 3 substantive turns, interleaved speakers)
    _la = [UtteranceTurn(1, 0.0, "speaker_1", "unknown", "這樣怎麼處理", thread_id="T1"),
           UtteranceTurn(2, 1.0, "speaker_2", "unknown", "今天出太陽", thread_id="T1"),
           UtteranceTurn(3, 2.0, "speaker_3", "unknown", "我覺得先打電話", thread_id="T1")]
    _lt = detect_mintrec_transitions(_la)
    assert [(x["type"], x["from_turn"], x["to_turn"], x["lag"]) for x in _lt] == [("PROBLEM_TO_GUIDANCE", 1, 3, 2)], _lt
    _far = [_la[0]] + [UtteranceTurn(k, float(k), "speaker_2", "unknown", "今天出太陽", thread_id="T1")
                       for k in (2, 3, 4)] + [UtteranceTurn(5, 5.0, "speaker_3", "unknown", "我覺得先打電話", thread_id="T1")]
    assert not detect_mintrec_transitions(_far), "answer beyond the 3-turn window is not a chain"
    # ---- v2: crisis chain liveness (meta discussion never escalates)
    _mc = [UtteranceTurn(1, 0.0, "speaker_1", "unknown", "像剛我們說的暴力性騷", thread_id="T1"),
           UtteranceTurn(2, 1.0, "speaker_2", "unknown", "這個都是第一時間非常緊急的", thread_id="T1")]
    _mct = detect_mintrec_transitions(_mc)
    assert _mct and _mct[0]["type"] == "CRISIS_ESCALATION_CHAIN" and not _mct[0]["live_hazard"], _mct
    assert _mct[0]["priority"] == "PROTOCOL_DESIGN_DISCUSSION"
    assert aggregate_cross_turn_slots(_mc, {}, transitions=_mct)["consensus_state"] != "IMMEDIATE_ESCALATION"
    assert detect_mintrec_transitions(_bt[10:12])[0]["live_hazard"], "救命/跌倒 is live"
    # ---- v2: recording-level role priors fill speakers the slice cannot resolve; slice cues still win
    _src: Dict[str, str] = {}
    _pr = infer_session_speaker_roles(
        [UtteranceTurn(1, 0.0, "speaker_1", "unknown", "今天出太陽"),
         UtteranceTurn(2, 1.0, "speaker_2", "unknown", "我們後端的知識庫系統"),
         UtteranceTurn(3, 2.0, "speaker_3", "unknown", "好")],
        priors={"speaker_1": ROLE_SUPERVISOR, "speaker_2": ROLE_DIRECTOR}, sources_out=_src)
    assert _pr == {"speaker_1": ROLE_SUPERVISOR, "speaker_2": ROLE_RESEARCHER, "speaker_3": "unknown"}, _pr
    assert _src == {"speaker_1": "recording_prior", "speaker_2": "session_cues", "speaker_3": "no_evidence"}, _src
    _ps = DialogueSessionState("S_PRIOR", "C", turns=[UtteranceTurn(1, 0.0, "speaker_1", "unknown", "今天出太陽")],
                               role_priors={"speaker_1": ROLE_CAREGIVER_LTC})
    assert engine.analyze_session(_ps)["speaker_roles"] == {"speaker_1": ROLE_CAREGIVER_LTC}
    assert results[0]["speaker_role_sources"]["speaker_2"] == "prefilled"
    # ---- v2: streaming lookahead in the pre-fetcher
    pf4 = AmbientIntentPreFetcher(engine)
    pf4.ingest_ambient_turn(UtteranceTurn(1, 0.0, "speaker_1", "unknown", "請你等一下過來"))
    pf4.ingest_ambient_turn(UtteranceTurn(2, 1.0, "speaker_1", "unknown", "今天出太陽"))
    _l4 = pf4.ingest_ambient_turn(UtteranceTurn(3, 2.0, "speaker_2", "unknown", "嗯"))
    assert _l4["mintrec_transition"]["from_turn"] == 1 and _l4["mintrec_transition"]["lag"] == 2, _l4
    assert [x["type"] for x in pf4.transitions] == ["REQUEST_TO_COMPLIANCE_OR_HESITATION"], pf4.transitions

    _gift_bicl_unit_tests()
    if "--quiet-test" in sys.argv:
        print(f"MPC_CORE_OK (env={'PI' if ON_PI else 'MAC_MOCK'}, scenarios={len(results)})")
        sys.exit(0)

    for r in results:
        print("=" * 72)
        print(f"[{r['session_id']}] 環境: {r['run_env']} | 狀態機: {r['pi_state']}")
        print(f"  偵測意圖: {r['detected_intents']} | 路由: {r['route_target']}")
        print(f"  保留對話輪次 (selected_idx): {r['selected_turn_indices']} | 剔除雜訊輪次: {r['filtered_out_turns']}")
        fu = r["dst_slots"].get("cross_turn_fusion", {})
        print(f"  DST slots: { {k: v for k, v in r['dst_slots'].items() if k != 'cross_turn_fusion'} }")
        print(f"  speaker_roles: {r['speaker_roles']} | consensus: {fu.get('consensus_state')}")
        print(f"  transitions: {[(x['type'], x['from_turn'], x['to_turn']) for x in r['mintrec_transitions']]}")
        for td in r["turns_detail"]:
            print(f"    - Turn {td['turn_id']} [{td['speaker_id']}: {td['who_to_whom_role']}] ({td['thread_id']}): {td['text']}")
