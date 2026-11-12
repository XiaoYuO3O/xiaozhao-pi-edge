import cmath
import itertools
import json
import math
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
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
    end_time: float = 0.0    # segment end (s); 0 = unknown (next turn's start is used)
    voiceprint: Optional[List[float]] = None             # v7.7 session-local 24-d turn voiceprint x_i
    voiceprint_subvectors: Optional[List[List[float]]] = None   # > 6 s turns: Must-Linked sub-window vectors
    voiceprint_source: str = ""                          # wav_75pct_anchor | proxy_acoustic | proxy_text
    near_field_score: float = -1.0                       # 0..1 headset / near-field likelihood (-1 = unknown)


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
    # iv2 49:15: "參數的問題…不合理…資料搜尋" + "不要讓這麼多的細節可以發散出來" (before the meeting 小結)
    ("搜尋參數不合理/會議細節發散", ("參數的問題", "參數不合理", "細節可以發散", "細節發散")),
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


def trailing_refusal(transitions: List[Dict], resolutions: List[Dict]) -> bool:
    """A REQUEST_TO_REFUSAL chain blocks consensus only while it is unanswered: no resolution is proposed after
    the refusal turn (to_turn >= the last resolution turn). A later resolution answers the objection."""
    last_res = max((r["turn_id"] for r in resolutions), default=0)
    return any(x["type"] == "REQUEST_TO_REFUSAL" and x["to_turn"] >= last_res for x in transitions)


def aggregate_cross_turn_slots(turns: List[UtteranceTurn], speaker_roles: Dict[str, str],
                               thread_of: Optional[Dict[int, str]] = None,
                               transitions: Optional[List[Dict]] = None,
                               safety_level: str = "NORMAL",
                               gsn: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Fuse multi-speaker contributions in the dominant substantive thread(s) into one DST slot.

    With `gsn` (gsn_propagate_session output) the fusion reads the 3-iter BIF states: issues raised in
    non-dominant threads that reach a core turn via E/U flow (weight >= GSN_IMPORT_MIN) are imported, and every
    resolution lists the issues whose features reached its turn (`gsn_linked_issues`)."""
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
    if gsn:
        issues = sorted(issues + gsn_imported_issues(gsn, core, issues, turns, speaker_roles, thread_of),
                        key=lambda d: d["turn_id"])
    first_issue = issues[0]["turn_id"] if issues else 0   # resolutions must follow the first raised issue
    resolutions = _catalog_hits(core, RESOLUTION_CATALOG, speaker_roles, "proposed_by", thread_of, first_issue)
    if gsn:
        for r in resolutions:
            r["gsn_linked_issues"] = gsn_linked_features(gsn, r["turn_id"], "issue:")
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
    elif resolutions and len(speakers) >= 2 and not trailing_refusal(dom_tr, resolutions):
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


# ------------- Upgrade F: GSN utterance-level graph-structured encoder (Hu et al., IJCAI 2019, Algorithm 1)
GSN_ALPHA = 0.25          # SQH floor (GSN Eq. 4)
GSN_ITERATIONS = 3        # N backward + N forward BIF iterations
GSN_IMPORT_MIN = 0.5      # min propagated weight for an off-thread issue to be imported into the fusion slot
GSNFeature = Tuple[float, int, int]   # (weight in [0, 1], origin turn id, hops from origin)


def gsn_sqh(norm: float, alpha: float = GSN_ALPHA) -> float:
    """GSN Eq. 4 squashing coefficient: (alpha + ||d||) / (1 + ||d||), in [alpha, 1) for ||d|| >= 0."""
    return (alpha + norm) / (1.0 + norm)


def gsn_initial_state(turn: UtteranceTurn, fine: str = "",
                      domain_fn: Optional[Callable[[str], List[str]]] = None) -> Dict[str, float]:
    """s_i^0: sparse binary features for domain intents, MIntRec fine intent, issue/resolution slots, live risk."""
    txt = _norm(turn.text)
    s: Dict[str, float] = {f"intent:{d}": 1.0 for d in (domain_fn(turn.text) if domain_fn else [])}
    if fine:
        s[f"mintrec:{fine}"] = 1.0
    for prefix, catalog in (("issue:", ISSUE_TRIGGER_CATALOG), ("res:", RESOLUTION_CATALOG)):
        s.update({prefix + label: 1.0 for label, cues in catalog if any(c in txt for c in cues)})
    if crisis_is_live(turn.text):
        s["risk:live_crisis"] = 1.0
    return s


def gsn_propagate_session(turns: List[UtteranceTurn], graph: Dict[str, Any],
                          initial: Optional[Dict[int, Dict[str, float]]] = None,
                          iterations: int = GSN_ITERATIONS, alpha: float = GSN_ALPHA,
                          speaker_weight: Optional[Callable[[int, int], float]] = None) -> Dict[str, Any]:
    """GSN UG-E (Algorithm 1) over the GIFT graph's substantive nodes.

    E: e_{i,j}=1 (i<j) when turn j replies to turn i (GIFT reply-to). U: u_{i,j}=1 (i<j) for chronologically
    consecutive turns of the same speaker (the O(N) chain form of GSN's speaker matrix). Each update aggregates
    the mean neighbour state dE_i / dU_i, gates it with eta_i = SQH(||dE_i||) / lambda_i = SQH(||dU_i||) and
    max-fuses it into s_i (a training-free stand-in for GSN's GRU fusion, so weights stay in [0, 1] and decay
    per hop). N backward iterations (E^T, U^T: children -> parents) are followed by N forward ones (E, U:
    parents -> children), so a child's content reaches its parallel siblings and descendants up to N hops.
    speaker_weight(i, j) (Upgrade G) turns U into a weighted adjacency: u_{i,j} = the acoustic-lexical same-speaker
    confidence of the chained pair instead of a hard 1, scaling dU_i (and so lambda_i). None keeps U binary."""
    ids = list(graph["node_ids"])
    idset = set(ids)
    speaker_of = graph["speaker_of"]
    parent = {c: p for c, p in graph["reply_to"].items() if c in idset and p in idset}
    children: Dict[int, List[int]] = {i: [] for i in ids}
    for c, p in parent.items():
        children[p].append(c)
    prev_same: Dict[int, int] = {}
    next_same: Dict[int, int] = {}
    last_of: Dict[str, int] = {}
    for i in ids:                                  # node_ids are chronological
        sp = speaker_of[i]
        if sp in last_of:
            prev_same[i], next_same[last_of[sp]] = last_of[sp], i
        last_of[sp] = i
    by_id = {t.turn_id: t for t in turns}
    initial = initial if initial is not None else {i: gsn_initial_state(by_id[i]) for i in ids}
    states: Dict[int, Dict[str, GSNFeature]] = {i: {f: (w, i, 0) for f, w in initial.get(i, {}).items()}
                                                for i in ids}

    def delta(nbrs: List[int], snap: Dict[int, Dict[str, GSNFeature]],
              wts: Optional[List[float]] = None) -> Tuple[Dict[str, GSNFeature], float]:
        acc: Dict[str, List] = {}
        for k, n in enumerate(nbrs):
            c = wts[k] if wts else 1.0
            for f, (w0, org, hop) in snap[n].items():
                w = c * w0
                a = acc.setdefault(f, [0.0, w, org, hop])
                a[0] += w
                if w > a[1]:
                    a[1:] = [w, org, hop]
        d = {f: (a[0] / len(nbrs), a[2], a[3] + 1) for f, a in acc.items()}
        return d, sum(v[0] ** 2 for v in d.values()) ** 0.5

    etas: List[float] = []
    lambdas: List[float] = []
    for direction in ("backward",) * iterations + ("forward",) * iterations:
        snap = {i: dict(s) for i, s in states.items()}
        for i in ids:
            if direction == "backward":           # GSN Eq. 12: E^T / U^T
                e_nb, u_nb = children[i], ([next_same[i]] if i in next_same else [])
            else:                                 # GSN Eq. 9-10: E / U
                e_nb, u_nb = ([parent[i]] if i in parent else []), ([prev_same[i]] if i in prev_same else [])
            for nbrs, gates, wfn in ((e_nb, etas, None), (u_nb, lambdas, speaker_weight)):
                if not nbrs:
                    continue
                d, norm = delta(nbrs, snap, [wfn(i, nb) for nb in nbrs] if wfn else None)
                gate = gsn_sqh(norm, alpha)
                gates.append(gate)
                for f, (w, org, hop) in d.items():
                    if org != i and gate * w > states[i].get(f, (0.0,))[0]:
                        states[i][f] = (round(gate * w, 4), org, hop)

    siblings = {p: sorted(cs) for p, cs in children.items() if len(cs) >= 2}
    fwd = {i: children[i] + ([next_same[i]] if i in next_same else []) for i in ids}
    multi_hop: List[Tuple[int, int, int]] = []
    for src in ids:                                # shortest-path BFS in E u U, depth <= iterations
        dist = {src: 0}
        frontier = [src]
        for depth in range(1, iterations + 1):
            frontier = [n for f in frontier for n in fwd[f] if n not in dist]
            for n in frontier:
                dist[n] = depth
            if depth >= 2:
                multi_hop.extend((src, n, depth) for n in sorted(set(frontier)))
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else 0.0   # noqa: E731
    u_conf = [speaker_weight(p, c) if speaker_weight else 1.0 for c, p in prev_same.items()]
    return {
        "node_ids": ids,
        "E": sorted((p, c) for c, p in parent.items()),
        "U": sorted((p, c) for c, p in prev_same.items()),
        "parallel_siblings": siblings,
        "multi_hop_paths": multi_hop,
        "states": states,
        "summary": {
            "iterations": iterations, "alpha": alpha, "schedule": "backward_then_forward",
            "reply_edges_E": len(parent), "speaker_edges_U": len(prev_same),
            "parallel_sibling_branches": len(siblings),
            "multi_hop_paths_3iter": len(multi_hop),
            "mean_eta_reply_weight": mean(etas), "mean_lambda_speaker_weight": mean(lambdas),
            "speaker_edge_weighting": "voiceprint_e2cp" if speaker_weight else "hard_0_1",
            "mean_u_speaker_confidence": mean(u_conf) if u_conf else 1.0,
        },
    }


def gsn_linked_features(gsn: Dict[str, Any], turn_id: int, prefix: str) -> List[Dict[str, Any]]:
    """Features with `prefix` in turn_id's propagated state (own or flowed in), strongest first."""
    st = gsn["states"].get(turn_id, {})
    return sorted(({"label": f[len(prefix):], "origin_turn": org, "weight": w, "hops": hop}
                   for f, (w, org, hop) in st.items() if f.startswith(prefix)),
                  key=lambda d: (-d["weight"], d["origin_turn"]))


def gsn_imported_issues(gsn: Dict[str, Any], core: List[UtteranceTurn], have: List[Dict],
                        turns: List[UtteranceTurn], roles: Dict[str, str],
                        thread_of: Dict[int, str]) -> List[Dict]:
    """Issues raised outside the dominant thread(s) whose features reached a core turn via E/U flow."""
    core_ids = {t.turn_id for t in core}
    known = {d["label"] for d in have}
    by_id = {t.turn_id: t for t in turns}
    best: Dict[str, Tuple[float, int, int, int]] = {}
    for tid in core_ids:
        for f, (w, org, hop) in gsn["states"].get(tid, {}).items():
            lab = f[len("issue:"):]
            if (f.startswith("issue:") and org not in core_ids and w >= GSN_IMPORT_MIN and lab not in known
                    and w > best.get(lab, (0.0,))[0]):
                best[lab] = (w, org, hop, tid)
    cues = dict(ISSUE_TRIGGER_CATALOG)
    out = []
    for lab, (w, org, hop, reached) in best.items():
        t = by_id[org]
        sp = t.speaker_id
        out.append({"label": lab, "raised_by": sp, "role": roles.get(sp, "unknown"), "turn_id": org,
                    "thread_id": thread_of.get(org, t.thread_id),
                    "evidence": next((c for c in cues[lab] if c in _norm(t.text)), ""), "mentions": 1,
                    "contributors": [sp], "via": "gsn_flow", "gsn_weight": w, "gsn_hops": hop,
                    "reached_turn": reached})
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
                              keyword_fn=None, speaker_prob: Optional[Callable[[int, int], float]] = None,
                              ) -> Tuple[Dict[int, str], Dict[str, Any]]:
    """Refine greedy preceding-only threads with Bi-CL's two levels: utterance vectors v_i vs. session
    prototypes p_k = mean(v_j, j in s_k), unsupervised EM (E: leave-one-out nearest prototype + local
    same-speaker continuity, M: recompute p_k) with K-estimation (merge prototypes while
    K > max(2, N_substantive // 6) or cos(p_a, p_b) >= BICL_MERGE_SIM). Micro-threads have no prototype
    besides themselves, so their turns flock to the dominant session. Noise turns keep their thread.
    -> (refined thread_of, bi_cl_disentangle_summary). Merged sessions keep the larger thread's id.
    speaker_prob(turn_i, turn_j) (Upgrade G voiceprint + E2CP) replaces the hard s(i) == s(j) continuity
    indicator with a continuous same-speaker confidence; None keeps the diarizer label."""
    thread_of = dict(thread_of or {t.turn_id: t.thread_id for t in turns})
    nodes = [t for t in turns if thread_of.get(t.turn_id, t.thread_id) not in NOISE_THREADS
             and not is_agreement_only(t.text)]
    raw = [thread_of.get(t.turn_id, t.thread_id) for t in nodes]
    n = len(nodes)
    k_max = max(2, n // BICL_TURNS_PER_SESSION)
    summary: Dict[str, Any] = {"substantive_turns": n, "k_upper_bound": k_max,
                               "raw_greedy_threads": len(set(raw)), "raw_thread_ids": sorted(set(raw)),
                               "speaker_continuity_weighting": "voiceprint_e2cp" if speaker_prob else "diarizer_label"}
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

    def same_speaker(i: int, j: int) -> float:
        if speaker_prob is not None:
            return speaker_prob(nodes[i].turn_id, nodes[j].turn_id)
        return 1.0 if nodes[j].speaker_id == nodes[i].speaker_id else 0.0

    def continuity(i: int, lab: str) -> float:
        return sum(BICL_SPEAKER_CONT_BONUS * same_speaker(i, j) for j in (i - 1, i + 1)
                   if 0 <= j < n and assign[j] == lab)

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


# ---------- Upgrade G (v7.7): acoustic-lexical turn voiceprints + E2CP constrained diarization (stdlib only)
#  - ECAPA-TDNN (Desplanques et al., Interspeech 2020): SE global channel rescaling, then channel-dependent
#    attentive statistics pooling (weighted mean mu~_c, weighted std sigma~_c). Training-free analogue: the
#    excitation is a per-channel reliability gate and the attention logits are energy / voicing / 75%-position /
#    per-channel outlier terms instead of learned weights.
#  - Turn-to-Diarize (Xia et al., ICASSP 2022): one embedding per turn taken around its 75% position, turns > 6 s
#    split into Must-Linked pieces, E2CP constraint propagation (alpha = 0.4), p-percentile refinement, eigengap K.
#  - Park & Georgiou (USC): lexical + acoustic cues; permutation-invariant (flip-aligned) sliding-window vote.
#  - M2MeT / AliMeeting, AISHELL-4: 0.25 s boundary collar; near-field (headset) vs far-field profiling.
# Privacy: vectors are session-local evidence for grouping the turns of THIS session. There is no enrollment DB,
# nothing is matched against stored identities, and analyze_session() never puts raw vectors in its payload.
VP_COLLAR_SEC = 0.25
VP_COLLAR_MIN_TURN_SEC = 0.8      # collar only when >= 0.3 s of core speech remains
VP_WINDOW_FRAC = (0.50, 0.90)     # analysis window of each (collar-trimmed) piece; holds its 75% anchor
VP_ANCHOR_FRAC = 0.75
VP_LONG_TURN_SEC = 6.0
VP_MIN_WINDOW_SEC = 0.10          # shorter windows fall back to the whole piece
VP_FRAME_SEC, VP_HOP_SEC = 0.025, 0.010
VP_MAX_FRAMES = 96                # per-window frame budget (uniform stride): pure-Python DSP stays ~0.1 s / window
VP_DSP_RATE = 8000                # frames are block-averaged to ~8 kHz: 25 ms = 200 samples -> 256-pt FFT
VP_NFFT = 256
VP_N_MEL, VP_N_CEPS = 20, 13
VP_MEL_HZ = (60.0, 3800.0)
VP_BANDS_HZ = ((60.0, 300.0), (300.0, 1000.0), (1000.0, 2000.0), (2000.0, 4000.0))
VP_F0_HZ = (70.0, 400.0)
VP_SILENCE_DB = 30.0              # frames this far below the window's loudest frame are silence
VP_VOICED_CEPS = 0.12             # cepstral pitch peak above this -> voiced frame
VP_DIM = 24
# frame channels: c0..c12, 4 band-energy fractions, centroid (kHz), flatness, ZCR, log2(F0/150 Hz), voicing
VP_CH_BAND, VP_CH_CENTROID, VP_CH_FLAT, VP_CH_ZCR, VP_CH_LOGF0, VP_CH_VOICING = 13, 17, 18, 19, 20, 21
# fixed population priors for channel standardisation: per-channel mean / std over the 5.9k active frames of the
# 43 Train turns (first 50 % of each recording's labelled hotspot turns) of src/train_val_test_eval.py (pooled
# over all speakers; unsupervised, no labels, no per-speaker enrollment; Val / Test turns never enter the fit)
VP_PRIOR_MEAN = [65.68, 0.645, -0.137, -0.977, -1.277, -1.074, -0.222, -0.468, -0.09, -0.142, -0.308, -0.321,
                 -0.16, 0.114, 0.448, 0.208, 0.246, 1.241, 0.224, 0.096, 0.341, 0.143]
VP_PRIOR_STD = [15.95, 2.814, 2.824, 2.244, 1.938, 1.355, 1.164, 1.173, 0.974, 0.935, 0.882, 0.79, 0.763, 0.132,
                0.24, 0.16, 0.237, 0.54, 0.133, 0.093, 0.489, 0.08]
VP_LOGSIG_PRIOR = [-0.498, -0.459, -0.438, -0.398, -0.93]   # mean log sigma~ of c1..c4, log2 F0 (same turns)
# x_i layout (24): mu~ c1..c12 | log sigma~ c1..c4 | mu~ / log sigma~ log2 F0 | mu~ ZCR, centroid, low band,
# top band, flatness | mu~ c0 (level, a distance cue). Per-dimension weights (applied before L2 normalisation)
# = clip(F_d / median F, 0.3, 2.0), F_d = adjacent-speaker / within-turn-piece variance ratio on the Train turns;
# the linear family won on Val d' (0.715 vs sqrt 0.660 / uniform 0.518). Held-out Test d' 0.341.
VP_DIM_WEIGHTS = [1.6, 0.9, 1.1, 1.4, 1.9, 0.9, 1.4, 1.1, 0.8, 0.7, 0.7, 0.6, 0.9, 0.7, 1.0, 0.8, 1.5, 0.9, 2.0,
                  2.0, 0.8, 1.9, 0.6, 1.0]
VP_SOURCE_WAV, VP_SOURCE_PROXY_AC, VP_SOURCE_PROXY_TEXT = "wav_75pct_anchor", "proxy_acoustic", "proxy_text"
# blend weight c of the voiceprint affinity against the diarizer label in the same-speaker probability
VP_CONFIDENCE = {VP_SOURCE_WAV: 0.5, VP_SOURCE_PROXY_AC: 0.25, VP_SOURCE_PROXY_TEXT: 0.0}
VP_ZH_CHARS_PER_SEC = 4.0         # proxy speech-rate / unknown-duration prior

_VP_FFT_PLANS: Dict[int, Tuple[List[int], List[complex]]] = {}
_VP_PLANS: Dict[int, Dict[str, Any]] = {}


def _vp_fft(x: List[complex]) -> List[complex]:
    """Iterative radix-2 FFT (len(x) a power of two)."""
    n = len(x)
    plan = _VP_FFT_PLANS.get(n)
    if plan is None:
        bits = n.bit_length() - 1
        plan = _VP_FFT_PLANS[n] = ([int(format(i, f"0{bits}b")[::-1], 2) for i in range(n)],
                                   [cmath.exp(-2j * math.pi * k / n) for k in range(n // 2)])
    rev, tw = plan
    a = [x[i] for i in rev]
    size = 2
    while size <= n:
        half, step = size >> 1, n // size
        for s in range(0, n, size):
            for k in range(half):
                j = s + k
                t = tw[k * step] * a[j + half]
                a[j + half] = a[j] - t
                a[j] = a[j] + t
        size <<= 1
    return a


def _vp_plan(sample_rate: int) -> Dict[str, Any]:
    plan = _VP_PLANS.get(sample_rate)
    if plan is not None:
        return plan
    r = max(1, int(round(sample_rate / VP_DSP_RATE)))
    rate = sample_rate / r
    flen = max(r * 8, int(round(VP_FRAME_SEC * sample_rate)))
    m = min(VP_NFFT, flen // r)
    half = VP_NFFT // 2
    binw = rate / VP_NFFT
    kb = lambda f: min(half, max(0, int(round(f / binw))))   # noqa: E731
    mel = lambda f: 2595.0 * math.log10(1.0 + f / 700.0)       # noqa: E731
    lo_hz, hi_hz = VP_MEL_HZ[0], min(VP_MEL_HZ[1], 0.475 * rate)
    pts = [700.0 * (10 ** ((mel(lo_hz) + (mel(hi_hz) - mel(lo_hz)) * i / (VP_N_MEL + 1)) / 2595.0) - 1.0)
           for i in range(VP_N_MEL + 2)]
    bank = []
    for j in range(VP_N_MEL):
        f0, f1, f2 = pts[j], pts[j + 1], pts[j + 2]
        fb = [(k, (k * binw - f0) / (f1 - f0) if k * binw <= f1 else (f2 - k * binw) / (f2 - f1))
              for k in range(int(f0 / binw), min(half, int(f2 / binw) + 1) + 1)]
        bank.append([(k, w) for k, w in fb if w > 0.0] or [(kb(f1), 1.0)])
    plan = _VP_PLANS[sample_rate] = {
        "r": r, "rate": rate, "flen": flen, "hop": max(1, int(round(VP_HOP_SEC * sample_rate))), "m": m,
        "ham": [0.54 - 0.46 * math.cos(2 * math.pi * i / max(1, m - 1)) for i in range(m)], "binw": binw,
        "mel": bank, "spec": (kb(lo_hz), kb(hi_hz)),
        "bands": [(kb(lo), kb(min(hi, rate / 2.0)) - 1) for lo, hi in VP_BANDS_HZ],
        "dct": [[math.cos(math.pi * k * (j + 0.5) / VP_N_MEL) * math.sqrt((1.0 if k == 0 else 2.0) / VP_N_MEL)
                 for j in range(VP_N_MEL)] for k in range(VP_N_CEPS)],
        "q": (int(math.ceil(rate / VP_F0_HZ[1])), min(half - 1, int(rate / VP_F0_HZ[0]))),
    }
    return plan


def _vp_frame(frame, plan: Dict[str, Any]) -> Tuple[List[float], float]:
    """One 25 ms int16 frame -> (22 raw channels, frame energy dB)."""
    n = len(frame)
    zc = sum(1 for a, b in zip(frame, frame[1:]) if (a >= 0) != (b >= 0)) / max(1, n - 1)
    edb = 10.0 * math.log10(sum(v * v for v in frame) / max(1, n) + 1.0)
    r = plan["r"]
    y = ([sum(frame[i:i + r]) / r for i in range(0, n - r + 1, r)] if r > 1 else [float(v) for v in frame])
    y = y[:plan["m"]]
    mu = sum(y) / len(y)
    ham = plan["ham"]
    w, prev = [], 0.0
    for i, v in enumerate(y):                       # DC removal, pre-emphasis, Hamming
        v -= mu
        w.append((v - 0.97 * prev) * ham[i])
        prev = v
    spec = _vp_fft(w + [0.0] * (VP_NFFT - len(w)))
    half = VP_NFFT // 2
    p = [z.real * z.real + z.imag * z.imag for z in spec[:half + 1]]
    lo, hi = plan["spec"]
    seg = p[lo:hi + 1]
    tot = sum(seg) + 1e-9
    floor = 1e-10 * max(seg) + 1e-9
    bands = [sum(p[a:b + 1]) / tot for a, b in plan["bands"]]
    centroid = sum((lo + i) * plan["binw"] * v for i, v in enumerate(seg)) / tot / 1000.0
    flat = math.exp(sum(math.log(v + floor) for v in seg) / len(seg)) / (tot / len(seg))
    mel = [math.log(sum(p[k] * wt for k, wt in fb) + 1.0) for fb in plan["mel"]]
    ceps = [sum(c * e for c, e in zip(row, mel)) for row in plan["dct"]]
    pmax = max(p) + 1e-9
    lg = [0.5 * math.log(v + 1e-8 * pmax) for v in p]
    cep = _vp_fft(lg + lg[-2:0:-1])                 # real cepstrum of the even log-magnitude spectrum
    q0, q1 = plan["q"]
    qpk = max(range(q0, q1 + 1), key=lambda q: cep[q].real)
    return ceps + bands + [centroid, flat, zc, math.log2(plan["rate"] / qpk / 150.0),
                           cep[qpk].real / VP_NFFT], edb


def voiceprint_windows(start_sec: float, end_sec: float) -> Tuple[List[Tuple[float, float, float]], bool]:
    """Turn-to-Diarize / AliMeeting sampling rule -> ([(win_start, win_end, anchor_75pct)], collar_applied).
    A VP_COLLAR_SEC collar is trimmed at both turn boundaries when the turn is >= VP_COLLAR_MIN_TURN_SEC; a turn
    longer than VP_LONG_TURN_SEC is split into ceil(dur / 6 s) equal pieces (Must-Linked downstream). Each piece is
    analysed over VP_WINDOW_FRAC = [0.50, 0.90] of its span, which holds its 75% anchor."""
    dur = max(0.0, end_sec - start_sec)
    collar = dur >= VP_COLLAR_MIN_TURN_SEC
    s, e = (start_sec + VP_COLLAR_SEC, end_sec - VP_COLLAR_SEC) if collar else (start_sec, end_sec)
    n = int(math.ceil(dur / VP_LONG_TURN_SEC - 1e-9)) if dur > VP_LONG_TURN_SEC else 1
    out = []
    for i in range(n):
        ps, pe = s + (e - s) * i / n, s + (e - s) * (i + 1) / n
        ws, we = ps + VP_WINDOW_FRAC[0] * (pe - ps), ps + VP_WINDOW_FRAC[1] * (pe - ps)
        if we - ws < VP_MIN_WINDOW_SEC:
            ws, we = ps, pe
        out.append((ws, we, ps + VP_ANCHOR_FRAC * (pe - ps)))
    return out, collar


def _vp_l2(x: List[float]) -> List[float]:
    nrm = math.sqrt(sum(v * v for v in x))
    return [v / nrm for v in x] if nrm > 1e-12 else [0.0] * len(x)


def _vp_pool(frames: List[List[float]], edbs: List[float], times: List[float], anchor: float, wlen: float,
             ) -> Tuple[List[float], List[float], List[bool], List[bool]]:
    """SE channel rescaling + channel-dependent attentive statistics pooling -> (mu~, sigma~, voiced, active)."""
    nt, nc = len(frames), len(VP_PRIOR_MEAN)
    emax = max(edbs)
    active = [e >= emax - VP_SILENCE_DB for e in edbs]
    voiced = [a and f[VP_CH_VOICING] >= VP_VOICED_CEPS for a, f in zip(active, frames)]
    h = [[(f[c] - VP_PRIOR_MEAN[c]) / VP_PRIOR_STD[c] for c in range(nc)] for f in frames]
    act = [row for row, a in zip(h, active) if a] or h
    z = [sum(row[c] for row in act) / len(act) for c in range(nc)]                       # squeeze
    sd = [math.sqrt(sum((row[c] - z[c]) ** 2 for row in act) / len(act)) for c in range(nc)]
    gate = [0.5 + 1.0 / (1.0 + math.exp(-2.0 * (abs(z[c]) / (sd[c] + 0.5) - 0.5))) for c in range(nc)]  # excite
    h = [[row[c] * gate[c] for c in range(nc)] for row in h]
    centre = [z[c] * gate[c] for c in range(nc)]
    emed = sorted(edbs)[nt // 2]
    sig = max(1e-3, 0.5 * wlen)
    base = [max(-2.0, min(1.0, (e - emed) / 6.0)) + (0.5 if v else 0.0) - 0.5 * ((t - anchor) / sig) ** 2
            - (0.0 if a else 4.0) for e, v, t, a in zip(edbs, voiced, times, active)]
    mu: List[float] = []
    sg: List[float] = []
    for c in range(nc):
        idx = [i for i in range(nt) if voiced[i]] if c == VP_CH_LOGF0 else list(range(nt))
        if not idx:                                   # no voiced frame: F0 stays at the population prior
            mu.append(0.0)
            sg.append(math.exp(VP_LOGSIG_PRIOR[4]))
            continue
        logits = [base[i] - 0.5 * min(4.0, abs(h[i][c] - centre[c])) for i in idx]   # channel-dependent
        top = max(logits)
        ws = [math.exp(v - top) for v in logits]
        tot = sum(ws)
        m = sum(w * h[i][c] for w, i in zip(ws, idx)) / tot
        var = sum(w * h[i][c] ** 2 for w, i in zip(ws, idx)) / tot - m * m
        mu.append(m)
        sg.append(math.sqrt(max(var, 1e-6)))
    return mu, sg, voiced, active


def _vp_embedding(mu: List[float], sg: List[float]) -> List[float]:
    x = (mu[1:13] + [math.log(sg[c]) - VP_LOGSIG_PRIOR[c - 1] for c in (1, 2, 3, 4)]
         + [mu[VP_CH_LOGF0], math.log(sg[VP_CH_LOGF0]) - VP_LOGSIG_PRIOR[4], mu[VP_CH_ZCR], mu[VP_CH_CENTROID],
            mu[VP_CH_BAND], mu[VP_CH_BAND + 3], mu[VP_CH_FLAT], mu[0]])
    return _vp_l2([v * w for v, w in zip(x, VP_DIM_WEIGHTS)])


def _vp_sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def _vp_near_field(edbs: List[float], frames: List[List[float]], active: List[bool]) -> Tuple[float, Dict]:
    """Near-field (headset / closest talker) vs far-field (1-3 m interlocutor): level, modulation depth (reverb
    fills the energy valleys), top-band share (air / room absorb highs) and spectral flatness (diffuse noise)."""
    act = [i for i, a in enumerate(active) if a] or list(range(len(edbs)))
    level = 10.0 * math.log10(sum(10 ** (edbs[i] / 10.0) for i in act) / len(act)) - 90.31   # dBFS
    srt = sorted(edbs)
    mod = srt[int(0.9 * (len(srt) - 1))] - srt[int(0.1 * (len(srt) - 1))]
    top = sum(frames[i][VP_CH_BAND + 3] for i in act) / len(act)
    flat = sum(frames[i][VP_CH_FLAT] for i in act) / len(act)
    # recording gain varies ~25 dB between devices, so level is a mild term; the gain-invariant cues carry it
    score = _vp_sigmoid(0.06 * (level + 45.0) + 0.12 * (mod - 15.0) + 3.0 * (top - 0.30) - 6.0 * (flat - 0.25))
    return score, {"level_dbfs": round(level, 1), "modulation_db": round(mod, 1), "top_band": round(top, 3),
                   "flatness": round(flat, 3)}


def _vp_proxy(rms_energy: float, zcr_pitch: float, speech_rate: float, duration: float) -> List[float]:
    """Deterministic acoustic-prosodic proxy x_i for JSON-only turns (no PCM): log RMS -> level slot, ZCR ->
    ZCR / F0 / c1-tilt slots, speech rate and log duration -> the c1/c2 dynamics slots."""
    x = [0.0] * VP_DIM
    if rms_energy > 0 and zcr_pitch > 0:
        zz = (zcr_pitch - 0.08) / 0.04
        x[0], x[16], x[18], x[23] = -0.4 * zz, 0.6 * zz, zz, 0.5 * (math.log(rms_energy + 1.0) - math.log(1500.0))
    x[12] = (speech_rate - VP_ZH_CHARS_PER_SEC) / 2.0
    x[13] = math.log(max(duration, 0.1) / 3.0)
    return _vp_l2(x)


def extract_turn_voiceprint(pcm=None, sample_rate: int = 16000, start_sec: float = 0.0,
                            end_sec: Optional[float] = None, offset_sec: float = 0.0, rms_energy: float = 0.0,
                            zcr_pitch: float = 0.0, text: str = "", duration: Optional[float] = None) -> Dict[str, Any]:
    """Turn-level voiceprint x_i (VP_DIM, L2-normalised) of the turn [start_sec, end_sec).

    pcm: int16 mono samples whose first sample sits at offset_sec (e.g. a 60 s hotspot slice with absolute turn
    times), or None. With PCM: collar + 75%-anchor windows (voiceprint_windows), 25 ms / 10 ms frames, SE +
    attentive statistics pooling per window; a > 6 s turn yields one sub-vector per piece (sub_voiceprints, to be
    Must-Linked) and x_i is their normalised mean. rms_energy / zcr_pitch are then measured on the anchor windows.
    Without PCM: a deterministic proxy from (rms_energy, zcr_pitch, speech rate, duration)."""
    end_sec = (offset_sec + len(pcm) / float(sample_rate)) if (end_sec is None and pcm is not None) else end_sec
    dur = duration if duration is not None else max(0.0, (end_sec or start_sec) - start_sec)
    if not dur:
        dur = max(1.0, len(_norm(text)) / VP_ZH_CHARS_PER_SEC)
    plan = _vp_plan(sample_rate) if pcm is not None else None
    if pcm is None or len(pcm) < plan["flen"]:
        rate = len(_norm(text)) / max(dur, 0.1)
        src = VP_SOURCE_PROXY_AC if rms_energy > 0 and zcr_pitch > 0 else VP_SOURCE_PROXY_TEXT
        x = _vp_proxy(rms_energy, zcr_pitch, rate, dur)
        nf = round(_vp_sigmoid((20.0 * math.log10(rms_energy / 32768.0) + 30.0) / 5.0), 3) if rms_energy > 0 else -1.0
        return {"voiceprint": [round(v, 5) for v in x], "sub_voiceprints": [[round(v, 5) for v in x]],
                "source": src, "confidence": VP_CONFIDENCE[src], "near_field_score": nf,
                "rms_energy": rms_energy, "zcr_pitch": zcr_pitch, "windows": [], "collar_applied": False,
                "frames": 0, "f0_hz": 0.0, "voiced_ratio": 0.0}
    flen, hop = plan["flen"], plan["hop"]
    last = len(pcm) - flen
    wins, collar = voiceprint_windows(start_sec, end_sec)
    subs: List[List[float]] = []
    all_f: List[List[float]] = []
    all_e: List[float] = []
    all_a: List[bool] = []
    all_v: List[bool] = []
    sq, zc, cnt = 0.0, 0, 0
    for ws, we, anchor in wins:
        a = max(0, min(last, int(round((ws - offset_sec) * sample_rate))))
        b = max(a, min(last, int(round((we - offset_sec) * sample_rate)) - flen))
        pos = list(range(a, b + 1, hop))
        if len(pos) > VP_MAX_FRAMES:
            pos = [pos[int(round(i * (len(pos) - 1) / (VP_MAX_FRAMES - 1)))] for i in range(VP_MAX_FRAMES)]
        frames, edbs, times = [], [], []
        for p0 in pos:
            fr = pcm[p0:p0 + flen]
            f, e = _vp_frame(fr, plan)
            frames.append(f)
            edbs.append(e)
            times.append(offset_sec + (p0 + flen / 2.0) / sample_rate)
        seg = pcm[a:max(a + 1, min(len(pcm), int(round((we - offset_sec) * sample_rate))))]
        sq += sum(v * v for v in seg)
        zc += sum(1 for u, v in zip(seg, seg[1:]) if (u >= 0) != (v >= 0))
        cnt += len(seg)
        mu, sg, voiced, active = _vp_pool(frames, edbs, times, anchor, we - ws)
        subs.append(_vp_embedding(mu, sg))
        all_f += frames
        all_e += edbs
        all_a += active
        all_v += voiced
    x = _vp_l2([sum(col) for col in zip(*subs)])
    nf, nf_feats = _vp_near_field(all_e, all_f, all_a)
    f0s = sorted(150.0 * 2 ** f[VP_CH_LOGF0] for f, v in zip(all_f, all_v) if v)
    return {"voiceprint": [round(v, 5) for v in x], "sub_voiceprints": [[round(v, 5) for v in s] for s in subs],
            "source": VP_SOURCE_WAV, "confidence": VP_CONFIDENCE[VP_SOURCE_WAV], "near_field_score": round(nf, 3),
            "near_field_features": nf_feats, "rms_energy": round(math.sqrt(sq / max(1, cnt)), 1),
            "zcr_pitch": round(zc / max(1, cnt - 1), 4),
            "windows": [[round(ws, 3), round(we, 3), round(an, 3)] for ws, we, an in wins],
            "collar_applied": collar, "frames": len(all_f),
            "f0_hz": round(f0s[len(f0s) // 2], 1) if f0s else 0.0,
            "voiced_ratio": round(sum(all_v) / max(1, len(all_v)), 3)}


# --- constrained diarization: lexical Must/Cannot-Link + E2CP + spectral clustering + flip-aligned window vote
E2CP_ALPHA = 0.4
VP_P_PERCENTILE = 0.5            # row-wise p-percentile soft threshold (entries below x 0.01)
VP_SOFT_THRESHOLD = 0.01
VP_MAX_SPEAKERS = 8
VP_PAIR_SAME_AFFINITY = 0.75     # < VP_MIN_NODES_MULTI-node blocks: single-link same speaker iff A^ >= this
VP_MIN_NODES_MULTI = 4           # blocks this large with >= 2 diarizer labels search K >= 2
VP_BLOCK_NODES = 48              # E2CP / eigen work is O(N^3) pure Python: long recordings run in chronological blocks
VP_VOTE_WIN, VP_VOTE_HOP = 6, 3
VP_LATCH_GAP_SEC = 1.5           # dangling-clause latch (= the analyzer's same-speaker merge gap)
VP_CONTINUATION_GAP_SEC = 2.0    # connective-opened continuation across a backchannel
VP_VOCATIVE_LOOKBACK_SEC = 60.0
VP_QUESTION_TAILS = ("嗎", "呢")
VP_QUESTION_CUES = ("對不對", "是不是", "有沒有", "要不要", "好不好", "可不可以", "會不會", "能不能", "可以嗎")
# yes/no-type openers: someone else answering (a self-answer opens with 因為 / 就是 / 其實, which are excluded)
VP_ANSWER_OPENERS = ("沒有", "不是", "不會", "不用", "不行", "不要", "沒錯", "可以", "應該", "對", "是", "有", "會",
                     "嗯", "好")
VP_RESPONSE_OPENERS = VP_ANSWER_OPENERS + ("OK", "ok", "Ok", "了解", "我懂", "我知道")
# MIntRec first pair parts that project a reply from someone else (request / proposal / advice / complaint)
VP_FIRST_PAIR_FINE = ("Arrange", "Advise", "Ask_for_help", "Prevent", "Complain")
VP_CONTINUATION_OPENERS = ("然後", "所以", "而且", "還有", "再來", "接著", "另外", "包括", "就是說", "或者", "以及")
# first-person self-reference per role: topical role words (派案, 安撫) are echoed across speakers, these are not
VP_SELF_REFERENCE_CUES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (ROLE_DIRECTOR, ("我是主任", "我當主任")),
    (ROLE_SUPERVISOR, ("我們督導", "我是督導", "我是居督", "我們居督", "我當督導")),
    (ROLE_RESEARCHER, ("我們後端", "我們的系統", "我們團隊", "我們研究", "我們學生")),
    (ROLE_CAREGIVER_LTC, ("我的夥伴", "我是照服員", "我是居服員", "我去案家", "我的個案")),
)
VP_DANGLING_TAILS = ("的", "跟", "和", "就", "在", "說", "然後", "所以", "因為", "但是", "可是", "如果", "那個", "這個")
_VP_PUNCT = "，。,.!！?？、~～…:：;； "


def _vp_core(text: str) -> str:
    return _norm(text).strip(_VP_PUNCT)


def _vp_is_question(text: str) -> bool:
    t = _norm(text).rstrip("。.!！~～… ")
    core = _vp_core(text)
    return bool(core) and (t.endswith(("?", "？")) or core.endswith(VP_QUESTION_TAILS)
                           or any(c in core for c in VP_QUESTION_CUES))


def _vp_is_answer(text: str, openers: Tuple[str, ...] = VP_ANSWER_OPENERS) -> bool:
    core = _vp_core(text)
    return bool(core) and core.startswith(openers) and not _vp_is_question(text)


def turn_end_times(turns: List[UtteranceTurn]) -> Dict[int, float]:
    """end_time when known, else the next turn's start, else start + text length at VP_ZH_CHARS_PER_SEC."""
    out = {}
    for i, t in enumerate(turns):
        est = t.timestamp + min(15.0, max(1.0, len(_norm(t.text)) / VP_ZH_CHARS_PER_SEC))
        if t.end_time > t.timestamp:
            out[t.turn_id] = t.end_time
        elif i + 1 < len(turns) and turns[i + 1].timestamp > t.timestamp:
            out[t.turn_id] = min(turns[i + 1].timestamp, est + 5.0)
        else:
            out[t.turn_id] = est
    return out


def voiceprint_lexical_constraints(turns: List[UtteranceTurn], ends: Optional[Dict[int, float]] = None,
                                   graph: Optional[Dict[str, Any]] = None,
                                   speaker_roles: Optional[Dict[str, str]] = None) -> List[Tuple[int, int, int, str]]:
    """Turn-level pairwise constraints Z_ij from lexical turn-taking (GIFT / Bi-CL signals) -> [(i, j, +-1, reason)].
    Cannot-Link (-1): a question answered by a yes/no-type reply (adjacent or GIFT reply-to: an <st> QA pair), an
    adjacent request / proposal / advice / complaint (MIntRec first pair part) met by a response token (可以 / 好 /
    沒問題 / OK ...), a vocative addressed to a role held by the previous holder's turn, a floor handoff (請主任...)
    to the next turn. Must-Link (+1): a same-label discourse latch (connective-opened continuation across
    backchannels, or a dangling clause latched within VP_LATCH_GAP_SEC) and lexical self-consistency (same label
    repeating one role's first-person self-reference, VP_SELF_REFERENCE_CUES; chained to the previous such turn)."""
    ends = ends or turn_end_times(turns)
    roles = speaker_roles or {}
    seq = sorted(turns, key=lambda t: (t.timestamp, t.turn_id))
    by_id = {t.turn_id: t for t in seq}
    out: List[Tuple[int, int, int, str]] = []

    def add(a: UtteranceTurn, b: UtteranceTurn, z: int, why: str) -> None:
        if a.turn_id != b.turn_id:
            out.append((min(a.turn_id, b.turn_id), max(a.turn_id, b.turn_id), z, why))

    for prev, cur in zip(seq, seq[1:]):
        if _vp_is_question(prev.text) and _vp_is_answer(cur.text):
            add(prev, cur, -1, "qa_turn_taking")
        elif (classify_mintrec_taxonomy(prev.text)["fine"] in VP_FIRST_PAIR_FINE
              and _vp_is_answer(cur.text, VP_RESPONSE_OPENERS)):
            add(prev, cur, -1, "adjacency_pair_response")
    for child, parent in ((graph or {}).get("reply_to") or {}).items():
        if child in by_id and parent in by_id and _vp_is_question(by_id[parent].text) and _vp_is_answer(by_id[child].text):
            add(by_id[parent], by_id[child], -1, "qa_turn_taking")
    for k, cur in enumerate(seq):
        txt = _norm(cur.text)
        voc = next((role for role, cues in VOCATIVE_ROLE_CUES if any(c in txt for c in cues)), None)
        if voc and voc != roles.get(cur.speaker_id):
            prev = next((p for p in reversed(seq[:k]) if roles.get(p.speaker_id) == voc
                         and cur.timestamp - p.timestamp <= VP_VOCATIVE_LOOKBACK_SEC), None)
            if prev is not None:
                add(prev, cur, -1, "vocative_addressing")
        if k + 1 < len(seq) and any(c in _vp_core(cur.text)[-12:] for c in DIRECTOR_HANDOFF_CUES):
            add(cur, seq[k + 1], -1, "floor_handoff")
    for k, cur in enumerate(seq):
        j = k - 1
        while j >= 0 and seq[j].speaker_id != cur.speaker_id and is_agreement_only(seq[j].text):
            j -= 1
        if j < 0 or seq[j].speaker_id != cur.speaker_id:
            continue
        prev = seq[j]
        gap = cur.timestamp - ends[prev.turn_id]
        opener = _vp_core(cur.text).startswith(VP_CONTINUATION_OPENERS)
        dangling = (_norm(prev.text)[-1:] not in _VP_PUNCT and _vp_core(prev.text).endswith(VP_DANGLING_TAILS))
        if (opener and gap <= VP_CONTINUATION_GAP_SEC) or (dangling and gap <= VP_LATCH_GAP_SEC):
            add(prev, cur, +1, "discourse_latch")
    last: Dict[Tuple[str, str], UtteranceTurn] = {}
    for cur in seq:
        txt = _norm(cur.text)
        for role, cues in VP_SELF_REFERENCE_CUES:
            if any(c in txt for c in cues):
                key = (cur.speaker_id, role)
                if key in last:
                    add(last[key], cur, +1, "lexical_self_consistency")
                last[key] = cur
    return out


def _vp_cos(a: List[float], b: List[float]) -> float:
    na = math.sqrt(sum(v * v for v in a))
    nb = math.sqrt(sum(v * v for v in b))
    return sum(u * v for u, v in zip(a, b)) / (na * nb) if na > 1e-12 and nb > 1e-12 else 0.0


def _vp_inverse(m: List[List[float]]) -> List[List[float]]:
    """Gauss-Jordan inverse with partial pivoting."""
    n = len(m)
    a = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(m)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(a[r][c]))
        a[c], a[p] = a[p], a[c]
        inv = 1.0 / a[c][c]
        rc = [v * inv for v in a[c]]
        a[c] = rc
        for r in range(n):
            f = a[r][c]
            if r != c and f:
                a[r] = [u - f * v for u, v in zip(a[r], rc)]
    return [row[n:] for row in a]


def _vp_matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    bt = list(zip(*b))
    return [[sum(u * v for u, v in zip(row, col)) for col in bt] for row in a]


def _vp_sym_normalize(w: List[List[float]]) -> List[List[float]]:
    dis = [1.0 / math.sqrt(s) if s > 1e-12 else 0.0 for s in (sum(r) for r in w)]
    return [[dis[i] * v * dis[j] for j, v in enumerate(row)] for i, row in enumerate(w)]


def e2cp_adjust_affinity(a: List[List[float]], z: List[List[float]], alpha: float = E2CP_ALPHA,
                         ) -> Tuple[List[List[float]], List[List[float]]]:
    """E2CP (Lu & Ip; Turn-to-Diarize Sec. 3.3) -> (A^, Q*). Closed form of the vertical-then-horizontal
    propagation F_v = aL F_v + (1-a)Z, F_h = a F_h L + (1-a)F_v over L = D^-1/2 A D^-1/2 (zero diagonal):
    Q* = (1-a)^2 (I - aL)^-1 Z (I - aL)^-1, clipped to [-1, 1]. Then A^_ij = 1 - (1 - Q*_ij)(1 - A_ij) when
    Q*_ij >= 0, else (1 + Q*_ij) A_ij. With no constraints A^ = A."""
    n = len(a)
    if not any(v for row in z for v in row):
        return [row[:] for row in a], [[0.0] * n for _ in range(n)]
    lb = _vp_sym_normalize([[0.0 if i == j else a[i][j] for j in range(n)] for i in range(n)])
    m = _vp_inverse([[(1.0 if i == j else 0.0) - alpha * lb[i][j] for j in range(n)] for i in range(n)])
    mz = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if z[i][j]:
                for r in range(n):
                    mz[r][j] += m[r][i] * z[i][j]
    f = _vp_matmul(mz, m)
    k = (1.0 - alpha) ** 2
    q = [[max(-1.0, min(1.0, k * v)) for v in row] for row in f]
    ah = [[1.0 if i == j else (1.0 - (1.0 - q[i][j]) * (1.0 - a[i][j]) if q[i][j] >= 0
                               else (1.0 + q[i][j]) * a[i][j]) for j in range(n)] for i in range(n)]
    return ah, q


def _vp_refine(ah: List[List[float]], p: float = VP_P_PERCENTILE, soft: float = VP_SOFT_THRESHOLD) -> List[List[float]]:
    """Row-wise p-percentile soft threshold, average symmetrisation, diagonal = row max (Wang et al. refinement)."""
    n = len(ah)
    x = [row[:] for row in ah]
    for i in range(n):
        off = sorted(x[i][j] for j in range(n) if j != i)
        if off:
            thr = off[int(p * (len(off) - 1))]
            x[i] = [v * soft if j != i and v < thr else v for j, v in enumerate(x[i])]
    y = [[0.5 * (x[i][j] + x[j][i]) for j in range(n)] for i in range(n)]
    for i in range(n):
        y[i][i] = max((y[i][j] for j in range(n) if j != i), default=1.0)
    return y


def _vp_eigh(s: List[List[float]], sweeps: int = 30) -> Tuple[List[float], List[List[float]]]:
    """Cyclic Jacobi eigen-decomposition of a symmetric matrix -> (eigenvalues desc, eigenvectors as rows)."""
    n = len(s)
    a = [row[:] for row in s]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        if sum(a[i][j] ** 2 for i in range(n) for j in range(i + 1, n)) < 1e-18:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) < 1e-15:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * apq)
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                sn = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p], a[k][q] = c * akp - sn * akq, sn * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k], a[q][k] = c * apk - sn * aqk, sn * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p], v[k][q] = c * vkp - sn * vkq, sn * vkp + c * vkq
    order = sorted(range(n), key=lambda i: -a[i][i])
    return [a[i][i] for i in order], [[v[k][i] for k in range(n)] for i in order]


def _vp_kmeans(u: List[List[float]], k: int, iters: int = 25) -> List[int]:
    """Deterministic k-means (farthest-point init from row 0)."""
    n = len(u)
    d2 = lambda x, y: sum((p - q) ** 2 for p, q in zip(x, y))   # noqa: E731
    cents = [u[0]]
    while len(cents) < min(k, n):
        cents.append(u[max(range(n), key=lambda i: (min(d2(u[i], c) for c in cents), -i))])
    lab: List[int] = []
    for _ in range(iters):
        new = [min(range(len(cents)), key=lambda c: (d2(u[i], cents[c]), c)) for i in range(n)]
        if new == lab:
            break
        lab = new
        for c in range(len(cents)):
            mem = [u[i] for i in range(n) if lab[i] == c]
            if mem:
                cents[c] = [sum(col) / len(mem) for col in zip(*mem)]
    return lab


def _vp_rownorm(rows: List[List[float]]) -> List[List[float]]:
    return [_vp_l2(r) if any(r) else r for r in rows]


def _vp_spectral(y: List[List[float]], ah: List[List[float]], min_k: int = 1,
                 max_k: int = VP_MAX_SPEAKERS) -> Tuple[List[int], int, List[float]]:
    """Eigengap speaker count over the normalised refined affinity + NJW spectral k-means -> (labels, K, eigs).
    The gap is searched over [min_k, max_k] (spectralcluster's min_clusters / max_clusters)."""
    n = len(y)
    if n < VP_MIN_NODES_MULTI:   # too few nodes for an eigengap (A = (1 + cos) / 2 puts unrelated pairs at 0.5):
        root = list(range(n))     # single-link components of A^ >= VP_PAIR_SAME_AFFINITY
        find = lambda i: i if root[i] == i else find(root[i])   # noqa: E731
        for i in range(n):
            for j in range(i + 1, n):
                if ah[i][j] >= VP_PAIR_SAME_AFFINITY:
                    root[find(j)] = find(i)
        ids: Dict[int, int] = {}
        lab = [ids.setdefault(find(i), len(ids)) for i in range(n)]
        return lab, len(ids), []
    evals, evecs = _vp_eigh(_vp_sym_normalize(y))
    kmax = max(1, min(max_k, VP_MAX_SPEAKERS, n - 1))
    k = max(range(min(min_k, kmax), kmax + 1), key=lambda kk: (evals[kk - 1] - evals[kk], -kk))
    if k == 1:
        return [0] * n, 1, evals
    emb = _vp_rownorm([[evecs[c][i] for c in range(k)] for i in range(n)])
    return _vp_kmeans(emb, k), k, evals


def _vp_window_vote(y: List[List[float]], glob: List[int]) -> Tuple[List[int], int]:
    """USC flip-aligned sliding-window majority vote: each window re-clusters its nodes locally (labels are
    window-relative), is aligned to the global labels by the best permutation, and votes; global labels vote once."""
    n = len(y)
    votes = [{g: 1.0} for g in glob]
    if n > VP_VOTE_WIN:
        starts = list(range(0, n - VP_VOTE_WIN + 1, VP_VOTE_HOP))
        if starts[-1] != n - VP_VOTE_WIN:
            starts.append(n - VP_VOTE_WIN)
        for s in starts:
            idx = list(range(s, s + VP_VOTE_WIN))
            present = sorted({glob[i] for i in idx})
            if len(present) < 2:
                continue
            local = _vp_kmeans(_vp_rownorm([[y[a][b] for b in idx] for a in idx]), len(present))
            k = max(local) + 1
            best = max(itertools.permutations(present, k),
                       key=lambda perm: sum(perm[local[t]] == glob[i] for t, i in enumerate(idx)))
            for t, i in enumerate(idx):
                votes[i][best[local[t]]] = votes[i].get(best[local[t]], 0.0) + 1.0
    final = [max(v, key=lambda lab: (v[lab], lab == glob[i], -lab)) for i, v in enumerate(votes)]
    return final, sum(1 for f, g in zip(final, glob) if f != g)


def _vp_turn_record(t: UtteranceTurn, end: float) -> Dict[str, Any]:
    if t.voiceprint:
        src = t.voiceprint_source or VP_SOURCE_WAV
        subs = t.voiceprint_subvectors or [t.voiceprint]
        return {"voiceprint": list(t.voiceprint), "sub_voiceprints": [list(s) for s in subs], "source": src,
                "confidence": VP_CONFIDENCE.get(src, VP_CONFIDENCE[VP_SOURCE_WAV]),
                "near_field_score": t.near_field_score}
    return extract_turn_voiceprint(None, start_sec=t.timestamp, end_sec=end, rms_energy=t.rms_energy,
                                   zcr_pitch=t.zcr_pitch, text=t.text)


def constrained_voiceprint_diarize(turns: List[UtteranceTurn], graph: Optional[Dict[str, Any]] = None,
                                   speaker_roles: Optional[Dict[str, str]] = None,
                                   alpha: float = E2CP_ALPHA) -> Dict[str, Any]:
    """Acoustic-lexical constrained diarization over turn voiceprints (Upgrade G).

    Nodes are turn pieces (a > 6 s turn contributes one node per sub-window, chained by Must-Links). Raw affinity
    A_ij = (1 + cos(x_i, x_j)) / 2; lexical constraints Z (voiceprint_lexical_constraints) are propagated by
    E2CP into A^ (e2cp_adjust_affinity); p-percentile refinement + eigengap + spectral k-means + flip-aligned
    window vote give refined clusters, mapped onto the diarizer's speaker_N labels by overlap. Turns without a
    voiceprint get the JSON-only proxy. -> {"summary", "same_speaker_prob", "affinity_raw", "affinity_e2cp",
    "refined_speaker", "turn_records"}; same_speaker_prob(i, j) = c A^_ij + (1 - c)[s_i == s_j] with c the
    smaller voiceprint confidence of the two turns (0 for text-only proxies -> exactly the diarizer label)."""
    seq = sorted(turns, key=lambda t: (t.timestamp, t.turn_id))
    ends = turn_end_times(seq)
    by_id = {t.turn_id: t for t in seq}
    rec = {t.turn_id: _vp_turn_record(t, ends[t.turn_id]) for t in seq}
    nodes: List[Tuple[int, List[float]]] = []
    nodes_of: Dict[int, List[int]] = {}
    for t in seq:
        for s in rec[t.turn_id]["sub_voiceprints"]:
            nodes_of.setdefault(t.turn_id, []).append(len(nodes))
            nodes.append((t.turn_id, s))
    pairs: Dict[Tuple[int, int], Tuple[int, str]] = {}
    conflicts = set()
    for tid, ns in nodes_of.items():
        for a, b in zip(ns, ns[1:]):
            pairs[(a, b)] = (1, "long_turn_split")
    for i, j, z, why in voiceprint_lexical_constraints(seq, ends, graph, speaker_roles):
        key = (nodes_of[i][-1], nodes_of[j][0]) if by_id[i].timestamp <= by_id[j].timestamp else \
              (nodes_of[j][-1], nodes_of[i][0])
        key = (min(key), max(key))
        if key in pairs and pairs[key][0] != z:
            conflicts.add(key)
        elif key not in pairs:
            pairs[key] = (z, why)
    for key in conflicts:
        pairs.pop(key, None)
    n = len(nodes)
    blocks = [list(range(s, min(n, s + VP_BLOCK_NODES))) for s in range(0, n, VP_BLOCK_NODES)]
    block_of = {i: bi for bi, blk in enumerate(blocks) for i in blk}
    a_raw: Dict[Tuple[int, int], float] = {}
    a_hat: Dict[Tuple[int, int], float] = {}
    node_label: Dict[int, str] = {}
    est_k, flips, dropped = [], 0, 0
    for bi, blk in enumerate(blocks):
        pos = {g: k for k, g in enumerate(blk)}
        m = len(blk)
        a = [[1.0 if p == q else 0.5 * (1.0 + _vp_cos(nodes[blk[p]][1], nodes[blk[q]][1])) for q in range(m)]
             for p in range(m)]
        z = [[0.0] * m for _ in range(m)]
        for (u, v), (val, _) in pairs.items():
            if u in pos and v in pos:
                z[pos[u]][pos[v]] = z[pos[v]][pos[u]] = float(val)
            elif block_of[u] == bi:
                dropped += 1
        ah, _ = e2cp_adjust_affinity(a, z, alpha)
        for p in range(m):
            for q in range(p + 1, m):
                a_raw[(blk[p], blk[q])], a_hat[(blk[p], blk[q])] = a[p][q], ah[p][q]
        y = _vp_refine(ah)
        # a multi-party block (>= 2 diarizer labels, >= VP_MIN_NODES_MULTI nodes) searches K >= 2
        n_lab = len({by_id[nodes[g][0]].speaker_id for g in blk})
        glob, k, _ = _vp_spectral(y, ah, 2 if n_lab >= 2 and m >= VP_MIN_NODES_MULTI else 1, n_lab + 1)
        final, f = _vp_window_vote(y, glob)
        est_k.append(len(set(final)))
        flips += f
        diar = [by_id[nodes[g][0]].speaker_id for g in blk]
        overlap: Dict[Tuple[int, str], int] = {}
        for c, sp in zip(final, diar):
            overlap[(c, sp)] = overlap.get((c, sp), 0) + 1
        mapping: Dict[int, str] = {}
        for (c, sp), _ in sorted(overlap.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])):
            if c not in mapping and sp not in mapping.values():
                mapping[c] = sp
        for c in sorted(set(final) - set(mapping)):
            mapping[c] = f"speaker_vp{bi + 1}_{c + 1}"
        for g, c in zip(blk, final):
            node_label[g] = mapping[c]

    def turn_aff(table: Dict[Tuple[int, int], float], i: int, j: int) -> Optional[float]:
        vals = [table[(min(p, q), max(p, q))] for p in nodes_of[i] for q in nodes_of[j]
                if (min(p, q), max(p, q)) in table]
        return sum(vals) / len(vals) if vals else None

    cache: Dict[Tuple[int, int], float] = {}

    def affinity_raw(i: int, j: int) -> float:
        v = turn_aff(a_raw, i, j)
        return v if v is not None else 0.5 * (1.0 + _vp_cos(rec[i]["voiceprint"], rec[j]["voiceprint"]))

    def affinity_e2cp(i: int, j: int) -> float:
        v = turn_aff(a_hat, i, j)
        return v if v is not None else affinity_raw(i, j)

    def same_speaker_prob(i: int, j: int) -> float:
        if i == j:
            return 1.0
        key = (min(i, j), max(i, j))
        if key not in cache:
            c = min(rec[i]["confidence"], rec[j]["confidence"]) if i in rec and j in rec else 0.0
            same = 1.0 if by_id[i].speaker_id == by_id[j].speaker_id else 0.0
            cache[key] = same if c <= 0.0 else round(c * affinity_e2cp(i, j) + (1.0 - c) * same, 4)
        return cache[key]

    refined: Dict[int, str] = {}
    for tid, ns in nodes_of.items():
        labs = [node_label[g] for g in ns]
        refined[tid] = max(labs, key=lambda lab: (labs.count(lab), lab == by_id[tid].speaker_id))
    ids = [t.turn_id for t in seq]
    intra = [(i, j) for k, i in enumerate(ids) for j in ids[k + 1:] if by_id[i].speaker_id == by_id[j].speaker_id
             and turn_aff(a_raw, i, j) is not None]
    inter = [(i, j) for k, i in enumerate(ids) for j in ids[k + 1:] if by_id[i].speaker_id != by_id[j].speaker_id
             and turn_aff(a_raw, i, j) is not None]
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None   # noqa: E731
    srcs = [rec[i]["source"] for i in ids]
    n_wav = srcs.count(VP_SOURCE_WAV)
    acoustic = [i for i in ids if rec[i]["confidence"] > 0.0]
    nf: Dict[str, List[float]] = {}
    for i in ids:
        if rec[i]["near_field_score"] is not None and rec[i]["near_field_score"] >= 0:
            nf.setdefault(by_id[i].speaker_id, []).append(rec[i]["near_field_score"])
    nf_mean = {sp: round(sum(v) / len(v), 3) for sp, v in sorted(nf.items())}
    primary = max(nf_mean, key=lambda sp: (nf_mean[sp], len(nf[sp]))) if nf_mean else "unknown"
    reasons: Dict[str, int] = {}
    for val, why in pairs.values():
        reasons[why] = reasons.get(why, 0) + 1
    must = sum(1 for val, _ in pairs.values() if val > 0)
    mode = (VP_SOURCE_WAV if n_wav == len(ids) else "mixed" if n_wav else
            VP_SOURCE_PROXY_AC if VP_SOURCE_PROXY_AC in srcs else VP_SOURCE_PROXY_TEXT)
    summary = {
        "voiceprint_dim": VP_DIM, "voiceprint_mode": mode,
        "turns_with_voiceprint": n_wav, "turns_with_proxy_voiceprint": len(ids) - n_wav,
        "nodes": n, "long_turn_splits": sum(1 for ns in nodes_of.values() if len(ns) > 1),
        "must_link_constraints": must, "cannot_link_constraints": len(pairs) - must,
        "constraints_by_reason": dict(sorted(reasons.items())), "constraint_conflicts_dropped": len(conflicts),
        "cross_block_constraints_dropped": dropped, "e2cp_alpha": alpha, "p_percentile": VP_P_PERCENTILE,
        "blocks": len(blocks),
        "mean_intra_speaker_affinity_raw": mean([turn_aff(a_raw, i, j) for i, j in intra]),
        "mean_intra_speaker_affinity_e2cp": mean([turn_aff(a_hat, i, j) for i, j in intra]),
        "mean_inter_speaker_affinity_raw": mean([turn_aff(a_raw, i, j) for i, j in inter]),
        "mean_inter_speaker_affinity_e2cp": mean([turn_aff(a_hat, i, j) for i, j in inter]),
        "diarizer_speakers": len({by_id[i].speaker_id for i in ids}),
        "estimated_speakers": max(est_k) if est_k else 0, "window_vote_flips": flips,
        "near_field_primary_speaker": primary, "near_field_by_speaker": nf_mean,
        "diarization_consistency_score": (round(sum(1 for i in acoustic if refined[i] == by_id[i].speaker_id)
                                                / len(acoustic), 3) if acoustic else None),
        "relabel_candidates": [{"turn_id": i, "diarizer": by_id[i].speaker_id, "voiceprint_speaker": refined[i]}
                               for i in acoustic if refined[i] != by_id[i].speaker_id],
        "bi_cl_gsn_speaker_weighting": "voiceprint_e2cp" if acoustic else "diarizer_label",
        "constraints": [{"link": "must" if val > 0 else "cannot", "nodes": [u, v],
                         "turns": [nodes[u][0], nodes[v][0]], "reason": why}
                        for (u, v), (val, why) in sorted(pairs.items())][:60],
    }
    return {"summary": summary, "same_speaker_prob": same_speaker_prob, "affinity_raw": affinity_raw,
            "affinity_e2cp": affinity_e2cp, "refined_speaker": refined, "turn_records": rec}


# synthetic test voices (unit tests here and in exp1_pi_audio_check): (F0 Hz, formants Hz, spectral tilt)
TEST_VOICE_LOW = (110.0, (600.0, 1100.0, 2500.0), 1.0)    # low pitch, bright harmonic timbre
TEST_VOICE_HIGH = (230.0, (850.0, 1900.0, 3000.0), 1.7)   # high pitch, darker timbre


def synth_test_voice(voice: Tuple[float, Tuple[float, ...], float], seconds: float, seed: int = 1,
                     amplitude: float = 9000.0, far_field: bool = False, sample_rate: int = 16000) -> List[int]:
    """Deterministic voiced 'speech' (int16): wavetable harmonics shaped by 3 formants and a 1/k^tilt slope, 5 Hz
    vibrato, ~4 Hz syllabic AM and a per-seed +-4% F0 / formant offset. far_field adds 3 room echoes, a one-pole
    high-frequency damping, -10 dB level and a noise floor (a 1-3 m interlocutor vs the headset wearer)."""
    f0, formants, tilt = voice
    rnd = (seed * 2654435761) % 2 ** 32
    off = 1.0 + 0.04 * (((rnd >> 8) % 2001) / 1000.0 - 1.0)
    f0 *= off
    size = 1024
    table = [0.0] * size
    for k in range(1, int(3800.0 / f0) + 1):
        amp = (1.0 / k ** tilt) * sum(1.0 / (1.0 + ((k * f0 - fm * off) / 120.0) ** 2) for fm in formants) + 0.02 / k
        ph = (k * 0.7 + seed) % (2 * math.pi)
        for i in range(size):
            table[i] += amp * math.sin(2 * math.pi * k * i / size + ph)
    peak = max(abs(v) for v in table) or 1.0
    table = [v / peak for v in table]
    n = int(seconds * sample_rate)
    out, phase, lcg = [0.0] * n, 0.0, rnd or 1
    syl = 3.5 + (rnd % 100) / 100.0
    for i in range(n):
        t = i / sample_rate
        phase = (phase + f0 * (1.0 + 0.02 * math.sin(2 * math.pi * 5.0 * t)) / sample_rate) % 1.0
        env = 0.55 + 0.45 * math.sin(2 * math.pi * syl * t) ** 2
        lcg = (1103515245 * lcg + 12345) % 2 ** 31
        out[i] = amplitude * env * table[int(phase * size)] + amplitude * 0.01 * (lcg / 2 ** 30 - 1.0)
    if far_field:
        dry = out
        out = [dry[i] + sum(g * dry[i - d] for d, g in ((331, 0.55), (757, 0.4), (1279, 0.3)) if i >= d)
               for i in range(n)]
        y = 0.0
        for i in range(n):
            y = 0.35 * out[i] + 0.65 * y
            lcg = (1103515245 * lcg + 12345) % 2 ** 31
            out[i] = 0.32 * y + amplitude * 0.006 * (lcg / 2 ** 30 - 1.0)
    return [max(-32768, min(32767, int(v))) for v in out]


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
        # Upgrade A: session-level roles (independent of thread_of; needed by G's vocative constraints)
        role_sources: Dict[str, str] = {}
        graph_role_evidence: Dict[str, Any] = {}
        speaker_roles = infer_session_speaker_roles(session.turns, pre_role,
                                                    role_priors or getattr(session, "role_priors", None),
                                                    role_sources, graph_role_evidence)
        # Upgrade G: turn voiceprints + lexical Must/Cannot-Link + E2CP -> continuous same-speaker confidence
        # (feeds E's speaker continuity and F's speaker adjacency U; text-only sessions keep the diarizer label)
        vp = constrained_voiceprint_diarize(session.turns, build_gift_graph(session.turns, thread_of), speaker_roles)
        same_spk = vp["same_speaker_prob"] if vp["summary"]["bi_cl_gsn_speaker_weighting"] == "voiceprint_e2cp" \
            else None
        # Upgrade E: Bi-CL session-prototype refinement of the (greedy) upstream threads
        thread_of, bi_cl_summary = bi_cl_disentangle_threads(session.turns, thread_of, speaker_prob=same_spk)

        # Upgrade A (cont.): who->whom
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
        fine_of = {t.turn_id: lb["fine"] for t, lb in zip(session.turns, mintrec_labels)}
        # Upgrade F: GSN 3-iter backward-then-forward flow over E (GIFT reply-to) and U (same speaker)
        by_tid = {t.turn_id: t for t in session.turns}
        gsn = gsn_propagate_session(session.turns, gift, {
            i: gsn_initial_state(by_tid[i], amendments.get(i, {}).get("gift_fine", fine_of[i]),
                                 self.detect_domain_intents) for i in gift["node_ids"]}, speaker_weight=same_spk)
        session.dst_slots["cross_turn_fusion"] = aggregate_cross_turn_slots(
            session.turns, speaker_roles, thread_of, transitions, safety_level, gsn)
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
            "gsn_flow_summary": gsn["summary"],
            "voiceprint_diarization_summary": vp["summary"],
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
                    "refined_speaker": vp["refined_speaker"].get(t.turn_id, t.speaker_id),
                    "voiceprint_source": vp["turn_records"][t.turn_id]["source"],
                    "near_field_score": vp["turn_records"][t.turn_id]["near_field_score"],
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
                rms_energy=float(seg.get("rms_energy") or 0.0),
                zcr_pitch=float(seg.get("zcr_pitch") or 0.0),
                end_time=float(seg.get("end") or 0.0),
                voiceprint=seg.get("voiceprint"),
                voiceprint_subvectors=seg.get("voiceprint_subvectors"),
                voiceprint_source=seg.get("voiceprint_source", ""),
                near_field_score=float(seg.get("near_field_score", -1.0)),
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

def _gsn_unit_tests() -> None:
    """v7.6 unit tests: GSN SQH bounds, sibling fusion via BIF, 3-iter multi-hop reach, U speaker chain."""
    U = UtteranceTurn
    # --- SQH (Eq. 4) in [alpha, 1), monotone in ||d||
    sq = [gsn_sqh(x) for x in (0.0, 0.1, 1.0, 3.0, 50.0, 1e6)]
    assert sq[0] == GSN_ALPHA and all(GSN_ALPHA <= v < 1.0 for v in sq) and sq == sorted(sq), sq
    # --- parallel siblings (GSN Fig 1 / 4a): 2, 3, 4 all reply to 1; 3's content reaches 2 and 4 only via BIF
    sib = [U(1, 0, "speaker_1", "unknown", "今天的個案狀況大家說一下"),
           U(2, 3, "speaker_2", "unknown", "我這邊阿嬤很穩定", addressee="speaker_1"),
           U(3, 6, "speaker_3", "unknown", "我那邊阿伯一直拉我", addressee="speaker_1"),
           U(4, 9, "speaker_4", "unknown", "我這週要排班表", addressee="speaker_1")]
    g = build_gift_graph(sib)
    r = gsn_propagate_session(sib, g, {3: {"issue:X": 1.0}})
    assert r["E"] == [(1, 2), (1, 3), (1, 4)] and r["parallel_siblings"] == {1: [2, 3, 4]}, r["E"]
    assert r["summary"]["parallel_sibling_branches"] == 1 and r["summary"]["speaker_edges_U"] == 0
    w1, o1, h1 = r["states"][1]["issue:X"]
    w2, o2, h2 = r["states"][2]["issue:X"]
    assert (o1, h1, o2, h2) == (3, 1, 3, 2) and r["states"][4]["issue:X"][1:] == (3, 2), r["states"]
    assert 0 < w2 < w1 < 1.0, (w1, w2)            # SQH gates decay per hop; mean over 3 children dilutes
    # --- 3-iter multi-hop: 1 -> 2 -> 3 -> 4 reached, the 4-hop node 5 is not
    chain = [U(i, 2.0 * i, f"speaker_{i}", "unknown", f"第{i}句不同的內容延伸討論") for i in range(1, 6)]
    r = gsn_propagate_session(chain, build_gift_graph(chain), {1: {"issue:Y": 1.0}})
    assert r["E"] == [(1, 2), (2, 3), (3, 4), (4, 5)], r["E"]
    assert [r["states"][i]["issue:Y"][2] for i in (2, 3, 4)] == [1, 2, 3] and "issue:Y" not in r["states"][5]
    assert r["summary"]["multi_hop_paths_3iter"] == 5 and (1, 4, 3) in r["multi_hop_paths"], r["multi_hop_paths"]
    # --- U speaker chain (GSN 3.2.3 / Fig 4b): A-B-A-B-A, speaker_1's stance flows 1 -> 3 -> 5 on U
    ab = [U(i, 2.0 * i, "speaker_1" if i % 2 else "speaker_2", "unknown", f"第{i}輪的意見") for i in range(1, 6)]
    r = gsn_propagate_session(ab, build_gift_graph(ab), {1: {"stance:Z": 1.0}})
    assert r["U"] == [(1, 3), (2, 4), (3, 5)] and r["summary"]["speaker_edges_U"] == 3, r["U"]
    w3, w5 = r["states"][3]["stance:Z"][0], r["states"][5]["stance:Z"][0]
    assert r["states"][5]["stance:Z"][1:] == (1, 2) and 0 < w5 < w3 < 1.0, r["states"][5]
    sm = r["summary"]
    assert (sm["iterations"], sm["alpha"]) == (3, 0.25)
    assert GSN_ALPHA <= sm["mean_lambda_speaker_weight"] < 1 and GSN_ALPHA <= sm["mean_eta_reply_weight"] < 1, sm
    # --- fusion: an off-thread issue reaching the dominant thread via U is imported and linked (iv2 49:15 shape)
    ft = [U(1, 0, "speaker_1", "unknown", "那是因為參數的問題他在做資料搜尋之後不合理", thread_id="T_SIDE"),
          U(2, 4, "speaker_2", "unknown", "我們今天大概就大家先做個小結", thread_id="T_MAIN"),
          U(3, 8, "speaker_1", "unknown", "照服員端要有緊急鈕的功能", thread_id="T_MAIN"),
          U(4, 12, "speaker_2", "unknown", "我們需要有多元問題的選項然後補充服務項目紀錄", thread_id="T_MAIN")]
    th = {t.turn_id: t.thread_id for t in ft}
    gr = build_gift_graph(ft, th)
    gs = gsn_propagate_session(ft, gr)
    roles = {"speaker_1": ROLE_RESEARCHER, "speaker_2": ROLE_SUPERVISOR}
    plain = aggregate_cross_turn_slots(ft, roles, th)
    fused = aggregate_cross_turn_slots(ft, roles, th, gsn=gs)
    assert plain["dominant_threads"] == ["T_MAIN"] and plain["issue_triggers"] == [], plain
    iss = fused["issue_triggers"]
    assert [(d["label"], d["turn_id"], d["via"]) for d in iss] == [("搜尋參數不合理/會議細節發散", 1, "gsn_flow")], iss
    assert {d["label"] for d in fused["proposed_resolutions"]} == {"照服員端緊急鈕+控制APP", "放進多元項目累積經驗"}
    assert all(any(li["origin_turn"] == 1 for li in d["gsn_linked_issues"]) for d in fused["proposed_resolutions"])
    s = DialogueSessionState("S_GSN", "C")
    s.turns = [U(t.turn_id, t.timestamp, t.speaker_id, "unknown", t.text, thread_id=t.thread_id) for t in ft]
    out = XiaoZhaoMPCEngine().analyze_session(s)["gsn_flow_summary"]
    assert set(out) >= {"iterations", "alpha", "reply_edges_E", "speaker_edges_U", "parallel_sibling_branches",
                        "multi_hop_paths_3iter", "mean_eta_reply_weight", "mean_lambda_speaker_weight"}, out


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


def _voiceprint_unit_tests() -> None:
    """v7.7 unit tests: (a) 75%-anchor + 0.25 s collar voiceprints on synthetic low / high voices, (b) E2CP
    Must-Link boost / Cannot-Link suppression, (c) voiceprint_diarization_summary feeding Bi-CL and GSN."""
    import os
    import tempfile
    lo, hi = TEST_VOICE_LOW, TEST_VOICE_HIGH
    close = lambda u, v: len(u) == len(v) and all(abs(a - b) < 1e-9 for a, b in zip(u, v))   # noqa: E731
    # --- (a) sampling rule: collar-trimmed [0.25, 3.75] -> window [0.50, 0.90] of 3.5 s, anchor at 75%
    w, col = voiceprint_windows(0.0, 4.0)
    assert col and len(w) == 1 and close(w[0], (2.0, 3.4, 2.875)), w
    w, col = voiceprint_windows(10.0, 23.0)                      # 13 s > 6 s -> 3 Must-Linked pieces
    piece = 12.5 / 3
    assert col and len(w) == 3 and close(w[0], (10.25 + 0.5 * piece, 10.25 + 0.9 * piece, 10.25 + 0.75 * piece)), w
    w, col = voiceprint_windows(0.0, 0.6)                        # too short for a collar
    assert not col and close(w[0], (0.3, 0.54, 0.45)), w
    # one stream: T1 lo | T2 hi | T3 = 1.4 s hi bleed + lo | T4 hi | T5 lo + 0.2 s hi tail bleed | T6 hi (7 s)
    pcm = (synth_test_voice(lo, 3.0, 1) + synth_test_voice(hi, 3.0, 2) + synth_test_voice(hi, 1.4, 3)
           + synth_test_voice(lo, 2.1, 4) + synth_test_voice(hi, 3.0, 5) + synth_test_voice(lo, 1.0, 6)
           + synth_test_voice(hi, 0.2, 7) + synth_test_voice(hi, 7.0, 8))
    spans = [(0.0, 3.0), (3.0, 6.0), (6.0, 9.5), (9.5, 12.5), (12.5, 13.7), (13.7, 20.7)]
    r = [extract_turn_voiceprint(pcm, 16000, s, e) for s, e in spans]
    cos = lambda i, j: _vp_cos(r[i]["voiceprint"], r[j]["voiceprint"])   # noqa: E731
    assert all(len(x["voiceprint"]) == VP_DIM and x["source"] == VP_SOURCE_WAV for x in r)
    assert abs(sum(v * v for v in r[0]["voiceprint"]) - 1.0) < 1e-3
    assert 100 < r[0]["f0_hz"] < 125 and 205 < r[1]["f0_hz"] < 250, (r[0]["f0_hz"], r[1]["f0_hz"])
    assert min(cos(0, 2), cos(1, 3), cos(1, 5)) > 0.9 > 0.5 > max(cos(0, 1), cos(2, 3), cos(0, 5)), \
        [[round(cos(i, j), 3) for j in range(6)] for i in range(6)]
    assert r[2]["windows"][0][0] >= 7.4 and cos(2, 0) > cos(2, 1) + 0.5, r[2]["windows"]   # anchor skips the bleed
    assert r[4]["collar_applied"] and r[4]["windows"][0][1] <= 13.5 and cos(4, 0) > cos(4, 1) + 0.5, r[4]["windows"]
    assert len(r[5]["sub_voiceprints"]) == 2 and all(_vp_cos(s, r[1]["voiceprint"]) > 0.9
                                                     for s in r[5]["sub_voiceprints"])
    assert extract_turn_voiceprint(pcm, 16000, 0.0, 3.0)["voiceprint"] == r[0]["voiceprint"], "deterministic"
    far = synth_test_voice(hi, 12.6, 11, far_field=True)        # same high voice from 1-3 m (reverb, damped highs)
    rf = extract_turn_voiceprint(far, 16000, 0.0, 2.5)
    assert r[1]["near_field_score"] > rf["near_field_score"] + 0.1, (r[1]["near_field_score"], rf["near_field_score"])
    px = extract_turn_voiceprint(None, rms_energy=2500.0, zcr_pitch=0.07, text="我們先設計一個流程", duration=3.0)
    assert px["source"] == VP_SOURCE_PROXY_AC and px["confidence"] == 0.25 and len(px["voiceprint"]) == VP_DIM
    assert px == extract_turn_voiceprint(None, rms_energy=2500.0, zcr_pitch=0.07, text="我們先設計一個流程", duration=3.0)
    pt = extract_turn_voiceprint(None, text="好", duration=1.0)
    assert pt["source"] == VP_SOURCE_PROXY_TEXT and pt["confidence"] == 0.0 and pt["near_field_score"] == -1.0

    # --- (b) E2CP: nodes 0, 1 = speaker X, 2, 3 = speaker Y; the raw affinity confuses 0 with 2
    a = [[1.0, .60, .80, .55], [.60, 1.0, .62, .58], [.80, .62, 1.0, .70], [.55, .58, .70, 1.0]]
    z = [[0.0] * 4 for _ in range(4)]
    for i, j, val in ((0, 1, 1.0), (2, 3, 1.0), (0, 2, -1.0)):
        z[i][j] = z[j][i] = val
    ah, q = e2cp_adjust_affinity(a, z)
    assert q[0][1] > 0 > q[0][2] and ah[0][1] > a[0][1] and ah[2][3] > a[2][3] and ah[0][2] < a[0][2], (q, ah)
    assert abs(ah[0][1] - (1 - (1 - q[0][1]) * (1 - a[0][1]))) < 1e-12 and abs(ah[0][2] - (1 + q[0][2]) * a[0][2]) < 1e-12
    intra = lambda m: (m[0][1] + m[2][3]) / 2                     # noqa: E731
    inter = lambda m: (m[0][2] + m[0][3] + m[1][2] + m[1][3]) / 4   # noqa: E731
    assert intra(ah) > intra(a) and inter(ah) < inter(a), (intra(a), intra(ah), inter(a), inter(ah))
    raw_lab, e2cp_lab = _vp_spectral(_vp_refine(a), a, 2, 3)[0], _vp_spectral(_vp_refine(ah), ah, 2, 3)[0]
    assert raw_lab[0] == raw_lab[2] and e2cp_lab[0] == e2cp_lab[1] != e2cp_lab[2] == e2cp_lab[3], (raw_lab, e2cp_lab)
    assert e2cp_adjust_affinity(a, [[0.0] * 4 for _ in range(4)])[0] == a, "no constraints: A^ = A"

    # --- (c) session: speaker_1 = dry low voice (headset), speaker_2 = far-field high voice; turn 7 is the far
    # high voice mislabelled speaker_1 by the diarizer
    ex = lambda src, s, e: extract_turn_voiceprint(src, 16000, s, e)   # noqa: E731
    recs = {1: r[0], 2: ex(far, 0.0, 2.5), 3: r[2], 4: ex(far, 2.5, 3.1), 5: r[4], 6: ex(far, 3.1, 10.1),
            7: ex(far, 10.1, 12.6), 8: ex(pcm, 0.5, 2.9)}
    script = {1: ("speaker_1", 0.0, 3.0, "我們後端的知識庫系統可以幫大家裝綁定碼"),
              2: ("speaker_2", 3.2, 5.7, "那這個居督可以綁哪一個居服員的個案嗎？"),
              3: ("speaker_1", 5.9, 9.4, "可以，我們後端的系統會幫居督綁定"),             # QA Cannot-Link 2-3
              4: ("speaker_2", 9.4, 10.0, "嗯嗯"),
              5: ("speaker_1", 10.4, 11.6, "然後每個居督都有自己的帳號"),                 # latch Must-Link 3-5
              6: ("speaker_2", 11.8, 18.8, "我覺得要先安撫居服員，然後回報居督，事後再回報主任"),   # 7 s split
              7: ("speaker_1", 19.0, 21.5, "那督導這邊派案的時候要怎麼處理"),
              8: ("speaker_1", 21.7, 24.1, "我們後端可以做一個派案的介面")}             # self-reference 1-3-8
    turns = [UtteranceTurn(k, ts, sp, "unknown", tx, end_time=te, rms_energy=recs[k]["rms_energy"],
                           zcr_pitch=recs[k]["zcr_pitch"], voiceprint=recs[k]["voiceprint"],
                           voiceprint_subvectors=recs[k]["sub_voiceprints"], voiceprint_source=recs[k]["source"],
                           near_field_score=recs[k]["near_field_score"]) for k, (sp, ts, te, tx) in script.items()]
    res = XiaoZhaoMPCEngine().analyze_session(DialogueSessionState("S_VP", "C", turns=turns))
    v = res["voiceprint_diarization_summary"]
    assert v["voiceprint_mode"] == VP_SOURCE_WAV and v["turns_with_voiceprint"] == 8 and v["long_turn_splits"] == 1, v
    assert v["constraints_by_reason"] == {"discourse_latch": 1, "lexical_self_consistency": 2, "long_turn_split": 1,
                                          "qa_turn_taking": 1}, v["constraints"]
    assert (v["must_link_constraints"], v["cannot_link_constraints"]) == (4, 1), v
    assert v["mean_intra_speaker_affinity_e2cp"] > v["mean_intra_speaker_affinity_raw"] \
        > v["mean_inter_speaker_affinity_raw"], v
    assert v["near_field_primary_speaker"] == "speaker_1" and v["estimated_speakers"] == 2, v
    assert v["diarization_consistency_score"] == 0.875 and v["relabel_candidates"] == [
        {"turn_id": 7, "diarizer": "speaker_1", "voiceprint_speaker": "speaker_2"}], v
    td = {d["turn_id"]: d for d in res["turns_detail"]}
    assert td[7]["refined_speaker"] == "speaker_2" and td[1]["voiceprint_source"] == VP_SOURCE_WAV
    blob = json.dumps(res, ensure_ascii=False)
    assert '"voiceprint": [' not in blob and "sub_voiceprints" not in blob, "payload never carries raw vectors"
    gs, bs = res["gsn_flow_summary"], res["bi_cl_disentangle_summary"]
    assert gs["speaker_edge_weighting"] == "voiceprint_e2cp" and 0.5 < gs["mean_u_speaker_confidence"] < 1.0, gs
    assert bs["speaker_continuity_weighting"] == "voiceprint_e2cp", bs
    d = constrained_voiceprint_diarize(turns, build_gift_graph(turns), res["speaker_roles"])
    prob = d["same_speaker_prob"]
    assert prob(1, 3) > 0.95 > max(prob(5, 7), prob(7, 8)) and 0.4 < prob(6, 7) <= 0.5, \
        (prob(1, 3), prob(5, 7), prob(7, 8), prob(6, 7))   # mislabelled U edges lose weight; 6-7 gains it
    calls: List[Tuple[int, int]] = []
    spy = lambda i, j: calls.append((i, j)) or prob(i, j)   # noqa: E731
    bi_cl_disentangle_threads(turns, speaker_prob=spy)
    n_bicl = len(calls)
    g = build_gift_graph(turns)
    weighted, hard = gsn_propagate_session(turns, g, speaker_weight=spy), gsn_propagate_session(turns, g)
    assert n_bicl > 0 and len(calls) > n_bicl, "Bi-CL and GSN both consume the same-speaker confidence"
    assert hard["summary"]["mean_u_speaker_confidence"] == 1.0 and \
        weighted["summary"]["mean_lambda_speaker_weight"] < hard["summary"]["mean_lambda_speaker_weight"]
    # JSON-only replay of the same segments (Whisper format with voiceprint fields) reproduces the verdict
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "vp_segments.json")
        Path(path).write_text(json.dumps({"segments": [
            {"start": t.timestamp, "end": t.end_time, "speaker": t.speaker_id, "text": t.text,
             "rms_energy": t.rms_energy, "zcr_pitch": t.zcr_pitch, "voiceprint": t.voiceprint,
             "voiceprint_subvectors": t.voiceprint_subvectors, "voiceprint_source": t.voiceprint_source,
             "near_field_score": t.near_field_score} for t in turns]}, ensure_ascii=False), encoding="utf-8")
        replay = XiaoZhaoMPCEngine().analyze_session(load_from_whisper_json(path))["voiceprint_diarization_summary"]
    assert replay["turns_with_voiceprint"] == 8 and replay["relabel_candidates"] == v["relabel_candidates"], replay
    # text-only: same lexical constraints are counted, but Bi-CL / GSN keep the hard diarizer label
    rt = XiaoZhaoMPCEngine().analyze_session(DialogueSessionState("S_TXT", "C", turns=[
        UtteranceTurn(t.turn_id, t.timestamp, t.speaker_id, "unknown", t.text, end_time=t.end_time) for t in turns]))
    vt = rt["voiceprint_diarization_summary"]
    assert vt["voiceprint_mode"] == VP_SOURCE_PROXY_TEXT and vt["turns_with_voiceprint"] == 0, vt
    assert (vt["must_link_constraints"], vt["cannot_link_constraints"]) == (3, 1) and \
        vt["diarization_consistency_score"] is None, vt
    assert vt["bi_cl_gsn_speaker_weighting"] == "diarizer_label" and \
        rt["gsn_flow_summary"]["speaker_edge_weighting"] == "hard_0_1"
    # 2-node session and the chronological block path (> VP_BLOCK_NODES nodes)
    pair = constrained_voiceprint_diarize(turns[:2])["summary"]
    assert pair["estimated_speakers"] == 2 and pair["diarization_consistency_score"] == 1.0, pair
    big = [UtteranceTurn(k, 2.0 * k, f"speaker_{k % 3 + 1}", "unknown", "我們先設計一個流程", end_time=2.0 * k + 1.8,
                         rms_energy=1000.0 + 300.0 * (k % 3), zcr_pitch=0.05 + 0.02 * (k % 3)) for k in range(1, 61)]
    db = constrained_voiceprint_diarize(big)
    assert db["summary"]["blocks"] == 2 and db["summary"]["voiceprint_mode"] == VP_SOURCE_PROXY_AC, db["summary"]
    assert 0.0 <= db["same_speaker_prob"](1, 59) <= 1.0 and db["same_speaker_prob"](3, 6) > db["same_speaker_prob"](3, 4)


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
    _ans = [UtteranceTurn(1, 0.0, "speaker_1", "unknown", "案家要我代購成藥", thread_id="T1"),
            UtteranceTurn(2, 1.0, "speaker_2", "unknown", "不用啦", thread_id="T1"),
            UtteranceTurn(3, 2.0, "speaker_2", "unknown", "請你用APP回報", thread_id="T1")]
    _ans_tr = [{"type": "REQUEST_TO_REFUSAL", "thread_id": "T1", "from_turn": 1, "to_turn": 2}]
    assert aggregate_cross_turn_slots(_ans, {}, transitions=_ans_tr)["consensus_state"] == "CONSENSUS_ACTIONABLE", \
        "a resolution proposed after the refusal answers it"
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
    _gsn_unit_tests()
    _voiceprint_unit_tests()
    assert all(r["voiceprint_diarization_summary"]["bi_cl_gsn_speaker_weighting"] == "diarizer_label"
               for r in results), "text-only benchmarks keep the diarizer label"
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
