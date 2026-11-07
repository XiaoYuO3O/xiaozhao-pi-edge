#!/usr/bin/env python3
"""XiaoZhao interview analyzer for long (~30 min) unedited multi-party recordings.

Pipeline: mp3/m4a -> 16k mono WAV -> cached Whisper STT -> seek-based acoustic speaker
clustering (ZCR/RMS) -> windowed addressee inference + thread disentanglement -> golden hotspots.
Privacy: no voiceprint DB; speakers are anonymous speaker_1/2/3 clusters per recording.
"""
import argparse
import bisect
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from src.mpc_core import (AGREEMENT_TOKENS_RE, AmbientIntentPreFetcher, DialogueSessionState,
                              UtteranceTurn, XiaoZhaoMPCEngine, classify_mintrec_taxonomy, infer_session_speaker_roles,
                              is_agreement_only)
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mpc_core import (AGREEMENT_TOKENS_RE, AmbientIntentPreFetcher, DialogueSessionState,
                          UtteranceTurn, XiaoZhaoMPCEngine, classify_mintrec_taxonomy, infer_session_speaker_roles,
                          is_agreement_only)

ROOT = Path(__file__).resolve().parent.parent
AUDIO_TMP = ROOT / "audio_tmp"
OUT_DIR = ROOT / "transcripts_out"

TARGET_SR = 16000
N_SPEAKERS = 3
FEATURE_MAX_SEC = 5.0          # max audio read per segment (bounded memory)
MERGE_GAP_S = 1.5
MERGE_MAX_CHARS = 120
INERTIA_MAX_DUR_S = 2.2        # fragment = shorter than this ...
INERTIA_MAX_CHARS = 10         # ... or fewer chars than this
INERTIA_GAP_S = 0.8
INERTIA_IQR_MULT = 1.2
LOG_RMS_WEIGHT = 0.5           # weight of normalized log-RMS vs normalized ZCR in speaker score
QA_LATCH_GAP_S = 4.0
ABA_GAP_S = 3.0
DISCOURSE_LATCH_GAP_S = 5.0
THREAD_JACCARD_MIN = 0.06
STRONG_SHARED_MIN = 2          # older thread must share >= this many keywords to beat the latch
BACKCHANNEL_MAX_CHARS = 5
BACKCHANNEL_HARD_MAX_CHARS = 16  # longer turns are never backchannel noise

WHISPER_INITIAL_PROMPT = ("繁體中文台灣長照機構訪談：嘉安、照服員、居服員、居督、督導、知識庫、法規、"
                          "長照基本法、小照、個案、家屬。")
ASR_FIXES = (("照不遠", "照服員"), ("照覆點", "照服員"), ("照服點", "照服員"), ("家安", "嘉安"),
             ("知識苦", "知識庫"), ("天馬行公", "天馬行空"),
             # real telemetry homophones (longer phrases before their substrings)
             ("頁島世界", "跌倒事件"), ("頁島", "跌倒"), ("暴力世界", "暴力事件"), ("暴力行朝", "暴力性騷"),
             ("慶祝的", "緊急的"), ("三口處理", "傷口處理"), ("都島", "督導"), ("BN24", "BA24"),
             ("BAN1", "BA01"), ("BN1", "BA01"), ("S7項目", "SC項目"), ("見面上", "介面上"),
             ("教界或傳審", "交接或傳承"),
             # 32-min telemetry homophones ("安家"->"案家" is a guarded regex in normalize_asr_text)
             ("褲子整個都撕掉", "褲子整個都濕掉"), ("蜜蜂遮了", "蜜蜂螫了"), ("累接下來", "累積下來"),
             ("照顧你", "照服員"), ("安正定", "安撫鎮定"), ("宗教信員", "宗教信仰"),
             ("一路機就", "一路急救"), ("整個廟", "整個尿"), ("玩衣服", "換衣服"), ("家屬重視", "家屬中風"),
             ("到院前懷疑他", "到院前死亡"), ("黃金很開歌", "黃金很臭"), ("公民醫院", "公立醫院"),
             ("躺白說", "坦白說"), ("心臟替的", "心臟停了"), ("咒事", "猝逝"), ("暴力性情", "暴力性騷"),
             ("緊急通報女", "緊急通報鈕"), ("緊急扭", "緊急鈕"), ("緊急膩", "緊急鈕"), ("把揮上", "法規上"),
             ("瑞格", "RAG"), ("加安", "嘉安"), ("加速要我", "家屬要我"), ("EBA16", "BA16"), ("白花", "白話"),
             ("天馬行光", "天馬行空"), ("見面出來", "介面出來"), ("見面裡面", "介面裡面"),
             ("照片第一線", "照服員第一線"), ("照五元", "照服員"), ("照護原", "照護員"), ("劇服人", "居服員"),
             ("居度", "居督"), ("居都端", "居督端"), ("主會端", "主任端"), ("主論", "主任"),
             # 52-min (interview2) telemetry homophones
             ("照古緣", "照服員"), ("照源進行", "照服員進行"), ("居夫人", "居服員"), ("居屋所督導", "居服督導"),
             ("居屬或是", "居督或是"), ("度導", "督導"), ("孤島", "督導"), ("剛剛主導", "剛剛督導"),
             ("主任跟主導", "主任跟督導"), ("各位主導", "各位督導"), ("鎮列行賣客風", "陣列型麥克風"),
             ("鎮列", "陣列"), ("賣客風", "麥克風"), ("賣者風", "麥克風"), ("賣克風", "麥克風"), ("收尋", "搜尋"),
             ("娜進", "納進"), ("把它漸進來", "把它納進來"), ("漸進來", "建進來"), ("真實的性命", "真實的姓名"),
             ("真實的信命", "真實的姓名"), ("那個性命是", "那個姓名是"), ("的性命是", "的姓名是"),
             ("照顧人情", "照服員"), ("照顧人", "照服員"), ("以資的氣候", "已知的架構"), ("馬上一定要手", "馬上一定要回"), ("色一個緊急", "設一個緊急"), ("四頭再手", "事後再回"), ("不是立即信", "不是立即性"), ("前照片", "請照服員"), ("照顧有", "照服員"), ("照護院", "照服員"), ("均度", "居督"), ("劇都", "居督"), ("每個居住", "每個居督"), ("綁居住", "綁居督"), ("居住管", "居督管"), ("房地嘛", "綁定碼"), ("新版本的想像", "新版本的小照"), ("新寶寶", "刑法法條"), ("g服務員", "居服員"), ("g 服務員", "居服員"), ("照護員", "照服員"), ("做的小節", "做個小結"), ("下次回憶", "下次會議"), ("暗族", "案主"), ("暗家", "案家"), ("咒勢", "猝逝"), ("知識負系統", "知識庫系統"), ("在性命上", "在姓名上"),
             ("耐群組", "LINE群組"), ("耐群", "LINE群"), ("活潰", "回饋"), ("知識不見完整", "知識庫建置完整"),
             ("摸大鼓", "摸大腿"), ("SSC", "SC"), ("S7的項目", "SC的項目"), ("Ga跟S7", "GA跟SC"),
             ("法徽", "法規"), ("收煉", "收斂"), ("出稿", "初稿"), ("省月", "審閱"), ("SOPP", "SOP"),
             # interview3 (whisper-small era) homophones
             ("篤島", "督導"), ("督島", "督導"), ("小皂", "小照"),
             # 居都 is only rewritten in unambiguous compounds (bare 居都 would break 鄰居都/家居都)
             ("綁居都", "綁居督"), ("每個居都", "每個居督"), ("劇毒館", "居督管"), ("居都館", "居督管"),
             ("居都身上", "居督身上"))
_ASR_LOOP_WORDS_RE = re.compile(r"(對啦|對啊|對|好|嗯|喔|是|可以|我懂)\1{2,}")  # same token x3+
_ASR_LOOP_SUBSTR_RE = re.compile(r"(.{1,3}?)\1{3,}")  # 1-3 char unit repeated >= 4x
# 找居住/跟居住/... -> 居督, but keep real words: 居住環境, 居住地, 居住在, 居住的, 居住於
_JUDU_CTX_RE = re.compile(r"(?<=[找跟請給問那個是由])居住(?!環境|地|在|的|於|條件|品質|權)")
_ELLIPSIS = r"(?:\.{2,}|…+)"
# 他...他... / 我…我…我 -> 他 / 我 (stutter unit 1-3 chars separated by ellipses)
_ELLIPSIS_STUTTER_RE = re.compile(rf"([^\s.…]{{1,3}}?){_ELLIPSIS}(?:\1{_ELLIPSIS})*\1(?!\1){_ELLIPSIS}?")
_ASR_SEP_LOOP_RE = re.compile(r"([^\s,，、。]{1,6})(?:[,，、\s]+\1){2,}")  # 他開始,他開始,他開始
ASR_LOOP_RUN_MIN = 3           # >= 3 identical consecutive segments = Whisper loop (對啦x50, 要x45)
_AGREE_SEP = r"[\s,，、。.!！~～…]"
_AGREE_TOK_RE = re.compile(AGREEMENT_TOKENS_RE)
_AGREE_RUN_RE = re.compile(rf"{AGREEMENT_TOKENS_RE}(?:{_AGREE_SEP}*{AGREEMENT_TOKENS_RE})*")
LOOP_UNIQUE_MAX_CHARS = 5
DOMAIN_TERMS = ("長照基本法", "照服員", "居服員", "知識庫", "嘉安", "居督", "督導", "法規", "長照",
                "小照", "個案", "家屬", "機構", "主任", "案家", "緊急鈕", "介面", "麥克風", "照護員", "多元")
AGREEMENT_TOKENS = ("對,對", "對對", "嗯嗯", "好,好", "對啊")
DISCOURSE_CONNECTORS = ("就是", "因為", "但是", "所以", "然後", "這些", "這樣", "其實", "而且", "那", "像")
STOP_WORDS = ("我們", "一個", "可以", "就是", "這個", "那個", "因為", "所以", "然後", "但是", "什麼",
              "他們", "你們", "沒有", "這樣", "其實", "而且", "這些")
STOP_CHARS = set("的了是我你他她它在有和也都就這那嗯對啊喔呢吧嗎呀哦哈很要會不說")
HOTSPOT_WIN_S = 60.0
HOTSPOT_STEP_S = 15.0
HOTSPOT_CHARS_PER_POINT = 60.0  # +1 score per 60 substantive chars (favours real talk over 1-word fragments)
HOTSPOT_CHARS_CAP = 5.0

QUESTION_MARKERS = ("?", "？", "嗎", "呢", "什麼", "怎麼", "對不對")
BACKCHANNELS = ("嗯", "對", "哈", "喔", "好", "是", "啊", "ok", "OK")
ROLE_CUES = ("主持人", "老師", "醫生", "醫師", "護理師", "社工", "居服員", "阿嬤", "阿公", "阿姨",
             "先生", "小姐", "媽媽", "爸爸")
SELF_INTRO = ("我是", "我叫")
NOISE_THREAD = "T_BACKCHANNEL_NOISE"


def mmss(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


# ---------------------------------------------------------------- audio I/O
def _is_16k_mono_wav(path: str) -> bool:
    try:
        with wave.open(path, "rb") as w:
            return w.getframerate() == TARGET_SR and w.getnchannels() == 1 and w.getsampwidth() == 2
    except (wave.Error, EOFError, OSError):
        return False


def convert_to_16k_wav(input_path: str) -> str:
    """Convert mp3/m4a/etc. to audio_tmp/<stem>_16k.wav (afconvert on macOS, ffmpeg fallback)."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)
    if input_path.lower().endswith(".wav") and _is_16k_mono_wav(input_path):
        return input_path
    AUDIO_TMP.mkdir(parents=True, exist_ok=True)
    out = AUDIO_TMP / f"{Path(input_path).stem}_16k.wav"
    if out.exists() and out.stat().st_size > 44:
        return str(out)
    cmds = []
    if platform.system() == "Darwin" and shutil.which("afconvert"):
        cmds.append(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", input_path, str(out)])
    if shutil.which("ffmpeg"):
        cmds.append(["ffmpeg", "-y", "-loglevel", "error", "-i", input_path, "-ar", str(TARGET_SR), "-ac", "1",
                     "-sample_fmt", "s16", str(out)])
    if not cmds:
        raise RuntimeError("Neither afconvert nor ffmpeg is available for audio conversion")
    last_err = ""
    for cmd in cmds:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and out.exists() and out.stat().st_size > 44:
            return str(out)
        last_err = (res.stderr or res.stdout).strip()
    raise RuntimeError(f"audio conversion failed: {last_err}")


def wav_duration(wav_path: str) -> float:
    with wave.open(wav_path, "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def _segment_features(wf: wave.Wave_read, rate: int, nframes: int, nch: int,
                      start: float, end: float) -> Tuple[float, float]:
    """Seek + bounded read (<= 5 s); returns (zero-crossing rate, RMS). Never loads the full file."""
    pos = min(max(0, int(start * rate)), max(0, nframes - 1))
    n = int(min(max(end - start, 0.0), FEATURE_MAX_SEC) * rate)
    n = min(n, nframes - pos)
    if n <= 1:
        return 0.0, 0.0
    wf.setpos(pos)
    raw = wf.readframes(n)
    raw = raw[: len(raw) - (len(raw) % 2)]
    zc, sq, cnt, prev = 0, 0.0, 0, 0
    for i, (s,) in enumerate(struct.iter_unpack("<h", raw)):
        if nch > 1 and i % nch:
            continue  # first channel only
        if (s >= 0) != (prev >= 0):
            zc += 1
        prev = s
        sq += s * s
        cnt += 1
    if cnt < 2:
        return 0.0, 0.0
    return zc / cnt, math.sqrt(sq / cnt)


def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = q * (len(sorted_vals) - 1)
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def _join_text(a: str, b: str) -> str:
    a, b = a.strip(), b.strip()
    if not a or not b:
        return a or b
    sep = " " if (a[-1].isascii() and a[-1].isalnum() and b[0].isascii() and b[0].isalnum()) else ""
    return a + sep + b


def _speaker_of(t: dict) -> str:
    return str(t.get("speaker") or t.get("speaker_id") or "speaker_1")


def collapse_asr_loops(segments: List[dict]) -> List[dict]:
    """Collapse >= ASR_LOOP_RUN_MIN consecutive segments with identical text (Whisper repetition
    hallucination, e.g. 對啦 x50 at 1 s spacing) into ONE segment spanning the run."""
    groups: List[List[int]] = []
    for idx, s in enumerate(segments):
        key = _clean(str(s.get("text", "")))
        if (groups and key and key == _clean(str(segments[groups[-1][0]].get("text", "")))
                and float(s["start"]) - float(segments[groups[-1][-1]]["end"]) < MERGE_GAP_S):
            groups[-1].append(idx)
        else:
            groups.append([idx])
    out: List[dict] = []
    for g in groups:
        if len(g) >= ASR_LOOP_RUN_MIN:
            first, last = segments[g[0]], segments[g[-1]]
            out.append({**first, "end": float(last["end"]), "src_idx": list(g), "asr_loop_collapsed": len(g)})
        else:
            out.extend({**segments[i], "src_idx": [i]} for i in g)
    return out


def _is_agree_loop(toks: List[str], has_sep: bool) -> bool:
    """A run of agreement tokens counts as a loop if a token repeats or >= 3 tokens; '好好' (adverb) is kept."""
    if len(toks) < 2:
        return False
    if len(toks) == 2 and not has_sep and toks[0] == toks[1] == "好":
        return False  # 好好休息 / 好好照顧
    return len(toks) >= 3 or has_sep or toks[0] == toks[1]


def _clean_turn_text(text: str) -> str:
    """Deterministic post-merge cleaner: collapse repeated agreement tokens (對啦對啦.. -> 對啦) and strip
    leading/trailing agreement loops from substantive turns. Agreement-only text -> its first token."""
    text = _fix_homophones(_strip_stutter(re.sub(r"\s+", " ", str(text or "")).strip()))
    if not text:
        return text
    if is_agreement_only(text):
        return _AGREE_TOK_RE.search(text).group(0)
    pieces: List[str] = []
    pos = 0
    for m in _AGREE_RUN_RE.finditer(text):
        start, end = m.start(), m.end()
        toks = _AGREE_TOK_RE.findall(text[start:end])
        # a single-char token glued to the previous word belongs to it (但是/還好/不對), not to the run
        if start > 0 and not re.match(_AGREE_SEP, text[start - 1]) and len(toks[0]) == 1:
            start += 1
            toks = toks[1:]
        run = text[start:end]
        has_sep = bool(re.search(_AGREE_SEP, run))
        if not _is_agree_loop(toks, has_sep):
            continue
        pieces.append(text[pos:start])
        at_edge = not text[:start].strip(" ,，、。") or not text[end:].strip(" ,，、。.!！~～…")
        if not at_edge:
            dedup = [t for i, t in enumerate(toks) if i == 0 or t != toks[i - 1]]
            pieces.append(("，" if has_sep else "").join(dedup))
        pos = end
    pieces.append(text[pos:])
    out = "".join(pieces).strip(" ,，、")
    return out or _AGREE_TOK_RE.search(text).group(0)


def merge_same_speaker(segments: List[dict]) -> List[dict]:
    """Consolidate same-speaker fragments (gap < 1.5 s). Pure agreement/backchannel segments are NEVER merged
    into a neighbour (no 對啦對啦.. chains, no 對啦 glued onto substantive turns)."""
    merged: List[dict] = []
    for idx, s in enumerate(segments):
        text = str(s.get("text", "")).strip()
        prev = merged[-1] if merged else None
        spk = _speaker_of(s)
        src = list(s.get("src_idx", [idx]))
        if (prev and prev["speaker"] == spk and float(s["start"]) - prev["end"] < MERGE_GAP_S
                and len(prev["text"]) + len(text) < MERGE_MAX_CHARS
                and not is_agreement_only(text) and not is_agreement_only(prev["text"])):
            prev["end"] = float(s["end"])
            prev["text"] = _join_text(prev["text"], text)
            prev["merged_from"].extend(src)
            if s.get("asr_loop_collapsed"):
                prev["asr_loop_collapsed"] = prev.get("asr_loop_collapsed", 0) + s["asr_loop_collapsed"]
        else:
            merged.append({"start": float(s["start"]), "end": float(s["end"]), "speaker": spk,
                           "speaker_id": spk, "text": text, "merged_from": src,
                           **({"zcr": s["zcr"], "rms": s["rms"]} if "zcr" in s else {}),
                           **({"asr_loop_collapsed": s["asr_loop_collapsed"]} if s.get("asr_loop_collapsed") else {})})
    for t in merged:  # cross-fragment homophones + agreement-loop cleanup on the consolidated turn
        t["text"] = _clean_turn_text(normalize_asr_text(t["text"]))
    return merged


def _robust_norm(vals: List[float]) -> List[float]:
    """(x - median) / IQR, IQR-guarded so a constant feature contributes 0."""
    s = sorted(vals)
    med, iqr = _quantile(s, 0.5), _quantile(s, 0.75) - _quantile(s, 0.25)
    return [(v - med) / iqr if iqr > 1e-9 else 0.0 for v in vals]


def _apply_speaker_inertia(segments: List[dict], scores: List[float], bins: List[int],
                           voiced_mask: List[bool]) -> List[int]:
    """Short fragments right after another segment keep the previous speaker unless acoustically far."""
    vs = sorted(sc for sc, v in zip(scores, voiced_mask) if v) or sorted(scores)
    limit = INERTIA_IQR_MULT * (_quantile(vs, 0.75) - _quantile(vs, 0.25))
    out = list(bins)
    for i in range(1, len(segments)):
        s, p = segments[i], segments[i - 1]
        dur = float(s["end"]) - float(s["start"])
        short = dur < INERTIA_MAX_DUR_S or len(_clean(str(s.get("text", "")))) < INERTIA_MAX_CHARS
        gap = float(s["start"]) - float(p["end"])
        if short and gap < INERTIA_GAP_S and abs(scores[i] - scores[i - 1]) <= limit:
            out[i] = out[i - 1]
    return out


def assign_speakers_by_acoustics(wav_path: str, segments: list) -> list:
    """Normalized ZCR+logRMS quantile clustering into speaker_1..3, temporal inertia, then merging."""
    if not segments:
        return []
    segments = collapse_asr_loops(segments)
    feats: List[Tuple[float, float]] = []
    with wave.open(wav_path, "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM WAV")
        rate, nframes, nch = wf.getframerate(), wf.getnframes(), wf.getnchannels()
        for s in segments:
            feats.append(_segment_features(wf, rate, nframes, nch, float(s["start"]), float(s["end"])))

    # combined score: robust-normalized ZCR + weighted robust-normalized log-RMS
    zn = _robust_norm([z for z, _ in feats])
    ln = _robust_norm([math.log1p(r) for _, r in feats])
    scores = [a + LOG_RMS_WEIGHT * b for a, b in zip(zn, ln)]

    # quantile cut points over voiced segments (silent ones fall back to median bin)
    rms_sorted = sorted(r for _, r in feats)
    floor = 0.1 * _quantile(rms_sorted, 0.5)
    voiced_mask = [r > floor for _, r in feats]
    voiced = sorted(sc for sc, v in zip(scores, voiced_mask) if v) or sorted(scores)
    k = min(N_SPEAKERS, len(voiced))
    cuts = sorted({_quantile(voiced, i / k) for i in range(1, k)})
    median_bin = bisect.bisect_right(cuts, _quantile(voiced, 0.5))
    bins = [bisect.bisect_right(cuts, sc) if v else median_bin for sc, v in zip(scores, voiced_mask)]
    bins = _apply_speaker_inertia(segments, scores, bins, voiced_mask)

    # relabel in order of first appearance -> speaker_1 is whoever talks first
    order: Dict[int, str] = {}
    out = []
    for s, b, (z, r) in zip(segments, bins, feats):
        if b not in order:
            order[b] = f"speaker_{len(order) + 1}"
        out.append({"start": float(s["start"]), "end": float(s["end"]), "text": str(s.get("text", "")),
                    "speaker": order[b], "zcr": round(z, 4), "rms": round(r, 1), "src_idx": s["src_idx"],
                    **({"asr_loop_collapsed": s["asr_loop_collapsed"]} if s.get("asr_loop_collapsed") else {})})
    return merge_same_speaker(out)


# ------------------------------------------------------- disentanglement
def _clean(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _is_question(text: str) -> bool:
    return any(m in text for m in QUESTION_MARKERS)


def _has_domain_noun(text: str) -> bool:
    return any(d in text for d in DOMAIN_TERMS)


def _is_backchannel(text: str) -> bool:
    if len(text) > BACKCHANNEL_HARD_MAX_CHARS:
        return False  # long turns carry content even with "對,對"/"嗯嗯"
    c = _clean(text)
    if _is_question(text):
        return False  # short questions ("幾點開始？") carry content
    if len(c) <= BACKCHANNEL_MAX_CHARS or all(ch in "".join(BACKCHANNELS) for ch in c):
        return True
    flat = re.sub(r"[，、,\s]+", ",", text)
    return (any(tok in flat or tok.replace(",", "") in c for tok in AGREEMENT_TOKENS)
            and not _has_domain_noun(text))


def _keywords(text: str) -> set:
    """Content unigrams (domain terms, ASCII words) + CJK bigrams without stopwords."""
    t = _clean(text).lower()
    for sw in STOP_WORDS:
        t = t.replace(sw, "|")
    kws = {d for d in DOMAIN_TERMS if d in text}
    kws |= {w for w in re.findall(r"[a-z0-9]{2,}", t)}
    kws |= {t[i:i + 2] for i in range(len(t) - 1)
            if "|" not in t[i:i + 2] and not (set(t[i:i + 2]) & STOP_CHARS)
            and not t[i:i + 2].isascii()}
    return kws


def _has_connector(text: str) -> bool:
    return any(c in text for c in DISCOURSE_CONNECTORS)


def _infer_addressee(i: int, turns: List[dict], role_map: Dict[str, str]) -> Tuple[str, str]:
    cur = turns[i]
    spk, text = cur["speaker"], cur["text"]
    # 1) role cue (vocative), resolved to a speaker who self-introduced with that role
    for cue in ROLE_CUES:
        if cue in text[:12] and not any(text.startswith(p) for p in SELF_INTRO):
            target = role_map.get(cue)
            if target and target != spk:
                return target, f"role_cue:{cue}"
    # 2) Question-Answer latch: answering the latest question by someone else within 4 s
    for j in range(i - 1, -1, -1):
        p = turns[j]
        if cur["start"] - p["end"] > QA_LATCH_GAP_S:
            break
        if p["speaker"] != spk and _is_question(p["text"]):
            return p["speaker"], "qa_latch"
    if i >= 1:
        prev = turns[i - 1]
        gap = cur["start"] - prev["end"]
        if prev["speaker"] != spk and gap < ABA_GAP_S:
            # 3) ABA turn-taking: A-B-A -> A addresses B; otherwise adjacency to previous speaker
            if i >= 2 and turns[i - 2]["speaker"] == spk:
                return prev["speaker"], "aba_turn_taking"
            return prev["speaker"], "adjacency"
    for cue in ROLE_CUES:
        if cue in text and not any(text.startswith(p) for p in SELF_INTRO):
            return f"ROLE:{cue}", f"role_cue:{cue}"
    return "ALL", "broadcast"


def general_mpc_disentangle(turns: list, window_sec: float = 300.0) -> dict:
    """Addressee inference + thread disentanglement over sliding windows of a long interview."""
    turns = sorted(({**t, "speaker": _speaker_of(t), "speaker_id": _speaker_of(t)} for t in turns),
                   key=lambda t: float(t["start"]))
    role_map: Dict[str, str] = {}
    threads: Dict[str, dict] = {}   # tid -> {"last": t, "kw": cumulative keyword set}
    out: List[dict] = []
    n_topics = 0
    last_content: Optional[str] = None
    prev_end: Optional[float] = None
    for i, t in enumerate(turns):
        text, start = t["text"], float(t["start"])
        for p in SELF_INTRO:
            if p in text:
                for cue in ROLE_CUES:
                    if cue in text.split(p, 1)[1][:8]:
                        role_map.setdefault(cue, t["speaker"])
        addressee, reason = _infer_addressee(i, turns, role_map)

        gap = start - prev_end if prev_end is not None else float("inf")
        prev_end = float(t["end"])
        raw_text = text
        text = _clean_turn_text(text) or raw_text  # idempotent: also cleans turns from external callers
        loop_unique = len(set(re.sub(r"[\s,，。？?!！]", "", text)))
        if (is_agreement_only(raw_text) or is_agreement_only(text)   # never spawns a T_TOPIC, any length
                or (len(text) <= BACKCHANNEL_HARD_MAX_CHARS and _is_backchannel(text))
                or (len(text) > BACKCHANNEL_HARD_MAX_CHARS and loop_unique <= LOOP_UNIQUE_MAX_CHARS)):
            tid, sim = NOISE_THREAD, 0.0
        else:
            kw = _keywords(text)
            # rank active threads (300 s window) by shared keywords, then Jaccard
            ranked = []
            for cand, st in threads.items():
                if start - st["last"] > window_sec:
                    continue
                ranked.append((len(kw & st["kw"]), _jaccard(kw, st["kw"]), cand))
            ranked.sort(reverse=True)
            shared, sim, best = ranked[0] if ranked else (0, 0.0, None)
            latch_ok = (last_content is not None and start - threads[last_content]["last"] <= window_sec)
            if (latch_ok and _has_connector(text) and gap < DISCOURSE_LATCH_GAP_S
                    and not _is_question(text)):
                # discourse continuity latch, unless an older thread clearly owns this content
                strong = best != last_content and shared >= STRONG_SHARED_MIN
                tid = best if strong else last_content
            elif best and (shared >= 1 or sim >= THREAD_JACCARD_MIN):
                tid = best
            elif reason == "qa_latch" and not _is_question(text):
                # answers are lexically different from questions: follow the asker's latest content thread
                tid = next((o["thread_id"] for o in reversed(out)
                            if o["speaker_id"] == addressee and o["thread_id"] != NOISE_THREAD), None)
                if tid is None or start - threads[tid]["last"] > window_sec:
                    n_topics += 1
                    tid = f"T_TOPIC_{n_topics}"
            else:
                n_topics += 1
                tid = f"T_TOPIC_{n_topics}"
            st = threads.setdefault(tid, {"last": start, "kw": set()})
            st["last"] = start
            st["kw"] |= kw
            last_content = tid

        out.append({"turn_id": i + 1, "start": round(start, 2), "end": round(float(t["end"]), 2),
                    "window": int(start // window_sec), "speaker": t["speaker"], "speaker_id": t["speaker"],
                    "addressee": addressee,
                    "addressee_reason": reason, "thread_id": tid, "thread_sim": round(sim, 3), "text": text})

    windows = []
    for w in sorted({o["window"] for o in out}):
        wt = [o for o in out if o["window"] == w]
        windows.append({"window": w, "start": mmss(w * window_sec), "end": mmss((w + 1) * window_sec),
                        "turns": len(wt), "speakers": sorted({o["speaker_id"] for o in wt}),
                        "threads": sorted({o["thread_id"] for o in wt if o["thread_id"] != NOISE_THREAD})})
    content = sorted({o["thread_id"] for o in out if o["thread_id"] != NOISE_THREAD},
                     key=lambda x: int(x.rsplit("_", 1)[1]))
    return {"window_sec": window_sec, "turns": out, "content_threads": content,
            "noise_turn_ids": [o["turn_id"] for o in out if o["thread_id"] == NOISE_THREAD],
            "role_map": role_map, "windows": windows}


def mine_golden_hotspots(turns, top_k: int = 5) -> List[dict]:
    """Top-k non-overlapping 60 s windows by speaker-switch density + thread interleaving.
    T_BACKCHANNEL_NOISE turns are excluded entirely, so only substantive multi-party talk is ranked."""
    turns = sorted((t for t in turns if t.get("thread_id") != NOISE_THREAD), key=lambda t: float(t["start"]))
    if not turns:
        return []
    starts = [float(t["start"]) for t in turns]
    spk = lambda t: t.get("speaker_id", t.get("speaker"))  # noqa: E731
    t0, t_end = starts[0], max(float(t["end"]) for t in turns)
    cands = []
    w = t0
    while True:
        lo = bisect.bisect_left(starts, w)
        hi = bisect.bisect_left(starts, w + HOTSPOT_WIN_S)
        seg = turns[lo:hi]
        speakers = {spk(t) for t in seg}
        if len(speakers) >= 2:
            switches = sum(1 for a, b in zip(seg, seg[1:]) if spk(a) != spk(b))
            content = [t.get("thread_id") for t in seg if t.get("thread_id") is not None]
            interleave = sum(1 for a, b in zip(content, content[1:]) if a != b)
            n_threads = len(set(content))
            chars = sum(len(_clean(t["text"])) for t in seg)
            score = (switches + 1.5 * interleave + 2.0 * (len(speakers) - 1) + (n_threads - 1)
                     + min(chars / HOTSPOT_CHARS_PER_POINT, HOTSPOT_CHARS_CAP))
            cands.append({"start": round(w, 2), "end": round(w + HOTSPOT_WIN_S, 2), "start_mmss": mmss(w),
                          "end_mmss": mmss(w + HOTSPOT_WIN_S), "score": round(score, 2),
                          "speakers": sorted(speakers), "speaker_switches": switches,
                          "thread_interleaves": interleave, "threads": sorted(set(content)),
                          "substantive_chars": chars, "turn_ids": [t.get("turn_id") for t in seg],
                          "preview": " / ".join(f"{spk(t)}: {t['text'][:20]}" for t in seg[:4])})
        if w + HOTSPOT_WIN_S >= t_end:
            break
        w += HOTSPOT_STEP_S
    cands.sort(key=lambda c: (-c["score"], c["start"]))
    picked: List[dict] = []
    for c in cands:
        if all(c["end"] <= p["start"] or c["start"] >= p["end"] for p in picked):
            picked.append(c)
        if len(picked) >= top_k:
            break
    for rank, p in enumerate(picked, 1):
        p["rank"] = rank
    return picked


def _to_turn(i: int, t: dict) -> UtteranceTurn:
    return UtteranceTurn(turn_id=i, timestamp=float(t["start"]), speaker_id=t["speaker"], inferred_role="unknown",
                         text=t["text"], rms_energy=float(t.get("rms", 0.0)), zcr_pitch=float(t.get("zcr", 0.0)))


def run_ambient_mintrec(turns: List[dict], session_id: str) -> dict:
    """Stream merged turns through MIntRec bi-modal labelling + AmbientIntentPreFetcher."""
    pf = AmbientIntentPreFetcher(XiaoZhaoMPCEngine(), session_id=session_id, case_id="INTERVIEW")
    mintrec_counts: Dict[str, int] = {}
    staged_turns = 0
    for i, t in enumerate(turns, start=1):
        turn = _to_turn(i, t)
        lb = classify_mintrec_taxonomy(turn.text, turn.rms_energy, turn.zcr_pitch)
        t["mintrec"] = lb
        mintrec_counts[lb["fine"]] = mintrec_counts.get(lb["fine"], 0) + 1
        res = pf.ingest_ambient_turn(turn)
        t["ambient_status"] = res["status"]
        staged_turns += res["status"] in ("STAGED_READY", "AUTO_FIRED_P0")
    return {"mintrec_counts": dict(sorted(mintrec_counts.items(), key=lambda kv: -kv[1])),
            "domain_intent_counts": dict(sorted(pf.domain_intent_counts.items(), key=lambda kv: -kv[1])),
            "ambient_staged_count": staged_turns,
            "ambient_stats": pf.stats(),
            "ambient_staged_packets": {k: {"stage_status": p["stage_status"], "turn_ids": p["turn_ids"]}
                                       for k, p in pf.staged_packets.items()}}


def recording_speaker_roles(turns: List[dict]) -> Dict[str, str]:
    """Upgrade A over the ENTIRE recording: a 60 s hotspot rarely carries role cues for every speaker, so
    the recording-level map is saved in the report and used as the fallback prior per hotspot window."""
    return infer_session_speaker_roles([_to_turn(i, t) for i, t in enumerate(turns, start=1)])


def run_mpc_engine(turns: List[dict], session_id: str) -> dict:
    session = DialogueSessionState(session_id=session_id, case_id="INTERVIEW")
    for i, t in enumerate(turns, start=1):
        session.turns.append(_to_turn(i, t))
    if not session.turns:
        return {}
    try:
        return XiaoZhaoMPCEngine().analyze_session(session)
    except Exception as e:  # engine is tuned for short sessions; never block the report
        return {"error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------- STT + cache
def _trim_wav(wav_path: str, max_sec: float) -> str:
    out = AUDIO_TMP / f"{Path(wav_path).stem}_first{int(max_sec // 60)}m.wav"
    if out.exists() and out.stat().st_size > 44:
        return str(out)
    with wave.open(wav_path, "rb") as src, wave.open(str(out), "wb") as dst:
        dst.setparams(src.getparams())
        remaining = int(max_sec * src.getframerate())
        while remaining > 0:
            chunk = src.readframes(min(remaining, 1 << 16))
            if not chunk:
                break
            dst.writeframes(chunk)
            remaining -= 1 << 16
    return str(out)


def normalize_asr_text(text: str) -> str:
    """Fix common domain homophone errors from Whisper."""
    text = _ASR_LOOP_WORDS_RE.sub(r"\1\1", text)          # Whisper repetition hallucination
    text = _ASR_LOOP_SUBSTR_RE.sub(r"\1\1", text)
    text = _ASR_SEP_LOOP_RE.sub(r"\1", text)
    return _fix_homophones(_strip_stutter(text))


def _strip_stutter(text: str) -> str:
    """Collapse ellipsis stutter runs: 他...他...說 -> 他說, 我…我…我要 -> 我要."""
    return _ELLIPSIS_STUTTER_RE.sub(r"\1", text)


def _fix_homophones(text: str) -> str:
    """Domain homophone table + guarded contextual rules (idempotent)."""
    for wrong, right in ASR_FIXES:
        text = text.replace(wrong, right)
    text = _JUDU_CTX_RE.sub("居督", text)
    return re.sub(r"(?<![嘉家])安家", "案家", text)       # keep 嘉安家屬 intact


def _normalize_segments(segs: List[dict]) -> List[dict]:
    return [{**s, "text": normalize_asr_text(str(s.get("text", "")))} for s in segs]


def transcribe(wav_path: str, model_size: str) -> List[dict]:
    from faster_whisper import WhisperModel  # optional dep
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segs, _ = model.transcribe(wav_path, vad_filter=True, beam_size=1, best_of=1, initial_prompt=WHISPER_INITIAL_PROMPT)
    return [{"start": round(s.start, 2), "end": round(s.end, 2), "text": normalize_asr_text(s.text.strip())}
            for s in segs]


def load_or_transcribe(audio: Path, wav: str, model: str, max_minutes: float) -> Tuple[List[dict], str]:
    cache = OUT_DIR / f"{audio.stem}_raw_stt.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))  # stale caches predate new ASR_FIXES
        covered = float(data.get("max_minutes", 0) or 0)
        if covered == 0 or (max_minutes and covered >= max_minutes):
            return _normalize_segments(data["segments"]), f"cache hit ({cache.name})"
    src = _trim_wav(wav, max_minutes * 60) if max_minutes else wav
    segs = transcribe(src, model)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"source": str(audio), "model": model, "max_minutes": max_minutes,
                                 "segments": segs}, ensure_ascii=False, indent=1), encoding="utf-8")
    return segs, f"whisper-{model} (cached -> {cache.name})"


def _print_summary(stem: str, duration: float, raw_n: int, merged: List[dict], dis: dict,
                   hotspots: List[dict], report_path: Path, stt_note: str, elapsed: float,
                   amb: Optional[dict] = None) -> None:
    counts: Dict[str, int] = {}
    for t in merged:
        counts[t["speaker"]] = counts.get(t["speaker"], 0) + 1
    total = max(1, len(merged))
    print(f"=== Interview MPC Report: {stem} ===")
    print(f"STT source        : {stt_note}")
    print(f"Total duration    : {mmss(duration)}")
    print(f"Raw segments      : {raw_n} -> Merged turns: {len(merged)}")
    print("Speakers          : " + ", ".join(f"{k}={v} ({100 * v / total:.0f}%)" for k, v in sorted(counts.items())))
    print(f"Threads           : {len(dis['content_threads'])} content + {len(dis['noise_turn_ids'])} backchannel turns")
    print("Top 3 Golden 60s Hotspots:")
    for h in hotspots[:3]:
        print(f"  #{h['rank']} {h['start_mmss']}-{h['end_mmss']} score={h['score']} "
              f"switches={h['speaker_switches']} interleaves={h['thread_interleaves']} {','.join(h['speakers'])}")
    if amb:
        print("MIntRec fine dist : " + (", ".join(f"{k}={v}" for k, v in amb["mintrec_counts"].items()) or "-"))
        print(f"Ambient prefetch  : {amb['ambient_staged_count']} staged turns -> "
              + (", ".join(f"{k}={v}" for k, v in amb["domain_intent_counts"].items()) or "no domain intents"))
    print(f"Report saved      : {report_path}  ({elapsed:.1f}s)")


def _turn_sample(t: dict, merged_by_start: Dict[float, dict]) -> dict:
    m = merged_by_start.get(round(float(t["start"]), 2), {})
    return {"turn_id": t["turn_id"], "start": t["start"], "end": t["end"], "start_mmss": mmss(t["start"]),
            "speaker": t["speaker_id"], "addressee": t["addressee"], "addressee_reason": t["addressee_reason"],
            "thread_id": t["thread_id"], "text": t["text"],
            "mintrec_fine": (m.get("mintrec") or {}).get("fine", "UNK"),
            "domain_intents": XiaoZhaoMPCEngine().detect_domain_intents(t["text"])}


def export_extracted_samples(report_paths: List[Path], out_path: Path, top_k: int = 3) -> dict:
    """Top clean multi-party hotspots (T_BACKCHANNEL_NOISE excluded) from each interview report, in a shape
    the backend can replay: `segments` feeds mpc_core.load_from_whisper_json; `mpc_result` is the
    XiaoZhaoMPCEngine.analyze_session verdict for that hotspot."""
    engine = XiaoZhaoMPCEngine()
    out = {"schema": "xiaozhao.real_interview_samples.v1", "privacy": "anonymous speaker_1/2/3 clusters; no voiceprint",
           "noise_policy": "T_BACKCHANNEL_NOISE turns removed; Whisper repetition loops collapsed", "interviews": []}
    for rp in report_paths:
        r = json.loads(rp.read_text(encoding="utf-8"))
        dis = r["general_disentangle"]
        by_id = {t["turn_id"]: t for t in dis["turns"]}
        merged_by_start = {round(float(t["start"]), 2): t for t in r["merged_turns"]}
        stem = Path(r["source"]).stem
        rec_roles = r.get("recording_speaker_roles") or recording_speaker_roles(r["merged_turns"])
        hotspots = []
        for h in r["golden_hotspots"][:top_k]:
            turns = [_turn_sample(by_id[i], merged_by_start) for i in h["turn_ids"]
                     if by_id[i]["thread_id"] != NOISE_THREAD]
            session = DialogueSessionState(session_id=f"S_{stem}_HS{h['rank']}", case_id=f"REAL_{stem.upper()}",
                                           role_priors=rec_roles)
            session.turns = [UtteranceTurn(k, t["start"], t["speaker"], "unknown", t["text"],
                                           addressee=t["addressee"], thread_id=t["thread_id"])
                             for k, t in enumerate(turns, start=1)]
            res = engine.analyze_session(session) if session.turns else {}
            hotspots.append({
                "rank": h["rank"], "window": f"{h['start_mmss']}-{h['end_mmss']}", "score": h["score"],
                "speakers": h["speakers"], "speaker_switches": h["speaker_switches"],
                "thread_interleaves": h["thread_interleaves"], "threads": h["threads"], "turns": turns,
                "segments": [{"start": t["start"], "end": t["end"], "speaker": t["speaker"], "text": t["text"]}
                             for t in turns],
                "mpc_result": {**{k: res.get(k) for k in ("detected_intents", "route_target", "safety_level",
                                                          "pi_state", "selected_turn_indices", "speaker_roles",
                                                          "speaker_role_sources",
                                                          "mintrec_transitions")},
                               "cross_turn_fusion": (res.get("dst_slots") or {}).get("cross_turn_fusion"),
                               "who_to_whom_role": {d["turn_id"]: d["who_to_whom_role"]
                                                    for d in res.get("turns_detail", [])}}})
        n = max(1, len(r["merged_turns"]))
        unk = r["mintrec_counts"].get("UNK", 0)
        out["interviews"].append({
            "source": stem, "duration_mmss": mmss(r["duration_sec"]), "raw_segments": r["raw_segment_count"],
            "merged_turns": r["merged_turn_count"], "content_threads": len(dis["content_threads"]),
            "backchannel_noise_turns": len(dis["noise_turn_ids"]), "mintrec_unk_rate": round(unk / n, 3),
            "recording_speaker_roles": rec_roles, "domain_intent_counts": r["domain_intent_counts"],
            "staged_packet_keys": sorted(r["ambient_staged_packets"]), "hotspots": hotspots})
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def analyze_file(path: str, model_size: str, max_minutes: float = 0.0) -> int:
    t_start = time.time()
    p = Path(path)
    if not p.exists():
        print(f"[ERR] not found: {path}")
        return 2
    try:
        wav = convert_to_16k_wav(str(p))
    except Exception as e:
        print(f"[ERR] conversion: {e}")
        return 2
    try:
        segments, note = load_or_transcribe(p, wav, model_size, max_minutes)
    except ImportError:
        print("[ERR] faster_whisper not installed and no cached transcript (pip install faster-whisper)")
        return 2
    limit = max_minutes * 60 if max_minutes else float("inf")
    segments = [s for s in segments if float(s["start"]) < limit and str(s.get("text", "")).strip()]
    duration = min(wav_duration(wav), limit)

    merged = assign_speakers_by_acoustics(wav, segments)
    dis = general_mpc_disentangle(merged)
    hotspots = mine_golden_hotspots(dis["turns"], top_k=5)
    recording_roles = recording_speaker_roles(merged)
    mpc = run_mpc_engine(merged, session_id=f"S_INTERVIEW_{p.stem}")
    amb = run_ambient_mintrec(merged, session_id=f"S_AMBIENT_{p.stem}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / f"{p.stem}_mpc_report.json"
    report = {"source": str(p), "wav": wav, "duration_sec": round(duration, 2), "max_minutes": max_minutes,
              "raw_segment_count": len(segments), "merged_turn_count": len(merged),
              "recording_speaker_roles": recording_roles,
              "golden_hotspots": hotspots, "general_disentangle": dis, "merged_turns": merged, "mpc_engine": mpc, **amb}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _print_summary(p.stem, duration, len(segments), merged, dis, hotspots, report_path, note,
                   time.time() - t_start, amb)
    return 0


# ----------------------------------------------------------------- self-test
def _write_tone_wav(path: Path, plan: List[Tuple[float, float, float]]) -> None:
    """plan: (start, end, freq_hz); 16 kHz mono, silence between tones."""
    n = int((max(e for _, e, _ in plan) + 0.2) * TARGET_SR)
    frames = bytearray()
    for i in range(n):
        t = i / TARGET_SR
        f = next((fr for s, e, fr in plan if s <= t < e), 0.0)
        frames += struct.pack("<h", int(9000 * math.sin(2 * math.pi * f * t)) if f else 0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_SR)
        w.writeframes(bytes(frames))


def _unit_checks() -> None:
    assert normalize_asr_text("家安的照不遠用知識苦") == "嘉安的照服員用知識庫", "ASR normalization"
    assert _is_backchannel("對對對，對啊") and _is_backchannel("嗯嗯") and not _is_backchannel("幾點開始？")
    assert not _is_backchannel("對對，照服員要排班喔"), "domain noun keeps turn as content"
    assert not _is_backchannel("對,對,嗯嗯我覺得那個時候大家其實都很辛苦"), ">16 chars is never backchannel"
    assert normalize_asr_text("頁島世界要通報都島") == "跌倒事件要通報督導", "telemetry homophones"
    # interview1/2 homophones
    assert normalize_asr_text("基本上私人醫院看到你到院前懷疑他") == "基本上私人醫院看到你到院前死亡"
    assert normalize_asr_text("現場黃金很開歌") == "現場黃金很臭"
    assert normalize_asr_text("督導管照古緣，家屬突然咒事") == "督導管照服員，家屬突然猝逝"
    assert normalize_asr_text("他開始,他開始,他開始,他開始") == "他開始"
    assert normalize_asr_text("生活起居都要人幫忙") == "生活起居都要人幫忙", "居都 fix must stay scoped"
    assert normalize_asr_text("督導的名字跟照顧人") == "督導的名字跟照服員"
    assert normalize_asr_text("其他的督導或照顧你") == "其他的督導或照服員"
    # agreement-loop cleaner
    assert _clean_turn_text("對啦" * 50) == "對啦"
    assert _clean_turn_text("這一課然後就配合" + "對啦" * 16) == "這一課然後就配合"
    assert _clean_turn_text("對啦" * 8 + "他一定就是照樣一下就念你") == "他一定就是照樣一下就念你"
    assert _clean_turn_text("但是對對對然後報警") == "但是對然後報警"
    assert normalize_asr_text("他...他...說篤島要找居住") == "他說督導要找居督"
    assert normalize_asr_text("我…我…我要問小皂") == "我要問小照"
    assert normalize_asr_text("居住環境跟居住地,是居住在台北") == "居住環境跟居住地,是居住在台北"
    assert _clean_turn_text("那居住...那居住...會處理") == "那居督會處理"
    assert normalize_asr_text("按鍵是綁居都還是問每個居都") == "按鍵是綁居督還是問每個居督"
    assert normalize_asr_text("鄰居都很好") == "鄰居都很好"
    assert _clean_turn_text("好好休息") == "好好休息" and _clean_turn_text("可以是一個方式") == "可以是一個方式"
    # Whisper cross-segment loop collapse + no backchannel merging
    loop = [{"start": 0.0, "end": 1.0, "text": "配合"}] + [
        {"start": float(i), "end": float(i + 1), "text": "對啦"} for i in range(1, 51)] + [
        {"start": 51.0, "end": 52.0, "text": "他一定就是照樣"}]
    col = collapse_asr_loops(loop)
    assert len(col) == 3 and col[1]["asr_loop_collapsed"] == 50 and col[1]["end"] == 51.0, col[1]
    m = merge_same_speaker([{**x, "speaker": "speaker_1"} for x in col])
    assert [t["text"] for t in m] == ["配合", "對啦", "他一定就是照樣"], [t["text"] for t in m]
    assert m[1]["merged_from"] == list(range(1, 51)), "merged_from must keep raw segment indices"
    noise = general_mpc_disentangle([{"start": 0.0, "end": 9.0, "speaker": "speaker_2", "text": "對啦" * 8},
                                     {"start": 9.0, "end": 12.0, "speaker": "speaker_3", "text": "對啦" * 3},
                                     {"start": 12.0, "end": 30.0, "speaker": "speaker_1", "text": "對，對，好" * 9}])
    assert all(o["thread_id"] == NOISE_THREAD for o in noise["turns"]) and not noise["content_threads"], noise
    hs_turns = [{"turn_id": i, "start": float(i), "end": float(i) + 0.9, "speaker_id": f"speaker_{i % 3 + 1}",
                 "thread_id": NOISE_THREAD, "text": "對啦"} for i in range(40)]
    assert mine_golden_hotspots(hs_turns) == [], "noise-only windows must never become hotspots"
    segs = [{"start": 0.0, "end": 3.0, "text": "這是一段超過十個字的長句子"},
            {"start": 3.2, "end": 3.6, "text": "好"},
            {"start": 4.0, "end": 7.0, "text": "另一位說話者的長句子內容"},
            {"start": 7.2, "end": 7.5, "text": "對"}]
    bins = _apply_speaker_inertia(segs, [0.0, 0.2, 2.0, 2.1], [0, 1, 2, 2], [True] * 4)
    assert bins == [0, 0, 2, 2], f"speaker inertia: {bins}"
    m = merge_same_speaker([{"start": 0.0, "end": 1.0, "speaker": "speaker_1", "text": "前半句"},
                            {"start": 2.4, "end": 3.0, "speaker": "speaker_1", "text": "後半句"}])
    assert len(m) == 1 and m[0]["merged_from"] == [0, 1], "fragment consolidation (gap < 1.5 s)"
    turns = [{"start": 0.0, "end": 2.0, "speaker": "speaker_1", "text": "復健課程三點開始"},
             {"start": 10.0, "end": 12.0, "speaker": "speaker_2", "text": "晚餐想吃稀飯和青菜"},
             {"start": 13.0, "end": 15.0, "speaker": "speaker_1", "text": "然後記得帶毛巾過去"},
             {"start": 40.0, "end": 42.0, "speaker": "speaker_2", "text": "稀飯要煮軟一點比較好"}]
    th = [o["thread_id"] for o in general_mpc_disentangle(turns)["turns"]]
    assert th[0] != th[1] and th[2] == th[1], f"discourse latch: {th}"
    assert th[3] == th[1], f"keyword centroid join: {th}"


def quiet_test() -> int:
    _unit_checks()
    AUDIO_TMP.mkdir(parents=True, exist_ok=True)
    wav = AUDIO_TMP / "interview_selftest.wav"
    plan = [(0.0, 2.6, 150.0), (2.8, 5.4, 600.0), (5.6, 8.2, 1500.0), (8.4, 11.0, 150.0),
            (12.0, 12.4, 600.0), (12.6, 15.2, 1500.0)]
    _write_tone_wav(wav, plan)
    texts = ["今天下午的復健課程幾點開始呢？", "復健課程是下午三點開始喔", "那晚餐大家想要吃什麼？",
             "復健課程三點開始我會準時到", "嗯嗯", "晚餐想吃稀飯和青菜就好"]
    segs = [{"start": s, "end": e, "text": tx} for (s, e, _), tx in zip(plan, texts)]
    merged = assign_speakers_by_acoustics(str(wav), segs)
    spk = {t["speaker"] for t in merged}
    assert len(spk) >= 2, f"expected >=2 speakers, got {spk}"
    assert merged[0]["speaker"] != merged[1]["speaker"], "low vs mid pitch should differ"
    assert merged[0]["speaker"] == merged[3]["speaker"], "same pitch should cluster together"
    dis = general_mpc_disentangle(merged, window_sec=300.0)
    assert len(dis["content_threads"]) >= 2, f"expected >=2 threads, got {dis['content_threads']}"
    assert dis["noise_turn_ids"], "backchannel should be tagged as noise"
    assert dis["turns"][1]["addressee"] == merged[0]["speaker"], "QA latch should address asker"
    hs = mine_golden_hotspots(dis["turns"], top_k=5)
    assert hs and hs[0]["speaker_switches"] >= 2 and hs[0]["start_mmss"] == "00:00", hs
    amb = run_ambient_mintrec(merged + [{"start": 20.0, "end": 22.0, "speaker": "speaker_1", "rms": 900.0,
                                         "zcr": 0.05, "text": "這個多元問題不在BA24，要問督導跟長照基本法"}],
                              session_id="S_SELFTEST")
    assert amb["mintrec_counts"].get("Agree", 0) >= 1, amb["mintrec_counts"]
    assert amb["ambient_staged_count"] >= 1 and "SUPERVISOR_DIVERSE_FEEDBACK" in amb["domain_intent_counts"], amb
    assert all("mintrec" in t for t in merged), "each merged turn should carry its MIntRec label"
    print("INTERVIEW_ANALYZER_OK")
    return 0


def main() -> int:
    if "--export-samples" in sys.argv:
        dst = ROOT / "backend_handover_package" / "real_interviews_extracted_samples.json"
        reports = sorted(OUT_DIR.glob("interview*_mpc_report.json"))
        res = export_extracted_samples(reports, dst)
        print(f"exported {sum(len(i['hotspots']) for i in res['interviews'])} hotspots from "
              f"{len(reports)} reports -> {dst}")
        return 0
    if "--quiet-test" in sys.argv:
        try:
            return quiet_test()
        except AssertionError as e:
            print(f"[FAIL] {e}", file=sys.stderr)
            return 1
    ap = argparse.ArgumentParser(description="Analyze 30-min unedited multi-party interview audio with XiaoZhao MPC")
    ap.add_argument("--audio", required=True, help="interview audio (.mp3/.m4a/.wav)")
    ap.add_argument("--model", default="base", help="faster-whisper model size")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="analyze only the first N minutes (0 = all)")
    args = ap.parse_args()
    return analyze_file(args.audio, args.model, args.max_minutes)


if __name__ == "__main__":
    sys.exit(main())

