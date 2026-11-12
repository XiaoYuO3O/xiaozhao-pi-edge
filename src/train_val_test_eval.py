#!/usr/bin/env python3
"""Train / Validation / Test benchmark for the v7.7 MPC pipeline (Mac corpus tooling, group B/D).

    python3 src/train_val_test_eval.py                 # export dataset + train + val grid + test tables
    python3 src/train_val_test_eval.py --write-core    # ... and write the Train-fitted VP_* params into mpc_core
    python3 src/train_val_test_eval.py --latency-only  # stdlib-only latency / memory probe (runs on the Pi)

Protocol — two chronological split schemes, run side by side (SCHEMES):
  * 50_25_25 = the professor's example ("20 min -> first 10 min Train / next 5 min Val / last 5 min Test", 10:5:5);
    80_10_10 = the lab whiteboard standard split (8:1:1).
  * Within-recording chronological split. For each Jia-An recording (interview1, interview2, interview3_full) the
    labelled golden-window turns are put in time order: first 50 % (80 %) Train, next 25 % (10 %) Val, final 25 %
    (10 %) Test. A window
    that straddles a boundary is cut into per-split segments; segments are the evaluation units, so no split ever
    sees a later turn of its recording (zero future leakage). The 5 scripted benchmark dialogues are not
    recordings: they form a separate fixed regression set (Table C) and are never trained on.
  * Full-recording Train evidence: besides the labelled Train turns, each recording's unlabelled merged turns that
    start before its Val cut feed (i) a cue-inferred speaker -> role prior (fills speakers without a labelled Train
    role) and (ii) up to FULLREC_VP_MAX extra turns of voiceprint channel statistics. Nothing after the Val cut.
  * Interleaved windows. Inside each split, every pair of segments (>= 3 turns, different windows) is merged into
    one interleaved session (2-turn blocks, A A B B ...); its upstream greedy thread / addressee hints are recomputed
    by the analyzer's own greedy stage on the interleaved order, so the pipeline must disentangle two topics. Gold threads / reply-to links come from the source segments. Segments of
    different recordings get disjoint speaker ids. These sessions are constructed, not recorded.
  * Fixed domain schema: ISSUE_TRIGGER_CATALOG / RESOLUTION_CATALOG / EXTENDED_ROLE_CUES / MIntRec rules stay as
    shipped (they were authored with every window visible: the leakage audit lists the Test-only keyword hits).
  * Train: VP_PRIOR_MEAN / VP_PRIOR_STD / VP_LOGSIG_PRIOR on Train frames; VP_DIM_WEIGHTS family (uniform /
    Fisher-sqrt / Fisher-linear, from Train variance ratios) picked by Val d'; recording-level speaker -> role
    priors (majority labelled role of each diarizer speaker in its recording's Train turns) and per-speaker
    voiceprint prototypes (M5, evaluation-only, in memory, never exported).
  * Validation: 3 x 4 x 4 grid over (BICL_MERGE_SIM, GSN iterations N, E2CP alpha) on Val segments + interleaved
    Val sessions; J = mean(thread F, addressee acc, slot F1, issue->resolution link F1, speaker-cluster F).
  * Test: the Val-selected configuration, scored once (Tables A, B). Table C = audio segmentation / noise
    statistics of the 3.6 h corpus + data simulation (synthetic boundary bleed / overlap SNR, near- vs far-field,
    interleaved multi-party sessions, 5 scripted home-care dialogues); it does not depend on the split.

Labels: ground_truth_* are DRAFT labels (DRAFT_LABELS: text-level labels drafted from the cleaned transcripts;
speaker = the acoustic diarizer label). Rows marked human_verified=1 in the exported CSV override the drafts.
"""
import argparse
import array
import copy
import csv
import dataclasses
import itertools
import json
import math
import os
import re
import sys
import time
import tracemalloc
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from src import mpc_core as core
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mpc_core as core

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "backend_handover_package" / "real_interviews_extracted_samples.json"
OUT_DIR = ROOT / "transcripts_out"
DATASET_JSON = OUT_DIR / "annotated_ground_truth_dataset.json"
DATASET_CSV = OUT_DIR / "annotated_ground_truth_dataset.csv"
RESULTS_JSON = OUT_DIR / "train_val_test_results.json"
TABLES_MD = OUT_DIR / "model_comparison_tables.md"
SPLITS = ("train", "val", "test")
# (scheme id, provenance, shares, ratio label)
SCHEMES = (("50_25_25", "Professor's example", {"train": 0.50, "val": 0.25, "test": 0.25}, "10 : 5 : 5"),
           ("80_10_10", "Lab whiteboard standard split", {"train": 0.80, "val": 0.10, "test": 0.10}, "8 : 1 : 1"))
PRIMARY_SCHEME = "50_25_25"   # its voiceprints go into the exported dataset / --write-core
FULLREC_VP_MAX = 24           # extra unlabelled pre-cut turns per recording for the voiceprint channel statistics
FULLREC_MIN_DUR = 1.5
JIA_AN = ("interview1", "interview2", "interview3_full")
INTERLEAVE_MIN_TURNS = 3
INTERLEAVE_BLOCK = 2
GRID_BICL = (0.30, 0.45, 0.65)
GRID_GSN_N = (1, 2, 3, 4)
GRID_ALPHA = (0.0, 0.2, 0.4, 0.7)
DEFAULTS = {"bicl": core.BICL_MERGE_SIM, "gsn": core.GSN_ITERATIONS, "alpha": core.E2CP_ALPHA}

# ------------------------------------------------------------------------------------------- draft labels
ROLE_CODE = {"R": core.ROLE_RESEARCHER, "D": core.ROLE_DIRECTOR, "S": core.ROLE_SUPERVISOR,
             "C": core.ROLE_CAREGIVER_LTC, "fam": "family", "eld": "elder", "cg": "caregiver", "?": "unknown"}
INTENT_CODE = {"KB": "EXPERIENCE_KB_CONTRIBUTION", "FB": "SUPERVISOR_DIVERSE_FEEDBACK", "SC": "SERVICE_CODE_QUERY",
               "EB": "EMERGENCY_BUTTON_ESCALATION", "LG": "LEGAL_REGULATION_QUERY", "JA": "JIA_AN_INTERNAL_RULE",
               "PV": "PRIVACY_CONSENT_RECORDING", "CA": "CASE_ASSIGNMENT_HIERARCHY",
               "WH": "WORKPLACE_HARASSMENT_REPORT", "P0": "P0_EMERGENCY_ALERT",
               "BA16": "BA16_MEDICINE_PURCHASE_INQUIRY", "CLEAN": "EXTRA_OUT_OF_SCOPE_CLEANING",
               "OOS": "OOS_ATYPICAL_FIELD_HAZARD", "GA": "GA09_SC09_RESPITE_FORM"}
# key  thread  reply(turn id | ALL | XZ=小照 | @speaker_N | -)  role  mintrec_fine  intents  slot(iss|res|-)  hypothetical
# Drafted by reading normalize_asr_text() output; no transcript text is stored here. "noise" threads are skipped
# by the thread metrics. Speaker identity cannot be read from text: the draft speaker is the diarizer label.
DRAFT_LABELS = """
interview1:94    kb     -    R Inform       -      -   0
interview1:97    kb     94   R Advise       KB,FB  res 0
interview1:99    kb     97   S Complain     KB     iss 0  # 我曾經遇過我的夥伴問我 ... 老師沒有教導
interview1:100   kb     99   ? Inform       -      -   0
interview1:101   kb     99   S Inform       -      -   0
interview1:102   kb     99   S Advise       FB,SC  res 1  # 如果是我們的家人 (hypothetical) ... 放進多元項目
interview1:103   kb     102  S Complain     -      iss 0
interview1:194   kbsys  -    R Inform       KB     -   0
interview1:195   kbsys  194  ? Inform       -      iss 0  # records are not read carefully
interview1:197   kbsys  195  R Inform       KB     -   0
interview1:198   kbsys  197  R Prevent      -      -   0
interview1:199   kbsys  198  R Inform       KB     res 0
interview1:200   kbsys  199  ? Inform       KB     -   0
interview1:201   kbsys  200  R Inform       KB     -   1  # 比如說家屬猝逝或家屬去世: example, not a live P0
interview1:203   kbsys  ALL  R Praise       FB     -   0
interview1:204   kbsys  ALL  R Arrange      KB,FB  res 0
interview1:117   kb     -    ? Advise       FB     -   0
interview1:120   emerg  -    S Complain     WH     iss 1  # 像剛我們說的暴力性騷: meta reference, not live
interview1:121   emerg  120  S Inform       EB     iss 0
interview1:123   noise  -    ? UNK          -      -   0
interview1:124   emerg  121  S Arrange      EB     res 0
interview1:125   emerg  124  ? Inform       EB     -   0
interview1:126   emerg  125  ? Inform       EB     -   0
interview1:127   emerg  126  ? Inform       EB     -   0
interview1:128   emerg  127  ? Inform       EB     -   0
interview1:130   emerg  128  R Advise       EB     iss 0
interview1:131   emerg  130  ? Inform       EB     res 0
interview1:133   emerg  131  ? Inform       -      -   0
interview2:139   legal  -    D Advise       LG,WH  iss 1  # 譬如說 ... 你都被打了
interview2:141   legal  139  R Advise       LG     res 0
interview2:142   legal  141  D Advise       LG,WH  -   1  # 譬如說我們被案家或案主暴力了: example
interview2:143   legal  142  D Inform       LG,WH  -   1
interview2:147   legal  143  R Agree        -      -   0
interview2:148   legal  147  D Praise       -      -   0
interview2:149   legal  148  D Advise       LG     res 0
interview2:150   legal  149  R Inform       JA     res 0  # 我們的目標就是為嘉安量身打造: read as the dev team
interview2:151   legal  150  R Inform       JA     -   0
interview2:152   legal  151  R Inform       -      -   0
interview2:153   legal  152  D Agree        -      -   0
interview2:216   param  -    ? UNK          -      -   0
interview2:217   param  216  R Oppose       -      iss 0
interview2:218   param  217  ? UNK          -      -   0
interview2:220   wrap   -    R Leave        -      -   0
interview2:221   wrap   ALL  R Leave        -      -   0  # 讓主任跟督導都辛苦: the host thanking them
interview2:222   wrap   ALL  R Arrange      -      -   0
interview2:223   wrap   ALL  R Inform       EB     res 0
interview2:224   wrap   ALL  R Arrange      EB,FB  res 0
interview2:225   wrap   ALL  R Arrange      FB,SC  res 0
interview2:180   cost   -    R Oppose       -      -   0
interview2:183   cost   180  R Inform       -      -   0
interview2:184   button -    D Inform       EB     -   0
interview2:185   button 184  R Advise       -      -   0
interview2:187   button 185  D Inform       EB     iss 0
interview2:189   cost   183  ? UNK          -      -   0
interview2:191   noise  -    ? UNK          -      -   0
interview2:193   push   ALL  R Arrange      -      res 0
interview3_full:111 vp   -    R Inform       PV     -   0
interview3_full:112 vp   111  R Advise       PV     -   0
interview3_full:113 vp   112  S Oppose       PV     iss 0
interview3_full:114 vp   113  R Inform       KB     -   0
interview3_full:115 vp   113  R Inform       PV     -   0
interview3_full:116 vp   115  S Advise       PV     -   0
interview3_full:117 vp   115  S Advise       -      res 0
interview3_full:118 vp   112  S Complain     PV     iss 1  # 比如說我今天 ... 四個案子
interview3_full:119 vp   118  ? Inform       -      -   0
interview3_full:120 vp   118  ? Advise       -      res 0
interview3_full:121 vp   120  R Oppose       -      -   0
interview3_full:59  fbui ALL  R Inform       -      -   0  # one presenter reading the feedback UI aloud
interview3_full:60  fbui ALL  R Inform       -      -   0
interview3_full:61  fbui ALL  R Inform       -      -   0
interview3_full:62  fbui ALL  R Inform       -      -   0  # UI option labels, not a complaint
interview3_full:63  fbui ALL  R Inform       -      -   0
interview3_full:64  fbui ALL  R Inform       -      -   0
interview3_full:65  fbui ALL  R Inform       -      -   0
interview3_full:66  fbui ALL  R Inform       -      -   1
interview3_full:518 bind -    R Arrange      -      -   0
interview3_full:519 bind 518  R Inform       -      -   0
interview3_full:520 bind 519  D Ask_for_help CA     iss 0
interview3_full:521 bind 520  R Arrange      CA     res 0
interview3_full:522 bind 521  D Inform       CA     -   0
interview3_full:523 bind 522  D Advise       CA     res 0
interview3_full:524 bind 523  S Oppose       CA     iss 0
interview3_full:525 bind 524  D Ask_for_help CA,EB  -   0
interview3_full:526 bind 525  S Advise       CA     res 0
interview3_full:527 bind 526  ? Inform       -      -   0
interview3_full:528 bind 527  D Advise       CA     res 0
SCEN_1_BA16_DISENTANGLE:1          task     @speaker_3 fam Ask_for_help BA16      iss 0
SCEN_1_BA16_DISENTANGLE:2          chitchat -          eld Complain     -         -   0
SCEN_1_BA16_DISENTANGLE:3          task     XZ         cg  Ask_for_help BA16      -   0
SCEN_2_MULTI_INTENT_HARASSMENT:1   task     @speaker_3 eld Arrange      WH        iss 0
SCEN_2_MULTI_INTENT_HARASSMENT:2   task     @speaker_3 fam Arrange      CLEAN     iss 0
SCEN_2_MULTI_INTENT_HARASSMENT:3   task     XZ         cg  Complain     WH,CLEAN  -   0
SCEN_3_OOD_MOUNTAIN_DOGS:1         task     XZ         cg  Ask_for_help OOS       iss 0
SCEN_4_P0_EMERGENCY_FALL:1         task     XZ         cg  Ask_for_help P0        iss 0
SCEN_5_GA09_SC09_RESPITE:1         task     @speaker_3 fam Arrange      GA        -   0
SCEN_5_GA09_SC09_RESPITE:2         chitchat -          eld Complain     -         -   0
SCEN_5_GA09_SC09_RESPITE:3         task     XZ         cg  Ask_for_help GA        -   0
"""
GT_FIELDS = ("ground_truth_speaker", "ground_truth_role", "ground_truth_thread", "ground_truth_addressee",
             "ground_truth_reply_to", "ground_truth_mintrec_fine", "ground_truth_intents", "ground_truth_slot",
             "is_hypothetical_example")
# lab-meeting display fixes on top of normalize_asr_text (資料擠 / 數沒派 are now in interview_analyzer.ASR_FIXES)
LAB_MEETING_FIXES = (("資料機", "資料集"), ("訓練機", "訓練集"), ("測試機", "測試集"), ("驗證機", "驗證集"),
                     ("核心演算吧", "核心演算法"), ("trainin say", "training set"),
                     ("validation say", "validation set"), ("testing say", "testing set"))


def parse_drafts() -> Dict[str, Dict[str, Any]]:
    out = {}
    for line in DRAFT_LABELS.strip().splitlines():
        body, _, note = line.partition("#")
        key, thread, reply, role, fine, intents, slot, hyp = body.split()
        out[key] = {"thread": thread, "reply": reply, "role": ROLE_CODE[role], "fine": fine,
                    "intents": [] if intents == "-" else [INTENT_CODE[c] for c in intents.split(",")],
                    "slot": "" if slot == "-" else {"iss": "issue", "res": "resolution"}[slot],
                    "hyp": int(hyp), "note": note.strip()}
    return out


# ------------------------------------------------------------------------------------------- units + split
def load_units() -> List[Dict[str, Any]]:
    from interview_analyzer import normalize_asr_text   # Mac corpus tooling only
    drafts = parse_drafts()
    units: List[Dict[str, Any]] = []
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    for iv in samples["interviews"]:
        for h in iv["hotspots"]:
            turns = []
            for t in sorted(h["turns"], key=lambda x: (x["start"], x["turn_id"])):
                turns.append({"key": f"{iv['source']}:{t['turn_id']}", "src": iv["source"], "turn_id": t["turn_id"],
                              "start": float(t["start"]), "end": float(t["end"]), "speaker_id": t["speaker"],
                              "role_input": "unknown", "asr_text_raw": t["text"],
                              "text": normalize_asr_text(t["text"]), "hint_addressee": t.get("addressee", "unknown"),
                              "hint_thread": t.get("greedy_thread_id") or "T0"})
            units.append({"unit_id": f"{iv['source']}@{h['window']}", "kind": "real_window",
                          "source": iv["source"], "window": h["window"], "turns": turns})
    for s in core.build_5_benchmark_scenarios():
        turns = [{"key": f"{s.session_id}:{t.turn_id}", "src": s.session_id, "turn_id": t.turn_id,
                  "start": t.timestamp, "end": 0.0, "speaker_id": t.speaker_id, "role_input": t.inferred_role,
                  "asr_text_raw": t.text, "text": normalize_asr_text(t.text), "hint_addressee": "unknown",
                  "hint_thread": "T0"} for t in s.turns]
        units.append({"unit_id": s.session_id, "kind": "scripted_dialogue", "source": "build_5_benchmark_scenarios",
                      "window": "", "turns": turns, "split": "scripted"})
    for u in units:
        for t in u["turns"]:
            d = drafts.get(t["key"])
            if d is None:
                raise SystemExit(f"no draft label for {t['key']}: add it to DRAFT_LABELS")
            t["gt"] = draft_gt(t, d, u)
            t["label_source"] = ("benchmark_script+claude_draft" if u["kind"] == "scripted_dialogue"
                                 else "claude_draft_text+diarizer_speaker")
            t["human_verified"] = 0
            t["annotator_notes"] = d["note"]
            if u["kind"] == "scripted_dialogue":
                t["split"] = "scripted"
    return units


def draft_gt(t: dict, d: dict, unit: dict) -> Dict[str, Any]:
    by_id = {x["turn_id"]: x for x in unit["turns"]}
    reply, addr = d["reply"], ""
    if reply.isdigit():
        addr = by_id[int(reply)]["speaker_id"]
    elif reply == "ALL":
        addr = "ALL"
    elif reply == "XZ":
        addr = "system_xiaozhao"
    elif reply.startswith("@"):
        addr = reply[1:]
    return {"ground_truth_speaker": t["speaker_id"], "ground_truth_role": d["role"],
            "ground_truth_thread": d["thread"], "ground_truth_addressee": addr,
            "ground_truth_reply_to": reply if reply.isdigit() else "",
            "ground_truth_mintrec_fine": d["fine"], "ground_truth_intents": d["intents"],
            "ground_truth_slot": d["slot"], "is_hypothetical_example": d["hyp"]}


def import_verified(units: List[dict], path: Path = DATASET_CSV) -> int:
    """Rows marked human_verified=1 in a previously exported (and hand-edited) CSV override the drafts."""
    if not path.exists():
        return 0
    by_key = {t["key"]: t for u in units for t in u["turns"]}
    n = 0
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            t = by_key.get(row.get("key", ""))
            if t is None or str(row.get("human_verified", "")).strip().lower() not in ("1", "y", "yes", "true"):
                continue
            for k in GT_FIELDS:
                v = (row.get(k) or "").strip()
                t["gt"][k] = ([x for x in v.split(";") if x] if k == "ground_truth_intents" else
                              int(v or 0) if k == "is_hypothetical_example" else v)
            t["human_verified"], t["label_source"] = 1, "human_verified_csv"
            t["annotator_notes"] = row.get("annotator_notes", t["annotator_notes"])
            n += 1
    return n


def chrono_split(windows: List[dict], share: Dict[str, float]) -> Tuple[List[dict], Dict[str, Dict[str, Any]]]:
    """Per recording: labelled turns in time order -> first share['train'] Train, next share['val'] Val, rest Test.
    Every window is cut into contiguous per-split segments (the evaluation units). -> (segments, cut report)."""
    by_src: Dict[str, List[dict]] = {}
    for w in windows:
        by_src.setdefault(w["source"], []).append(w)
    segs, cuts = [], {}
    for src, wins in sorted(by_src.items()):
        wins.sort(key=lambda w: w["turns"][0]["start"])
        turns = [t for w in wins for t in w["turns"]]
        n = len(turns)
        c1 = int(round(share["train"] * n))
        c2 = min(n - 1, max(c1 + 1, int(round((share["train"] + share["val"]) * n))))
        for i, t in enumerate(turns):
            t["split"] = "train" if i < c1 else "val" if i < c2 else "test"
        cuts[src] = {"turns": n, "train": c1, "val": c2 - c1, "test": n - c2,
                     "val_starts": _mmss(turns[c1]["start"]), "test_starts": _mmss(turns[c2]["start"]),
                     "val_start_sec": turns[c1]["start"]}
        for w in wins:
            for s in SPLITS:
                ts = [t for t in w["turns"] if t["split"] == s]
                if ts:
                    segs.append({**w, "unit_id": f"{w['unit_id']}#{s}", "window_id": w["unit_id"], "split": s,
                                 "turns": ts})
    return segs, cuts


def _mmss(sec: float) -> str:
    return f"{int(sec) // 60:02d}:{int(sec) % 60:02d}"


def _remap_sp(sp: str, off: int) -> str:
    m = re.match(r"^speaker_(\d+)$", sp or "")
    return f"speaker_{int(m.group(1)) + off}" if m and off else sp


def interleave(a: dict, b: dict) -> dict:
    """Two segments -> one interleaved session (A A B B A A ...). B's speakers are shifted to disjoint ids when it
    comes from another recording; times are re-laid on one timeline (durations kept); the upstream greedy thread /
    addressee hints are recomputed by interview_analyzer.general_mpc_disentangle on the interleaved sequence."""
    off = 0 if a["source"] == b["source"] else 3
    seq_a, seq_b = list(a["turns"]), list(b["turns"])
    order: List[Tuple[dict, int, str]] = []
    while seq_a or seq_b:
        for seq, o, tag in ((seq_a, 0, "A"), (seq_b, off, "B")):
            for _ in range(INTERLEAVE_BLOCK):
                if seq:
                    order.append((seq.pop(0), o, tag))
    turns, clock = [], 0.0
    for t, o, tag in order:
        dur = max(0.5, t["end"] - t["start"])
        gt = dict(t["gt"])
        gt["ground_truth_speaker"] = _remap_sp(gt["ground_truth_speaker"], o)
        gt["ground_truth_addressee"] = _remap_sp(gt["ground_truth_addressee"], o)
        if gt["ground_truth_thread"] != "noise":
            gt["ground_truth_thread"] = f"{t['src']}:{gt['ground_truth_thread']}"
        turns.append({**t, "key": f"{t['key']}~mix", "start": clock, "end": clock + dur,
                      "speaker_id": _remap_sp(t["speaker_id"], o), "hint_addressee": "unknown", "hint_thread": "T0",
                      "gt": gt, "mix_side": tag})
        clock += dur + 0.8
    # upstream hints from the analyzer's own greedy TD/MF + addressee stage, run on the interleaved sequence
    from interview_analyzer import general_mpc_disentangle
    greedy = general_mpc_disentangle([{"start": t["start"], "end": t["end"], "speaker": t["speaker_id"],
                                       "text": t["text"]} for t in turns])["turns"]
    for t, g in zip(turns, greedy):
        t["hint_thread"], t["hint_addressee"] = g["greedy_thread_id"], g["addressee"]
    priors = dict(a.get("role_priors") or {})
    priors.update({_remap_sp(sp, off): r for sp, r in (b.get("role_priors") or {}).items()})
    return {"unit_id": f"MIX[{a['unit_id']} + {b['unit_id']}]", "kind": "interleaved", "source": "interleaved",
            "window": "", "split": a["split"], "turns": turns, "role_priors": priors}


def build_interleaved(segs: List[dict]) -> List[dict]:
    out = []
    for s in SPLITS:
        pool = [u for u in segs if u["split"] == s and len(u["turns"]) >= INTERLEAVE_MIN_TURNS]
        for a, b in itertools.combinations(pool, 2):
            if a["window_id"] != b["window_id"]:
                out.append(interleave(a, b))
    return out


# ------------------------------------------------------------------------------------------- audio + voiceprints
VP_KEYS = ("VP_PRIOR_MEAN", "VP_PRIOR_STD", "VP_LOGSIG_PRIOR", "VP_DIM_WEIGHTS")


def shipped_vp_params() -> Dict[str, List[float]]:
    return {k: list(getattr(core, k)) for k in VP_KEYS}


@contextmanager
def vp_params(params: Dict[str, List[float]]):
    saved = shipped_vp_params()
    try:
        for k, v in params.items():
            setattr(core, k, list(v))
        yield
    finally:
        for k, v in saved.items():
            setattr(core, k, v)


@contextmanager
def record_pool(sink: List):
    orig = core._vp_pool

    def wrapped(frames, edbs, times, anchor, wlen):
        out = orig(frames, edbs, times, anchor, wlen)
        sink.append((frames, edbs, out))
        return out

    core._vp_pool = wrapped
    try:
        yield
    finally:
        core._vp_pool = orig


VP_CACHE_PATH = ROOT / "audio_tmp" / "eval_voiceprint_cache.json"
VP_CACHE_VERSION = 1
_VP_CACHE: Dict[str, Any] = {}
_VP_CACHE_STATE: Dict[str, Any] = {"loaded": False, "dirty": False, "hits": 0, "misses": 0, "new": set()}


def vp_cache() -> Dict[str, Any]:
    """Per-turn extract_window records keyed by (turn, span, slice offset, VP params, mpc_core voiceprint code
    hash). Exact: JSON floats round-trip, and any change to the extractor's code or params is a different key."""
    if not _VP_CACHE_STATE["loaded"]:
        _VP_CACHE_STATE["loaded"] = True
        try:
            d = json.loads(VP_CACHE_PATH.read_text(encoding="utf-8"))
            if d.get("version") == VP_CACHE_VERSION:
                _VP_CACHE.update(d["records"])
        except (OSError, ValueError, KeyError):
            pass
    return _VP_CACHE


def save_vp_cache() -> None:
    """Raw frames are kept only for records extracted with the shipped params (the only ones
    fit_channel_priors reads); every other record keeps its pooled mu / sigma."""
    keep = params_hash(shipped_vp_params())
    for ck, rec in _VP_CACHE.items():
        if isinstance(rec, dict) and "pool" in rec and ck.split("|")[3] != keep and rec["pool"] \
                and rec["pool"][0][0]:
            rec["pool"] = [[[], [], out] for _, _, out in rec["pool"]]
            _VP_CACHE_STATE["dirty"] = True
    if _VP_CACHE_STATE["dirty"]:
        VP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = VP_CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": VP_CACHE_VERSION, "records": _VP_CACHE}, separators=(",", ":")),
                       encoding="utf-8")
        tmp.replace(VP_CACHE_PATH)
        _VP_CACHE_STATE["dirty"] = False


_CODE_HASH: List[str] = []


def vp_code_hash() -> str:
    """Hash of the voiceprint extractor source (+ its constants) so an edited extractor invalidates the cache."""
    if not _CODE_HASH:
        import hashlib
        import inspect
        parts = [inspect.getsource(f) for f in (core.extract_turn_voiceprint, core.voiceprint_windows, core._vp_pool,
                                                 core._vp_near_field)]
        parts += [repr(getattr(core, k)) for k in dir(core) if k.startswith("VP_") and k not in VP_KEYS]
        _CODE_HASH.append(hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:12])
    return _CODE_HASH[0]


def params_hash(params: Dict[str, List[float]]) -> str:
    import hashlib
    return hashlib.sha1(json.dumps({k: params.get(k) for k in VP_KEYS}, sort_keys=True).encode()).hexdigest()[:12]


class LazyAudio:
    """A window's decoded PCM, decoded only on the first voiceprint-cache miss."""

    def __init__(self, unit: dict):
        from interview_analyzer import HOTSPOT_SLICE_PAD_S
        self.unit = unit
        self.st = max(0.0, min(t["start"] for t in unit["turns"]) - HOTSPOT_SLICE_PAD_S)
        self._pcm: Optional[Tuple[array.array, int, float]] = None

    def get(self) -> Tuple[array.array, int, float]:
        if self._pcm is None:
            self._pcm = load_window_pcm(self.unit)
        return self._pcm


def lazy_audio(unit: dict) -> Optional[LazyAudio]:
    from interview_analyzer import find_source_audio
    return LazyAudio(unit) if find_source_audio(unit["source"]) is not None else None


def load_window_pcm(unit: dict) -> Optional[Tuple[array.array, int, float]]:
    from interview_analyzer import (HOTSPOT_SLICE_MAX_S, HOTSPOT_SLICE_PAD_S, find_source_audio,
                                    hotspot_wav_slice)
    audio = find_source_audio(unit["source"])
    if audio is None:
        return None
    st = max(0.0, min(t["start"] for t in unit["turns"]) - HOTSPOT_SLICE_PAD_S)
    dur = min(max(t["end"] for t in unit["turns"]) + HOTSPOT_SLICE_PAD_S - st, HOTSPOT_SLICE_MAX_S)
    with hotspot_wav_slice(str(audio), st, dur) as path:
        with wave.open(path, "rb") as w:
            rate = w.getframerate()
            pcm = array.array("h")
            pcm.frombytes(w.readframes(w.getnframes()))
    if sys.byteorder == "big":
        pcm.byteswap()
    return pcm, rate, st


def x_raw(mu: List[float], sg: List[float], logsig: List[float]) -> List[float]:
    """_vp_embedding's 24-d x_i before the per-dimension weights and L2 normalisation."""
    return (mu[1:13] + [math.log(sg[c]) - logsig[c - 1] for c in (1, 2, 3, 4)]
            + [mu[core.VP_CH_LOGF0], math.log(sg[core.VP_CH_LOGF0]) - logsig[4], mu[core.VP_CH_ZCR],
               mu[core.VP_CH_CENTROID], mu[core.VP_CH_BAND], mu[core.VP_CH_BAND + 3], mu[core.VP_CH_FLAT], mu[0]])


def _rms_zcr(seg) -> Tuple[float, float]:
    n = len(seg)
    if n < 2:
        return 0.0, 0.0
    rms = math.sqrt(sum(v * v for v in seg) / n)
    return rms, sum(1 for a, b in zip(seg, seg[1:]) if (a >= 0) != (b >= 0)) / (n - 1)


def extract_window(win: dict, audio: LazyAudio, params: Dict[str, List[float]],
                   only: Optional[set] = None, frames: bool = False) -> Dict[str, dict]:
    """turn key -> voiceprint dict + per-piece pooling records (raw frames, mu / sigma) + per-piece rms / zcr.
    Served from the voiceprint cache when the same turn was extracted with the same params and extractor code."""
    cache = vp_cache()
    tag = f"{params_hash(params)}|{vp_code_hash()}|{audio.st:.3f}"
    out = {}
    todo = []
    for t in win["turns"]:
        if only is not None and t["key"] not in only:
            continue
        ck = f"{t['key']}|{t['start']:.3f}|{t['end']:.3f}|{tag}"
        if ck in cache and not (frames and cache[ck]["pool"] and not cache[ck]["pool"][0][0]):
            out[t["key"]] = cache[ck]
            _VP_CACHE_STATE["hits"] += 1
        else:
            todo.append((t, ck))
    if not todo:
        return out
    pcm, rate, st = audio.get()
    span = len(pcm) / float(rate)
    with vp_params(params):
        for t, ck in todo:
            s, e = t["start"] - st, min(t["end"], st + span) - st
            a, b = max(0, int(s * rate)), max(0, int(e * rate))
            seg = pcm[a:b]
            sink: List = []
            t0 = time.perf_counter()
            with record_pool(sink):
                vp = core.extract_turn_voiceprint(seg if len(seg) else None, rate, s, e, offset_sec=a / float(rate),
                                                  text=t["text"])
            vp_ms = 1000.0 * (time.perf_counter() - t0)
            t1 = time.perf_counter()
            pieces_ac = [_rms_zcr(pcm[int(ws * rate):int(we * rate)]) for ws, we, _ in vp["windows"]]
            ac_ms = 1000.0 * (time.perf_counter() - t1)
            rec = json.loads(json.dumps({"vp": vp, "pool": sink, "pieces_ac": pieces_ac, "vp_ms": vp_ms,
                                         "ac_ms": ac_ms}))   # same JSON shape on a miss as on a hit
            out[t["key"]] = cache[ck] = rec
            _VP_CACHE_STATE["new"].add(ck)
            _VP_CACHE_STATE["misses"] += 1
            _VP_CACHE_STATE["dirty"] = True
    return out


def vp_peak_kib(t: dict, audio: LazyAudio, params: Dict[str, List[float]]) -> float:
    """Peak traced memory of one real extract_turn_voiceprint call (measured once, then cached)."""
    cache = vp_cache()
    ck = f"mem|{t['key']}|{params_hash(params)}|{vp_code_hash()}|{audio.st:.3f}"
    if ck not in cache:
        pcm, rate, st = audio.get()
        s, e = t["start"] - st, min(t["end"], st + len(pcm) / float(rate)) - st
        a, b = max(0, int(s * rate)), max(0, int(e * rate))
        with vp_params(params):
            cache[ck] = peak_kib(lambda: core.extract_turn_voiceprint(pcm[a:b], rate, s, e, offset_sec=a / float(rate),
                                                                      text=t["text"]))
        _VP_CACHE_STATE["new"].add(ck)
        _VP_CACHE_STATE["dirty"] = True
    return cache[ck]


def fit_channel_priors(recs: List[dict]) -> Tuple[List[float], List[float], int]:
    nc = len(core.VP_PRIOR_MEAN)
    s1, s2, n = [0.0] * nc, [0.0] * nc, 0
    for r in recs:
        for frames, edbs, _ in r["pool"]:
            emax = max(edbs)
            for f, e in zip(frames, edbs):
                if e >= emax - core.VP_SILENCE_DB:
                    n += 1
                    for c in range(nc):
                        s1[c] += f[c]
                        s2[c] += f[c] * f[c]
    mean = [s1[c] / n for c in range(nc)]
    std = [math.sqrt(max(1e-9, s2[c] / n - mean[c] ** 2)) for c in range(nc)]
    return ([round(mean[0], 2)] + [round(v, 3) for v in mean[1:]],
            [round(std[0], 2)] + [round(v, 3) for v in std[1:]], n)


def fit_logsig(recs: List[dict]) -> List[float]:
    acc = [[] for _ in range(5)]
    for r in recs:
        for _, _, (mu, sg, voiced, _) in r["pool"]:
            for i, c in enumerate((1, 2, 3, 4)):
                acc[i].append(math.log(sg[c]))
            if any(voiced):
                acc[4].append(math.log(sg[core.VP_CH_LOGF0]))
    return [round(sum(v) / len(v), 3) for v in acc]


def piece_vectors(r: dict, logsig: List[float], weights: List[float]) -> List[List[float]]:
    return [core._vp_l2([v * w for v, w in zip(x_raw(mu, sg, logsig), weights)]) for _, _, (mu, sg, _, _) in r["pool"]]


def speaker_pairs(unit: dict) -> List[Tuple[int, int]]:
    """Adjacent turn pairs (unit order) whose ground-truth speaker differs."""
    ts = unit["turns"]
    return [(i, i + 1) for i in range(len(ts) - 1)
            if ts[i]["gt"]["ground_truth_speaker"] != ts[i + 1]["gt"]["ground_truth_speaker"]]


def dprime(within: List[float], cross: List[float]) -> Optional[float]:
    if len(within) < 2 or len(cross) < 2:
        return None
    mw, mc = sum(within) / len(within), sum(cross) / len(cross)
    vw = sum((v - mw) ** 2 for v in within) / (len(within) - 1)
    vc = sum((v - mc) ** 2 for v in cross) / (len(cross) - 1)
    return (mw - mc) / math.sqrt(max(1e-12, 0.5 * (vw + vc)))


def pair_scores(units: List[dict], vecs_of, sim) -> Tuple[List[float], List[float]]:
    """within = pieces of one turn (physically one talker); cross = pieces of adjacent different-speaker turns."""
    within, cross = [], []
    for u in units:
        vecs = [vecs_of(u, i) for i in range(len(u["turns"]))]
        for v in vecs:
            within += [sim(v[p], v[q]) for p in range(len(v)) for q in range(p + 1, len(v))]
        for i, j in speaker_pairs(u):
            cross += [sim(a, b) for a in vecs[i] for b in vecs[j]]
    return within, cross


def fisher_ratios(units: List[dict], recs: Dict[str, dict], logsig: List[float]) -> List[float]:
    d = core.VP_DIM
    ws, wdf, cs, cn = [0.0] * d, 0, [0.0] * d, 0
    for u in units:
        xs = [[x_raw(mu, sg, logsig) for _, _, (mu, sg, _, _) in recs[t["key"]]["pool"]] for t in u["turns"]]
        for pieces in xs:
            if len(pieces) >= 2:
                m = [sum(col) / len(pieces) for col in zip(*pieces)]
                for p in pieces:
                    for k in range(d):
                        ws[k] += (p[k] - m[k]) ** 2
                wdf += len(pieces) - 1
        for i, j in speaker_pairs(u):
            for a in xs[i]:
                for b in xs[j]:
                    for k in range(d):
                        cs[k] += 0.5 * (a[k] - b[k]) ** 2
                    cn += 1
    return [(cs[k] / max(1, cn)) / max(1e-9, ws[k] / max(1, wdf)) for k in range(d)]


# ------------------------------------------------------------------------------------------- pipeline configs
# In-run memo of constrained_voiceprint_diarize (a pure function of its inputs): the Val grid re-runs the same
# sessions under 48 configs and only alpha changes the diarization. Used for the grid / Train checks only, never
# for runs whose latency or memory is reported.
_DIAR_MEMO: Dict[Any, Dict[str, Any]] = {}


@contextmanager
def patched(bicl: bool = True, bicl_sim: float = core.BICL_MERGE_SIM, gift: bool = True, gsn: bool = True,
            gsn_iters: int = core.GSN_ITERATIONS, alpha: float = core.E2CP_ALPHA, lexical: bool = True,
            graph_roles: bool = True, transitions: bool = True, capture: Optional[List] = None, memo: bool = False):
    saved: Dict[str, Any] = {}

    def put(name: str, val: Any) -> None:
        saved.setdefault(name, getattr(core, name))
        setattr(core, name, val)

    o_cvd, o_gsn, o_agg, o_e2cp = (core.constrained_voiceprint_diarize, core.gsn_propagate_session,
                                   core.aggregate_cross_turn_slots, core.e2cp_adjust_affinity)
    try:
        put("BICL_MERGE_SIM", bicl_sim)
        def cvd(turns, graph=None, speaker_roles=None, **_):
            if not memo:
                return o_cvd(turns, graph, speaker_roles, alpha=alpha)
            key = (alpha, lexical, repr([vars(t) for t in turns]),
                   json.dumps(graph, sort_keys=True, default=str), repr(sorted((speaker_roles or {}).items())))
            if key not in _DIAR_MEMO:
                _DIAR_MEMO[key] = o_cvd(turns, graph, speaker_roles, alpha=alpha)
            return _DIAR_MEMO[key]
        put("constrained_voiceprint_diarize", cvd)
        put("gsn_propagate_session", lambda *a, **k: o_gsn(*a, **{**k, "iterations": gsn_iters}))
        if not gsn:
            put("aggregate_cross_turn_slots", lambda *a, **k: o_agg(*a[:5], **{**k, "gsn": None}))
        if not bicl:
            put("bi_cl_disentangle_threads", lambda turns, thread_of=None, **_: (
                dict(thread_of or {t.turn_id: t.thread_id for t in turns}), {"disabled": True}))
        if not gift:
            put("gift_amend_intents", lambda *a, **k: {})
            put("gift_boost_transitions", lambda turns, labels, graph, amendments, transitions, *a, **k: transitions)
        if not (gift and graph_roles):
            put("gift_infer_roles", lambda *a, **k: {})
        if not transitions:   # no MIntRec transition chains (REQUEST_TO_REFUSAL / COMPLIANCE ...) reach the fusion
            put("detect_mintrec_transitions", lambda *a, **k: [])
            put("gift_boost_transitions", lambda *a, **k: [])
        if not lexical:
            put("voiceprint_lexical_constraints", lambda *a, **k: [])
        if capture is not None:
            def e2cp(a, z, alpha_=alpha):
                ah, q = o_e2cp(a, z, alpha_)
                capture.append(ah)
                return ah, q
            put("e2cp_adjust_affinity", e2cp)
        yield
    finally:
        for name, val in saved.items():
            setattr(core, name, val)


TABLE_B_CONFIGS = (
    ("C1", "Baseline Sliding Window / Sequential (HRED-style, no disentanglement, raw P0 keywords)", None),
    ("C2", "Greedy TD/MF + Hypothetical P0 Guard", dict(bicl=False, gift=False, gsn=False, transitions=False)),
    ("C3", "+ Bi-CL Session-Prototype Disentanglement", dict(bicl=True, gift=False, gsn=False, transitions=False)),
    ("C4", "+ GIFT 4-Edge Topology & Replied-By Amendment", dict(bicl=True, gift=True, gsn=False, transitions=False)),
    ("C5", "+ GSN 3-iter BIF + MIntRec Transition Chains (Full v7.7)",
     dict(bicl=True, gift=True, gsn=True, transitions=True)),
)
TABLE_A_MODELS = (
    ("M1", "Acoustic Baseline (ZCR/RMS K-means)"),
    ("M2", "Unconstrained 24-d Spectral Voiceprint (StatPool only, alpha=0) + Train role priors"),
    ("M3", "Lexical self-cues only (EXTENDED_ROLE_CUES; no Train priors, no GIFT graph, no voiceprint)"),
    ("M4", "Ours v7.7 (StatPool 75%-Anchor + Turn-to-Diarize E2CP + GIFT Graph) + Train role priors"),
    ("M5", "Train-split speaker voiceprint prototypes + role priors (nearest prototype; offline eval only)"),
)


def make_session(unit: dict) -> core.DialogueSessionState:
    turns = []
    for i, t in enumerate(unit["turns"], 1):
        vp = t.get("vp") or {}
        turns.append(core.UtteranceTurn(
            i, t["start"], t["speaker_id"], t["role_input"], t["text"], addressee=t["hint_addressee"],
            thread_id=t["hint_thread"], rms_energy=float(vp.get("rms_energy") or 0.0),
            zcr_pitch=float(vp.get("zcr_pitch") or 0.0), end_time=t["end"], voiceprint=vp.get("voiceprint"),
            voiceprint_subvectors=vp.get("sub_voiceprints"), voiceprint_source=vp.get("source", ""),
            near_field_score=float(vp.get("near_field_score", -1.0))))
    return core.DialogueSessionState(session_id=unit["unit_id"], case_id="TVT_EVAL", turns=turns,
                                     role_priors=dict(unit.get("role_priors") or {}))


def run_engine(unit: dict, cfg: dict) -> Tuple[dict, float]:
    session = make_session(unit)
    eng = core.XiaoZhaoMPCEngine()
    with patched(**cfg):
        t0 = time.perf_counter()
        payload = eng.analyze_session(session)
        ms = 1000.0 * (time.perf_counter() - t0)
    return payload, ms


def _thread_links(issues: set, res: set, thread: List[str]) -> set:
    return {(i, r) for i in issues for r in res if i < r and thread[i - 1] == thread[r - 1]}


def predict_config(unit: dict, cfg: Optional[dict]) -> Dict[str, Any]:
    """Per-turn predictions (index-aligned with unit['turns'], engine turn ids = index + 1)."""
    ts = unit["turns"]
    eng = core.XiaoZhaoMPCEngine()
    if cfg is None:   # C1: one sequential context, previous-speaker addressee, raw keyword P0, unfiltered fusion
        t0 = time.perf_counter()
        sess = make_session(unit)
        prev_other = [next((ts[j]["speaker_id"] for j in range(i - 1, -1, -1)
                            if ts[j]["speaker_id"] != t["speaker_id"]), "") for i, t in enumerate(ts)]
        fusion = core.aggregate_cross_turn_slots(sess.turns, {}, {x.turn_id: "T_ALL" for x in sess.turns})
        issues = {x["turn_id"] for x in fusion["issue_triggers"]}
        res = {x["turn_id"] for x in fusion["proposed_resolutions"]}
        thread = ["T_ALL"] * len(ts)
        return {"thread": thread, "addr": [(a, "") for a in prev_other],
                "p0": [any(k in t["text"] for k in eng.P0_EMERGENCY_KW) for t in ts], "issues": issues,
                "resolutions": res, "links": _thread_links(issues, res, thread), "speakers": None,
                "consensus": fusion["consensus_state"], "ms": 1000.0 * (time.perf_counter() - t0)}
    payload, ms = run_engine(unit, cfg)
    td = payload["turns_detail"]
    use_gift = cfg.get("gift", True)
    addr = []
    for d in td:
        sp = d["addressee_speaker"] or ""
        if use_gift and d.get("gift_reply_to") and sp not in ("system_xiaozhao", "ALL"):   # vocatives win
            sp = td[d["gift_reply_to"] - 1]["speaker_id"]
        addr.append((sp, d["who_to_whom_role"].split(" -> ")[-1]))
    fusion = payload["dst_slots"]["cross_turn_fusion"]
    thread = [d["source_thread_id"] for d in td]
    issues = {x["turn_id"] for x in fusion["issue_triggers"]}
    res = {x["turn_id"] for x in fusion["proposed_resolutions"]}
    links = _thread_links(issues, res, thread)
    if cfg.get("gsn", True):
        links |= {(g["origin_turn"], r["turn_id"]) for r in fusion["proposed_resolutions"]
                  for g in r.get("gsn_linked_issues", []) if g["origin_turn"] != r["turn_id"]}
    has_vp = any(t.get("vp") for t in ts)
    return {"thread": thread, "addr": addr, "p0": [eng.has_live_p0(t["text"]) for t in ts], "issues": issues,
            "resolutions": res, "links": links, "speakers": [d["refined_speaker"] for d in td] if has_vp else None,
            "consensus": fusion["consensus_state"], "ms": ms, "payload": payload}


# ------------------------------------------------------------------------------------------- metrics
def purity_counts(pred: List[str], gt: List[str]) -> Tuple[int, int, int]:
    def major(a: List[str], b: List[str]) -> int:
        groups: Dict[str, Dict[str, int]] = {}
        for x, y in zip(a, b):
            groups.setdefault(x, {}).setdefault(y, 0)
            groups[x][y] += 1
        return sum(max(g.values()) for g in groups.values())
    return major(pred, gt), major(gt, pred), len(pred)


def pr_f(p_num: int, i_num: int, n: int) -> Tuple[float, float, float]:
    if not n:
        return 0.0, 0.0, 0.0
    p, i = p_num / n, i_num / n
    return p, i, (2 * p * i / (p + i) if p + i else 0.0)


def f1(r: float, p: Optional[float]) -> float:
    return 2 * r * p / (r + p) if p and r + p else 0.0


def gt_links(unit: dict) -> Dict[Tuple[int, int], int]:
    """(issue idx, resolution idx) pairs of one labelled thread, issue first -> reply-chain hops (99 = not on the
    resolution's reply chain, i.e. linked only through the thread)."""
    ts = unit["turns"]
    pos = {(t["src"], str(t["turn_id"])): k + 1 for k, t in enumerate(ts)}
    parent = {k + 1: pos.get((t["src"], t["gt"]["ground_truth_reply_to"])) for k, t in enumerate(ts)}
    out = {}
    for i, a in enumerate(ts, 1):
        for r, b in enumerate(ts, 1):
            ga, gb = a["gt"], b["gt"]
            if (i < r and ga["ground_truth_slot"] == "issue" and gb["ground_truth_slot"] == "resolution"
                    and ga["ground_truth_thread"] == gb["ground_truth_thread"] != "noise"):
                hop, cur = 0, r
                while cur and cur != i and hop < 50:
                    cur, hop = parent.get(cur), hop + 1
                out[(i, r)] = hop if cur == i else 99
    return out


def gt_consensus(unit: dict) -> str:
    """Gold consensus state: the engine's own state rule (aggregate_cross_turn_slots) applied to the gold labels.
    IMMEDIATE_ESCALATION if a gold live P0 intent or a live crisis chain occurs; CONSENSUS_ACTIONABLE if some gold
    thread has a resolution (after its first issue, when it has one), >= 2 gold speakers and no unanswered
    REQUEST_TO_REFUSAL chain (core.trailing_refusal; chains = MINTREC_TRANSITION_RULES over gold fine intents /
    threads / speakers); else OPEN_DISCUSSION."""
    ts = unit["turns"]
    sess = make_session(unit)
    turns = [dataclasses.replace(t, speaker_id=x["gt"]["ground_truth_speaker"]) for t, x in zip(sess.turns, ts)]
    thread_of = {t.turn_id: ("T_BACKCHANNEL_NOISE" if x["gt"]["ground_truth_thread"] == "noise"
                             else "G:" + x["gt"]["ground_truth_thread"]) for t, x in zip(turns, ts)}
    chains = core.detect_mintrec_transitions(turns, [{"fine": x["gt"]["ground_truth_mintrec_fine"]} for x in ts],
                                             thread_of)
    if (any("P0_EMERGENCY_ALERT" in x["gt"]["ground_truth_intents"] for x in ts)
            or any(c["type"] == "CRISIS_ESCALATION_CHAIN" and c.get("live_hazard") for c in chains)):
        return "IMMEDIATE_ESCALATION"
    for th in {v for v in thread_of.values() if v != "T_BACKCHANNEL_NOISE"}:
        idx = [i for i, t in enumerate(turns) if thread_of[t.turn_id] == th]
        slots = [ts[i]["gt"]["ground_truth_slot"] for i in idx]
        first_issue = slots.index("issue") if "issue" in slots else -1
        mine = [c for c in chains if c["thread_id"] == th]
        gold_res = [{"turn_id": turns[i].turn_id} for k, i in enumerate(idx)
                    if k > first_issue and slots[k] == "resolution"]
        if gold_res and len({turns[i].speaker_id for i in idx}) >= 2 and not core.trailing_refusal(mine, gold_res):
            return "CONSENSUS_ACTIONABLE"
    return "OPEN_DISCUSSION"


def score_table_b(units: List[dict], preds: Dict[str, dict]) -> Dict[str, Any]:
    pn = inn = n = 0
    sp_p = sp_i = sp_n = 0
    a_ok = a_n = 0
    s_hit = s_gt = s_pred = s_tp_pred = 0
    l_tp = l_gt = l_pred = l_tp_pred = m_tp = m_gt = 0
    far_fp = far_n = rec_tp = rec_n = 0
    c_ok = c_n = 0
    c_gt: Dict[str, int] = {}
    ms = turns = 0.0
    for u in units:
        pr, ts = preds[u["unit_id"]], u["turns"]
        g_state = u.get("gt_consensus") or gt_consensus(u)
        c_gt[g_state] = c_gt.get(g_state, 0) + 1
        c_n += 1
        c_ok += int(pr["consensus"] == g_state)
        idx = [i for i, t in enumerate(ts) if t["gt"]["ground_truth_thread"] != "noise"]
        a, b, c = purity_counts([pr["thread"][i] for i in idx], [ts[i]["gt"]["ground_truth_thread"] for i in idx])
        pn, inn, n = pn + a, inn + b, n + c
        if pr.get("speakers"):
            a, b, c = purity_counts(pr["speakers"], [t["gt"]["ground_truth_speaker"] for t in ts])
            sp_p, sp_i, sp_n = sp_p + a, sp_i + b, sp_n + c
        roles = {t["speaker_id"]: t["gt"]["ground_truth_role"] for t in ts} if u["kind"] == "scripted_dialogue" \
            else {}
        present = {(t["src"], str(t["turn_id"])) for t in ts}
        for i, t in enumerate(ts):
            g = t["gt"]
            rt = g["ground_truth_reply_to"]
            if g["ground_truth_addressee"] and (not rt or (t["src"], rt) in present):
                a_n += 1
                sp, role = pr["addr"][i]
                a_ok += int(sp == g["ground_truth_addressee"] or
                            (not sp and role != "unknown" and roles.get(g["ground_truth_addressee"]) == role))
            want = {"issue": pr["issues"], "resolution": pr["resolutions"]}
            if g["ground_truth_slot"]:
                s_gt += 1
                s_hit += int(i + 1 in want[g["ground_truth_slot"]])
            if u["kind"] == "interleaved":   # same texts as the source segments: P0 is counted once, there
                continue
            raw_kw = any(k in t["text"] for k in core.XiaoZhaoMPCEngine.P0_EMERGENCY_KW)
            if "P0_EMERGENCY_ALERT" in g["ground_truth_intents"]:
                rec_n += 1
                rec_tp += int(pr["p0"][i])
            elif raw_kw:
                far_n += 1
                far_fp += int(pr["p0"][i])
        gt_slot = {i + 1: t["gt"]["ground_truth_slot"] for i, t in enumerate(ts)}
        s_pred += len(pr["issues"]) + len(pr["resolutions"])
        s_tp_pred += sum(1 for x in pr["issues"] if gt_slot.get(x) == "issue")
        s_tp_pred += sum(1 for x in pr["resolutions"] if gt_slot.get(x) == "resolution")
        gl = gt_links(u)
        l_gt += len(gl)
        l_tp += sum(1 for k in gl if k in pr["links"])
        m_gt += sum(1 for h in gl.values() if h >= 2)
        m_tp += sum(1 for k, h in gl.items() if h >= 2 and k in pr["links"])
        l_pred += len(pr["links"])
        l_tp_pred += sum(1 for k in pr["links"] if k in gl)
        ms += pr["ms"]
        turns += len(ts)
    p, i, f = pr_f(pn, inn, n)
    comp = s_hit / s_gt if s_gt else 0.0
    prec = s_tp_pred / s_pred if s_pred else None
    l_rec = l_tp / l_gt if l_gt else 0.0
    l_prec = l_tp_pred / l_pred if l_pred else None
    _, _, spf = pr_f(sp_p, sp_i, sp_n)
    return {"thread_purity": p, "thread_cohesion": i, "thread_f": f, "thread_turns": n,
            "speaker_purity": sp_p / sp_n if sp_n else None, "speaker_inverse_purity": sp_i / sp_n if sp_n else None,
            "speaker_f": spf if sp_n else None, "speaker_turns": sp_n,
            "addressee_acc": a_ok / a_n if a_n else 0.0, "addressee_n": a_n,
            "slot_completeness": comp, "slot_precision": prec, "slot_f1": f1(comp, prec), "slot_gt": s_gt,
            "link_recall": l_rec, "link_precision": l_prec, "link_f1": f1(l_rec, l_prec), "link_gt": l_gt,
            "multihop_link_recall": m_tp / m_gt if m_gt else None, "multihop_link_gt": m_gt, "multihop_link_tp": m_tp,
            "consensus_acc": c_ok / c_n if c_n else None, "consensus_n": c_n, "consensus_ok": c_ok,
            "consensus_gt_states": c_gt,
            "mhc_acc": (m_tp + c_ok) / (m_gt + c_n) if m_gt + c_n else None, "mhc_n": m_gt + c_n,
            "p0_far": far_fp / far_n if far_n else None, "p0_far_n": far_n, "p0_far_fp": far_fp,
            "p0_recall": rec_tp / rec_n if rec_n else None, "p0_recall_n": rec_n,
            "ms_per_turn": ms / turns if turns else 0.0}


def objective(m: dict) -> float:
    parts = [m["thread_f"], m["addressee_acc"], m["slot_f1"], m["link_f1"]]
    if m["speaker_f"] is not None:
        parts.append(m["speaker_f"])
    return sum(parts) / len(parts)


# ------------------------------------------------------------------------------------------- Train-learned priors
def learn_role_priors(train_units: List[dict]) -> Dict[str, Dict[str, str]]:
    votes: Dict[str, Dict[str, Dict[str, int]]] = {}
    for u in train_units:
        for t in u["turns"]:
            r = t["gt"]["ground_truth_role"]
            if r != "unknown":
                v = votes.setdefault(u["source"], {}).setdefault(t["speaker_id"], {})
                v[r] = v.get(r, 0) + 1
    return {src: {sp: max(v.items(), key=lambda kv: (kv[1], kv[0]))[0] for sp, v in sps.items()}
            for src, sps in votes.items()}


def learn_prototypes(train_units: List[dict]) -> Dict[str, Dict[str, List[float]]]:
    """Per recording, per diarizer speaker: L2 mean of the Train turns' voiceprints (in memory only)."""
    acc: Dict[str, Dict[str, List[float]]] = {}
    for u in train_units:
        for t in u["turns"]:
            v = (t.get("vp") or {}).get("voiceprint")
            if v:
                cur = acc.setdefault(u["source"], {}).setdefault(t["speaker_id"], [0.0] * len(v))
                for k, x in enumerate(v):
                    cur[k] += x
    return {src: {sp: core._vp_l2(v) for sp, v in sps.items()} for src, sps in acc.items()}


def _report(src: str) -> Optional[dict]:
    path = OUT_DIR / f"{src}_mpc_report.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fullrec_role_priors(cuts: Dict[str, dict]) -> Dict[str, Dict[str, str]]:
    """Per recording: session-level cue (+ GIFT graph) role inference over the unlabelled, non-noise merged turns
    that start before the Val cut (no labels, no future turns). -> {src: {speaker: role}} (resolved speakers only)."""
    from interview_analyzer import normalize_asr_text
    out: Dict[str, Dict[str, str]] = {}
    for src, c in cuts.items():
        d = _report(src)
        if d is None:
            continue
        noise = set(d["general_disentangle"].get("noise_turn_ids") or [])
        turns: List[core.UtteranceTurn] = []
        for k, t in enumerate(d["merged_turns"], 1):
            if k not in noise and float(t["start"]) < c["val_start_sec"]:
                turns.append(core.UtteranceTurn(len(turns) + 1, float(t["start"]), t["speaker"], "unknown",
                                                normalize_asr_text(t["text"]), end_time=float(t["end"])))
        roles = core.infer_session_speaker_roles(turns) if turns else {}
        out[src] = {sp: r for sp, r in roles.items() if r != "unknown"}
    return out


def fullrec_vp_turns(cuts: Dict[str, dict], windows: List[dict]) -> List[dict]:
    """Up to FULLREC_VP_MAX unlabelled substantive turns per recording, evenly spaced over [0, Val cut), outside the
    golden windows: pseudo one-turn windows for the unsupervised voiceprint channel statistics."""
    out = []
    for src, c in cuts.items():
        d = _report(src)
        if d is None:
            continue
        noise = set(d["general_disentangle"].get("noise_turn_ids") or [])
        spans = [(min(t["start"] for t in w["turns"]), max(t["end"] for t in w["turns"]))
                 for w in windows if w["source"] == src]
        cand = [(k, t) for k, t in enumerate(d["merged_turns"], 1)
                if k not in noise and float(t["end"]) < c["val_start_sec"]
                and float(t["end"]) - float(t["start"]) >= FULLREC_MIN_DUR
                and not any(a - 1.0 <= float(t["start"]) <= b + 1.0 for a, b in spans)]
        step = max(1, len(cand) // FULLREC_VP_MAX)
        for k, t in cand[::step][:FULLREC_VP_MAX]:
            key = f"{src}:full{k}"
            out.append({"unit_id": key, "source": src, "turns": [
                {"key": key, "start": float(t["start"]), "end": float(t["end"]), "text": t["text"]}]})
    return out


# ------------------------------------------------------------------------------------------- Table A models
def roles_for(unit: dict, speakers: List[str], graph_roles: bool, priors: Optional[Dict[str, str]]) -> List[str]:
    sess = make_session(unit)
    turns = [dataclasses.replace(t, speaker_id=sp) for t, sp in zip(sess.turns, speakers)]
    with patched(graph_roles=graph_roles):
        roles = core.infer_session_speaker_roles(turns, {t.turn_id: t.inferred_role for t in turns}, priors or {})
    return [roles.get(sp, "unknown") for sp in speakers]


def m1_cluster(recs: List[dict], ac_norm) -> List[str]:
    (m0, m1), (s0, s1) = ac_norm
    feats = [[(math.log(r["vp"]["rms_energy"] + 1.0) - m0) / s0, (r["vp"]["zcr_pitch"] - m1) / s1] for r in recs]
    return [f"speaker_{c + 1}" for c in core._vp_kmeans(feats, max(1, min(3, len(feats))))]


def m2_cluster(unit: dict) -> List[str]:
    sess = make_session(unit)
    with patched(lexical=False, alpha=0.0):
        vp = core.constrained_voiceprint_diarize(sess.turns, None, None)
    return [vp["refined_speaker"][i + 1] for i in range(len(unit["turns"]))]


def m5_assign(unit: dict, protos: Dict[str, List[float]]) -> List[str]:
    out = []
    for t in unit["turns"]:
        v = (t.get("vp") or {}).get("voiceprint")
        out.append(max(protos, key=lambda sp: core._vp_cos(v, protos[sp])) if v and protos else t["speaker_id"])
    return out


def table_a_unit(unit: dict, recs: List[dict], ac_norm, best: dict, protos: Dict[str, List[float]]) -> Dict[str, Any]:
    ts = unit["turns"]
    priors = unit.get("role_priors") or {}
    out: Dict[str, Any] = {}
    (m0, m1), (s0, s1) = ac_norm

    def ac_vec(rms: float, zcr: float) -> List[float]:
        return [(math.log(rms + 1.0) - m0) / s0, (zcr - m1) / s1]

    t0 = time.perf_counter()
    lab = m1_cluster(recs, ac_norm)   # arbitrary cluster names: no recording priors apply
    out["M1"] = {"speakers": lab, "roles": roles_for(unit, lab, False, None),
                 "ms": 1000.0 * (time.perf_counter() - t0) + sum(r["ac_ms"] for r in recs)}
    t0 = time.perf_counter()
    lab = m2_cluster(unit)            # refined clusters are mapped onto diarizer names -> priors apply
    out["M2"] = {"speakers": lab, "roles": roles_for(unit, lab, False, priors),
                 "ms": 1000.0 * (time.perf_counter() - t0) + sum(r["vp_ms"] for r in recs)}
    t0 = time.perf_counter()
    lab = [t["speaker_id"] for t in ts]
    out["M3"] = {"speakers": lab, "roles": roles_for(unit, lab, False, None),   # direct lexical self-cues only
                 "ms": 1000.0 * (time.perf_counter() - t0)}
    cap: List = []
    payload, ms = run_engine(unit, {**best_cfg(best), "capture": cap})
    td = payload["turns_detail"]
    out["M4"] = {"speakers": [d["refined_speaker"] for d in td], "roles": [d["inferred_role"] for d in td],
                 "ms": ms + sum(r["vp_ms"] for r in recs)}
    # role-ID attribution (which M4 component resolves which speaker)
    no_graph, _ = run_engine(unit, {**best_cfg(best), "graph_roles": False})
    out["_role_ablation"] = {
        "cues_graph": roles_for(unit, lab, True, None),
        "cues_priors": roles_for(unit, lab, False, priors),
        "m4_no_graph": [d["inferred_role"] for d in no_graph["turns_detail"]],
        "m4_no_priors": [d["inferred_role"] for d in
                         run_engine({**unit, "role_priors": {}}, best_cfg(best))[0]["turns_detail"]]}
    out["_role_sources"] = dict(payload.get("speaker_role_sources") or {})
    t0 = time.perf_counter()
    lab = m5_assign(unit, protos)
    out["M5"] = {"speakers": lab, "roles": [priors.get(sp, "unknown") for sp in lab],
                 "ms": 1000.0 * (time.perf_counter() - t0) + sum(r["vp_ms"] for r in recs)}
    pieces = [len(r["vp"]["sub_voiceprints"]) for r in recs]
    node0 = [sum(pieces[:i]) for i in range(len(pieces))]
    ah_full: Dict[Tuple[int, int], float] = {}
    off = 0
    for ah in cap:
        for p in range(len(ah)):
            for q in range(len(ah)):
                ah_full[(off + p, off + q)] = ah[p][q]
        off += len(ah)
    out["_dprime_inputs"] = {"ac": [[ac_vec(*pa) for pa in r["pieces_ac"]] for r in recs],
                             "vp": [r["vp"]["sub_voiceprints"] for r in recs],
                             "nodes": [[node0[i] + k for k in range(pieces[i])] for i in range(len(recs))],
                             "e2cp": ah_full}
    out["_gt_speakers"] = [t["gt"]["ground_truth_speaker"] for t in ts]
    return out


def best_cfg(best: dict) -> dict:
    return dict(bicl=True, gift=True, gsn=True, bicl_sim=best["bicl"], gsn_iters=best["gsn"], alpha=best["alpha"])


def score_table_a(units: List[dict], res: Dict[str, dict], any_verified_speaker: bool) -> Dict[str, Dict[str, Any]]:
    euclid_sim = lambda a, b: -math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))   # noqa: E731
    table: Dict[str, Dict[str, Any]] = {}
    for mid, _ in TABLE_A_MODELS:
        pn = inn = n = r_ok = r_n = 0
        ms = turns = 0.0
        for u in units:
            r = res[u["unit_id"]]
            a, b, c = purity_counts(r[mid]["speakers"], r["_gt_speakers"])
            pn, inn, n = pn + a, inn + b, n + c
            for t, role in zip(u["turns"], r[mid]["roles"]):
                g = t["gt"]["ground_truth_role"]
                if g != "unknown":
                    r_n += 1
                    r_ok += int(role == g)
            ms += r[mid]["ms"]
            turns += len(u["turns"])
        if mid == "M1":
            w, c_ = pair_scores(units, lambda u, i: res[u["unit_id"]]["_dprime_inputs"]["ac"][i], euclid_sim)
        elif mid == "M2":
            w, c_ = pair_scores(units, lambda u, i: res[u["unit_id"]]["_dprime_inputs"]["vp"][i], core._vp_cos)
        elif mid == "M4":
            w, c_ = pair_scores(units, lambda u, i: [(u["unit_id"], k) for k in
                                                    res[u["unit_id"]]["_dprime_inputs"]["nodes"][i]],
                                lambda a, b: res[a[0]]["_dprime_inputs"]["e2cp"][(a[1], b[1])])
        else:
            w, c_ = [], []
        p, i, _ = pr_f(pn, inn, n)
        table[mid] = {"dprime": dprime(w, c_), "within_pairs": len(w), "cross_pairs": len(c_),
                      "speaker_purity": p, "speaker_inverse_purity": i,
                      "purity_is_label_source": mid == "M3" and not any_verified_speaker,
                      "role_acc": r_ok / r_n if r_n else 0.0, "role_n": r_n,
                      "ms_per_turn": ms / turns if turns else 0.0}
    abl: Dict[str, Dict[str, Any]] = {}
    for key in ("cues_graph", "cues_priors", "m4_no_graph", "m4_no_priors"):
        ok = n_ = 0
        for u in units:
            for t, role in zip(u["turns"], res[u["unit_id"]]["_role_ablation"][key]):
                if t["gt"]["ground_truth_role"] != "unknown":
                    n_ += 1
                    ok += int(role == t["gt"]["ground_truth_role"])
        abl[key] = {"role_acc": ok / n_ if n_ else 0.0, "role_n": n_}
    src_count: Dict[str, int] = {}
    graph_turns = []
    for u in units:
        rs = res[u["unit_id"]]["_role_sources"]
        for sp, srcname in rs.items():
            src_count[srcname] = src_count.get(srcname, 0) + 1
            if srcname == core.GRAPH_INFERRED_SOURCE:
                graph_turns.append(f"{u['unit_id'].split('#')[0]}:{sp}")
    table["_role_attribution"] = {**abl, "m4_speaker_role_sources": src_count, "graph_inferred": graph_turns}
    return table


# ------------------------------------------------------------------------------------------- schema audit
def schema_audit(train_units: List[dict], test_units: List[dict]) -> Dict[str, Any]:
    """The schema stays fixed; this lists keywords that fire on Test turns but on no Train turn (authored with the
    Test windows visible) and the Train agreement of each role cue."""
    tr = [core._norm(t["text"]) for u in train_units for t in u["turns"]]
    te = [core._norm(t["text"]) for u in test_units for t in u["turns"]]
    report: Dict[str, Any] = {}
    for name in ("EXTENDED_ROLE_CUES", "ISSUE_TRIGGER_CATALOG", "RESOLUTION_CATALOG"):
        cat = getattr(core, name)
        kws = [c for _, cues in cat for c in cues]
        report[name] = {"keywords": len(kws), "fire_on_train": sum(1 for c in kws if any(c in x for x in tr)),
                        "test_only": [c for c in kws if any(c in x for x in te) and not any(c in x for x in tr)]}
    prec = []
    for role, cues in core.EXTENDED_ROLE_CUES:
        for c in cues:
            hits = [t for u in train_units for t in u["turns"] if c in core._norm(t["text"])]
            if hits:
                prec.append({"role": role, "cue": c, "hits": len(hits),
                             "agree": sum(1 for t in hits if t["gt"]["ground_truth_role"] == role),
                             "known": sum(1 for t in hits if t["gt"]["ground_truth_role"] != "unknown")})
    report["role_cue_agreement_train"] = prec
    return report


# ------------------------------------------------------------------------------------------- export
def export_dataset(units: List[dict], meta: dict, scheme_splits: Optional[Dict[str, Dict[str, str]]] = None) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    for u in units:
        for t in u["turns"]:
            vp = t.get("vp") or {}
            rows.append({
                "key": t["key"], "split": t["split"],
                **{col: m.get(t["key"], t["split"]) for col, m in (scheme_splits or {}).items()},
                "unit_id": u["unit_id"], "kind": u["kind"],
                "source": u["source"], "window": u["window"], "turn_id": t["turn_id"],
                "start": round(t["start"], 2), "end": round(t["end"], 2), "speaker_id": t["speaker_id"],
                "asr_text_raw": t["asr_text_raw"], "text_clean": t["text"], **t["gt"],
                "label_source": t["label_source"], "human_verified": t["human_verified"],
                "annotator_notes": t["annotator_notes"],
                "voiceprint_source": vp.get("source", "none (scripted text)"),
                "near_field_score": vp.get("near_field_score", -1.0), "f0_hz": vp.get("f0_hz", 0.0),
                "voiceprint_24d": [round(v, 4) for v in vp.get("voiceprint") or []],
                "voiceprint_subvectors": [[round(v, 4) for v in s] for s in vp.get("sub_voiceprints") or []]
                if vp.get("source") == core.VP_SOURCE_WAV else [],
            })
    DATASET_JSON.write_text(json.dumps({**meta, "turns": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    cols = [k for k in rows[0] if k != "voiceprint_subvectors"]
    with DATASET_CSV.open("w", encoding="utf-8-sig", newline="") as f:   # BOM: Excel opens UTF-8 Chinese
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "ground_truth_intents": ";".join(r["ground_truth_intents"]),
                        "voiceprint_24d": " ".join(f"{v:.4f}" for v in r["voiceprint_24d"])})


def write_core_params(params: Dict[str, List[float]], path: Path = ROOT / "src" / "mpc_core.py") -> bool:
    src = path.read_text(encoding="utf-8")
    new = src
    for k in VP_KEYS:
        vals = ", ".join(repr(round(v, 3)) for v in params[k])
        pat = re.compile(rf"^{k} = \[[^\]]*\]", re.M | re.S)
        lines, cur = [], f"{k} = ["
        for tok in vals.split(", "):
            piece = tok + ", "
            if len(cur) + len(piece) > 116:
                lines.append(cur.rstrip())
                cur = " " * (len(k) + 4)
            cur += piece
        lines.append(cur.rstrip(", ") + "]")
        new = pat.sub(lambda _m: "\n".join(lines), new, count=1)
    if new != src:
        path.write_text(new, encoding="utf-8")
    return new != src


# ------------------------------------------------------------------------------------------- report helpers
def pct(v: Optional[float], nd: int = 1) -> str:
    return "n/a" if v is None else f"{100.0 * v:.{nd}f}"


def f3(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def md_table(header: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def peak_kib(fn) -> float:
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1] / 1024.0
    finally:
        tracemalloc.stop()


def interview4_summary() -> List[str]:
    path = OUT_DIR / "interview4_mpc_report.json"
    if not path.exists():
        return ["(transcripts_out/interview4_mpc_report.json not found)"]
    d = json.loads(path.read_text(encoding="utf-8"))
    mt = d["merged_turns"]
    span = [t for t in mt if 300.0 <= t["start"] < 840.0]
    fine: Dict[str, int] = {}
    for t in span:
        lb = t.get("mintrec") or {}
        k = lb.get("fine", "UNK") if isinstance(lb, dict) else str(lb)
        fine[k] = fine.get(k, 0) + 1
    eng = core.XiaoZhaoMPCEngine()
    intents: Dict[str, int] = {}
    for t in span:
        for i in eng.detect_domain_intents(t["text"]):
            intents[i] = intents.get(i, 0) + 1
    by_sp: Dict[str, int] = {}
    for t in span:
        by_sp[t["speaker"]] = by_sp.get(t["speaker"], 0) + 1
    lines = [f"interview4 (lab meeting, {d['duration_sec'] / 60:.1f} min, {d['merged_turn_count']} merged turns; "
             "excluded from the Jia-An handover corpus and from this benchmark)",
             f"- 05:00-14:00: {len(span)} turns, diarizer speakers {dict(sorted(by_sp.items()))}",
             f"- MIntRec fine labels: {dict(sorted(fine.items(), key=lambda kv: -kv[1]))}",
             f"- domain intents fired: {intents or 'none'} (home-care catalog; this is a lab meeting, so few apply)"]
    hs = next((h for h in d["golden_hotspots"] if h["start_mmss"] == "13:16"), None)
    if hs:
        lines.append(f"- Hotspot #{hs['rank']} {hs['start_mmss']}-{hs['end_mmss']}: score {hs['score']}, "
                     f"{hs['speaker_switches']} speaker switches, threads {hs['threads']}")
        for tid in hs["turn_ids"]:
            t = mt[tid - 1]
            lb = t.get("mintrec") or {}
            lines.append(f"    {_mmss(t['start'])} {t['speaker']:9s} "
                         f"{(lb.get('fine') if isinstance(lb, dict) else lb) or '':12s} {_lab_fix(t['text'])}")
    return lines


def _lab_fix(text: str) -> str:
    try:
        from interview_analyzer import normalize_asr_text
        text = normalize_asr_text(text)
    except ImportError:
        pass
    for a, b in LAB_MEETING_FIXES:
        text = text.replace(a, b)
    return text


# ------------------------------------------------------------------------------------------- Table C: audio + simulation
SIM_LEAD_BLEED = (0.2, 0.5, 1.0, 1.5)    # s of the previous talker inside the ASR turn boundary (start)
SIM_TAIL_BLEED = (0.2, 0.5)              # s of the next talker inside the boundary (end)
SIM_SNR_DB = (20.0, 10.0, 5.0, 0.0)      # whole-turn overlap: target-to-interferer power ratio
SIM_SEEDS = (3, 4)
SIM_TURN_S = 3.0


def corpus_segmentation_stats() -> List[Dict[str, Any]]:
    """Per recording: raw Whisper segments -> ASR repetition loops collapsed -> merged turns -> backchannel noise."""
    from interview_analyzer import collapse_asr_loops
    rows = []
    for src in JIA_AN + ("interview4",):
        d = _report(src)
        raw_p = OUT_DIR / f"{src}_raw_stt.json"
        if d is None or not raw_p.exists():
            continue
        segs = json.loads(raw_p.read_text(encoding="utf-8"))["segments"]
        col = collapse_asr_loops(segs)
        loops = [x["asr_loop_collapsed"] for x in col if x.get("asr_loop_collapsed")]
        noise = len(d["general_disentangle"].get("noise_turn_ids") or [])
        rows.append({"source": src, "duration_sec": d["duration_sec"], "raw_segments": len(segs),
                     "report_raw_segments": d["raw_segment_count"], "asr_loop_runs": len(loops),
                     "asr_loop_segments_absorbed": sum(n - 1 for n in loops), "max_loop": max(loops, default=0),
                     "merged_turns": d["merged_turn_count"], "noise_turns": noise,
                     "substantive_turns": d["merged_turn_count"] - noise,
                     "ambient_ignored_noise": (d.get("ambient_stats") or {}).get("ignored_noise"),
                     "in_corpus": src in JIA_AN})
    return rows


@contextmanager
def naive_sampler():
    """Whole-turn analysis window: no 0.25 s collar, no 75 % anchor (the ablation of voiceprint_windows)."""
    orig = core.voiceprint_windows
    core.voiceprint_windows = lambda s, e: ([(s, e, 0.5 * (s + e))], False)
    try:
        yield
    finally:
        core.voiceprint_windows = orig


def _overlap(wins: List[Tuple[float, float, float]], a: float, b: float) -> float:
    tot = sum(we - ws for ws, we, _ in wins) or 1.0
    return sum(max(0.0, min(we, b) - max(ws, a)) for ws, we, _ in wins) / tot


def simulate_audio() -> Dict[str, Any]:
    """Cached wrapper of _simulate_audio (deterministic synthetic signals; keyed by extractor + simulator code)."""
    import hashlib
    import inspect
    src = inspect.getsource(_simulate_audio) + inspect.getsource(core.synth_test_voice) + repr(
        (SIM_LEAD_BLEED, SIM_TAIL_BLEED, SIM_SNR_DB, SIM_SEEDS, SIM_TURN_S, core.TEST_VOICE_LOW, core.TEST_VOICE_HIGH))
    ck = f"sim|{vp_code_hash()}|{params_hash(shipped_vp_params())}|{hashlib.sha1(src.encode()).hexdigest()[:12]}"
    cache = vp_cache()
    if ck not in cache:
        cache[ck] = json.loads(json.dumps(_simulate_audio()))
        _VP_CACHE_STATE["dirty"] = True
    return cache[ck]


def _simulate_audio() -> Dict[str, Any]:
    """Deterministic synthetic benchmark on core.synth_test_voice (two talkers, low / high voice):
    (a) boundary bleed: the other talker occupies the first (last) b s of the ASR turn span;
    (b) whole-turn overlap at SIM_SNR_DB; (c) near-field (headset) vs far-field (1-3 m, reverb) near_field_score.
    Correct = the turn voiceprint is closer (cosine) to the target's clean reference than to the interferer's."""
    voices = {"low": core.TEST_VOICE_LOW, "high": core.TEST_VOICE_HIGH}
    rate = 16000
    ref = {k: core.extract_turn_voiceprint(core.synth_test_voice(v, SIM_TURN_S, 21 + i), rate, 0.0, SIM_TURN_S)
           ["voiceprint"] for i, (k, v) in enumerate(voices.items())}
    bleed: Dict[str, Dict[str, List[Tuple[int, float, float]]]] = {}
    for tgt, other in (("low", "high"), ("high", "low")):
        for seed in SIM_SEEDS:
            cases = [("lead", b) for b in SIM_LEAD_BLEED] + [("tail", b) for b in SIM_TAIL_BLEED]
            for side, b in cases:
                tv = core.synth_test_voice(voices[tgt], SIM_TURN_S, seed)
                ov = core.synth_test_voice(voices[other], b, seed + 50)
                pcm = ov + tv if side == "lead" else tv + ov
                span = SIM_TURN_S + b
                ba, bb = (0.0, b) if side == "lead" else (SIM_TURN_S, span)
                for name, ctx in (("v7.7 collar + 75% anchor", None), ("whole-turn window", naive_sampler)):
                    if ctx is None:
                        vp = core.extract_turn_voiceprint(pcm, rate, 0.0, span)
                    else:
                        with ctx():
                            vp = core.extract_turn_voiceprint(pcm, rate, 0.0, span)
                    m = core._vp_cos(vp["voiceprint"], ref[tgt]) - core._vp_cos(vp["voiceprint"], ref[other])
                    bleed.setdefault(name, {}).setdefault(f"{side} {b:.1f} s", []).append(
                        (int(m > 0), m, _overlap(vp["windows"], ba, bb)))
    snr: Dict[str, List[Tuple[int, float]]] = {}
    for tgt, other in (("low", "high"), ("high", "low")):
        for seed in SIM_SEEDS:
            tv = core.synth_test_voice(voices[tgt], SIM_TURN_S, seed)
            ov = core.synth_test_voice(voices[other], SIM_TURN_S, seed + 50)
            pt = sum(v * v for v in tv) / len(tv)
            po = sum(v * v for v in ov) / len(ov)
            for db in SIM_SNR_DB:
                g = math.sqrt(pt / (po * 10 ** (db / 10.0)))
                mix = [max(-32768, min(32767, int(a + g * c))) for a, c in zip(tv, ov)]
                vp = core.extract_turn_voiceprint(mix, rate, 0.0, SIM_TURN_S)
                m = core._vp_cos(vp["voiceprint"], ref[tgt]) - core._vp_cos(vp["voiceprint"], ref[other])
                snr.setdefault(f"{db:.0f} dB", []).append((int(m > 0), m))
    near, far = [], []
    for v in voices.values():
        for seed in (1, 2, 3, 4):
            near.append(core.extract_turn_voiceprint(core.synth_test_voice(v, SIM_TURN_S, seed), rate, 0.0,
                                                     SIM_TURN_S)["near_field_score"])
            far.append(core.extract_turn_voiceprint(core.synth_test_voice(v, SIM_TURN_S, seed, far_field=True), rate,
                                                    0.0, SIM_TURN_S)["near_field_score"])
    auc = sum((a > b) + 0.5 * (a == b) for a in near for b in far) / (len(near) * len(far))
    thr = 0.5 * (sum(near) / len(near) + sum(far) / len(far))
    acc = (sum(x >= thr for x in near) + sum(x < thr for x in far)) / (len(near) + len(far))
    return {"bleed": bleed, "snr": snr,
            "near_far": {"near": near, "far": far, "auc": auc, "midpoint_threshold": thr, "midpoint_acc": acc,
                         "dprime": dprime(near, far)}}


# ------------------------------------------------------------------------------------------- Table D: P-series prefetcher
P1_WINDOW = 4   # post-hoc sliding window: the last 4 turns before the query
P_SERIES = (("P1", "Post-hoc Sliding Window (no background pool; analyze_session on the last 4 turns at query time)"),
            ("P2", "Basic Keyword Background Pool (no MIntRec transition priority boost, no backchannel filter)"),
            ("P3", "Ours: AmbientIntentPreFetcher + MIntRec Transition Priority Boost + P0 Auto-Fire"))
# probe templates: Arrange-type requests (chain sources), informational mentions (no chain), noise, traps
P_REQUEST = {"BA16_MEDICINE_PURCHASE_INQUIRY": "麻煩順便幫我買普拿疼",
             "GA09_SC09_RESPITE_FORM": "麻煩幫我們申請喘息服務",
             "EXTRA_OUT_OF_SCOPE_CLEANING": "麻煩順便幫我們把廁所跟馬桶刷一刷"}
P_INFO = {"BA16_MEDICINE_PURCHASE_INQUIRY": "上次醫生說普拿疼一天最多吃四顆",
          "GA09_SC09_RESPITE_FORM": "我們下禮拜想申請 GA09 喘息服務跟 SC09 短照"}
P_TRAP = {"BA16_MEDICINE_PURCHASE_INQUIRY": "電視八點檔那個家屬也在申請喘息服務",   # chitchat carrying GA09 words
          "GA09_SC09_RESPITE_FORM": "便當店老闆娘說她也要買普拿疼",               # chitchat carrying BA16 words
          "EXTRA_OUT_OF_SCOPE_CLEANING": "便當店老闆娘說她也要買普拿疼"}
P_GAP = ("今天午餐的雞腿便當冷掉了不好吃。", "嗯嗯", "今天中午的排骨便當菜色很鹹欸，電視八點檔又在重播。", "對啊", "喔喔")
P_REPLY = {"agree": "好", "hesitation": "嗯…我再看看"}
P_QUERY = "小照，家屬剛剛說的那樣可以嗎？"


class KeywordPool(core.AmbientIntentPreFetcher):
    """P2: stage every domain-keyword hit (no chitchat / backchannel filter, no MIntRec gating, no transition
    tracking or priority boost, no P0 auto-fire); firing reuses the shipped fire_on_caregiver_query."""

    def ingest_ambient_turn(self, turn: core.UtteranceTurn) -> dict:
        self.ingested_count += 1
        intents = self.engine.detect_domain_intents(turn.text or "")
        mt = core.classify_mintrec_taxonomy(turn.text or "", turn.rms_energy, turn.zcr_pitch)   # payload metadata only
        for key in intents:
            pkt = self.staged_packets.setdefault(key, {
                "intent_key": key, "stage_status": "STAGED_READY", "turn_ids": [], "texts": [], "mintrec": [],
                "speakers": [], "first_ts": turn.timestamp, "last_ts": turn.timestamp,
                "staging_priority": self.BASE_STAGING_PRIORITY, "transition_chains": []})
            pkt["turn_ids"].append(turn.turn_id)
            pkt["texts"].append(turn.text)
            pkt["mintrec"].append({"turn_id": turn.turn_id, **mt})
            if turn.speaker_id not in pkt["speakers"]:
                pkt["speakers"].append(turn.speaker_id)
            pkt["last_ts"] = turn.timestamp
        if not intents:
            self.ignored_count += 1
        return {"turn_id": turn.turn_id, "status": "STAGED_READY" if intents else "OBSERVED", "staged_keys": intents}


def _p_turns(spec: List[Tuple[str, str]]) -> List[core.UtteranceTurn]:
    return [core.UtteranceTurn(i, 10.0 * i, sp, "unknown", text) for i, (sp, text) in enumerate(spec, 1)]


def p_run(system: str, turns: List[core.UtteranceTurn], query: str) -> Tuple[List[str], List[int], float]:
    """-> (fired intents in rank order, fired turn ids, query-time latency ms) for one probe."""
    ts = turns[-1].timestamp + 10.0 if turns else 0.0
    if system == "P1":
        t0 = time.perf_counter()
        win = [dataclasses.replace(t, turn_id=k) for k, t in enumerate(turns[-P1_WINDOW:], 1)]
        q = core.UtteranceTurn(len(win) + 1, ts, "speaker_3", "caregiver", query)
        pay = core.XiaoZhaoMPCEngine().analyze_session(core.DialogueSessionState("P1", "P", turns=win + [q]))
        ms = 1000.0 * (time.perf_counter() - t0)
        ids = [turns[-len(win):][k - 1].turn_id for k in pay["selected_turn_indices"] if k <= len(win)]
        return [i for i in pay["detected_intents"] if i != "GENERAL_CARE_LOG"], ids, ms
    pf = KeywordPool() if system == "P2" else core.AmbientIntentPreFetcher()
    for t in turns:
        pf.ingest_ambient_turn(t)
    pay = pf.fire_on_caregiver_query(query, ts)
    return [i for i in pay["detected_intents"] if i != "GENERAL_CARE_LOG"], pay["selected_turn_indices"], \
        pay["dispatch_latency_ms"]


def p_probes() -> Dict[str, List[dict]]:
    """Constructed probe set (data simulation; every factor level is listed in the report)."""
    ref = []
    for tgt, req in P_REQUEST.items():
        for trap in ("none", "before", "after"):
            for gap in (0, 2, 5):
                spec = [("speaker_1", P_TRAP[tgt])] if trap == "before" else []
                spec.append(("speaker_2", req))
                spec += [("speaker_1", P_GAP[k % len(P_GAP)]) for k in range(gap)]
                if trap == "after":
                    spec.append(("speaker_1", P_TRAP[tgt]))
                ref.append({"factors": {"trap": trap, "gap": gap}, "turns": _p_turns(spec), "query": P_QUERY,
                            "gold": {tgt}})
    for s in core.build_5_benchmark_scenarios():   # the natural referential queries of the scripted dialogues
        q = next((k for k, t in enumerate(s.turns) if t.text.startswith("小照") and "剛剛" in t.text), None)
        if q:
            ref.append({"factors": {"trap": "scripted", "gap": q - 1}, "turns": s.turns[:q], "query": s.turns[q].text,
                        "gold": set(core.XiaoZhaoMPCEngine().detect_domain_intents(s.turns[q].text)) or
                        {i for t in s.turns[:q] for i in core.XiaoZhaoMPCEngine().detect_domain_intents(t.text)}})
    comp = []
    pairs = (("BA16_MEDICINE_PURCHASE_INQUIRY", "GA09_SC09_RESPITE_FORM"),
             ("EXTRA_OUT_OF_SCOPE_CLEANING", "GA09_SC09_RESPITE_FORM"),
             ("GA09_SC09_RESPITE_FORM", "BA16_MEDICINE_PURCHASE_INQUIRY"))
    for chain, other in pairs:
        for order in ("other_first", "chain_first"):
            for reply in P_REPLY:
                for tail in (0, 3):
                    c = [("speaker_2", P_REQUEST[chain]), ("speaker_3", P_REPLY[reply])]
                    o = [("speaker_4", P_INFO[other])]
                    spec = (o + c if order == "other_first" else c + o)
                    spec += [("speaker_1", P_GAP[2 * k % len(P_GAP)]) for k in range(tail)]
                    comp.append({"factors": {"order": order, "reply": reply, "tail": tail},
                                 "turns": _p_turns(spec), "query": P_QUERY, "gold": {chain}})
    return {"referential": ref, "compliance": comp}


def p_noise_rejection() -> Dict[str, Dict[str, Any]]:
    """Real corpus: every merged turn of the 3 Jia-An recordings, streamed WITHOUT the analyzer's thread labels.
    Gold noise = interview_analyzer's T_BACKCHANNEL_NOISE turns (an upstream rule-based labeller, not a human).
    P2 / P3: rejected = the turn's text is in no staged packet. P1: rejected = not in the query context of its 4-turn window."""
    from interview_analyzer import normalize_asr_text
    out: Dict[str, Dict[str, Any]] = {pid: {"noise": 0, "rejected": 0, "domain": 0, "domain_kept": 0}
                                      for pid, _ in P_SERIES}
    eng = core.XiaoZhaoMPCEngine()
    for src in JIA_AN:
        d = _report(src)
        if d is None:
            continue
        noise = set(d["general_disentangle"].get("noise_turn_ids") or [])
        turns = [core.UtteranceTurn(k, float(t["start"]), t["speaker"], "unknown", normalize_asr_text(t["text"]))
                 for k, t in enumerate(d["merged_turns"], 1)]
        domain = {t.turn_id for t in turns if t.turn_id not in noise and eng.detect_domain_intents(t.text)}
        for pid in ("P2", "P3"):
            pf = KeywordPool(eng) if pid == "P2" else core.AmbientIntentPreFetcher(eng)
            for t in turns:
                pf.ingest_ambient_turn(t)
            # in the pool = the turn's text sits in a staged packet (a backchannel that only closes a chain and
            # boosts an existing packet is not in it); nothing is popped while streaming (no queries)
            staged = {i for pk in pf.staged_packets.values() for i in pk["turn_ids"]}
            staged |= {i for pay in pf.auto_fired for i in pay["selected_turn_indices"]}
            o = out[pid]
            o["noise"] += len(noise)
            o["rejected"] += len(noise - staged)
            o["domain"] += len(domain)
            o["domain_kept"] += len(domain & staged)
        kept = set()
        for i in range(0, len(turns), P1_WINDOW):
            win = [dataclasses.replace(t, turn_id=k) for k, t in enumerate(turns[i:i + P1_WINDOW], 1)]
            q = core.UtteranceTurn(len(win) + 1, win[-1].timestamp + 1.0, "speaker_x", "caregiver", P_QUERY)
            pay = eng.analyze_session(core.DialogueSessionState("P1", "P", turns=win + [q]))
            kept |= {turns[i + k - 1].turn_id for k in pay["selected_turn_indices"] if k <= len(win)}
        o = out["P1"]
        o["noise"] += len(noise)
        o["rejected"] += len(noise - kept)
        o["domain"] += len(domain)
        o["domain_kept"] += len(domain & kept)
    for o in out.values():
        o["rejection_rate"] = o["rejected"] / o["noise"] if o["noise"] else None
        o["domain_retention"] = o["domain_kept"] / o["domain"] if o["domain"] else None
    return out


def p_series() -> Dict[str, Any]:
    probes = p_probes()
    res: Dict[str, Any] = {}
    for pid, _ in P_SERIES:
        r: Dict[str, Any] = {}
        lat = []
        for kind in ("referential", "compliance"):
            rows = []
            for pr in probes[kind]:
                intents, ids, ms = p_run(pid, pr["turns"], pr["query"])
                lat.append(ms)
                rows.append({**pr["factors"], "ok": int(bool(intents) and intents[0] in pr["gold"]),
                             "top1": intents[0] if intents else "", "gold": sorted(pr["gold"])})
            r[kind] = {"acc": sum(x["ok"] for x in rows) / len(rows), "n": len(rows), "rows": rows}
        r["fire_ms_mean"] = sum(lat) / len(lat)
        r["fire_ms_max"] = max(lat)
        res[pid] = r
    # P0: P3 auto-fires while ingesting the hazard turn; P1 / P2 dispatch only when someone queries
    pf = core.AmbientIntentPreFetcher()
    pf.ingest_ambient_turn(core.UtteranceTurn(1, 0.0, "speaker_2", "family", P_REQUEST["BA16_MEDICINE_PURCHASE_INQUIRY"]))
    t0 = time.perf_counter()
    r0 = pf.ingest_ambient_turn(core.UtteranceTurn(2, 5.0, "speaker_1", "elder", "救命！我跌倒了好痛"))
    res["P3"]["p0_autofire_ms"] = 1000.0 * (time.perf_counter() - t0) if r0["status"] == "AUTO_FIRED_P0" else None
    res["noise"] = p_noise_rejection()
    res["probe_counts"] = {k: len(v) for k, v in probes.items()}
    return res


def p_breakdown(rows: List[dict], key: str) -> str:
    levels: Dict[Any, List[int]] = {}
    for x in rows:
        levels.setdefault(x[key], []).append(x["ok"])
    return ", ".join(f"{k}: {sum(v)}/{len(v)}" for k, v in levels.items())


def table_d_md(ps: Dict[str, Any]) -> List[str]:
    rows = []
    for pid, name in P_SERIES:
        r, nz = ps[pid], ps["noise"][pid]
        p0 = (f"auto-fired at ingest ({r['p0_autofire_ms']:.2f} ms)" if r.get("p0_autofire_ms") is not None
              else "— (waits for a query)")
        rows.append([f"{pid}: {name}",
                     f"{pct(r['referential']['acc'])} ({sum(x['ok'] for x in r['referential']['rows'])}/"
                     f"{r['referential']['n']})",
                     f"{pct(nz['rejection_rate'])} ({nz['rejected']}/{nz['noise']}); domain turns kept "
                     f"{pct(nz['domain_retention'])}",
                     f"{pct(r['compliance']['acc'])} ({sum(x['ok'] for x in r['compliance']['rows'])}/"
                     f"{r['compliance']['n']})",
                     f"{r['fire_ms_mean']:.2f} (max {r['fire_ms_max']:.2f})", p0])
    md = ["## Table D — Ambient prefetcher (P-series): background pool vs post-hoc window",
          "",
          md_table(["System", "Referential query resolution acc %", "Backchannel noise rejection %",
                    "Priority ranking acc % (REQUEST_TO_COMPLIANCE_OR_HESITATION)", "Fire latency ms (query → payload)",
                    "Live P0 dispatch"], rows),
          f"- Referential / ranking probes are CONSTRUCTED (data simulation, {ps['probe_counts']['referential']} + "
          f"{ps['probe_counts']['compliance']}): the labelled corpus has 0 gold compliance chains and only the "
          "scripted dialogues contain natural 小照 referential queries (included). Correct = the top-ranked fired "
          "intent is the gold referent. They test the mechanisms by design, so read them as a mechanism check with "
          "the per-factor breakdown below, not as field accuracy.",
          "- Referential probes: family request → `gap` non-domain turns (chitchat / backchannel) → query "
          "「" + P_QUERY + "」; `trap` = a chitchat turn carrying another intent's keywords before / after the "
          "request. Ranking probes: an Arrange request answered by the caregiver (好 / 嗯…我再看看) plus an "
          "informational mention of another intent (`order`), then `tail` chitchat turns; gold = the answered request.",
          "- Noise rejection: all merged turns of the 3 Jia-An recordings streamed without upstream thread labels; "
          "gold noise = interview_analyzer's T_BACKCHANNEL_NOISE (rule-based, not human). 'Domain turns kept' = "
          "non-noise turns with a domain intent that still reach the pool (P2/P3) or the query context (P1).",
          "- P1 latency is the post-hoc analyze_session over its window; P2 / P3 latency is the pop + merge of "
          "pre-staged packets (dispatch_latency_ms). P-series has no trained parameters, so it does not depend on "
          "the split scheme."]
    for pid, _ in P_SERIES:
        r = ps[pid]
        md.append(f"- {pid} breakdown — referential by trap: {p_breakdown(r['referential']['rows'], 'trap')}; by "
                  f"gap: {p_breakdown(r['referential']['rows'], 'gap')}; ranking by order: "
                  f"{p_breakdown(r['compliance']['rows'], 'order')}; by tail: "
                  f"{p_breakdown(r['compliance']['rows'], 'tail')}")
    return md


# ------------------------------------------------------------------------------------------- latency-only (Pi)
def latency_only() -> int:
    """Stdlib-only probe: voiceprint extraction on a synthetic 4 s voice + C5 analyze_session per turn on the
    exported dataset (stored voiceprints) or, without it, on the 5 benchmark dialogues."""
    pcm = core.synth_test_voice(core.TEST_VOICE_LOW, 4.0)
    t0 = time.perf_counter()
    for _ in range(3):
        core.extract_turn_voiceprint(pcm, 16000, 0.0, 4.0)
    vp_ms = 1000.0 * (time.perf_counter() - t0) / 3
    vp_kib = peak_kib(lambda: core.extract_turn_voiceprint(pcm, 16000, 0.0, 4.0))
    sessions = []
    if DATASET_JSON.exists():
        rows = json.loads(DATASET_JSON.read_text(encoding="utf-8"))["turns"]
        units: Dict[str, dict] = {}
        for r in rows:
            u = units.setdefault(r["unit_id"], {"unit_id": r["unit_id"], "turns": []})
            u["turns"].append({"turn_id": r["turn_id"], "start": r["start"], "end": r["end"],
                               "speaker_id": r["speaker_id"], "role_input": "unknown", "text": r["text_clean"],
                               "hint_addressee": "unknown", "hint_thread": "T0",
                               "vp": {"voiceprint": r["voiceprint_24d"] or None,
                                      "sub_voiceprints": r.get("voiceprint_subvectors") or None,
                                      "source": r["voiceprint_source"] if r["voiceprint_24d"] else "",
                                      "near_field_score": r["near_field_score"]}})
        sessions = list(units.values())
    n = ms = 0.0
    peak = 0.0
    for u in sessions:
        _, t = run_engine(u, {})
        n, ms = n + len(u["turns"]), ms + t
        peak = max(peak, peak_kib(lambda: run_engine(u, {})))
    if not sessions:
        for s in core.build_5_benchmark_scenarios():
            t0 = time.perf_counter()
            core.XiaoZhaoMPCEngine().analyze_session(s)
            n, ms = n + len(s.turns), ms + 1000.0 * (time.perf_counter() - t0)
    print(f"run_env={'RaspberryPi5_ARM64' if core.ON_PI else 'Mac_Mock_Env'}")
    print(f"voiceprint extraction: {vp_ms:.1f} ms per 4 s turn, peak {vp_kib:.0f} KiB")
    print(f"C5 analyze_session: {ms / max(1, n):.1f} ms per turn over {int(n)} turns, peak {peak:.0f} KiB / session")
    return 0


# ------------------------------------------------------------------------------------------- main
TABLE_B_HEADER = ["Config", "Thread purity / cohesion", "Addressee acc %", "Slot completeness / precision %",
                  "Issue→resolution link recall all / multi-hop %", "Multi-hop & consensus acc %",
                  "Hypothetical P0 FAR %", "Live P0 recall %", "Latency ms/turn", "Peak mem KiB"]
SUMMARY_B_HEADER = ["Config", "Thread purity / cohesion", "Addressee acc %", "Slot completeness %",
                    "Multi-Hop & Consensus acc %", "Hypothetical P0 FAR %", "Pi mem KiB / latency ms/turn"]
SUMMARY_A_HEADER = ["Model", "d' (within / cross pairs)", "Speaker purity / inverse purity", "Role ID acc %",
                    "Latency ms/turn", "Peak mem KiB"]


def mhc_cell(m: dict) -> str:
    return (f"{pct(m['mhc_acc'])} (multi-hop {m['multihop_link_tp']}/{m['multihop_link_gt']}, "
            f"consensus {m['consensus_ok']}/{m['consensus_n']})")


def table_b_rows(tb: Dict[str, dict], mem: Dict[str, float]) -> List[List[str]]:
    rows = []
    for cid, name, _ in TABLE_B_CONFIGS:
        m = tb[cid]
        rows.append([f"{cid}: {name}", f"{m['thread_purity']:.3f} / {m['thread_cohesion']:.3f}",
                     f"{pct(m['addressee_acc'])} (n={m['addressee_n']})",
                     f"{pct(m['slot_completeness'])} / {pct(m['slot_precision'])}",
                     f"{pct(m['link_recall'])} / {pct(m['multihop_link_recall'])} (n={m['link_gt']}/"
                     f"{m['multihop_link_gt']})", mhc_cell(m),
                     f"{pct(m['p0_far'])} ({m['p0_far_fp']}/{m['p0_far_n']})",
                     f"{pct(m['p0_recall'])} (n={m['p0_recall_n']})", f"{m['ms_per_turn']:.2f}",
                     f"{mem.get(cid, 0.0):.0f}"])
    return rows


def summary_b_rows(tb: Dict[str, dict], mem: Dict[str, float]) -> List[List[str]]:
    return [[f"{cid}: {name}", f"{tb[cid]['thread_purity']:.3f} / {tb[cid]['thread_cohesion']:.3f}",
             f"{pct(tb[cid]['addressee_acc'])} (n={tb[cid]['addressee_n']})",
             f"{pct(tb[cid]['slot_completeness'])} (n={tb[cid]['slot_gt']})", mhc_cell(tb[cid]),
             f"{pct(tb[cid]['p0_far'])} ({tb[cid]['p0_far_fp']}/{tb[cid]['p0_far_n']})",
             f"{mem.get(cid, 0.0):.0f} / {tb[cid]['ms_per_turn']:.2f}"] for cid, name, _ in TABLE_B_CONFIGS]


def summary_a_rows(ta: Dict[str, dict], mem: Dict[str, float]) -> List[List[str]]:
    rows = []
    for mid, name in TABLE_A_MODELS:
        m = ta[mid]
        pur = ("— (= label source)" if m["purity_is_label_source"]
               else f"{m['speaker_purity']:.3f} / {m['speaker_inverse_purity']:.3f}")
        dp = (f"{f3(m['dprime'])} ({m['within_pairs']} / {m['cross_pairs']})" if mid in ("M1", "M2", "M4")
              else "—")
        rows.append([f"{mid}: {name}", dp, pur, f"{pct(m['role_acc'])} (n={m['role_n']})",
                     f"{m['ms_per_turn']:.2f}", f"{mem.get(mid, 0.0):.0f}"])
    return rows


def role_attribution_line(ta: Dict[str, dict]) -> str:
    a = ta["_role_attribution"]
    return (f"Role ID attribution (Test, n={ta['M3']['role_n']}): M3 cues only {pct(ta['M3']['role_acc'])} → "
            f"cues + GIFT graph (no priors) {pct(a['cues_graph']['role_acc'])} → cues + Train priors (no graph) "
            f"{pct(a['cues_priors']['role_acc'])}; M4 without GIFT roles {pct(a['m4_no_graph']['role_acc'])}, "
            f"M4 without Train priors {pct(a['m4_no_priors']['role_acc'])}, M4 full {pct(ta['M4']['role_acc'])}. "
            f"M4 speaker role sources: {a['m4_speaker_role_sources']}; graph-inferred speakers: "
            f"{a['graph_inferred'] or 'none'}")


def cfg_with(cfg: Optional[dict], best: dict) -> Optional[dict]:
    return None if cfg is None else {**cfg, "bicl_sim": best["bicl"], "gsn_iters": best["gsn"], "alpha": best["alpha"]}


_FR_AUDIO: Dict[str, Optional[LazyAudio]] = {}


def fr_audio(pseudo: dict) -> Optional[LazyAudio]:
    if pseudo["unit_id"] not in _FR_AUDIO:
        _FR_AUDIO[pseudo["unit_id"]] = lazy_audio(pseudo)
    return _FR_AUDIO[pseudo["unit_id"]]


def run_scheme(scheme: tuple, base_windows: List[dict], scripted: List[dict],
               audio: Dict[str, LazyAudio], n_verified: int) -> Dict[str, Any]:
    sid, origin, share, ratio = scheme
    tag = f"{int(100 * share['train'])} / {int(100 * share['val'])} / {int(100 * share['test'])}"
    md: List[str] = []
    emit = lambda line="": md.append(line)   # noqa: E731
    log = lambda s: print(f"[{sid}] {s}", flush=True)   # noqa: E731

    windows = copy.deepcopy(base_windows)
    segs, cuts = chrono_split(windows, share)
    for u in segs:   # gold consensus is a property of the unit (shared by every config)
        u["gt_consensus"] = gt_consensus(u)
    seg_by = {s: [u for u in segs if u["split"] == s] for s in SPLITS}
    real_turns = sum(len(w["turns"]) for w in windows)
    for s in SPLITS:
        assert all(t["split"] == s for u in seg_by[s] for t in u["turns"])
    for src in cuts:
        ts = sorted((t for u in segs if u["source"] == src for t in u["turns"]), key=lambda t: t["start"])
        order = [SPLITS.index(t["split"]) for t in ts]
        assert order == sorted(order), f"{src}: split is not chronological"
    log(f"split {tag}: " + ", ".join(f"{s} {sum(len(u['turns']) for u in seg_by[s])} turns" for s in SPLITS))

    emit(f"## Scheme {sid} — {origin}: chronological {tag} ({ratio})")
    emit()
    emit(f"### {sid} · 1. Split — per recording, chronological: first {int(100 * share['train'])} % Train / next "
         f"{int(100 * share['val'])} % Val / last {int(100 * share['test'])} % Test")
    rows = [[src, str(c["turns"]), str(c["train"]), f"{c['val']} (from {c['val_starts']})",
             f"{c['test']} (from {c['test_starts']})"] for src, c in cuts.items()]
    tot = {s: sum(len(u["turns"]) for u in seg_by[s]) for s in SPLITS}
    rows.append(["all 3 recordings", str(real_turns)] + [f"{tot[s]} ({pct(tot[s] / real_turns)} %)" for s in SPLITS])
    emit(md_table(["recording", "labelled turns", "Train", "Val", "Test"], rows))
    emit(f"- Evaluation units = {len(segs)} window segments: " + "; ".join(
        f"{s}: " + ", ".join(u["unit_id"].split("#")[0] for u in seg_by[s]) for s in SPLITS))

    # ---------------------------------------------------------------- audio + Train fitting
    train_keys = {t["key"] for u in seg_by["train"] for t in u["turns"]}
    fr_wins = [w for w in fullrec_vp_turns(cuts, windows) if fr_audio(w) is not None]
    fr_keys = {w["turns"][0]["key"] for w in fr_wins}
    shipped = shipped_vp_params()
    recs_old: Dict[str, dict] = {}
    for w in windows:
        if w["unit_id"] in audio:
            recs_old.update(extract_window(w, audio[w["unit_id"]], shipped, frames=True))
    recs_fr_old: Dict[str, dict] = {}
    for w in fr_wins:
        recs_fr_old.update(extract_window(w, fr_audio(w), shipped, frames=True))
    _, _, n_lab_frames = fit_channel_priors([recs_old[k] for k in train_keys if k in recs_old])
    mean, std, n_frames = fit_channel_priors([recs_old[k] for k in train_keys if k in recs_old]
                                             + list(recs_fr_old.values()))
    priors_only_ship_w = {**shipped, "VP_PRIOR_MEAN": mean, "VP_PRIOR_STD": std}
    recs_tr: Dict[str, dict] = {}
    for w in windows:
        if w["unit_id"] in audio:
            recs_tr.update(extract_window(w, audio[w["unit_id"]], priors_only_ship_w, only=train_keys))
    recs_fr_tr: Dict[str, dict] = {}
    for w in fr_wins:
        recs_fr_tr.update(extract_window(w, fr_audio(w), priors_only_ship_w))
    logsig = fit_logsig(list(recs_tr.values()) + list(recs_fr_tr.values()))
    with_audio = {s: [u for u in seg_by[s] if u["window_id"] in audio] for s in SPLITS}
    ratios = fisher_ratios(with_audio["train"], recs_tr, logsig)
    med = sorted(ratios)[len(ratios) // 2]
    cands = {
        "shipped": shipped["VP_DIM_WEIGHTS"],
        "uniform": [1.0] * core.VP_DIM,
        "fisher_sqrt": [round(min(2.0, max(0.3, math.sqrt(r / med))), 1) for r in ratios],
        "fisher_linear": [round(min(2.0, max(0.3, r / med)), 1) for r in ratios],
    }
    priors_only = {"VP_PRIOR_MEAN": mean, "VP_PRIOR_STD": std, "VP_LOGSIG_PRIOR": logsig}
    recs_pri: Dict[str, dict] = {}
    for w in windows:
        if w["unit_id"] in audio:
            recs_pri.update(extract_window(w, audio[w["unit_id"]], {**shipped, **priors_only}))

    def d_for(units_: List[dict], recs: Dict[str, dict], ls: List[float], wts: List[float]) -> Optional[float]:
        within, cross = pair_scores(units_, lambda u, i: piece_vectors(recs[u["turns"][i]["key"]], ls, wts),
                                    core._vp_cos)
        return dprime(within, cross)

    cand_d = {s: {k: d_for(with_audio[s], recs_pri, logsig, wv) for k, wv in cands.items()} for s in SPLITS}
    sel_split = "val" if any(cand_d["val"][k] is not None for k in cands if k != "shipped") else "train"
    best_w = max((k for k in cands if k != "shipped"),
                 key=lambda k: (cand_d[sel_split][k] if cand_d[sel_split][k] is not None else -9, k))
    fitted = {**priors_only, "VP_DIM_WEIGHTS": list(cands[best_w])}
    recs_new: Dict[str, dict] = {}
    for w in windows:
        if w["unit_id"] in audio:
            recs_new.update(extract_window(w, audio[w["unit_id"]], fitted))
    for w in windows:
        for t in w["turns"]:
            t["vp"] = recs_new[t["key"]]["vp"] if t["key"] in recs_new else None
    changed = {k: fitted[k] != shipped[k] for k in VP_KEYS}
    lab_priors = learn_role_priors(seg_by["train"])
    fr_priors = fullrec_role_priors(cuts)
    role_priors = {src: {**fr_priors.get(src, {}), **lab_priors.get(src, {})}
                   for src in set(lab_priors) | set(fr_priors)}
    protos = learn_prototypes(seg_by["train"])
    for u in segs:
        u["role_priors"] = dict(role_priors.get(u["source"], {}))
    mixes = build_interleaved(segs)
    for m in mixes:
        m["gt_consensus"] = gt_consensus(m)
    eval_by = {s: seg_by[s] + [m for m in mixes if m["split"] == s] for s in SPLITS}
    n_mix = {s: sum(1 for m in mixes if m["split"] == s) for s in SPLITS}
    log(f"train fit: {n_frames} voiceprint frames ({len(fr_wins)} full-recording turns), weights={best_w}")

    emit()
    emit(f"### {sid} · 2. Training stage (first {int(100 * share['train'])} % of each recording)")
    emit(f"- Voiceprint channel priors refit on {n_frames} active frames: {n_lab_frames} from the "
         f"{len(train_keys)} labelled Train turns + {n_frames - n_lab_frames} from {len(fr_wins)} unlabelled "
         f"full-recording turns before each Val cut (≤ {FULLREC_VP_MAX} per recording, outside the golden windows).")
    rows = [[k + (f" ← selected on {sel_split.title()}" if k == best_w else " (reference: earlier calibration)"
                  if k == "shipped" else ""), f3(cand_d["train"][k]), f3(cand_d["val"][k]), f3(cand_d["test"][k])]
            for k in cands]
    emit(md_table(["VP_DIM_WEIGHTS candidate (Train-fitted priors)", "Train d'", "Val d' (selects)",
                   "Test d' (not used)"], rows))
    if sel_split == "train":
        emit("- Val has no turn > 6 s (no within-turn piece pairs), so Val d' is undefined; weights selected on "
             "Train d' instead.")
    d_old = {s: d_for(with_audio[s], recs_old, shipped["VP_LOGSIG_PRIOR"], shipped["VP_DIM_WEIGHTS"]) for s in SPLITS}
    d_new = {s: d_for(with_audio[s], recs_new, logsig, fitted["VP_DIM_WEIGHTS"]) for s in SPLITS}
    emit(md_table(["voiceprint params", "Train d'", "Val d'", "Test d'"],
                  [["currently in mpc_core.py"] + [f3(d_old[s]) for s in SPLITS],
                   ["Train-fitted (this run)"] + [f3(d_new[s]) for s in SPLITS]]))
    emit(f"- changed vs mpc_core.py: {', '.join(k for k, v in changed.items() if v) or 'none'}")
    emit("- Speaker → role priors: labelled Train majority (wins) + full-recording cue/graph inference over the "
         "unlabelled turns before the Val cut (fills the rest): " + "; ".join(
             f"{src}: " + ", ".join(f"{sp}={r.split('_')[0]}"
                                    + ("" if sp in lab_priors.get(src, {}) else " (full-rec)")
                                    for sp, r in sorted(p.items()))
             for src, p in sorted(role_priors.items())))
    emit(f"- Voiceprint prototypes: {sum(len(p) for p in protos.values())} (recording, speaker) means from Train "
         "turns, used only by M5 in memory; never exported or written to mpc_core (privacy rule: no enrollment "
         "in the deployed engine).")
    audit = schema_audit(seg_by["train"], seg_by["test"])
    for name in ("EXTENDED_ROLE_CUES", "ISSUE_TRIGGER_CATALOG", "RESOLUTION_CATALOG"):
        r = audit[name]
        emit(f"- Fixed schema {name}: {r['keywords']} keywords kept as shipped, {r['fire_on_train']} fire on Train; "
             f"fire on Test but never on Train (authored with Test visible): {r['test_only'] or 'none'}")
    tr_c2 = score_table_b(eval_by["train"], {u["unit_id"]: predict_config(u, {**TABLE_B_CONFIGS[1][2], "memo": True})
                                             for u in eval_by["train"]})
    tr_c3 = score_table_b(eval_by["train"], {u["unit_id"]: predict_config(u, {**TABLE_B_CONFIGS[2][2], "memo": True})
                                             for u in eval_by["train"]})
    emit(f"- Bi-CL check on Train (segments + {n_mix['train']} interleaved): "
         f"thread F greedy {f3(tr_c2['thread_f'])} → Bi-CL {f3(tr_c3['thread_f'])}.")

    # ---------------------------------------------------------------- Validation grid
    emit()
    emit(f"### {sid} · 3. Validation stage — grid on {len(seg_by['val'])} Val segments + {n_mix['val']} interleaved "
         "Val sessions (J = mean of thread F, addressee acc, slot F1, link F1, speaker-cluster F)")
    grid = []
    for b in GRID_BICL:
        for n in GRID_GSN_N:
            for a in GRID_ALPHA:
                cfg = dict(bicl=True, gift=True, gsn=True, bicl_sim=b, gsn_iters=n, alpha=a, memo=True)
                m = score_table_b(eval_by["val"], {u["unit_id"]: predict_config(u, cfg) for u in eval_by["val"]})
                grid.append({"bicl": b, "gsn": n, "alpha": a, **m, "J": objective(m)})
    dist = lambda g: (abs(g["bicl"] - DEFAULTS["bicl"]) / 0.35 + abs(g["gsn"] - DEFAULTS["gsn"]) / 3.0  # noqa: E731
                      + abs(g["alpha"] - DEFAULTS["alpha"]) / 0.7)
    top = max(g["J"] for g in grid)
    best = min((g for g in grid if top - g["J"] < 1e-9), key=dist)
    n_tied = sum(1 for g in grid if top - g["J"] < 1e-9)
    rows = [[f"{g['bicl']:.2f}", str(g["gsn"]), f"{g['alpha']:.1f}", f3(g["thread_f"]), pct(g["addressee_acc"]),
             pct(g["slot_f1"]), pct(g["link_f1"]), pct(g["multihop_link_recall"]), f3(g["speaker_f"]), f"{g['J']:.4f}",
             ("**selected**" if g is best else "default" if dist(g) == 0 else "")] for g in grid]
    emit(md_table(["BICL sim", "GSN N", "E2CP α", "thread F", "addr %", "slot F1 %", "link F1 %", "multi-hop R %",
                   "speaker F", "J", ""], rows))
    distinct = len({round(g["J"], 9) for g in grid})
    emit(f"- selected (BICL={best['bicl']:.2f}, N={best['gsn']}, α={best['alpha']:.1f}) with J={best['J']:.4f}; "
         f"{n_tied} of {len(grid)} configs tie at the top, {distinct} distinct J values "
         f"(ties go to the shipped default BICL={DEFAULTS['bicl']}, N={DEFAULTS['gsn']}, α={DEFAULTS['alpha']}).")
    for hp, vals in (("bicl", GRID_BICL), ("gsn", GRID_GSN_N), ("alpha", GRID_ALPHA)):
        emit(f"- best J by {hp}: " + ", ".join(f"{v}: {max(g['J'] for g in grid if g[hp] == v):.4f}" for v in vals))
    log(f"val grid: selected BICL={best['bicl']}, N={best['gsn']}, alpha={best['alpha']} ({n_tied} tied)")

    # ---------------------------------------------------------------- Test: Table A
    test_real = with_audio["test"]
    ac_tr = [(math.log(recs_new[k]["vp"]["rms_energy"] + 1.0), recs_new[k]["vp"]["zcr_pitch"])
             for k in train_keys if k in recs_new]
    m0 = sum(x for x, _ in ac_tr) / len(ac_tr)
    m1 = sum(y for _, y in ac_tr) / len(ac_tr)
    s0 = math.sqrt(sum((x - m0) ** 2 for x, _ in ac_tr) / len(ac_tr)) or 1.0
    s1 = math.sqrt(sum((y - m1) ** 2 for _, y in ac_tr) / len(ac_tr)) or 1.0
    ac_norm = ((m0, m1), (s0, s1))
    recs_of = lambda u: [recs_new[t["key"]] for t in u["turns"]]   # noqa: E731
    res_a = {u["unit_id"]: table_a_unit(u, recs_of(u), ac_norm, best, protos.get(u["source"], {}))
             for u in test_real}
    ta = score_table_a(test_real, res_a, n_verified > 0)
    big = max(test_real, key=lambda u: len(u["turns"]))
    mem_vp = vp_peak_kib(big["turns"][0], audio[big["window_id"]], fitted)
    mem_a = {"M1": peak_kib(lambda: roles_for(big, m1_cluster(recs_of(big), ac_norm), False, None)),
             "M2": peak_kib(lambda: m2_cluster(big)) + mem_vp,
             "M3": peak_kib(lambda: roles_for(big, [t["speaker_id"] for t in big["turns"]], False, None)),
             "M4": peak_kib(lambda: run_engine(big, best_cfg(best))) + mem_vp,
             "M5": peak_kib(lambda: m5_assign(big, protos.get(big["source"], {}))) + mem_vp}
    n_cues = sum(len(c) for _, c in core.EXTENDED_ROLE_CUES)
    params_a = {"M1": "0 fitted (2-d K-means, k=3)", "M2": "73 VP floats + role priors",
                "M3": f"{n_cues} schema cues (no priors)", "M4": "73 VP floats + schema + priors + 3 hyper-params",
                "M5": f"{sum(len(p) for p in protos.values())} × 24-d prototypes + role priors"}
    n_test_turns = sum(len(u["turns"]) for u in test_real)
    emit()
    emit(f"### {sid} · 4. Testing stage — last {int(100 * share['test'])} % of each recording ({len(test_real)} "
         f"segments, {n_test_turns} turns), Val-selected config")
    emit()
    emit(f"#### Table A [{tag}] — Speaker diarization & role identification")
    rows = [r[:-1] + [r[-1], params_a[mid]] for r, (mid, _) in zip(summary_a_rows(ta, mem_a), TABLE_A_MODELS)]
    emit(md_table(SUMMARY_A_HEADER + ["Fitted params"], rows))
    emit("- " + role_attribution_line(ta))
    log(f"table A: role ID M3 {pct(ta['M3']['role_acc'])} vs M4 {pct(ta['M4']['role_acc'])} "
        f"(n={ta['M4']['role_n']})")

    # ---------------------------------------------------------------- Test: Table B
    test_units = eval_by["test"]
    tb: Dict[str, dict] = {}
    tb_mix: Dict[str, Optional[dict]] = {}
    for cid, _, cfg in TABLE_B_CONFIGS:
        preds = {u["unit_id"]: predict_config(u, cfg_with(cfg, best)) for u in test_units}
        tb[cid] = score_table_b(test_units, preds)
        mx = [u for u in test_units if u["kind"] == "interleaved"]
        tb_mix[cid] = score_table_b(mx, preds) if mx else None
        tb[cid]["consensus_pred"] = {u["unit_id"]: [preds[u["unit_id"]]["consensus"], u["gt_consensus"]]
                                     for u in test_units}
    big_b = max(test_units, key=lambda u: len(u["turns"]))
    mem_b = {cid: peak_kib(lambda c=cfg: predict_config(big_b, cfg_with(c, best))) for cid, _, cfg in TABLE_B_CONFIGS}
    emit()
    emit(f"#### Table B [{tag}] — Disentanglement, addressee, slot fusion, multi-hop / consensus, P0 guard (Test: "
         f"{len(seg_by['test'])} segments + {n_mix['test']} interleaved sessions, "
         f"{sum(len(u['turns']) for u in test_units)} turns)")
    emit(md_table(TABLE_B_HEADER, table_b_rows(tb, mem_b)))
    if any(tb_mix.values()):
        emit("- Interleaved Test sessions only — thread purity / cohesion: " + "; ".join(
            f"{cid} {tb_mix[cid]['thread_purity']:.3f} / {tb_mix[cid]['thread_cohesion']:.3f}"
            for cid, _, _ in TABLE_B_CONFIGS))
    emit(f"- Gold consensus states of the Test units: {tb['C5']['consensus_gt_states']}; "
         "C5 predicted vs gold per unit: " + "; ".join(
             f"{uid.replace('#test', '')}: {p} / {g}" for uid, (p, g) in tb["C5"]["consensus_pred"].items()))
    log(f"table B: multi-hop & consensus C4 {pct(tb['C4']['mhc_acc'])} vs C5 {pct(tb['C5']['mhc_acc'])} "
        f"(n={tb['C5']['mhc_n']})")

    # ---------------------------------------------------------------- scripted home-care dialogues
    tc = {cid: score_table_b(scripted, {u["unit_id"]: predict_config(u, cfg_with(cfg, best)) for u in scripted})
          for cid, _, cfg in TABLE_B_CONFIGS}

    results = {
        "scheme": sid, "origin": origin, "shares": share, "ratio": ratio,
        "recording_cuts": cuts, "segments": {u["unit_id"]: u["split"] for u in segs},
        "interleaved": {u["unit_id"]: u["split"] for u in mixes}, "interleaved_per_split": n_mix,
        "fullrec_vp_turns": len(fr_wins), "vp_frames": {"labelled": n_lab_frames, "total": n_frames},
        "vp_fitted": fitted, "vp_before": shipped, "vp_changed": changed, "vp_weight_candidates_dprime": cand_d,
        "vp_weight_selected": best_w, "vp_weight_selected_on": sel_split, "vp_dprime_before": d_old,
        "vp_dprime_fitted": d_new, "fisher_ratios": [round(r, 3) for r in ratios],
        "role_priors_labelled": lab_priors, "role_priors_fullrec": fr_priors, "role_priors": role_priors,
        "schema_audit": audit, "grid": grid, "selected": {k: best[k] for k in ("bicl", "gsn", "alpha")},
        "test_turns_table_a": n_test_turns, "table_a": ta, "table_a_mem_kib": mem_a, "table_b": tb,
        "table_b_mem_kib": mem_b, "table_b_interleaved_only": tb_mix, "table_c_scripted": tc}
    return {"md": md, "results": results, "segs": segs, "fitted": fitted, "tag": tag}


def run_scheme_worker(scheme: tuple, windows: List[dict], scripted: List[dict], audio: Dict[str, LazyAudio],
                      n_verified: int) -> Tuple[Dict[str, Any], Dict[str, Any], Tuple[int, int]]:
    """Process-pool entry: one split scheme -> (run, new voiceprint-cache records, (hits, misses))."""
    run = run_scheme(scheme, windows, scripted, audio, n_verified)
    cache = vp_cache()
    return run, {k: cache[k] for k in _VP_CACHE_STATE["new"]}, (_VP_CACHE_STATE["hits"], _VP_CACHE_STATE["misses"])


def table_c_md(corpus: List[dict], sim: Dict[str, Any], runs: Dict[str, dict]) -> List[str]:
    md: List[str] = []
    emit = lambda line="": md.append(line)   # noqa: E731
    emit("## Table C — Audio segmentation, noise filtering & data simulation (錄音檔：切分、雜訊 · 數據模擬)")
    emit()
    emit("### C-1. Real recordings: segmentation & noise (Whisper segments → loop collapse → merged turns → "
         "T_BACKCHANNEL_NOISE)")
    rows, tot = [], {k: 0 for k in ("duration_sec", "raw_segments", "asr_loop_runs", "asr_loop_segments_absorbed",
                                    "merged_turns", "noise_turns", "substantive_turns", "ambient_ignored_noise")}
    for r in corpus:
        rows.append([r["source"] + ("" if r["in_corpus"] else " (lab meeting, not in corpus)"),
                     f"{r['duration_sec'] / 60:.1f}", str(r["raw_segments"]),
                     f"{r['asr_loop_runs']} runs / {r['asr_loop_segments_absorbed']} segs (max ×{r['max_loop']})",
                     str(r["merged_turns"]), f"{r['raw_segments'] / max(1, r['merged_turns']):.2f}",
                     f"{r['noise_turns']} ({pct(r['noise_turns'] / max(1, r['merged_turns']))} %)",
                     str(r["substantive_turns"]), str(r["ambient_ignored_noise"])])
        if r["in_corpus"]:
            for k in tot:
                tot[k] += r[k] or 0
    rows.append([f"**3 Jia-An recordings ({tot['duration_sec'] / 3600:.2f} h)**", f"{tot['duration_sec'] / 60:.1f}",
                 str(tot["raw_segments"]), f"{tot['asr_loop_runs']} runs / {tot['asr_loop_segments_absorbed']} segs",
                 str(tot["merged_turns"]), f"{tot['raw_segments'] / max(1, tot['merged_turns']):.2f}",
                 f"{tot['noise_turns']} ({pct(tot['noise_turns'] / max(1, tot['merged_turns']))} %)",
                 str(tot["substantive_turns"]), str(tot["ambient_ignored_noise"])])
    emit(md_table(["recording", "min", "raw Whisper segments", "ASR repetition loops collapsed", "merged turns",
                   "segments / turn", "T_BACKCHANNEL_NOISE turns", "substantive turns",
                   "ambient packets ignored as noise"], rows))
    emit("- Loops = ≥ 3 consecutive identical segments (collapse_asr_loops on the stored *_raw_stt.json); merged "
         "turns / noise turns are read from each *_mpc_report.json (general_disentangle.noise_turn_ids).")
    emit()
    emit("### C-2. Simulated boundary bleed — 0.25 s collar + 75 %-anchor window vs whole-turn window "
         f"(synthetic low / high voices, {SIM_TURN_S:.0f} s target turn, {len(SIM_SEEDS)} seeds × 2 target voices)")
    names = list(sim["bleed"])
    rows = []
    for case in sim["bleed"][names[0]]:
        row = [case]
        for nm in names:
            xs = sim["bleed"][nm][case]
            row += [f"{100.0 * sum(x[0] for x in xs) / len(xs):.0f} %", f"{sum(x[1] for x in xs) / len(xs):+.3f}",
                    f"{100.0 * sum(x[2] for x in xs) / len(xs):.0f} %"]
        rows.append(row)
    allc = {nm: [x for v in sim["bleed"][nm].values() for x in v] for nm in names}
    rows.append(["**all cases**"] + [c for nm in names for c in (
        f"{100.0 * sum(x[0] for x in allc[nm]) / len(allc[nm]):.0f} %",
        f"{sum(x[1] for x in allc[nm]) / len(allc[nm]):+.3f}",
        f"{100.0 * sum(x[2] for x in allc[nm]) / len(allc[nm]):.0f} %")])
    emit(md_table(["bleed inside the turn span"] + [f"{nm}: {k}" for nm in names
                                                     for k in ("correct speaker", "cos margin", "bleed in window")],
                  rows))
    emit("- correct speaker = cos(turn vp, target ref) > cos(turn vp, interferer ref); bleed in window = share of the "
         "analysis window that lies inside the bleed span (0 % = rejected by the sampler).")
    emit()
    emit("### C-3. Simulated whole-turn overlap (interferer mixed under the full target turn)")
    emit(md_table(["target-to-interferer SNR", "correct speaker", "mean cos margin"],
                  [[k, f"{100.0 * sum(x[0] for x in v) / len(v):.0f} % ({sum(x[0] for x in v)}/{len(v)})",
                    f"{sum(x[1] for x in v) / len(v):+.3f}"] for k, v in sim["snr"].items()]))
    emit("- The collar / anchor only rejects bleed at turn boundaries; full overlap is not separated (no source "
         "separation on the Pi), so this row shows where the voiceprint evidence degrades.")
    emit()
    nf = sim["near_far"]
    emit("### C-4. Simulated near-field (headset wearer) vs far-field (1–3 m interlocutor) — near_field_score")
    emit(md_table(["condition", "n", "mean near_field_score", "min / max"],
                  [["near-field", str(len(nf["near"])), f"{sum(nf['near']) / len(nf['near']):.3f}",
                    f"{min(nf['near']):.3f} / {max(nf['near']):.3f}"],
                   ["far-field (3 echoes, HF damping, −10 dB, noise floor)", str(len(nf["far"])),
                    f"{sum(nf['far']) / len(nf['far']):.3f}", f"{min(nf['far']):.3f} / {max(nf['far']):.3f}"]]))
    emit(f"- separation: AUC {nf['auc']:.3f}, d' {f3(nf['dprime'])}, accuracy {pct(nf['midpoint_acc'])} % at the "
         f"midpoint threshold {nf['midpoint_threshold']:.3f}.")
    emit()
    emit("### C-5. Simulated multi-party interleaving (two real segments → one A A B B … session, gold threads kept)")
    rows = []
    for sid, run in runs.items():
        r = run["results"]
        mx = r["table_b_interleaved_only"]
        rows.append([run["tag"], " / ".join(str(r["interleaved_per_split"][s]) for s in SPLITS)]
                    + [(f"{mx[c]['thread_purity']:.3f} / {mx[c]['thread_cohesion']:.3f}" if mx.get(c) else "n/a")
                       for c in ("C1", "C2", "C3", "C5")])
    emit(md_table(["split scheme", "interleaved sessions Train / Val / Test", "Test thread purity / cohesion: C1",
                   "C2 greedy", "C3 Bi-CL", "C5 full"], rows))
    emit()
    emit("### C-6. Simulated home-care benchmark — 5 scripted dialogues (build_5_benchmark_scenarios; never trained "
         "on)")
    tcs = {sid: run["results"]["table_c_scripted"] for sid, run in runs.items()}
    same = len({json.dumps({c: {k: x for k, x in m.items() if k != "ms_per_turn"} for c, m in v.items()},
                           sort_keys=True, default=str) for v in tcs.values()}) == 1
    rows = []
    for sid, tc in tcs.items():
        for cid, _, _ in TABLE_B_CONFIGS:
            m = tc[cid]
            rows.append([cid + ("" if same else f" [{runs[sid]['tag']}]"),
                         f"{m['thread_purity']:.3f} / {m['thread_cohesion']:.3f}",
                         f"{pct(m['addressee_acc'])} (n={m['addressee_n']})", pct(m["slot_completeness"]),
                         f"{pct(m['consensus_acc'])} ({m['consensus_ok']}/{m['consensus_n']})",
                         f"{pct(m['p0_far'])} ({m['p0_far_fp']}/{m['p0_far_n']})",
                         f"{pct(m['p0_recall'])} (n={m['p0_recall_n']})"])
        if same:
            break
    emit(md_table(["Config", "Thread purity / cohesion", "Addressee acc %", "Slot completeness %",
                   "Consensus acc %", "Hypothetical P0 FAR %", "Live P0 recall %"], rows))
    if same:
        emit("- identical under both split schemes (both Val grids select the same configuration).")
    return md


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-core", action="store_true",
                    help=f"write the {PRIMARY_SCHEME} Train-fitted VP_* params into src/mpc_core.py")
    ap.add_argument("--latency-only", action="store_true")
    ap.add_argument("--serial", action="store_true", help="run the split schemes one after another (no workers)")
    args = ap.parse_args()
    if args.latency_only:
        return latency_only()

    units = load_units()
    n_verified = import_verified(units)
    windows = [u for u in units if u["kind"] == "real_window"]
    scripted = [u for u in units if u["kind"] == "scripted_dialogue"]
    for u in scripted:
        u["gt_consensus"] = gt_consensus(u)
    n_turns = sum(len(w["turns"]) for w in windows) + sum(len(u["turns"]) for u in scripted)
    t_start = time.perf_counter()
    audio = {w["unit_id"]: a for w in windows for a in [lazy_audio(w)] if a is not None}

    def tables_cd() -> Tuple[List[dict], Dict[str, Any], Dict[str, Any]]:
        print("[table C] corpus segmentation stats + synthetic bleed / overlap / near-far simulation ...", flush=True)
        corpus_, sim_ = corpus_segmentation_stats(), simulate_audio()
        print("[table D] P-series ambient prefetcher benchmark ...", flush=True)
        return corpus_, sim_, p_series()

    if args.serial:
        runs = {sc[0]: run_scheme(sc, windows, scripted, audio, n_verified) for sc in SCHEMES}
        corpus, sim, ps = tables_cd()
    else:   # one worker process per split scheme; Tables C / D are computed here meanwhile
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=len(SCHEMES), mp_context=multiprocessing.get_context("spawn")) as ex:
            futs = {sc[0]: ex.submit(run_scheme_worker, sc, windows, scripted, audio, n_verified) for sc in SCHEMES}
            corpus, sim, ps = tables_cd()
            runs = {}
            for sid, fut in futs.items():
                runs[sid], new, (hits, misses) = fut.result()
                vp_cache().update(new)
                _VP_CACHE_STATE["dirty"] |= bool(new)
                _VP_CACHE_STATE["hits"] += hits
                _VP_CACHE_STATE["misses"] += misses

    head: List[str] = [
        "# XiaoZhao v7.7 — Train / Validation / Test benchmark, two chronological split schemes",
        "",
        f"Labels: {n_verified} / {n_turns} turns human-verified; the rest are DRAFT labels (text-level labels drafted "
        "from the cleaned transcripts, speaker = acoustic diarizer label). Numbers are provisional until verified.",
        "",
        "Schemes: " + "; ".join(f"**{sc[0]}** = {sc[1]} ({sc[3]})" for sc in SCHEMES)
        + ". Both cut each recording's labelled turns chronologically (no future leakage) and select "
          "hyper-parameters on their own Val split.",
        "",
        "Pipeline division of labour: STT (XiaoZhao Whisper API → local faster_whisper fallback) → Edge MPC on the Pi "
        "(this benchmark: addressee, TD/MF + Bi-CL threads, GIFT, GSN, MIntRec, voiceprint E2CP, DST fusion) → "
        "backend LLM (consumes the analyze_session payload; not scored here).",
        "",
        "## Summary — Test-split tables, both schemes",
    ]
    for sid, run in runs.items():
        r = run["results"]
        head += ["", f"### Table A [{run['tag']}] — Speaker diarization & role ID ({r['test_turns_table_a']} Test "
                     "turns)",
                 md_table(SUMMARY_A_HEADER, summary_a_rows(r["table_a"], r["table_a_mem_kib"])),
                 "- " + role_attribution_line(r["table_a"])]
    for sid, run in runs.items():
        r = run["results"]
        n_b = sum(len(u["turns"]) for u in run["segs"] if u["split"] == "test")
        head += ["", f"### Table B [{run['tag']}] — MPC disentanglement, GIFT, GSN & MIntRec (Test: "
                     f"{n_b} segment turns + {r['interleaved_per_split']['test']} interleaved sessions; Val-selected "
                     f"BICL={r['selected']['bicl']}, N={r['selected']['gsn']}, α={r['selected']['alpha']})",
                 md_table(SUMMARY_B_HEADER, summary_b_rows(r["table_b"], r["table_b_mem_kib"]))]
    head += [
        "",
        "Notes",
        "- d': within = pieces of one >6 s turn (one talker by construction); cross = pieces of adjacent turns whose "
        "(draft) speaker differs. M4's within pairs are E2CP Must-Link constraints (optimistic); M1's cross pairs "
        "come from a diarizer built on the same RMS / ZCR features (inflated). n/a = fewer than 2 pairs in the split.",
        "- Speaker purity is against the draft speaker column (the acoustic diarizer) until speakers are verified by "
        "listening. Role ID: M3 uses only lexical self-cues; M2 / M4 / M5 add the Train speaker → role priors; M4 "
        "also runs gift_infer_roles, which only resolves speakers that have neither own cues nor a prior. Roles are "
        "keyed by the diarizer label, so voiceprints change speaker purity / d', not role ID.",
        "- Multi-Hop & Consensus acc = pooled accuracy over (a) gold issue → resolution pairs whose reply chain is "
        "≥ 2 hops or off the chain (hit = linked by same predicted thread or, C5, by GSN gsn_linked_issues) and (b) "
        "one consensus_state per Test unit vs the gold state (the engine's state rule — a resolution after the first "
        "issue, ≥ 2 speakers, no REQUEST_TO_REFUSAL left unanswered by a later resolution — applied to gold threads, speakers, MIntRec fine "
        "intents and slots). C1–C4 run without GSN and without MIntRec transition chains, "
        "so a trailing REQUEST_TO_REFUSAL cannot hold their state at OPEN_DISCUSSION.",
        "- Latency includes per-turn feature extraction and is measured on this Mac (`--latency-only` on the Pi).",
        "",
    ]
    tail = table_c_md(corpus, sim, runs) + [""] + table_d_md(ps)
    print("\n".join(head + tail))

    md = head + tail + [""]
    for run in runs.values():
        md += run["md"] + [""]
    md += ["## interview4 lab meeting (05:00–14:00, Hotspot #3 13:16–14:16)"] + interview4_summary()

    prim = runs[PRIMARY_SCHEME]
    other_split = {f"split_{sid}": {t["key"]: t["split"] for u in run["segs"] for t in u["turns"]}
                   for sid, run in runs.items()}
    meta = {"schema": "xiaozhao.annotated_ground_truth.v3",
            "privacy": "anonymous speaker_N diarizer clusters; 24-d voiceprints are session-local evidence "
                       "(no enrollment, no identity matching); local file, gitignored",
            "split_protocol": f"'split' = {PRIMARY_SCHEME}; split_<scheme> columns per scheme: per recording, "
                              "chronological over the labelled golden-window turns; scripted dialogues = 'scripted'",
            "recording_cuts": {sid: run["results"]["recording_cuts"] for sid, run in runs.items()},
            "human_verified_turns": n_verified, "total_turns": n_turns,
            "label_provenance": {"claude_draft_text+diarizer_speaker": "text labels drafted from normalize_asr_text "
                                 "output; ground_truth_speaker = acoustic diarizer label (needs listening)",
                                 "benchmark_script+claude_draft": "speaker / role scripted in "
                                 "build_5_benchmark_scenarios; other labels drafted",
                                 "human_verified_csv": "re-imported from the CSV (human_verified=1)"},
            "voiceprint_params": f"Train-fitted on scheme {PRIMARY_SCHEME} (src/train_val_test_eval.py)"}
    export_dataset(sorted(prim["segs"], key=lambda u: (u["source"], u["turns"][0]["start"])) + scripted, meta,
                   other_split)
    RESULTS_JSON.write_text(json.dumps({
        "schemes": {sid: run["results"] for sid, run in runs.items()},
        "table_c_audio": {"corpus_segmentation": corpus, "simulation": sim},
        "table_d_prefetcher": ps,
        "human_verified_turns": n_verified}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    TABLES_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    save_vp_cache()
    print(f"\nvoiceprint cache {VP_CACHE_PATH.relative_to(ROOT)}: {_VP_CACHE_STATE['hits']} hits, "
          f"{_VP_CACHE_STATE['misses']} misses; total {time.perf_counter() - t_start:.1f} s")
    print(f"wrote {DATASET_JSON.relative_to(ROOT)}, {DATASET_CSV.relative_to(ROOT)}, "
          f"{RESULTS_JSON.relative_to(ROOT)}, {TABLES_MD.relative_to(ROOT)}")
    if args.write_core:
        print("src/mpc_core.py: VP_PRIOR_MEAN / VP_PRIOR_STD / VP_LOGSIG_PRIOR / VP_DIM_WEIGHTS "
              + ("updated (Train-fitted)" if write_core_params(prim["fitted"]) else "unchanged"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
