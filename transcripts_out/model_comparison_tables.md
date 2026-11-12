# XiaoZhao v7.7 — Train / Validation / Test benchmark, two chronological split schemes

Labels: 0 / 97 turns human-verified; the rest are DRAFT labels (text-level labels drafted from the cleaned transcripts, speaker = acoustic diarizer label). Numbers are provisional until verified.

Schemes: **50_25_25** = Professor's example (10 : 5 : 5); **80_10_10** = Lab whiteboard standard split (8 : 1 : 1). Both cut each recording's labelled turns chronologically (no future leakage) and select hyper-parameters on their own Val split.

Pipeline division of labour: STT (XiaoZhao Whisper API → local faster_whisper fallback) → Edge MPC on the Pi (this benchmark: addressee, TD/MF + Bi-CL threads, GIFT, GSN, MIntRec, voiceprint E2CP, DST fusion) → backend LLM (consumes the analyze_session payload; not scored here).

## Summary — Test-split tables, both schemes

### Table A [50 / 25 / 25] — Speaker diarization & role ID (22 Test turns)
| Model | d' (within / cross pairs) | Speaker purity / inverse purity | Role ID acc % | Latency ms/turn | Peak mem KiB |
|---|---|---|---|---|---|
| M1: Acoustic Baseline (ZCR/RMS K-means) | 0.598 (24 / 48) | 0.727 / 0.727 | 21.1 (n=19) | 12.81 | 7 |
| M2: Unconstrained 24-d Spectral Voiceprint (StatPool only, alpha=0) + Train role priors | 0.278 (24 / 48) | 0.591 / 0.773 | 52.6 (n=19) | 392.02 | 486 |
| M3: Lexical self-cues only (EXTENDED_ROLE_CUES; no Train priors, no GIFT graph, no voiceprint) | — | — (= label source) | 31.6 (n=19) | 0.55 | 7 |
| M4: Ours v7.7 (StatPool 75%-Anchor + Turn-to-Diarize E2CP + GIFT Graph) + Train role priors | 0.447 (24 / 48) | 0.682 / 0.818 | 57.9 (n=19) | 392.89 | 512 |
| M5: Train-split speaker voiceprint prototypes + role priors (nearest prototype; offline eval only) | — | 0.591 / 0.818 | 31.6 (n=19) | 388.17 | 432 |
- Role ID attribution (Test, n=19): M3 cues only 31.6 → cues + GIFT graph (no priors) 31.6 → cues + Train priors (no graph) 57.9; M4 without GIFT roles 57.9, M4 without Train priors 31.6, M4 full 57.9. M4 speaker role sources: {'session_cues': 3, 'recording_prior': 6}; graph-inferred speakers: none

### Table A [80 / 10 / 10] — Speaker diarization & role ID (9 Test turns)
| Model | d' (within / cross pairs) | Speaker purity / inverse purity | Role ID acc % | Latency ms/turn | Peak mem KiB |
|---|---|---|---|---|---|
| M1: Acoustic Baseline (ZCR/RMS K-means) | 0.421 (13 / 15) | 1.000 / 0.889 | 12.5 (n=8) | 15.87 | 5 |
| M2: Unconstrained 24-d Spectral Voiceprint (StatPool only, alpha=0) + Train role priors | -0.149 (13 / 15) | 0.778 / 1.000 | 37.5 (n=8) | 378.81 | 751 |
| M3: Lexical self-cues only (EXTENDED_ROLE_CUES; no Train priors, no GIFT graph, no voiceprint) | — | — (= label source) | 25.0 (n=8) | 0.04 | 5 |
| M4: Ours v7.7 (StatPool 75%-Anchor + Turn-to-Diarize E2CP + GIFT Graph) + Train role priors | 1.201 (13 / 15) | 0.778 / 1.000 | 50.0 (n=8) | 379.94 | 796 |
| M5: Train-split speaker voiceprint prototypes + role priors (nearest prototype; offline eval only) | — | 0.889 / 1.000 | 25.0 (n=8) | 378.07 | 729 |
- Role ID attribution (Test, n=8): M3 cues only 25.0 → cues + GIFT graph (no priors) 25.0 → cues + Train priors (no graph) 50.0; M4 without GIFT roles 50.0, M4 without Train priors 25.0, M4 full 50.0. M4 speaker role sources: {'session_cues': 1, 'recording_prior': 7}; graph-inferred speakers: none

### Table B [50 / 25 / 25] — MPC disentanglement, GIFT, GSN & MIntRec (Test: 22 segment turns + 3 interleaved sessions; Val-selected BICL=0.45, N=2, α=0.7)
| Config | Thread purity / cohesion | Addressee acc % | Slot completeness % | Multi-Hop & Consensus acc % | Hypothetical P0 FAR % | Pi mem KiB / latency ms/turn |
|---|---|---|---|---|---|---|
| C1: Baseline Sliding Window / Sequential (HRED-style, no disentanglement, raw P0 keywords) | 0.667 / 1.000 | 42.6 (n=54) | 30.0 (n=30) | 50.0 (multi-hop 0/6, consensus 6/6) | 100.0 (1/1) | 8 / 0.05 |
| C2: Greedy TD/MF + Hypothetical P0 Guard | 0.773 / 0.864 | 46.3 (n=54) | 30.0 (n=30) | 50.0 (multi-hop 0/6, consensus 6/6) | 0.0 (0/1) | 216 / 7.12 |
| C3: + Bi-CL Session-Prototype Disentanglement | 0.758 / 0.924 | 46.3 (n=54) | 30.0 (n=30) | 50.0 (multi-hop 0/6, consensus 6/6) | 0.0 (0/1) | 244 / 7.54 |
| C4: + GIFT 4-Edge Topology & Replied-By Amendment | 0.758 / 0.924 | 51.9 (n=54) | 30.0 (n=30) | 50.0 (multi-hop 0/6, consensus 6/6) | 0.0 (0/1) | 244 / 7.80 |
| C5: + GSN 3-iter BIF + MIntRec Transition Chains (Full v7.7) | 0.758 / 0.924 | 51.9 (n=54) | 30.0 (n=30) | 50.0 (multi-hop 0/6, consensus 6/6) | 0.0 (0/1) | 243 / 10.32 |

### Table B [80 / 10 / 10] — MPC disentanglement, GIFT, GSN & MIntRec (Test: 9 segment turns + 3 interleaved sessions; Val-selected BICL=0.45, N=3, α=0.0)
| Config | Thread purity / cohesion | Addressee acc % | Slot completeness % | Multi-Hop & Consensus acc % | Hypothetical P0 FAR % | Pi mem KiB / latency ms/turn |
|---|---|---|---|---|---|---|
| C1: Baseline Sliding Window / Sequential (HRED-style, no disentanglement, raw P0 keywords) | 0.667 / 1.000 | 19.0 (n=21) | 50.0 (n=18) | 100.0 (multi-hop 0/0, consensus 6/6) | 100.0 (1/1) | 6 / 0.06 |
| C2: Greedy TD/MF + Hypothetical P0 Guard | 0.667 / 1.000 | 23.8 (n=21) | 50.0 (n=18) | 100.0 (multi-hop 0/0, consensus 6/6) | 0.0 (0/1) | 77 / 2.43 |
| C3: + Bi-CL Session-Prototype Disentanglement | 0.667 / 1.000 | 23.8 (n=21) | 50.0 (n=18) | 100.0 (multi-hop 0/0, consensus 6/6) | 0.0 (0/1) | 134 / 2.97 |
| C4: + GIFT 4-Edge Topology & Replied-By Amendment | 0.667 / 1.000 | 23.8 (n=21) | 50.0 (n=18) | 100.0 (multi-hop 0/0, consensus 6/6) | 0.0 (0/1) | 134 / 3.00 |
| C5: + GSN 3-iter BIF + MIntRec Transition Chains (Full v7.7) | 0.667 / 1.000 | 23.8 (n=21) | 50.0 (n=18) | 100.0 (multi-hop 0/0, consensus 6/6) | 0.0 (0/1) | 133 / 2.98 |

Notes
- d': within = pieces of one >6 s turn (one talker by construction); cross = pieces of adjacent turns whose (draft) speaker differs. M4's within pairs are E2CP Must-Link constraints (optimistic); M1's cross pairs come from a diarizer built on the same RMS / ZCR features (inflated). n/a = fewer than 2 pairs in the split.
- Speaker purity is against the draft speaker column (the acoustic diarizer) until speakers are verified by listening. Role ID: M3 uses only lexical self-cues; M2 / M4 / M5 add the Train speaker → role priors; M4 also runs gift_infer_roles, which only resolves speakers that have neither own cues nor a prior. Roles are keyed by the diarizer label, so voiceprints change speaker purity / d', not role ID.
- Multi-Hop & Consensus acc = pooled accuracy over (a) gold issue → resolution pairs whose reply chain is ≥ 2 hops or off the chain (hit = linked by same predicted thread or, C5, by GSN gsn_linked_issues) and (b) one consensus_state per Test unit vs the gold state (the engine's state rule — a resolution after the first issue, ≥ 2 speakers, no REQUEST_TO_REFUSAL left unanswered by a later resolution — applied to gold threads, speakers, MIntRec fine intents and slots). C1–C4 run without GSN and without MIntRec transition chains, so a trailing REQUEST_TO_REFUSAL cannot hold their state at OPEN_DISCUSSION.
- Latency includes per-turn feature extraction and is measured on this Mac (`--latency-only` on the Pi).

## Table C — Audio segmentation, noise filtering & data simulation (錄音檔：切分、雜訊 · 數據模擬)

### C-1. Real recordings: segmentation & noise (Whisper segments → loop collapse → merged turns → T_BACKCHANNEL_NOISE)
| recording | min | raw Whisper segments | ASR repetition loops collapsed | merged turns | segments / turn | T_BACKCHANNEL_NOISE turns | substantive turns | ambient packets ignored as noise |
|---|---|---|---|---|---|---|---|---|
| interview1 | 32.2 | 1132 | 4 runs / 113 segs (max ×54) | 241 | 4.70 | 84 (34.9 %) | 157 | 128 |
| interview2 | 51.7 | 797 | 9 runs / 99 segs (max ×47) | 234 | 3.41 | 63 (26.9 %) | 171 | 126 |
| interview3_full | 133.2 | 3853 | 15 runs / 62 segs (max ×22) | 834 | 4.62 | 219 (26.3 %) | 615 | 474 |
| interview4 (lab meeting, not in corpus) | 53.9 | 1920 | 0 runs / 0 segs (max ×0) | 461 | 4.16 | 135 (29.3 %) | 326 | 310 |
| **3 Jia-An recordings (3.62 h)** | 217.1 | 5782 | 28 runs / 274 segs | 1309 | 4.42 | 366 (28.0 %) | 943 | 728 |
- Loops = ≥ 3 consecutive identical segments (collapse_asr_loops on the stored *_raw_stt.json); merged turns / noise turns are read from each *_mpc_report.json (general_disentangle.noise_turn_ids).

### C-2. Simulated boundary bleed — 0.25 s collar + 75 %-anchor window vs whole-turn window (synthetic low / high voices, 3 s target turn, 2 seeds × 2 target voices)
| bleed inside the turn span | v7.7 collar + 75% anchor: correct speaker | v7.7 collar + 75% anchor: cos margin | v7.7 collar + 75% anchor: bleed in window | whole-turn window: correct speaker | whole-turn window: cos margin | whole-turn window: bleed in window |
|---|---|---|---|---|---|---|
| lead 0.2 s | 100 % | +0.685 | 0 % | 100 % | +0.640 | 6 % |
| lead 0.5 s | 100 % | +0.683 | 0 % | 100 % | +0.575 | 14 % |
| lead 1.0 s | 100 % | +0.684 | 0 % | 100 % | +0.473 | 25 % |
| lead 1.5 s | 100 % | +0.685 | 0 % | 100 % | +0.356 | 33 % |
| tail 0.2 s | 100 % | +0.685 | 0 % | 100 % | +0.639 | 6 % |
| tail 0.5 s | 100 % | +0.682 | 0 % | 100 % | +0.574 | 14 % |
| **all cases** | 100 % | +0.684 | 0 % | 100 % | +0.543 | 17 % |
- correct speaker = cos(turn vp, target ref) > cos(turn vp, interferer ref); bleed in window = share of the analysis window that lies inside the bleed span (0 % = rejected by the sampler).

### C-3. Simulated whole-turn overlap (interferer mixed under the full target turn)
| target-to-interferer SNR | correct speaker | mean cos margin |
|---|---|---|
| 20 dB | 100 % (4/4) | +0.669 |
| 10 dB | 100 % (4/4) | +0.540 |
| 5 dB | 100 % (4/4) | +0.311 |
| 0 dB | 50 % (2/4) | -0.033 |
- The collar / anchor only rejects bleed at turn boundaries; full overlap is not separated (no source separation on the Pi), so this row shows where the voiceprint evidence degrades.

### C-4. Simulated near-field (headset wearer) vs far-field (1–3 m interlocutor) — near_field_score
| condition | n | mean near_field_score | min / max |
|---|---|---|---|
| near-field | 8 | 0.657 | 0.586 / 0.730 |
| far-field (3 echoes, HF damping, −10 dB, noise floor) | 8 | 0.444 | 0.353 / 0.558 |
- separation: AUC 1.000, d' 3.251, accuracy 93.8 % at the midpoint threshold 0.551.

### C-5. Simulated multi-party interleaving (two real segments → one A A B B … session, gold threads kept)
| split scheme | interleaved sessions Train / Val / Test | Test thread purity / cohesion: C1 | C2 greedy | C3 Bi-CL | C5 full |
|---|---|---|---|---|---|
| 50 / 25 / 25 | 15 / 6 / 3 | 0.523 / 1.000 | 0.682 / 0.818 | 0.659 / 0.886 | 0.659 / 0.886 |
| 80 / 10 / 10 | 36 / 3 / 3 | 0.500 / 1.000 | 0.500 / 1.000 | 0.500 / 1.000 | 0.500 / 1.000 |

### C-6. Simulated home-care benchmark — 5 scripted dialogues (build_5_benchmark_scenarios; never trained on)
| Config | Thread purity / cohesion | Addressee acc % | Slot completeness % | Consensus acc % | Hypothetical P0 FAR % | Live P0 recall % |
|---|---|---|---|---|---|---|
| C1 | 0.818 / 1.000 | 0.0 (n=9) | 60.0 | 80.0 (4/5) | n/a (0/0) | 100.0 (n=1) |
| C2 | 1.000 / 1.000 | 100.0 (n=9) | 60.0 | 100.0 (5/5) | n/a (0/0) | 100.0 (n=1) |
| C3 | 1.000 / 1.000 | 100.0 (n=9) | 60.0 | 100.0 (5/5) | n/a (0/0) | 100.0 (n=1) |
| C4 | 1.000 / 1.000 | 88.9 (n=9) | 60.0 | 100.0 (5/5) | n/a (0/0) | 100.0 (n=1) |
| C5 | 1.000 / 1.000 | 88.9 (n=9) | 60.0 | 100.0 (5/5) | n/a (0/0) | 100.0 (n=1) |
- identical under both split schemes (both Val grids select the same configuration).

## Table D — Ambient prefetcher (P-series): background pool vs post-hoc window

| System | Referential query resolution acc % | Backchannel noise rejection % | Priority ranking acc % (REQUEST_TO_COMPLIANCE_OR_HESITATION) | Fire latency ms (query → payload) | Live P0 dispatch |
|---|---|---|---|---|---|
| P1: Post-hoc Sliding Window (no background pool; analyze_session on the last 4 turns at query time) | 50.0 (15/30) | 100.0 (366/366); domain turns kept 16.4 | 16.7 (4/24) | 5.51 (max 13.14) | — (waits for a query) |
| P2: Basic Keyword Background Pool (no MIntRec transition priority boost, no backchannel filter) | 70.0 (21/30) | 99.5 (364/366); domain turns kept 100.0 | 50.0 (12/24) | 0.20 (max 1.45) | — (waits for a query) |
| P3: Ours: AmbientIntentPreFetcher + MIntRec Transition Priority Boost + P0 Auto-Fire | 100.0 (30/30) | 98.1 (359/366); domain turns kept 100.0 | 100.0 (24/24) | 0.26 (max 3.92) | auto-fired at ingest (0.16 ms) |
- Referential / ranking probes are CONSTRUCTED (data simulation, 30 + 24): the labelled corpus has 0 gold compliance chains and only the scripted dialogues contain natural 小照 referential queries (included). Correct = the top-ranked fired intent is the gold referent. They test the mechanisms by design, so read them as a mechanism check with the per-factor breakdown below, not as field accuracy.
- Referential probes: family request → `gap` non-domain turns (chitchat / backchannel) → query 「小照，家屬剛剛說的那樣可以嗎？」; `trap` = a chitchat turn carrying another intent's keywords before / after the request. Ranking probes: an Arrange request answered by the caregiver (好 / 嗯…我再看看) plus an informational mention of another intent (`order`), then `tail` chitchat turns; gold = the answered request.
- Noise rejection: all merged turns of the 3 Jia-An recordings streamed without upstream thread labels; gold noise = interview_analyzer's T_BACKCHANNEL_NOISE (rule-based, not human). 'Domain turns kept' = non-noise turns with a domain intent that still reach the pool (P2/P3) or the query context (P1).
- P1 latency is the post-hoc analyze_session over its window; P2 / P3 latency is the pop + merge of pre-staged packets (dispatch_latency_ms). P-series has no trained parameters, so it does not depend on the split scheme.
- P1 breakdown — referential by trap: none: 4/9, before: 4/9, after: 4/9, scripted: 3/3; by gap: 0: 6/9, 2: 6/9, 5: 0/9, 1: 3/3; ranking by order: other_first: 2/12, chain_first: 2/12; by tail: 0: 4/12, 3: 0/12
- P2 breakdown — referential by trap: none: 9/9, before: 0/9, after: 9/9, scripted: 3/3; by gap: 0: 6/9, 2: 6/9, 5: 6/9, 1: 3/3; ranking by order: other_first: 0/12, chain_first: 12/12; by tail: 0: 6/12, 3: 6/12
- P3 breakdown — referential by trap: none: 9/9, before: 9/9, after: 9/9, scripted: 3/3; by gap: 0: 9/9, 2: 9/9, 5: 9/9, 1: 3/3; ranking by order: other_first: 12/12, chain_first: 12/12; by tail: 0: 12/12, 3: 12/12

## Scheme 50_25_25 — Professor's example: chronological 50 / 25 / 25 (10 : 5 : 5)

### 50_25_25 · 1. Split — per recording, chronological: first 50 % Train / next 25 % Val / last 25 % Test
| recording | labelled turns | Train | Val | Test |
|---|---|---|---|---|
| interview1 | 28 | 14 | 7 (from 19:42) | 7 (from 26:22) |
| interview2 | 28 | 14 | 7 (from 46:04) | 7 (from 49:34) |
| interview3_full | 30 | 15 | 7 (from 17:39) | 8 (from 89:16) |
| all 3 recordings | 86 | 43 (50.0 %) | 21 (24.4 %) | 22 (25.6 %) |
- Evaluation units = 15 window segments: train: interview1@16:15-17:15, interview1@19:00-20:00, interview2@41:45-42:45, interview2@45:45-46:45, interview3_full@09:30-10:30, interview3_full@17:00-18:00; val: interview1@19:00-20:00, interview1@26:15-27:15, interview2@45:45-46:45, interview2@49:15-50:15, interview3_full@17:00-18:00, interview3_full@89:00-90:00; test: interview1@26:15-27:15, interview2@49:15-50:15, interview3_full@89:00-90:00

### 50_25_25 · 2. Training stage (first 50 % of each recording)
- Voiceprint channel priors refit on 24225 active frames: 5867 from the 43 labelled Train turns + 18358 from 72 unlabelled full-recording turns before each Val cut (≤ 24 per recording, outside the golden windows).
| VP_DIM_WEIGHTS candidate (Train-fitted priors) | Train d' | Val d' (selects) | Test d' (not used) |
|---|---|---|---|
| shipped (reference: earlier calibration) | 0.663 | 0.466 | 0.278 |
| uniform | 0.453 | 0.409 | 0.294 |
| fisher_sqrt | 0.598 | 0.434 | 0.302 |
| fisher_linear ← selected on Val | 0.677 | 0.483 | 0.278 |
| voiceprint params | Train d' | Val d' | Test d' |
|---|---|---|---|
| currently in mpc_core.py | 0.720 | 0.715 | 0.341 |
| Train-fitted (this run) | 0.677 | 0.483 | 0.278 |
- changed vs mpc_core.py: VP_PRIOR_MEAN, VP_PRIOR_STD, VP_LOGSIG_PRIOR, VP_DIM_WEIGHTS
- Speaker → role priors: labelled Train majority (wins) + full-recording cue/graph inference over the unlabelled turns before the Val cut (fills the rest): interview1: speaker_1=supervisor, speaker_2=supervisor, speaker_3=supervisor; interview2: speaker_1=director, speaker_2=researcher, speaker_3=director; interview3_full: speaker_1=researcher, speaker_2=researcher, speaker_3=supervisor
- Voiceprint prototypes: 9 (recording, speaker) means from Train turns, used only by M5 in memory; never exported or written to mpc_core (privacy rule: no enrollment in the deployed engine).
- Fixed schema EXTENDED_ROLE_CUES: 23 keywords kept as shipped, 6 fire on Train; fire on Test but never on Train (authored with Test visible): ['主任跟督導', '我們後端', '學生們', '知識庫系統', '做個小結']
- Fixed schema ISSUE_TRIGGER_CATALOG: 68 keywords kept as shipped, 20 fire on Train; fire on Test but never on Train (authored with Test visible): ['猝逝', '去世', '綁居督', '個案綁', '細節可以發散']
- Fixed schema RESOLUTION_CATALOG: 42 keywords kept as shipped, 14 fire on Train; fire on Test but never on Train (authored with Test visible): ['切片', 'AI知識庫', '知識庫系統', '進知識庫', '多元選項', '多元問題的選項', '緊急鈕', '緊急控制', '控制的APP', 'APP', '加一個介面', '做綁定', '給我綁']
- Bi-CL check on Train (segments + 15 interleaved): thread F greedy 0.756 → Bi-CL 0.769.

### 50_25_25 · 3. Validation stage — grid on 6 Val segments + 6 interleaved Val sessions (J = mean of thread F, addressee acc, slot F1, link F1, speaker-cluster F)
| BICL sim | GSN N | E2CP α | thread F | addr % | slot F1 % | link F1 % | multi-hop R % | speaker F | J |  |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.30 | 1 | 0.0 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.691 | 0.6747 |  |
| 0.30 | 1 | 0.2 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.705 | 0.6775 |  |
| 0.30 | 1 | 0.4 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.691 | 0.6747 |  |
| 0.30 | 1 | 0.7 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.710 | 0.6785 |  |
| 0.30 | 2 | 0.0 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.691 | 0.6747 |  |
| 0.30 | 2 | 0.2 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.705 | 0.6775 |  |
| 0.30 | 2 | 0.4 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.691 | 0.6747 |  |
| 0.30 | 2 | 0.7 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.710 | 0.6785 |  |
| 0.30 | 3 | 0.0 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.691 | 0.6712 |  |
| 0.30 | 3 | 0.2 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.705 | 0.6741 |  |
| 0.30 | 3 | 0.4 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.691 | 0.6712 |  |
| 0.30 | 3 | 0.7 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.710 | 0.6750 |  |
| 0.30 | 4 | 0.0 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.691 | 0.6712 |  |
| 0.30 | 4 | 0.2 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.705 | 0.6741 |  |
| 0.30 | 4 | 0.4 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.691 | 0.6712 |  |
| 0.30 | 4 | 0.7 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.710 | 0.6750 |  |
| 0.45 | 1 | 0.0 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.691 | 0.6747 |  |
| 0.45 | 1 | 0.2 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.705 | 0.6775 |  |
| 0.45 | 1 | 0.4 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.691 | 0.6747 |  |
| 0.45 | 1 | 0.7 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.710 | 0.6785 |  |
| 0.45 | 2 | 0.0 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.691 | 0.6747 |  |
| 0.45 | 2 | 0.2 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.705 | 0.6775 |  |
| 0.45 | 2 | 0.4 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.691 | 0.6747 |  |
| 0.45 | 2 | 0.7 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.710 | 0.6785 | **selected** |
| 0.45 | 3 | 0.0 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.691 | 0.6712 |  |
| 0.45 | 3 | 0.2 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.705 | 0.6741 |  |
| 0.45 | 3 | 0.4 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.691 | 0.6712 | default |
| 0.45 | 3 | 0.7 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.710 | 0.6750 |  |
| 0.45 | 4 | 0.0 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.691 | 0.6712 |  |
| 0.45 | 4 | 0.2 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.705 | 0.6741 |  |
| 0.45 | 4 | 0.4 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.691 | 0.6712 |  |
| 0.45 | 4 | 0.7 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.710 | 0.6750 |  |
| 0.65 | 1 | 0.0 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.691 | 0.6747 |  |
| 0.65 | 1 | 0.2 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.705 | 0.6775 |  |
| 0.65 | 1 | 0.4 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.691 | 0.6747 |  |
| 0.65 | 1 | 0.7 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.710 | 0.6785 |  |
| 0.65 | 2 | 0.0 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.691 | 0.6747 |  |
| 0.65 | 2 | 0.2 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.705 | 0.6775 |  |
| 0.65 | 2 | 0.4 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.691 | 0.6747 |  |
| 0.65 | 2 | 0.7 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.710 | 0.6785 |  |
| 0.65 | 3 | 0.0 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.691 | 0.6712 |  |
| 0.65 | 3 | 0.2 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.705 | 0.6741 |  |
| 0.65 | 3 | 0.4 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.691 | 0.6712 |  |
| 0.65 | 3 | 0.7 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.710 | 0.6750 |  |
| 0.65 | 4 | 0.0 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.691 | 0.6712 |  |
| 0.65 | 4 | 0.2 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.705 | 0.6741 |  |
| 0.65 | 4 | 0.4 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.691 | 0.6712 |  |
| 0.65 | 4 | 0.7 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.710 | 0.6750 |  |
- selected (BICL=0.45, N=2, α=0.7) with J=0.6785; 6 of 48 configs tie at the top, 6 distinct J values (ties go to the shipped default BICL=0.45, N=3, α=0.4).
- best J by bicl: 0.3: 0.6785, 0.45: 0.6785, 0.65: 0.6785
- best J by gsn: 1: 0.6785, 2: 0.6785, 3: 0.6750, 4: 0.6750
- best J by alpha: 0.0: 0.6747, 0.2: 0.6775, 0.4: 0.6747, 0.7: 0.6785

### 50_25_25 · 4. Testing stage — last 25 % of each recording (3 segments, 22 turns), Val-selected config

#### Table A [50 / 25 / 25] — Speaker diarization & role identification
| Model | d' (within / cross pairs) | Speaker purity / inverse purity | Role ID acc % | Latency ms/turn | Peak mem KiB | Fitted params |
|---|---|---|---|---|---|---|
| M1: Acoustic Baseline (ZCR/RMS K-means) | 0.598 (24 / 48) | 0.727 / 0.727 | 21.1 (n=19) | 12.81 | 7 | 0 fitted (2-d K-means, k=3) |
| M2: Unconstrained 24-d Spectral Voiceprint (StatPool only, alpha=0) + Train role priors | 0.278 (24 / 48) | 0.591 / 0.773 | 52.6 (n=19) | 392.02 | 486 | 73 VP floats + role priors |
| M3: Lexical self-cues only (EXTENDED_ROLE_CUES; no Train priors, no GIFT graph, no voiceprint) | — | — (= label source) | 31.6 (n=19) | 0.55 | 7 | 23 schema cues (no priors) |
| M4: Ours v7.7 (StatPool 75%-Anchor + Turn-to-Diarize E2CP + GIFT Graph) + Train role priors | 0.447 (24 / 48) | 0.682 / 0.818 | 57.9 (n=19) | 392.89 | 512 | 73 VP floats + schema + priors + 3 hyper-params |
| M5: Train-split speaker voiceprint prototypes + role priors (nearest prototype; offline eval only) | — | 0.591 / 0.818 | 31.6 (n=19) | 388.17 | 432 | 9 × 24-d prototypes + role priors |
- Role ID attribution (Test, n=19): M3 cues only 31.6 → cues + GIFT graph (no priors) 31.6 → cues + Train priors (no graph) 57.9; M4 without GIFT roles 57.9, M4 without Train priors 31.6, M4 full 57.9. M4 speaker role sources: {'session_cues': 3, 'recording_prior': 6}; graph-inferred speakers: none

#### Table B [50 / 25 / 25] — Disentanglement, addressee, slot fusion, multi-hop / consensus, P0 guard (Test: 3 segments + 3 interleaved sessions, 66 turns)
| Config | Thread purity / cohesion | Addressee acc % | Slot completeness / precision % | Issue→resolution link recall all / multi-hop % | Multi-hop & consensus acc % | Hypothetical P0 FAR % | Live P0 recall % | Latency ms/turn | Peak mem KiB |
|---|---|---|---|---|---|---|---|---|---|
| C1: Baseline Sliding Window / Sequential (HRED-style, no disentanglement, raw P0 keywords) | 0.667 / 1.000 | 42.6 (n=54) | 30.0 / 42.9 | 0.0 / 0.0 (n=6/6) | 50.0 (multi-hop 0/6, consensus 6/6) | 100.0 (1/1) | n/a (n=0) | 0.05 | 8 |
| C2: Greedy TD/MF + Hypothetical P0 Guard | 0.773 / 0.864 | 46.3 (n=54) | 30.0 / 45.0 | 0.0 / 0.0 (n=6/6) | 50.0 (multi-hop 0/6, consensus 6/6) | 0.0 (0/1) | n/a (n=0) | 7.12 | 216 |
| C3: + Bi-CL Session-Prototype Disentanglement | 0.758 / 0.924 | 46.3 (n=54) | 30.0 / 45.0 | 0.0 / 0.0 (n=6/6) | 50.0 (multi-hop 0/6, consensus 6/6) | 0.0 (0/1) | n/a (n=0) | 7.54 | 244 |
| C4: + GIFT 4-Edge Topology & Replied-By Amendment | 0.758 / 0.924 | 51.9 (n=54) | 30.0 / 45.0 | 0.0 / 0.0 (n=6/6) | 50.0 (multi-hop 0/6, consensus 6/6) | 0.0 (0/1) | n/a (n=0) | 7.80 | 244 |
| C5: + GSN 3-iter BIF + MIntRec Transition Chains (Full v7.7) | 0.758 / 0.924 | 51.9 (n=54) | 30.0 / 42.9 | 0.0 / 0.0 (n=6/6) | 50.0 (multi-hop 0/6, consensus 6/6) | 0.0 (0/1) | n/a (n=0) | 10.32 | 243 |
- Interleaved Test sessions only — thread purity / cohesion: C1 0.523 / 1.000; C2 0.682 / 0.818; C3 0.659 / 0.886; C4 0.659 / 0.886; C5 0.659 / 0.886
- Gold consensus states of the Test units: {'CONSENSUS_ACTIONABLE': 6}; C5 predicted vs gold per unit: interview1@26:15-27:15: CONSENSUS_ACTIONABLE / CONSENSUS_ACTIONABLE; interview2@49:15-50:15: CONSENSUS_ACTIONABLE / CONSENSUS_ACTIONABLE; interview3_full@89:00-90:00: CONSENSUS_ACTIONABLE / CONSENSUS_ACTIONABLE; MIX[interview1@26:15-27:15 + interview2@49:15-50:15]: CONSENSUS_ACTIONABLE / CONSENSUS_ACTIONABLE; MIX[interview1@26:15-27:15 + interview3_full@89:00-90:00]: CONSENSUS_ACTIONABLE / CONSENSUS_ACTIONABLE; MIX[interview2@49:15-50:15 + interview3_full@89:00-90:00]: CONSENSUS_ACTIONABLE / CONSENSUS_ACTIONABLE

## Scheme 80_10_10 — Lab whiteboard standard split: chronological 80 / 10 / 10 (8 : 1 : 1)

### 80_10_10 · 1. Split — per recording, chronological: first 80 % Train / next 10 % Val / last 10 % Test
| recording | labelled turns | Train | Val | Test |
|---|---|---|---|---|
| interview1 | 28 | 22 | 3 (from 26:26) | 3 (from 26:52) |
| interview2 | 28 | 22 | 3 (from 49:41) | 3 (from 49:57) |
| interview3_full | 30 | 24 | 3 (from 89:23) | 3 (from 89:49) |
| all 3 recordings | 86 | 68 (79.1 %) | 9 (10.5 %) | 9 (10.5 %) |
- Evaluation units = 15 window segments: train: interview1@16:15-17:15, interview1@19:00-20:00, interview1@26:15-27:15, interview2@41:45-42:45, interview2@45:45-46:45, interview2@49:15-50:15, interview3_full@09:30-10:30, interview3_full@17:00-18:00, interview3_full@89:00-90:00; val: interview1@26:15-27:15, interview2@49:15-50:15, interview3_full@89:00-90:00; test: interview1@26:15-27:15, interview2@49:15-50:15, interview3_full@89:00-90:00

### 80_10_10 · 2. Training stage (first 80 % of each recording)
- Voiceprint channel priors refit on 25565 active frames: 9069 from the 68 labelled Train turns + 16496 from 72 unlabelled full-recording turns before each Val cut (≤ 24 per recording, outside the golden windows).
| VP_DIM_WEIGHTS candidate (Train-fitted priors) | Train d' | Val d' (selects) | Test d' (not used) |
|---|---|---|---|
| shipped (reference: earlier calibration) | 0.532 | 0.287 | -0.022 |
| uniform ← selected on Val | 0.428 | 0.567 | -0.149 |
| fisher_sqrt | 0.528 | 0.393 | -0.139 |
| fisher_linear | 0.556 | 0.232 | -0.062 |
| voiceprint params | Train d' | Val d' | Test d' |
|---|---|---|---|
| currently in mpc_core.py | 0.664 | 0.241 | 0.182 |
| Train-fitted (this run) | 0.428 | 0.567 | -0.149 |
- changed vs mpc_core.py: VP_PRIOR_MEAN, VP_PRIOR_STD, VP_LOGSIG_PRIOR, VP_DIM_WEIGHTS
- Speaker → role priors: labelled Train majority (wins) + full-recording cue/graph inference over the unlabelled turns before the Val cut (fills the rest): interview1: speaker_1=supervisor, speaker_2=supervisor, speaker_3=supervisor; interview2: speaker_1=director, speaker_2=researcher, speaker_3=director; interview3_full: speaker_1=researcher, speaker_2=researcher, speaker_3=supervisor
- Voiceprint prototypes: 9 (recording, speaker) means from Train turns, used only by M5 in memory; never exported or written to mpc_core (privacy rule: no enrollment in the deployed engine).
- Fixed schema EXTENDED_ROLE_CUES: 23 keywords kept as shipped, 6 fire on Train; fire on Test but never on Train (authored with Test visible): ['我們後端']
- Fixed schema ISSUE_TRIGGER_CATALOG: 68 keywords kept as shipped, 26 fire on Train; fire on Test but never on Train (authored with Test visible): ['猝逝', '去世']
- Fixed schema RESOLUTION_CATALOG: 42 keywords kept as shipped, 20 fire on Train; fire on Test but never on Train (authored with Test visible): ['AI知識庫', '進知識庫', '多元選項', '多元問題的選項', '緊急鈕', '緊急控制', '控制的APP', 'APP', '給我綁']
- Bi-CL check on Train (segments + 36 interleaved): thread F greedy 0.739 → Bi-CL 0.753.

### 80_10_10 · 3. Validation stage — grid on 3 Val segments + 3 interleaved Val sessions (J = mean of thread F, addressee acc, slot F1, link F1, speaker-cluster F)
| BICL sim | GSN N | E2CP α | thread F | addr % | slot F1 % | link F1 % | multi-hop R % | speaker F | J |  |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.30 | 1 | 0.0 | 0.755 | 44.4 | 35.3 | 0.0 | n/a | 0.674 | 0.4453 |  |
| 0.30 | 1 | 0.2 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.30 | 1 | 0.4 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.30 | 1 | 0.7 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.30 | 2 | 0.0 | 0.755 | 44.4 | 35.3 | 0.0 | n/a | 0.674 | 0.4453 |  |
| 0.30 | 2 | 0.2 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.30 | 2 | 0.4 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.30 | 2 | 0.7 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.30 | 3 | 0.0 | 0.755 | 44.4 | 35.3 | 0.0 | n/a | 0.674 | 0.4453 |  |
| 0.30 | 3 | 0.2 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.30 | 3 | 0.4 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.30 | 3 | 0.7 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.30 | 4 | 0.0 | 0.755 | 44.4 | 35.3 | 0.0 | n/a | 0.674 | 0.4453 |  |
| 0.30 | 4 | 0.2 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.30 | 4 | 0.4 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.30 | 4 | 0.7 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.45 | 1 | 0.0 | 0.755 | 44.4 | 35.3 | 0.0 | n/a | 0.674 | 0.4453 |  |
| 0.45 | 1 | 0.2 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.45 | 1 | 0.4 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.45 | 1 | 0.7 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.45 | 2 | 0.0 | 0.755 | 44.4 | 35.3 | 0.0 | n/a | 0.674 | 0.4453 |  |
| 0.45 | 2 | 0.2 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.45 | 2 | 0.4 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.45 | 2 | 0.7 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.45 | 3 | 0.0 | 0.755 | 44.4 | 35.3 | 0.0 | n/a | 0.674 | 0.4453 | **selected** |
| 0.45 | 3 | 0.2 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.45 | 3 | 0.4 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 | default |
| 0.45 | 3 | 0.7 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.45 | 4 | 0.0 | 0.755 | 44.4 | 35.3 | 0.0 | n/a | 0.674 | 0.4453 |  |
| 0.45 | 4 | 0.2 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.45 | 4 | 0.4 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.45 | 4 | 0.7 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.65 | 1 | 0.0 | 0.755 | 44.4 | 35.3 | 0.0 | n/a | 0.674 | 0.4453 |  |
| 0.65 | 1 | 0.2 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.65 | 1 | 0.4 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.65 | 1 | 0.7 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.65 | 2 | 0.0 | 0.755 | 44.4 | 35.3 | 0.0 | n/a | 0.674 | 0.4453 |  |
| 0.65 | 2 | 0.2 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.65 | 2 | 0.4 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.65 | 2 | 0.7 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.65 | 3 | 0.0 | 0.755 | 44.4 | 35.3 | 0.0 | n/a | 0.674 | 0.4453 |  |
| 0.65 | 3 | 0.2 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.65 | 3 | 0.4 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.65 | 3 | 0.7 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.65 | 4 | 0.0 | 0.755 | 44.4 | 35.3 | 0.0 | n/a | 0.674 | 0.4453 |  |
| 0.65 | 4 | 0.2 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.65 | 4 | 0.4 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
| 0.65 | 4 | 0.7 | 0.755 | 44.4 | 12.5 | 0.0 | n/a | 0.705 | 0.4058 |  |
- selected (BICL=0.45, N=3, α=0.0) with J=0.4453; 12 of 48 configs tie at the top, 2 distinct J values (ties go to the shipped default BICL=0.45, N=3, α=0.4).
- best J by bicl: 0.3: 0.4453, 0.45: 0.4453, 0.65: 0.4453
- best J by gsn: 1: 0.4453, 2: 0.4453, 3: 0.4453, 4: 0.4453
- best J by alpha: 0.0: 0.4453, 0.2: 0.4058, 0.4: 0.4058, 0.7: 0.4058

### 80_10_10 · 4. Testing stage — last 10 % of each recording (3 segments, 9 turns), Val-selected config

#### Table A [80 / 10 / 10] — Speaker diarization & role identification
| Model | d' (within / cross pairs) | Speaker purity / inverse purity | Role ID acc % | Latency ms/turn | Peak mem KiB | Fitted params |
|---|---|---|---|---|---|---|
| M1: Acoustic Baseline (ZCR/RMS K-means) | 0.421 (13 / 15) | 1.000 / 0.889 | 12.5 (n=8) | 15.87 | 5 | 0 fitted (2-d K-means, k=3) |
| M2: Unconstrained 24-d Spectral Voiceprint (StatPool only, alpha=0) + Train role priors | -0.149 (13 / 15) | 0.778 / 1.000 | 37.5 (n=8) | 378.81 | 751 | 73 VP floats + role priors |
| M3: Lexical self-cues only (EXTENDED_ROLE_CUES; no Train priors, no GIFT graph, no voiceprint) | — | — (= label source) | 25.0 (n=8) | 0.04 | 5 | 23 schema cues (no priors) |
| M4: Ours v7.7 (StatPool 75%-Anchor + Turn-to-Diarize E2CP + GIFT Graph) + Train role priors | 1.201 (13 / 15) | 0.778 / 1.000 | 50.0 (n=8) | 379.94 | 796 | 73 VP floats + schema + priors + 3 hyper-params |
| M5: Train-split speaker voiceprint prototypes + role priors (nearest prototype; offline eval only) | — | 0.889 / 1.000 | 25.0 (n=8) | 378.07 | 729 | 9 × 24-d prototypes + role priors |
- Role ID attribution (Test, n=8): M3 cues only 25.0 → cues + GIFT graph (no priors) 25.0 → cues + Train priors (no graph) 50.0; M4 without GIFT roles 50.0, M4 without Train priors 25.0, M4 full 50.0. M4 speaker role sources: {'session_cues': 1, 'recording_prior': 7}; graph-inferred speakers: none

#### Table B [80 / 10 / 10] — Disentanglement, addressee, slot fusion, multi-hop / consensus, P0 guard (Test: 3 segments + 3 interleaved sessions, 27 turns)
| Config | Thread purity / cohesion | Addressee acc % | Slot completeness / precision % | Issue→resolution link recall all / multi-hop % | Multi-hop & consensus acc % | Hypothetical P0 FAR % | Live P0 recall % | Latency ms/turn | Peak mem KiB |
|---|---|---|---|---|---|---|---|---|---|
| C1: Baseline Sliding Window / Sequential (HRED-style, no disentanglement, raw P0 keywords) | 0.667 / 1.000 | 19.0 (n=21) | 50.0 / 60.0 | 0.0 / n/a (n=0/0) | 100.0 (multi-hop 0/0, consensus 6/6) | 100.0 (1/1) | n/a (n=0) | 0.06 | 6 |
| C2: Greedy TD/MF + Hypothetical P0 Guard | 0.667 / 1.000 | 23.8 (n=21) | 50.0 / 60.0 | 0.0 / n/a (n=0/0) | 100.0 (multi-hop 0/0, consensus 6/6) | 0.0 (0/1) | n/a (n=0) | 2.43 | 77 |
| C3: + Bi-CL Session-Prototype Disentanglement | 0.667 / 1.000 | 23.8 (n=21) | 50.0 / 60.0 | 0.0 / n/a (n=0/0) | 100.0 (multi-hop 0/0, consensus 6/6) | 0.0 (0/1) | n/a (n=0) | 2.97 | 134 |
| C4: + GIFT 4-Edge Topology & Replied-By Amendment | 0.667 / 1.000 | 23.8 (n=21) | 50.0 / 60.0 | 0.0 / n/a (n=0/0) | 100.0 (multi-hop 0/0, consensus 6/6) | 0.0 (0/1) | n/a (n=0) | 3.00 | 134 |
| C5: + GSN 3-iter BIF + MIntRec Transition Chains (Full v7.7) | 0.667 / 1.000 | 23.8 (n=21) | 50.0 / 60.0 | 0.0 / n/a (n=0/0) | 100.0 (multi-hop 0/0, consensus 6/6) | 0.0 (0/1) | n/a (n=0) | 2.98 | 133 |
- Interleaved Test sessions only — thread purity / cohesion: C1 0.500 / 1.000; C2 0.500 / 1.000; C3 0.500 / 1.000; C4 0.500 / 1.000; C5 0.500 / 1.000
- Gold consensus states of the Test units: {'CONSENSUS_ACTIONABLE': 6}; C5 predicted vs gold per unit: interview1@26:15-27:15: CONSENSUS_ACTIONABLE / CONSENSUS_ACTIONABLE; interview2@49:15-50:15: CONSENSUS_ACTIONABLE / CONSENSUS_ACTIONABLE; interview3_full@89:00-90:00: CONSENSUS_ACTIONABLE / CONSENSUS_ACTIONABLE; MIX[interview1@26:15-27:15 + interview2@49:15-50:15]: CONSENSUS_ACTIONABLE / CONSENSUS_ACTIONABLE; MIX[interview1@26:15-27:15 + interview3_full@89:00-90:00]: CONSENSUS_ACTIONABLE / CONSENSUS_ACTIONABLE; MIX[interview2@49:15-50:15 + interview3_full@89:00-90:00]: CONSENSUS_ACTIONABLE / CONSENSUS_ACTIONABLE

## interview4 lab meeting (05:00–14:00, Hotspot #3 13:16–14:16)
interview4 (lab meeting, 53.9 min, 461 merged turns; excluded from the Jia-An handover corpus and from this benchmark)
- 05:00-14:00: 73 turns, diarizer speakers {'speaker_1': 17, 'speaker_2': 37, 'speaker_3': 19}
- MIntRec fine labels: {'UNK': 38, 'Agree': 12, 'Oppose': 6, 'Advise': 6, 'Arrange': 6, 'Inform': 2, 'Complain': 2, 'Prevent': 1}
- domain intents fired: {'JIA_AN_INTERNAL_RULE': 1} (home-care catalog; this is a lab meeting, so few apply)
- Hotspot #3 13:16-14:16: score 24.85, 7 speaker switches, threads ['T_TOPIC_2', 'T_TOPIC_3', 'T_TOPIC_6']
    13:21 speaker_1 UNK          你要有個比較表說這個測試完哪一個
    13:25 speaker_1 UNK          他可以挖在順位、拍照面
    13:27 speaker_2 UNK          那我決定用這個老師我已經解決了太好了就這樣這麼簡單一件事情
    13:39 speaker_3 Arrange      有概念喔好 那現在有沒有期待了那我下次不吹你了
    13:43 speaker_1 Complain     你有概念你就要那個你要自己加油有那個那個流程圖這樣畫很細
    13:51 speaker_2 UNK          自己看的真的很好
    13:52 speaker_3 UNK          但是如果你去比賽就這種圖給評審委員看對 他不會看的那麼多組對不對印度社我隨便幫你擺多組、連擺多組
    14:03 speaker_3 UNK          還有這種圖刪掉
    14:06 speaker_3 UNK          對不對
    14:08 speaker_3 UNK          OK嗎
    14:09 speaker_1 UNK          好 下一位下面一位你不就
    14:12 speaker_2 UNK          我我那個東西很容易登跟我開會一樣演套
