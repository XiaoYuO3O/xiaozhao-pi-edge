# XiaoZhao v7.7 — Train / Validation / Test benchmark (within-recording chronological 10 : 5 : 5)

Labels: 0 / 97 turns human-verified; the rest are DRAFT labels (text-level labels drafted from the cleaned transcripts, speaker = acoustic diarizer label). Numbers are provisional until verified.

## 1. Split — per recording, chronological: first 50 % Train / next 25 % Val / last 25 % Test
| recording | labelled turns | Train | Val | Test |
|---|---|---|---|---|
| interview1 | 28 | 14 | 7 (from 19:42) | 7 (from 26:22) |
| interview2 | 28 | 14 | 7 (from 46:04) | 7 (from 49:34) |
| interview3_full | 30 | 15 | 7 (from 17:39) | 8 (from 89:16) |
| all 3 recordings | 86 | 43 (50.0 %) | 21 (24.4 %) | 22 (25.6 %) |
- Evaluation units = 15 window segments: train: interview1@16:15-17:15, interview1@19:00-20:00, interview2@41:45-42:45, interview2@45:45-46:45, interview3_full@09:30-10:30, interview3_full@17:00-18:00; val: interview1@19:00-20:00, interview1@26:15-27:15, interview2@45:45-46:45, interview2@49:15-50:15, interview3_full@17:00-18:00, interview3_full@89:00-90:00; test: interview1@26:15-27:15, interview2@49:15-50:15, interview3_full@89:00-90:00
- Scripted regression set (not split, never trained on): 5 benchmark dialogues → Table C.

## 2. Training stage (first 50 % of each recording)
- Voiceprint channel priors refit on 5867 active frames of the 43 Train turns.
| VP_DIM_WEIGHTS candidate (Train-fitted priors) | Train d' | Val d' (selects) | Test d' (not used) |
|---|---|---|---|
| shipped (reference: earlier calibration) | 0.720 | 0.715 | 0.341 |
| uniform | 0.489 | 0.518 | 0.335 |
| fisher_sqrt | 0.635 | 0.660 | 0.364 |
| fisher_linear ← selected on Val | 0.720 | 0.715 | 0.341 |
| voiceprint params | Train d' | Val d' | Test d' |
|---|---|---|---|
| currently in mpc_core.py | 0.720 | 0.715 | 0.341 |
| Train-fitted (this run) | 0.720 | 0.715 | 0.341 |
- changed vs mpc_core.py: none
- Recording-level speaker → role priors learned from Train labels: interview1: speaker_1=supervisor, speaker_2=supervisor, speaker_3=supervisor; interview2: speaker_1=director, speaker_2=researcher, speaker_3=director; interview3_full: speaker_1=researcher, speaker_2=researcher, speaker_3=supervisor
- Voiceprint prototypes: 9 (recording, speaker) means from Train turns, used only by M5 in memory; never exported or written to mpc_core (privacy rule: no enrollment in the deployed engine).
- Fixed schema EXTENDED_ROLE_CUES: 23 keywords kept as shipped, 6 fire on Train; fire on Test but never on Train (authored with Test visible): ['主任跟督導', '我們後端', '學生們', '知識庫系統', '做個小結']
- Fixed schema ISSUE_TRIGGER_CATALOG: 68 keywords kept as shipped, 20 fire on Train; fire on Test but never on Train (authored with Test visible): ['猝逝', '去世', '綁居督', '個案綁', '細節可以發散']
- Fixed schema RESOLUTION_CATALOG: 42 keywords kept as shipped, 14 fire on Train; fire on Test but never on Train (authored with Test visible): ['切片', 'AI知識庫', '知識庫系統', '進知識庫', '多元選項', '多元問題的選項', '緊急鈕', '緊急控制', '控制的APP', 'APP', '加一個介面', '做綁定', '給我綁']
- Role-cue weights: one hit per cue, too few to re-weight; Train agreement 量身打造→director 0/1; 居服員的問題→supervisor 1/1; 也在騎車→supervisor 1/1; 我們督導→supervisor 0/0; 做實驗→researcher 1/1; 我的夥伴→caregiver 0/1
- Bi-CL check on Train (segments + 15 interleaved): thread F greedy 0.756 → Bi-CL 0.769.

## 3. Validation stage — grid on 6 Val segments + 6 interleaved Val sessions (J = mean of thread F, addressee acc, slot F1, link F1, speaker-cluster F)
| BICL sim | GSN N | E2CP α | thread F | addr % | slot F1 % | link F1 % | multi-hop R % | speaker F | J |  |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.30 | 1 | 0.0 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.705 | 0.6775 |  |
| 0.30 | 1 | 0.2 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.691 | 0.6747 |  |
| 0.30 | 1 | 0.4 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.686 | 0.6737 |  |
| 0.30 | 1 | 0.7 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.696 | 0.6756 |  |
| 0.30 | 2 | 0.0 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.705 | 0.6775 |  |
| 0.30 | 2 | 0.2 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.691 | 0.6747 |  |
| 0.30 | 2 | 0.4 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.686 | 0.6737 |  |
| 0.30 | 2 | 0.7 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.696 | 0.6756 |  |
| 0.30 | 3 | 0.0 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.705 | 0.6741 |  |
| 0.30 | 3 | 0.2 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.691 | 0.6712 |  |
| 0.30 | 3 | 0.4 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.686 | 0.6703 |  |
| 0.30 | 3 | 0.7 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.696 | 0.6722 |  |
| 0.30 | 4 | 0.0 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.705 | 0.6741 |  |
| 0.30 | 4 | 0.2 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.691 | 0.6712 |  |
| 0.30 | 4 | 0.4 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.686 | 0.6703 |  |
| 0.30 | 4 | 0.7 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.696 | 0.6722 |  |
| 0.45 | 1 | 0.0 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.705 | 0.6775 |  |
| 0.45 | 1 | 0.2 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.691 | 0.6747 |  |
| 0.45 | 1 | 0.4 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.686 | 0.6737 |  |
| 0.45 | 1 | 0.7 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.696 | 0.6756 |  |
| 0.45 | 2 | 0.0 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.705 | 0.6775 | **selected** |
| 0.45 | 2 | 0.2 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.691 | 0.6747 |  |
| 0.45 | 2 | 0.4 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.686 | 0.6737 |  |
| 0.45 | 2 | 0.7 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.696 | 0.6756 |  |
| 0.45 | 3 | 0.0 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.705 | 0.6741 |  |
| 0.45 | 3 | 0.2 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.691 | 0.6712 |  |
| 0.45 | 3 | 0.4 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.686 | 0.6703 | default |
| 0.45 | 3 | 0.7 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.696 | 0.6722 |  |
| 0.45 | 4 | 0.0 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.705 | 0.6741 |  |
| 0.45 | 4 | 0.2 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.691 | 0.6712 |  |
| 0.45 | 4 | 0.4 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.686 | 0.6703 |  |
| 0.45 | 4 | 0.7 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.696 | 0.6722 |  |
| 0.65 | 1 | 0.0 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.705 | 0.6775 |  |
| 0.65 | 1 | 0.2 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.691 | 0.6747 |  |
| 0.65 | 1 | 0.4 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.686 | 0.6737 |  |
| 0.65 | 1 | 0.7 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.696 | 0.6756 |  |
| 0.65 | 2 | 0.0 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.705 | 0.6775 |  |
| 0.65 | 2 | 0.2 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.691 | 0.6747 |  |
| 0.65 | 2 | 0.4 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.686 | 0.6737 |  |
| 0.65 | 2 | 0.7 | 0.782 | 63.0 | 88.9 | 38.1 | n/a | 0.696 | 0.6756 |  |
| 0.65 | 3 | 0.0 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.705 | 0.6741 |  |
| 0.65 | 3 | 0.2 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.691 | 0.6712 |  |
| 0.65 | 3 | 0.4 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.686 | 0.6703 |  |
| 0.65 | 3 | 0.7 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.696 | 0.6722 |  |
| 0.65 | 4 | 0.0 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.705 | 0.6741 |  |
| 0.65 | 4 | 0.2 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.691 | 0.6712 |  |
| 0.65 | 4 | 0.4 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.686 | 0.6703 |  |
| 0.65 | 4 | 0.7 | 0.782 | 63.0 | 88.9 | 36.4 | n/a | 0.696 | 0.6722 |  |
- selected (BICL=0.45, N=2, α=0.0) with J=0.6775; 6 of 48 configs tie at the top, 8 distinct J values (ties go to the shipped default BICL=0.45, N=3, α=0.4).
- best J by bicl: 0.3: 0.6775, 0.45: 0.6775, 0.65: 0.6775
- best J by gsn: 1: 0.6775, 2: 0.6775, 3: 0.6741, 4: 0.6741
- best J by alpha: 0.0: 0.6775, 0.2: 0.6747, 0.4: 0.6737, 0.7: 0.6756

## 4. Testing stage — last 25 % of each recording (3 segments, 22 turns), Val-selected config

### Table A — Speaker diarization & role identification
| Model | d' within-turn vs cross-speaker | Speaker purity / inverse purity | Role ID acc % | Latency ms/turn | Peak mem KiB | Fitted params |
|---|---|---|---|---|---|---|
| M1: Acoustic Baseline (ZCR/RMS K-means) | 0.598 (w=24, c=48) | 0.727 / 0.727 | 21.1 (n=19) | 11.07 | 7 | 0 fitted (2-d K-means, k=3) |
| M2: Unconstrained 24-d Spectral Voiceprint (StatPool only, alpha=0) + Train role priors | 0.341 (w=24, c=48) | 0.591 / 0.818 | 47.4 (n=19) | 289.19 | 484 | 73 VP floats + role priors |
| M3: Lexical Cues + Train role priors (No Graph, No Voiceprint) | — | — (= label source) | 57.9 (n=19) | 0.06 | 7 | 23 schema cues + role priors |
| M4: Ours v7.7 (StatPool 75%-Anchor + Turn-to-Diarize E2CP + GIFT Graph) + Train role priors | 1.413 (w=24, c=48) | 0.591 / 0.818 | 57.9 (n=19) | 294.45 | 515 | 73 VP floats + schema + priors + 3 hyper-params |
| M5: Train-split speaker voiceprint prototypes + role priors (nearest prototype; offline eval only) | — | 0.591 / 0.818 | 31.6 (n=19) | 274.90 | 435 | 9 × 24-d prototypes + role priors |
- d': within = pieces of one >6 s turn (one talker by construction); cross = pieces of adjacent turns whose (draft) speaker differs. M4's within pairs are E2CP Must-Link constraints (optimistic); M1's cross pairs come from a diarizer built on the same RMS / ZCR features (inflated).
- Speaker purity is against the draft speaker column (the acoustic diarizer), i.e. agreement with the diarizer until speakers are verified by listening. Role priors come from the same recording's Train turns.
- Latency includes per-turn feature extraction and is measured on this Mac (`--latency-only` on the Pi).

### Table B — Disentanglement, addressee, slot fusion, P0 guard (Test: 3 segments + 3 interleaved sessions, 66 turns)
| Config | Thread purity / cohesion | Addressee acc % | Slot completeness / precision % | Issue→resolution link recall all / multi-hop % | Hypothetical P0 FAR % | Live P0 recall % | Latency ms/turn | Peak mem KiB |
|---|---|---|---|---|---|---|---|---|
| C1: Baseline Sliding Window / Sequential (HRED-style, no disentanglement, raw P0 keywords) | 0.667 / 1.000 | 42.6 (n=54) | 30.0 / 42.9 | 0.0 / 0.0 (n=6/6) | 100.0 (1/1) | n/a (n=0) | 0.06 | 8 |
| C2: Greedy TD/MF + Hypothetical P0 Guard | 0.773 / 0.864 | 46.3 (n=54) | 30.0 / 45.0 | 0.0 / 0.0 (n=6/6) | 0.0 (0/1) | n/a (n=0) | 32.83 | 215 |
| C3: + Bi-CL Session-Prototype Disentanglement | 0.758 / 0.924 | 46.3 (n=54) | 30.0 / 45.0 | 0.0 / 0.0 (n=6/6) | 0.0 (0/1) | n/a (n=0) | 13.78 | 244 |
| C4: + GIFT 4-Edge Topology & Replied-By Amendment | 0.758 / 0.924 | 51.9 (n=54) | 30.0 / 45.0 | 0.0 / 0.0 (n=6/6) | 0.0 (0/1) | n/a (n=0) | 15.50 | 244 |
| C5: + GSN Bidirectional Flow + MIntRec Transitions (Full v7.7) | 0.758 / 0.924 | 51.9 (n=54) | 30.0 / 42.9 | 0.0 / 0.0 (n=6/6) | 0.0 (0/1) | n/a (n=0) | 9.17 | 243 |
- Interleaved Test sessions only — thread purity / cohesion: C1 0.523 / 1.000; C2 0.682 / 0.818; C3 0.659 / 0.886; C4 0.659 / 0.886; C5 0.659 / 0.886
- Links: labelled issue → later resolution in the same thread; multi-hop = the resolution's reply chain is ≥ 2 hops from the issue, or does not reach it. Predicted links = fused issue/resolution pairs in one predicted thread, plus (C5) GSN gsn_linked_issues. P0 is counted on real segments only.

### Table C — Scripted regression set (5 benchmark dialogues; not a held-out split)
| Config | Thread purity / cohesion | Addressee acc % | Slot completeness % | Hypothetical P0 FAR % | Live P0 recall % |
|---|---|---|---|---|---|
| C1 | 0.818 / 1.000 | 0.0 (n=9) | 60.0 | n/a (0/0) | 100.0 (n=1) |
| C2 | 1.000 / 1.000 | 100.0 (n=9) | 60.0 | n/a (0/0) | 100.0 (n=1) |
| C3 | 1.000 / 1.000 | 100.0 (n=9) | 60.0 | n/a (0/0) | 100.0 (n=1) |
| C4 | 1.000 / 1.000 | 88.9 (n=9) | 60.0 | n/a (0/0) | 100.0 (n=1) |
| C5 | 1.000 / 1.000 | 88.9 (n=9) | 60.0 | n/a (0/0) | 100.0 (n=1) |

## 5. interview4 lab meeting (05:00–14:00, Hotspot #3 13:16–14:16)
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
