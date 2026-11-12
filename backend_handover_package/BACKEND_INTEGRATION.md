# 小照 (CareCopilot) 多人多輪意圖偵測模組 — 後端對接指南

## 1. 檔案清單
- `mpc_core.py`: 核心演算法（受話者辨識、TD/MF 與 Bi-CL 話題解纏結、GIFT/GSN 圖傳播、v7.7 聲紋 + E2CP 約束式說話者分群、OOD/P0 安全路由）。零外部套件依賴（只用 Python 標準函式庫），可直接複製進後端專案目錄。
- `BACKEND_INTEGRATION.md`: 本文件。
- `demo_real_clip.json`: Whisper STT 輸入格式範本（頂層 `{"segments": [...]}`），內容取自真實訪談 interview2 49:15 熱點，可直接用於下方快速呼叫範例。v7.7 起每個 segment 另帶真實音訊的聲紋欄位（見第 8 節），由 `interview_analyzer.py --export-samples` 自動產生。
- `mpc_5_scenarios_handover.json`: 5 大長照情境經本引擎分析後的標準輸出 JSON（供對齊 DB 與 LINE LIFF 欄位），即 `/api/iot/v1/utterances` 的 request body 格式。
- `real_interviews_extracted_samples.json`: 三場真實未剪輯訪談（interview1 / interview2 / interview3_full）各前 3 名多人熱點對話（共 9 個），已移除 `T_BACKCHANNEL_NOISE` 附和雜訊與 Whisper 重複幻覺。每個熱點的 `segments` 包成 `{"segments": [...]}` 後即可餵給 `load_from_whisper_json`；`mpc_result` 為本引擎的分析結果，可作為回歸測試基準。v7.7 起 9 個熱點的 `turns` / `segments` 皆含由原始錄音切出 60 秒片段計算的每輪聲紋欄位，`mpc_result.voiceprint_diarization_summary` 為該熱點的分群結果。

### 樹莓派 → 後端 HTTP 介面
| 端點 | Request body | 說明 |
|---|---|---|
| `POST /api/iot/v1/utterances` | `analyze_session()` 的完整輸出（格式同 `mpc_5_scenarios_handover.json` 的單一元素） | `client_event_id` 是 uuid5(`session_id`_`last_turn_id`)，後端必須以它做冪等去重。回應碼規則：2xx 視為成功；**409 視為「已收過」，Pi 端當作成功並移出佇列**；連線失敗、5xx、408/425/429 時，Pi 會把完整 payload 放進離線佇列，稍後依序重送；其他 4xx 不會重試：該 payload 會移到 Pi 端的死信檔 `offline_queue.rejected.jsonl`（含 `http_status`、`rejected_at`、完整 `payload`），不留在重送佇列。請讓 4xx 只用於「內容本身有誤、重送也不會成功」的情況。 |
| `POST /api/transcripts/{id}/confirm` | `{"confirmed_role_map": {"speaker_1": "<role>", ...}}` | LINE LIFF 確認 speaker → 角色對應後回寫。 |

## 2. 後端伺服器 3 行快速呼叫法 (Python)
當小照後端收到語音並透過 Whisper 轉出逐字稿後，在送入 GraphRAG / LLM 之前加入以下 3 行：

```python
from mpc_core import XiaoZhaoMPCEngine, load_from_whisper_json

# 1. 載入 Whisper 轉出的 segments JSON
session = load_from_whisper_json("demo_real_clip.json", session_id="S_001", case_id="JIA_AN_001")

# 2. 執行受話者推論與話題解纏結
result = XiaoZhaoMPCEngine().analyze_session(session)

# 3. 取得剔除閒聊後的乾淨上下文與意圖標籤，再送入 GraphRAG
clean_turns = [t for t in result["turns_detail"] if t["turn_id"] in result["selected_turn_indices"]]
detected_intents = result["detected_intents"]   # 例如 ['BA16_MEDICINE_PURCHASE_INQUIRY']
route_target = result["route_target"]           # 例如 'RAG_STANDARD_SOP' 或 '0E_Emergency_Broadcast'
```

## 3. 由真實訪談新增的意圖與路由
| detected_intent | route_target |
|---|---|
| `EMERGENCY_BUTTON_ESCALATION`（緊急鈕／對講機） | `SUPERVISOR_URGENT_CALL_CHANNEL` |
| `DEATH_AT_HOME_AFTERCARE_SOP`（宗教信仰、安撫鎮定、到院前） | `RAG_EXPERIENCE_KB_AFTERCARE` |
| `CASE_ASSIGNMENT_HIERARCHY`（主任 > 督導 > 照服員、權限） | `CASE_MGMT_ADMIN_QUEUE` |
| `SERVICE_CODE_QUERY`（BA/GA/SC 項目） | `RAG_SERVICE_CODE_KB` |
| `EXPERIENCE_KB_CONTRIBUTION`（歷史經驗、案例） | `EXPERIENCE_KB_INGEST` |
| `PRIVACY_CONSENT_RECORDING`（收音、麥克風、真實姓名） | `PRIVACY_REVIEW_QUEUE` |

「過世／到院前死亡／心臟停」已併入 `P0_EMERGENCY_ALERT`。`is_agreement_only(text)` 可用來在後端過濾純附和輪次（例如 對啦×50）。

## 4. v7.3 多人多輪 (MPC) 新增輸出欄位
`analyze_session(session, role_priors=None)` 的輸出新增：

| 欄位 | 說明 |
|---|---|
| `speaker_roles` | `speaker_N -> role`。居家情境沿用 `caregiver/family/elder`；會議/訪談情境推論 `supervisor_居督`、`caregiver_照服員`、`researcher_主持團隊`、`director_主任`。不建聲紋庫。 |
| `speaker_role_sources` | 每位說話者角色來源：`session_cues` / `prefilled` / `recording_prior` / `graph_inferred` / `no_evidence`。 |
| `turns_detail[].who_to_whom_role` | 例如 `caregiver_照服員 -> supervisor_居督`；另有 `inferred_role`、`addressee_speaker`、`mintrec_fine`、`source_thread_id`。 |
| `mintrec_transitions` | 同話題串、不同說話者、往後 3 個實質輪次內的意圖轉換：`PROBLEM_TO_GUIDANCE`、`REQUEST_TO_COMPLIANCE_OR_HESITATION`（`rag_intervention=true`）、`REQUEST_TO_REFUSAL`、`REPORT_TO_ACTION_PLAN`、`CRISIS_ESCALATION_CHAIN`。危機鏈的 `live_hazard=false`（例如「像剛我們說的暴力性騷」這類回顧／舉例）時 `priority=PROTOCOL_DESIGN_DISCUSSION`，不升級。 |
| `dst_slots.cross_turn_fusion` | `issue_triggers`（`raised_by`）、`proposed_resolutions`（`proposed_by`）、`participating_roles`、`consensus_state` ∈ `CONSENSUS_ACTIONABLE` / `IMMEDIATE_ESCALATION` / `OPEN_DISCUSSION`（`REQUEST_TO_REFUSAL` 只有在其後沒有新的 resolution 時才阻擋 `CONSENSUS_ACTIONABLE`：`trailing_refusal`）。 |

`role_priors`（或 `DialogueSessionState.role_priors`）可傳入整段錄音推論出的角色，作為短片段無角色線索時的後備；每份 `real_interviews_extracted_samples.json` 訪談附 `recording_speaker_roles`。
`AmbientIntentPreFetcher` 在意圖鏈完成時提高封包 `staging_priority`；送出的 payload 附 `mintrec_transitions` 與 `rag_intervention_required`。

## 5. v7.4 GIFT 4-edge graph (Gu et al., *GIFT: Graph-Induced Fine-Tuning for MPC Understanding*)
Substantive turns form a graph with edges `reply-self` / `reply-to` / `replied-by` / `indirect-reply`. A turn's `reply-to` parent is the latest earlier turn of its addressee speaker in the same thread; otherwise it is the discourse latch, which also covers A-B-A adjacency, within 180 s. Edge weights φ: `GIFT_EDGE_WEIGHTS = {reply-self: 1.0, reply-to: 0.92, replied-by: 0.88, indirect-reply: 0.45}`.

| 欄位 | 說明 |
|---|---|
| `gift_graph_summary.edge_counts` | 4 種邊的有序配對數（`indirect-reply = N² − N − 2R`）。 |
| `gift_graph_summary.edge_weights_phi` | 上述 4 個 φ(e_{q,v})。 |
| `gift_graph_summary.replied_by_amendments` | 經 `replied-by` / `reply-to` 邊修正意圖或角色的輪次數；明細見 `intent_amendments`、`graph_inferred_roles`、`graph_role_evidence`。 |
| `gift_graph_summary.anchor_turn_topology` | 主議題觸發輪（無則取最後實質輪）的 `reply_to_turn_id` 與 `replied_by_turn_ids`。 |
| `turns_detail[].gift_fine` / `gift_reply_to` | 圖修正後的有效意圖、直接回覆的父輪次。 |
| `mintrec_transitions[].gift_edge` / `gift_phi` / `gift_weight` | 意圖鏈是否落在 `reply-to` 邊上；`graph_induced=true` 為修正意圖後新增的鏈。 |

When a speaker has no lexical self-cue in a meeting, the speaker's role is inferred from the `reply-to` / `replied-by` edges they exchange with participants whose roles are already known. The partners' roles are excluded, and topic cues are weighted by φ. `speaker_roles` keeps the canonical role (for example `supervisor_居督`), and `speaker_role_sources` marks it as `graph_inferred`. Example: in interview3_full, `speaker_3` resolves to `supervisor_居督 (graph_inferred)`.
Since v7.5, an `Oppose` reply to a weak turn marks that turn as a proposal (`Arrange`), which unlocks `REQUEST_TO_REFUSAL`.

## 6. v7.5 Bi-CL session-prototype disentanglement (Zhang et al., *Conversation Disentanglement with Bi-Level Contrastive Learning*)
`bi_cl_disentangle_threads(turns, thread_of)` refines greedy, preceding-only thread splits at two levels:
- **Utterance level:** each turn gets a vector v_i built from the n-grams and domain tokens of the context window (u_{i-1}, u_i, u_{i+1}), plus the speaker s(i) and the mentions m_i.
- **Session level:** each thread k has a prototype p_k, the mean of its members' vectors.

Refinement runs as unsupervised EM: each turn is reassigned to the nearest leave-one-out prototype, with a bonus for local same-speaker continuity. The number of threads is estimated with K ≤ max(2, ⌊N_substantive / 6⌋). Over that budget, every turn in a micro-thread (≤ 2 turns) joins its nearest established session. Within budget, a turn moves only if its own content overlaps the target session, or if it is a single-turn thread sitting between two turns of that session. This keeps genuinely interleaved topics apart. `general_mpc_disentangle()` runs the same refinement in 300 s blocks. In the output, `thread_id` is the Bi-CL thread and `greedy_thread_id` is the original greedy split. Hotspots are still ranked by greedy interleave.

| 欄位 | 說明 |
|---|---|
| `bi_cl_disentangle_summary.estimated_k` | 估計的 session 數 K（上限見 `k_upper_bound`）。 |
| `bi_cl_disentangle_summary.raw_greedy_threads` / `refined_prototype_threads` | 貪婪切分串數 → 原型精煉後串數（另附 `raw_thread_ids` / `refined_thread_ids` / `reassigned_turns`）。 |
| `bi_cl_disentangle_summary.prototype_cohesion_score` | 留一法 cos(v_i, p_k(i)) 平均（單輪串記 0，碎片化不加分）；對照 `raw_cohesion_score`。 |


## 7. v7.6 GSN utterance-level graph-structured encoder (Hu et al., *GSN: A Graph-Structured Network for Multi-Party Dialogues*, IJCAI 2019)
`gsn_propagate_session(turns, gift_graph, initial)` implements Algorithm 1 (UG-E) on the GIFT graph's substantive nodes:
- **E (reply edges):** e_{i,j} = 1 (i < j) when turn j replies to turn i (the GIFT `reply-to` link).
- **U (speaker edges):** u_{i,j} = 1 (i < j) for chronologically consecutive turns of the same speaker (O(N) chain form).
- **State s_i⁰:** sparse features `intent:*` (domain intents), `mintrec:*` (GIFT-amended fine intent), `issue:*` / `res:*` (catalog slots), `risk:live_crisis`.
- **Gating:** η_i = SQH(‖ΔE_i‖), λ_i = SQH(‖ΔU_i‖), with SQH(x) = (α + x) / (1 + x), α = 0.25, so gates lie in [0.25, 1). Δ is the mean neighbour state. It is max-fused into s_i, so weights stay in [0, 1] and decay with each hop. This is a training-free stand-in for GSN's GRU.
- **Schedule (BIF):** 3 backward iterations (Eᵀ, Uᵀ: children → parents), then 3 forward iterations (E, U: parents → children). Content from one child reaches its parallel siblings (child → parent → sibling) and descendants up to 3 hops away.

`aggregate_cross_turn_slots(..., gsn=...)` reads the propagated states. Each `proposed_resolutions[*]` gains `gsn_linked_issues` (`label`, `origin_turn`, `weight`, `hops`). An issue raised outside the dominant thread(s) whose feature reaches a core turn with weight ≥ 0.5 is imported with `via: "gsn_flow"`, `gsn_weight`, `gsn_hops` and `reached_turn`. Separately, a new issue catalog entry, `搜尋參數不合理/會議細節發散`, fills the iv2 49:15 gap. That window is a single thread, so the catalog entry is what fills it. GSN adds the issue→resolution linkage (turn 6 → turns 7 and 8, 1 hop).

| 欄位 | 說明 |
|---|---|
| `gsn_flow_summary.iterations` / `alpha` / `schedule` | 3 / 0.25 / `backward_then_forward`。 |
| `gsn_flow_summary.reply_edges_E` | **E** 中的有向回覆邊數。 |
| `gsn_flow_summary.speaker_edges_U` | **U** 中同說話者的時序邊數。 |
| `gsn_flow_summary.parallel_sibling_branches` | 有 ≥ 2 個平行回覆的父輪數（GSN Fig 1）。 |
| `gsn_flow_summary.multi_hop_paths_3iter` | E ∪ U 中最短距離為 2–3 跳的有序 (i, j) 配對數。 |
| `gsn_flow_summary.mean_eta_reply_weight` / `mean_lambda_speaker_weight` | 6 次迭代中所有 η_i / λ_i 的平均值。 |

## 8. v7.7 Acoustic-lexical voiceprint + E2CP constrained diarization
Sources: ECAPA-TDNN (Desplanques et al., Interspeech 2020); Turn-to-Diarize (Xia et al., ICASSP 2022); Park & Georgiou (USC), *Multimodal Speaker Segmentation and Diarization using Lexical and Acoustic Cues*; M2MeT / AliMeeting (Yu et al., ICASSP 2022); AISHELL-4 (Fu et al.).
Everything is pure Python standard library, so the Pi needs no ML packages. The diarizer's `speaker_N` labels stay authoritative for roles and who→whom. v7.7 adds a continuous same-speaker confidence and flags turns whose voiceprint disagrees with their label.

**Turn voiceprint** (`extract_turn_voiceprint`; WAV wrapper in the Pi's `exp1_pi_audio_check.py`):
- **Sampling:** a 0.25 s collar is trimmed at both turn boundaries (turns ≥ 0.8 s). Analysis covers [0.50, 0.90] of the turn, around its 75% anchor. A turn > 6 s is split into ⌈dur / 6⌉ pieces, one sub-vector each, chained by Must-Links.
- **Frames:** 25 ms window, 10 ms hop, ≤ 96 frames per window. Each frame gives 13 cepstra (DCT of 20 log mel bands), 4 band-energy shares, spectral centroid, flatness, ZCR, cepstral F0 and voicing.
- **Pooling:** an SE-style per-channel reliability gate, then channel-dependent attentive statistics pooling (weighted mean and std; attention favours loud, voiced frames near the anchor). The result is a 24-d L2-normalised `x_i`. This is a training-free analogue: there are no learned weights.
- **`near_field_score` (0..1):** headset or closest talker vs. a 1–3 m interlocutor. It combines modulation depth, top-band share and flatness, plus a mild level term, because device gain varies by about 25 dB.
- **JSON-only input:** a deterministic proxy built from `rms_energy`, `zcr_pitch`, speech rate and duration.

**Constrained diarization** (`constrained_voiceprint_diarize`, run inside `analyze_session`):
- **Raw affinity:** A_ij = (1 + cos(x_i, x_j)) / 2.
- **Cannot-Link (−1):**
  - a question answered by a yes/no-type reply (adjacent, or along a GIFT reply-to edge);
  - a request, proposal, advice or complaint answered by a response token (可以 / 好 / 沒問題 / OK …);
  - a vocative addressed to another role (請主任 / 督導你 …);
  - a floor handoff to the next turn.
- **Must-Link (+1):**
  - sub-windows of a > 6 s turn;
  - a same-label discourse latch (a connective-opened continuation across a backchannel, or a dangling clause within 1.5 s);
  - a same label repeating one role's first-person self-reference (我們督導 / 我們後端 …).
- **E2CP (α = 0.4):** Q* = (1 − α)² (I − αL̄)⁻¹ Z (I − αL̄)⁻¹ over L̄ = D^-½ A D^-½. Then Â_ij = 1 − (1 − Q*_ij)(1 − A_ij) when Q* ≥ 0, else (1 + Q*_ij) A_ij.
- **Clustering:** row-wise 50th-percentile soft threshold, average symmetrisation, eigengap K (K ≥ 2 when ≥ 2 diarizer labels and ≥ 4 nodes), spectral k-means, then a flip-aligned 6-turn sliding-window majority vote. Clusters are mapped onto `speaker_N` by overlap. Sessions over 48 nodes run in chronological blocks.
- **Downstream weighting:** same-speaker probability P_ij = c·Â_ij + (1 − c)·[s_i = s_j], where c = 0.5 for WAV voiceprints, 0.25 for acoustic proxies and 0 for text-only. P_ij weights Bi-CL's speaker-continuity term and GSN's speaker adjacency **U** in place of a hard 0/1. **Text-only sessions produce exactly the v7.6 output**, plus the new summary fields.

**Privacy:** vectors are session-local evidence for grouping the turns of one session. There is no enrollment DB and no identity matching. **`analyze_session()` never puts raw vectors in the payload.** On the Pi they stay in memory. Only the 9 exported research hotspots store per-turn vectors.

Whisper segment input (`load_from_whisper_json`) accepts these optional fields: `end`, `rms_energy`, `zcr_pitch`, `voiceprint` (24 floats), `voiceprint_subvectors`, `voiceprint_source`, `near_field_score`.

| 欄位 | 說明 |
|---|---|
| `voiceprint_diarization_summary.voiceprint_mode` | `wav_75pct_anchor`（全部真實音訊）/ `mixed` / `proxy_acoustic` / `proxy_text`（純文字，分群不影響下游）。 |
| `turns_with_voiceprint` / `turns_with_proxy_voiceprint` | 有真實音訊聲紋 / 只有代理向量的輪數。 |
| `must_link_constraints` / `cannot_link_constraints` / `constraints_by_reason` | 節點對約束數與原因統計（`long_turn_split`、`discourse_latch`、`lexical_self_consistency`、`qa_turn_taking`、`adjacency_pair_response`、`vocative_addressing`、`floor_handoff`）。`constraints` 列出前 60 筆。 |
| `mean_intra_speaker_affinity_raw` / `_e2cp` | 同一 diarizer 標籤的輪對平均親和度（E2CP 前 / 後）；`mean_inter_speaker_affinity_*` 為不同標籤者。 |
| `estimated_speakers` / `diarizer_speakers` | 特徵值間隙估計的說話者數 / 上游標籤數。 |
| `diarization_consistency_score` | 有聲學證據的輪中，聲紋分群（對應到 `speaker_N` 後）與 diarizer 標籤一致的比例；純文字時為 `null`。 |
| `relabel_candidates` | 聲紋分群與標籤不一致的輪（`turn_id`、`diarizer`、`voiceprint_speaker`），可在 LINE LIFF 確認時提示。 |
| `near_field_primary_speaker` / `near_field_by_speaker` | 平均近場分數最高的標籤（配戴耳機 / 最靠近麥克風者）與各標籤平均。 |
| `turns_detail[*].refined_speaker` / `voiceprint_source` / `near_field_score` | 每輪的聲紋分群標籤、來源與近場分數。 |
| `gsn_flow_summary.speaker_edge_weighting` / `mean_u_speaker_confidence` | `voiceprint_e2cp` 或 `hard_0_1`；**U** 邊的平均同說話者信心。 |
| `bi_cl_disentangle_summary.speaker_continuity_weighting` | `voiceprint_e2cp` 或 `diarizer_label`。 |

Real-audio results on the 9 golden hotspots:
- **Coverage:** all 9 hotspots have real voiceprints (86 turns).
- **Constraints:** 54 Must-Links and 6 Cannot-Links.
- **E2CP shift:** intra-speaker affinity rises by +0.005 on average after E2CP; inter-speaker affinity moves by +0.002.
- **Agreement:** voiceprint clusters agree with the upstream ZCR/RMS diarizer on 50–73% of turns.
- **No verdict changes:** routes, intents, roles, transitions and consensus are unchanged on all 9 hotspots.

These are training-free features on a single far-field mic: treat `relabel_candidates` as review hints, not corrections.
