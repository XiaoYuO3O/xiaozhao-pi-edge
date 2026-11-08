# 小照 (CareCopilot) 多人多輪意圖偵測模組 — 後端對接指南

## 1. 檔案清單
- `mpc_core.py`: 核心演算法（受話者辨識 + TD/MF 話題解纏結 + OOD/P0 安全路由），零外部套件依賴，可直接複製進後端專案目錄。
- `demo_real_clip.json`: Whisper STT 轉出後的輸入格式範本。
- `mpc_5_scenarios_handover.json`: 5 大長照情境經本引擎分析後的標準輸出 JSON（供對齊 DB 與 LINE LIFF 欄位）。`handover_payloads_for_backend.json` 為相同內容的舊檔名，保留供既有程式相容。
- `iot_api_client.py`: 樹莓派邊緣端發送至 `/api/iot/v1/utterances` 與 `/api/transcripts/{id}/confirm` 的對接客戶端（邊緣端使用，依賴 Pi 端其他模組）。
- `real_interviews_extracted_samples.json`: 三場真實未剪輯訪談（interview1 / interview2 / interview3_full）各前 3 名多人熱點對話（共 9 個），已移除 `T_BACKCHANNEL_NOISE` 附和雜訊與 Whisper 重複幻覺。每個熱點的 `segments` 可直接餵給 `load_from_whisper_json`，`mpc_result` 為本引擎的分析結果，可作為回歸測試基準。

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
| `dst_slots.cross_turn_fusion` | `issue_triggers`（`raised_by`）、`proposed_resolutions`（`proposed_by`）、`participating_roles`、`consensus_state` ∈ `CONSENSUS_ACTIONABLE` / `IMMEDIATE_ESCALATION` / `OPEN_DISCUSSION`。 |

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

