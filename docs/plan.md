

下面我按你的要求分三部分给出：

1. **从 PDF 完整提取出的“可复制英文原文（清理排版版）”**
2. **逐条 review：每一环节可能漏掉/不够可执行的细节 + 我建议补齐的点**
3. **“补齐后的完整中文版本（可直接复制用作评估方案/实验计划）”**

---

## 1) 完整可复制英文原文（清理排版版）

> 说明：这是对 PDF 提取文本的**忠实转写**，我仅做了**排版清理**（修掉明显的断行、列表对齐、去掉 PDF 杂乱行号/重复数字），不改变原意。

```text
MuSciClaims_eng

1. One-sentence Summary
We need to answer an unavoidable question: “Can our model truly audit scientific claims by
seeing figures + reading captions, rather than merely guessing from text?”
MuSciClaims characterizes this capability boundary via three-way classification (SUPPORT /
CONTRADICT / NEUTRAL) plus diagnostic dimensions. (arXiv)

1. What “Necessary Capabilities” Are We Actually Validating?
We decompose a “scientific vetting agent” into minimal capability axioms. For each axiom,
we specify the required observable evidence (experiments), forming a non-separable
validation loop.

1.1 Capability Axioms
A1 Evidence-Consistency
Given (Figure, Caption, Claim), the model’s predicted label must be consistent with the
evidence:
• SUPPORT: evidence supports the claim
• CONTRADICT: evidence contradicts the claim
• NEUTRAL: evidence is irrelevant / insufficient

A2 Cross-Modal Necessity
If we provide only the Caption or only the Figure, performance should drop significantly;
otherwise, it indicates the model is not truly fusing the two modalities. The MuSciClaims paper
explicitly notes that “models can often obtain most of their score using only a single
modality,” which is exactly what must be diagnosed. (arXiv)

A3 Evidence Localization
Vetting is not a black-box decision: the model should be able to indicate which panels (Panel
A/B/…) it used. The dataset provides gold panel annotations for the most relevant panels,
enabling quantitative evaluation of localization. (Hugging Face)

A4 Epistemic Sensitivity
For the same base claim, comparing the supported version vs. the perturbed-contradicting
version, the model should flip its judgment (from SUPPORT → CONTRADICT). The
contradictory samples in MuSciClaims are produced via manual perturbations of supported
claims, specifically designed to test this property. (arXiv)

A5 Reproducibility & Auditability
Under the same input and the same settings, outputs should be deterministic; outputs must
follow a strict, parseable schema, enabling replay and large-scale statistics.

Why these cannot be removed:
These five axioms form the minimal logical closure required for a “vetting agent” to be valid.
If we only do A1 (classification score), then text gaming / support bias / failure to read figures
could all be hidden. The MuSciClaims paper also emphasizes systematic failure modes such as
bias toward SUPPORT, poor localization, and weak cross-modal aggregation. (arXiv)

2. Dataset and Input Contract (Contract)

2.1 Dataset Facts and Scale
MuSciClaims contains 1,515 samples, spanning physics/chemistry/biology, with balanced
classes (505 per class). (arXiv)

Key fields in the Hugging Face dataset (minimum required) include:
• base_claim_id: ID of the original/base claim
• claim_id: ID of the variant (support/contra/neutral)
• claim_text: claim text
• label_3class: SUPPORT/CONTRADICT/NEUTRAL
• label_2class: SUPPORT/NON_SUPPORT (optional 2-class evaluation)
• associated_figure_filepath: path to the figure image
• associated_figure_panels: list of relevant panels (gold for localization)
• caption: figure caption text
• paper_id / associated_figure_number and other metadata (Hugging Face)

Why this cannot be removed:
Without explicit input and label fields, the experiment cannot be executed; without base_claim_id,
A4 (sensitivity) cannot be defined; without associated_figure_panels, A3 cannot be quantified.

2.2 Data Loading and Image Download Specification (Must Be Fixed)
• Dataset loading: datasets.load_dataset("StonyBrookNLP/MuSciClaims") (split:
  test, 1,515 rows) (Hugging Face)
• Image retrieval: for each row, use
  hf_hub_download(repo_id="StonyBrookNLP/MuSciClaims",
                  filename=row["associated_figure_filepath"], repo_type="dataset")
  to download locally. (Hugging Face)
• Image preprocessing (minimal and reproducible):
  a. PIL.Image.open → convert to RGB
  b. No cropping (cropping multi-panel figures breaks A3/A2)
  c. If the model has max resolution/patch limits, apply aspect-ratio-preserving resize to the
     model’s max side length, and record original size / resize ratio (log it)

Why this cannot be removed:
Multi-panel figures in VLMs are extremely prone to information loss due to resizing/cropping;
if you do not record this, reproducibility fails and error analysis cannot determine whether
“the model is bad” or “preprocessing destroyed the figure.”

3. Evaluation Targets and Controls

3.1 Models That Must Be Evaluated
• Ours: the model released on Hugging Face on Saturday (target)
• Base: a baseline checkpoint with the same architecture/interface as Ours (or an unadapted
  version) (internal baseline)

Why this cannot be removed:
Without Base, we cannot demonstrate that “our training/adaptation produced improvements”—
we can only show “the model has some score.”

3.2 Input Ablation Baselines That Must Be Included (To Prove A2)
For each model, we must run the following input conditions (VLM case):
1. F+C (Full): Figure + Caption + Claim
2. C-only: Caption + Claim (image null/empty)
3. F-only: Figure + Claim (no caption)

The paper explicitly notes that many models can get most of their score using only a single
modality, so these ablations are required to prove real cross-modal fusion. (arXiv)

If our Hugging Face model is text-only (cannot take images):
• The main experiment runs only C-only and (optional) Claim-only (see 6.4), and we mark the
  “VLM track” as “to be completed once a visual-interface version is available.”
• We also provide a “tool-augmented track” (see 6.5) that converts Figure → structured text,
  feeds it into the text model, and reports separately: “model-only” vs “model+tool.”

Why this cannot be removed:
Without input ablations, we cannot prove the model is actually “looking at the figure”; a high
score might come entirely from captions or claim language patterns.

4. Experiment Matrix (Non-separable: Each Item Maps to an Axiom)
We run the following experimental blocks for each model (Ours/Base). Each block corresponds
to at least one of A1–A5; removing any block makes some axiom unverifiable.

4.1 Main Task: 3-class Claim Verification (Validates A1)
• Input: the three conditions from 3.2 (Full / C-only / F-only)
• Output: label ∈ {SUPPORT, NEUTRAL, CONTRADICT}
• Metrics: Macro-F1 (primary) + per-class P/R/F1 + confusion matrix

4.2 Evidence Localization: Panel Prediction (Validates A3 and Explains A1 Failures)
• Task: model outputs the list of panels it used (e.g., ["Panel A", "Panel C"])
• Gold: HF field associated_figure_panels (relevant panel set per example) (Hugging Face)
• Metric: set-based F1 (defined in 7.3)

4.3 Cross-Modal Aggregation Diagnosis (Validates A2)
Compute from 4.1 across input conditions:
• Δ_full-caption = F1(Full) - F1(C-only)
• Δ_full-figure = F1(Full) - F1(F-only)
• Synergy gain: Synergy = F1(Full) - max(F1(C-only), F1(F-only))

Synergy near 0 indicates the model is not truly fusing modalities (it is merely “using the
stronger modality”). The paper also discusses this phenomenon. (arXiv)

4.4 Epistemic Sensitivity (Validates A4)
• Group samples by base_claim_id: a base claim typically has both a support variant and
  a contradict variant (created via perturbation). (arXiv)
• Core metric: Flip-Rate (support→contradict flip rate)
  For each base_claim_id:
  ▪ take prediction ŷ_s on the SUPPORT variant
  ▪ take prediction ŷ_c on the CONTRADICT variant
  If ŷ_s = SUPPORT and ŷ_c = CONTRADICT, count as one “correct flip”
• Output: Flip-Rate + failure mode breakdown (e.g., SUPPORT→SUPPORT,
  CONTRADICT→SUPPORT, etc.)

Why this cannot be removed:
A4 is the most critical anti-fraud capability of a vetting agent. The paper’s abstract explicitly
notes models tend to predict SUPPORT and are insensitive to subtle perturbations. (arXiv)

4.5 Reproducibility (Validates A5)
• Fix decoding parameters (temperature=0, etc.)
• Repeat the same setup twice: require per-example output identical (decision agreement
  rate = 100%).
• If inconsistent: log the inconsistent examples and system environment (GPU, library versions,
  flash-attn enabled, parallel sampling, etc.)

Why this cannot be removed:
Without this, results cannot serve as a reliable benchmark for a “vetting system”; any score
could be contaminated by engineering randomness.

5. Prompting and Output Schema (Must Be Strict and Machine-Readable)
The MuSciClaims paper provides strict JSON output formatting prompts (decision-only /
reasoning+decision / panels+reasoning+decision). We adopt this as the only valid output
protocol. (arXiv)

5.1 Schema 1: Decision-only (D)
• Output must be single-key JSON:
  {"decision": "SUPPORT"} (or NEUTRAL/CONTRADICT)

5.2 Schema 2: Short Reasoning + Decision (R→D)
• Output must be two-key JSON (in order):
  {"reasoning": "...(1–2 sentences)...", "decision": "SUPPORT"}

5.3 Schema 3: Panel + Reasoning + Decision (For Localization)
• Output must be three-key JSON (in order):
  {"figure_panels": ["Panel A"], "reasoning": "...", "decision": "SUPPORT"}

5.4 Parsing and Error Handling (Must Be Fixed)
• If strict JSON parse fails → trigger one “format-fix retry” (stronger constraint: output JSON
  only, no extra text)
• If still fails on second attempt → mark as invalid_output and report its rate separately
• If label not in the 3 allowed classes → treat as invalid

Why this cannot be removed:
Without a machine-checkable contract, large-scale evaluation is impossible; also, the model
cannot be integrated into an automated vetting pipeline (let alone on-chain audit).

6. Inference Settings (Engineering Must Be Fixed)

6.1 Decoding Parameters (Main Evaluation)
• temperature = 0 (greedy)
• top_p = 1.0
• max_new_tokens: recommend 128 (D) / 256 (R→D and panels)
• Disable tool/function calling (unless using the tool-augmented track in 6.5)

6.2 Batching and Throughput (Record, Don’t Guess)
• For each inference, record: input tokens, output tokens, latency, memory usage (if available)
• Any timeout/OOM: log and rerun (same parameters)

6.3 Visual Input (VLM case)
• Use the full figure image at associated_figure_filepath (multi-panel) (Hugging Face)
• If resize is required: record original and resized dimensions; ensure Full/F-only use identical
  preprocessing.

6.4 Additional “Gaming Detection” Ablation (Recommended in Main Table; Strengthens A2)
• Claim-only: provide only the Claim (no figure/caption)
Purpose: test whether the model relies on strong language priors (e.g., common biological
phrasing causing support bias).
Why this is also “non-removable” (strictly speaking):
if Claim-only performance is close to Full, we know the model is not using evidence. This is a
more extreme sanity check than C-only/F-only, and is necessary for a vetting system.

6.5 Tool-Augmented Track (Only if Text-only / or as an extension)
• Use an independent figure reader to produce figure_description (OCR + chart structure
  extraction/summarization)
• Input: figure_description + caption + claim
• Output: same schema as 5.1/5.2
• Reporting must split into two columns:
  ◦ Ours(text-only)
  ◦ Ours(text-only + tool)
Avoid attributing “tool capability” to the model itself.

7. Metrics and Statistical Methods

7.1 Primary Metrics (A1)
• Macro-F1 (3 classes): primary metric
• Per-class Precision/Recall/F1
• Confusion matrix (especially: CONTRADICT/NEUTRAL wrongly predicted as SUPPORT)

7.2 Support Bias (A1 + A4)
• Pred(SUPPORT): fraction predicted as SUPPORT
• FN_contradict_as_support: fraction CONTRADICT → SUPPORT
• FN_neutral_as_support: fraction NEUTRAL → SUPPORT
The paper notes a tendency to predict “SUPPORTED”; this bias must be explicitly quantified. (arXiv)

7.3 Evidence Localization Metric (A3)
Let gold panel set be P and predicted set be P_hat:
• Precision = |P ∩ P_hat| / |P_hat| (if |P_hat| = 0, Precision = 1 iff |P| = 0)
• Recall    = |P ∩ P_hat| / |P|     (if |P| = 0, Recall = 1)
• F1 = 2PR/(P+R)
Gold comes from associated_figure_panels. (Hugging Face)

7.4 Cross-Modal Synergy (A2)
• Synergy = F1(Full) - max(F1(C-only), F1(F-only))
The larger the synergy, the stronger the evidence the model truly fuses the two modalities
rather than relying on a single modality. (arXiv)

7.5 Epistemic Sensitivity (A4)
• Flip-Rate: as defined in 4.4
• Failure type breakdown:
  ◦ SUPPORT→SUPPORT (insensitive to perturbations)
  ◦ SUPPORT→NEUTRAL (uncertainty increases but contradiction not detected)
  ◦ SUPPORT→CONTRADICT (ideal)
and reverse-direction errors (e.g., predicting CONTRADICT for a supported claim)

7.6 Statistical Significance (Avoid “Looks Improved”)
• For differences in Macro-F1 and CONTRADICT-F1: use paired bootstrap (resample examples)
  to compute 95% CI; if CI does not cross 0, treat improvement as significant.
• Since this is a paired comparison on the same test set, this is the lowest-cost and most robust
  significance approach.

8. Implementation Plan (Engineer to File-Level; Avoid “Slogan Plans”) [Must Not Be Deleted]

8.1 Directory and Artifact Specification (Evaluation = Audit)
Suggested artifact structure (all reproducible via scripts):
• runs/{model}/{condition}/{prompt_mode}/predictions.jsonl
  ◦ Each line includes: claim_id, base_claim_id, label_gold, label_pred,
    panels_gold, panels_pred, reasoning(optional), latency_ms,
    tokens_in/out, image_meta
• scores/{model}/summary.csv
  ◦ Macro-F1, per-class F1, Synergy, Flip-Rate, SupportBias, etc.
• analysis/{model}/error_slices.md
  ◦ Top 30 high-risk failure cases (e.g., CONTRADICT→SUPPORT)

8.2 Run Matrix (Must Be Written Once, Explicitly)
For each model (Ours/Base):

Main task (A1/A2):
• Full + D
• Full + R→D
• C-only + D
• C-only + R→D
• F-only + D (VLM only)
• F-only + R→D (VLM only)
• Claim-only + D (strongly recommended)
• Claim-only + R→D (optional)

Evidence localization (A3):
• Full + Panels+R→D (or Panels+D)

Sensitivity (A4):
• Compute by aggregating main-task outputs by base_claim_id (no extra inference required)

Reproducibility (A5):
• Repeat key runs (at least Full+D and Full+R→D) twice and check consistency

9. Risks, Validity, and Leakage Audit (Must Consider from First Principles)

9.1 Data Leakage / Training Contamination (Validity Threat)
MuSciClaims comes from specific journals (e.g., Cell, JACS, Nature Physics). If our training
corpus contains these papers/figures, evaluation may be contaminated. The HF data card also
indicates source journals. (Hugging Face)
Minimal audit actions:
• Use paper_id list to cross-check against our training corpus index (if available)
• If overlaps exist: mark “potential contamination” in the report and separately report scores
  on overlapping papers vs overall

Why this cannot be removed:
Without leakage audit, the score cannot be used to externally claim “generalizable vetting capability.”

9.2 Multi-panel Figure Information Loss
• Resizing/compression may make axes/legends unreadable
• Mitigation: log image sizes; increase max input resolution if needed; visually inspect failed examples

9.3 Output Format Drift
• Mitigation: strict JSON schema + one format-fix retry + invalid rate reporting

10. Deliverables
1. One-page results summary:
   ◦ Ours vs Base: Macro-F1, CONTRADICT-F1, Synergy, Flip-Rate, SupportBias
   ◦ Two confusion matrices (Ours/Base)
   ◦ 5–10 “vetting highlight cases” (Base wrong, Ours correct)
2. Reproducible evaluation package:
   ◦ evaluation scripts, environment file (conda/pip freeze), run configs, all predictions.jsonl,
     scores.csv
3. Failure-mode report (for next iteration):
   ◦ error types: localization failure / cross-modal fusion failure / visual reading failure /
     language-prior gaming
   ◦ 3–5 representative examples per type + remediation suggestions (e.g., higher resolution,
     panel-aware prompting, perturbation-contrast training)

11. Appendix: Prompt Templates (Strict JSON, Ready to Use)
These templates align with MuSciClaims prompt specifications: strict JSON, fixed keys,
extensible to panel prediction. (arXiv)

A. Decision-only (D)
You are an AI model tasked with verifying claims related to visual evidence using zero-shot learning.
Your job is to analyze the given image(s) and its provided caption(s) to decide whether it SUPPORT or
CONTRADICT or NEUTRAL the provided claim.
CLAIM: {claim_text}
IMAGE CAPTION(S): {caption}
After completing your analysis, output exactly one JSON object with exactly one key: "decision".
For "decision", output exactly one word — either "SUPPORT" or "CONTRADICT" or "NEUTRAL"
(uppercase, no extra text).
The output must start with { and end with }.

B. Reasoning → Decision (R→D, 1–2 sentences)
You are an AI model tasked with verifying claims related to visual evidence using zero-shot learning.
CLAIM: {claim_text}
IMAGE CAPTION(S): {caption}
After completing your analysis, output exactly one JSON object with exactly two keys in this order:
"reasoning" and "decision".
- "reasoning": one- or two-sentence evidence-based explanation (refer to axes/legend/trends/panels).
- "decision": one of "SUPPORT" / "CONTRADICT" / "NEUTRAL".
No extra text.

C. Panels + Reasoning + Decision (Evidence Localization)
You are an AI model tasked with verifying claims related to visual evidence using zero-shot learning.
CLAIM: {claim_text}
IMAGE CAPTION(S): {caption}
Output exactly one JSON object with exactly three keys in this order: "figure_panels", "reasoning",
"decision".
- "figure_panels": list ONLY panel labels needed (e.g., ["Panel A","Panel C"]), or [].
- "reasoning": one- or two-sentence explanation grounded in those panels.
- "decision": "SUPPORT" / "CONTRADICT" / "NEUTRAL".
No extra text.
```

---

## 2) 逐条 review：每一环节“容易漏掉的细节” + 我建议补齐的点

下面我按原文结构，从**可执行性/可审计性/可复现实验**角度逐条挑“缺口”。（你说的“补齐”，我会在第 3 部分直接写进中文版最终稿里。）

### 1. One-sentence Summary（目标定义）

**可能漏掉：**

* “审计 scientific claims” 的**边界条件**没写清：

  * 是只针对 **(Figure, Caption, Claim)** 的局部证据？
  * 还是允许模型用外部常识/背景知识？
* “不过度猜测”如何落地：需要写明**禁止外部检索**、是否允许使用论文正文等。

**建议补齐：**

* 明确：本评估默认**不允许**访问论文正文与外部知识库（除非单列“tool-augmented track”），否则 A2/A4 容易被“外部知识”污染。

---

### 1.1 Capability Axioms（A1–A5 定义）

**可能漏掉：**

* A1 “consistent with evidence”仍然需要**判定依据的最小要求**：

  * 例如：如果图像看不清/caption 不完整，应该倾向 NEUTRAL 还是允许“合理推断”？
* A3 panel 标注：数据集里的 panel label 形式可能不统一（"A" vs "Panel A"），需要**规范化规则**。
* A5 “deterministic” 在很多推理栈（flash-attn、cuda kernel、并行采样）下不一定 100% 可达，需要写“**达不到时如何处理**”。

**建议补齐：**

* 给出**判定优先级**：证据不足 → NEUTRAL；不允许用常识补洞。
* A3：定义 panel label normalize（大写、去空格、统一加 “Panel ” 前缀）。
* A5：定义“确定性等级”：

  * Level-1：固定 temperature=0 + 单线程/单 batch + 固定 seed；
  * Level-2：再加 deterministic 算子与禁用非确定性 kernel；
  * 并规定：做不到 100% 时至少要报告“差异率 + 栈信息”。

---

### 2. Dataset and Input Contract（数据与输入契约）

#### 2.1 Dataset Facts and Scale

**可能漏掉：**

* **数据版本锁定**：HF dataset 有可能更新（字段、文件路径、修正），不锁定 revision 会导致结果不可复现。
* train/val/test 的说明：文档假设 split=test 且 1515 行，但实际评估脚本应当**显式断言**行数/字段存在。

**建议补齐：**

* 写明：`load_dataset(..., revision=<commit_hash>)` 或至少记录 dataset 的 commit hash / dataset_info。
* 加“启动自检”：检查 split 名称、行数、字段齐全、图片可下载比例。

#### 2.2 Data Loading and Image Download Specification

**可能漏掉：**

* 图片下载可能遇到：缺失文件、网络失败、缓存污染、路径不合法。需要明确：

  * 重试次数
  * 失败样本怎么处理（剔除？算 invalid?）
* 图像解码异常：PIL 打不开、颜色 profile、alpha 通道、巨大图导致 OOM。

**建议补齐：**

* 明确：下载失败/解码失败 → 记为 `invalid_input_image`，并**单独报率**；主指标可以：

  * 方案 A：剔除并报告剔除比例
  * 方案 B：强制该样本预测 NEUTRAL（更保守）
    两者选其一并固定。
* 写明图像处理：RGBA→RGB，保留原始尺寸日志，禁止隐式压缩。

---

### 3. Evaluation Targets and Controls（模型与对照）

#### 3.1 Models

**可能漏掉：**

* “Ours：Saturday release”太口语化：需要**精确到 repo_id + revision/commit/tag**。
* “Base：same architecture”需要写：

  * tokenizer 是否相同
  * prompt 模板是否相同
  * max context 是否相同
    否则对照不公平。

**建议补齐：**

* 固定：`model_id`, `revision`, `dtype`, `device`, `transformers` 版本。
* Base：明确是“未微调 checkpoint”还是“上一个版本”。

#### 3.2 Input Ablations（证明 A2）

**可能漏掉：**

* “image null/empty”怎么实现要写清：

  * VLM 接口里是传 `None`？传全黑图？传空列表？不同实现会影响模型行为。
* F-only：caption 空字符串 vs 不提供 caption 字段也不同。

**建议补齐：**

* 明确定义输入契约：

  * C-only：`images=[]` 或 `image=None`（统一一种），caption 正常提供；
  * F-only：caption=""（空字符串）且明确告知模型“caption not provided”；
  * Claim-only：caption="" 且 image=None，并明确告知“no figure/caption provided”。

---

### 4. Experiment Matrix（实验矩阵）

#### 4.1 Main Task 指标

**可能漏掉：**

* invalid_output 如何进入 Macro-F1：

  * 如果直接丢掉会虚高
  * 如果当作某类会偏置
    需要固定规则。

**建议补齐：**

* 采用“双指标口径”：

  1. **Valid-only Macro-F1**（只在可解析输出上算）
  2. **End-to-end Macro-F1**（把 invalid_output 当作单独错误：等价于预测一个“INVALID”再映射为错误）
     并同时报告 invalid rate。

#### 4.2 Panel Prediction

**可能漏掉：**

* 模型输出 panel 列表时：可能输出 "A"、"panel a"、"Fig 1A"、"left"。需要**严格允许集**与 normalize。

**建议补齐：**

* Panel label 白名单：`{"Panel A","Panel B",...,"Panel Z"}`（按数据集最大值动态生成也行）。
* 超出白名单 → 记为 invalid_panels，并报告其率。

#### 4.4 Epistemic Sensitivity（Flip-Rate）

**可能漏掉：**

* base_claim_id 组里未必齐全：可能缺 support 或缺 contradict，需要写处理。
* Flip-rate 只统计“SUPPORT→CONTRADICT”的理想翻转，那如果模型输出 NEUTRAL 是怎么计入？

**建议补齐：**

* 先定义可用对：必须同时存在 SUPPORT 与 CONTRADICT 变体。
* 报 3 个量：

  * strict flip-rate（只算 SUPPORT→CONTRADICT）
  * partial flip-rate（SUPPORT→{NEUTRAL,CONTRADICT} 视作“敏感但不够强”）
  * failure breakdown（按 3x3 迁移矩阵统计）

#### 4.5 Reproducibility

**可能漏掉：**

* “repeat twice”不够：至少要记录：seed、库版本、GPU 型号、CUDA、是否启用 TF32、flash-attn、并行度。

**建议补齐：**

* 输出一个 `run_metadata.json`：包含环境与关键开关。
* 允许出现不确定性时的处置：

  * 如果 decision 不一致：以第一次为准并标记 `nondeterministic=true`；
  * 或者取多数投票（但这会引入额外计算与偏差，建议只用于诊断，不用于主榜）。

---

### 5. Prompting and Output Schema（输出协议）

**可能漏掉：**

* JSON 解析：需要写明用什么 parser、是否允许 trailing text、是否允许换行。
* “format-fix retry” 的 prompt 文案没给。
* key 顺序：JSON 本身无序，但你要求“in order”，实现上需要决定是否强制。

**建议补齐：**

* 解析规则：

  * 只接受**第一个** JSON object；禁止前后缀文本；
  * 允许换行但必须是合法 JSON；
  * decision 值必须严格匹配三类。
* format-fix retry 模板：固定一句话强约束（我会写进中文版附录）。
* key 顺序：在评估端不要以顺序作为失败条件（否则误伤）；把“顺序要求”仅作为 prompt 约束。

---

### 6. Inference Settings（推理工程）

**可能漏掉：**

* max_new_tokens 不同 prompt_mode 需要不同上限，但也要防止“长输出导致截断 invalid”。
* batching：batch size 变化会影响显存与吞吐，也可能影响某些实现的确定性。

**建议补齐：**

* 固定 batch_size（或写成 sweep 但主结果固定一个）。
* 固定 `max_new_tokens` 并记录是否发生截断（输出到达上限视为风险信号）。
* 明确禁用：sampling、beam search（除非另开实验）。

---

### 7. Metrics and Statistical Methods（统计）

**可能漏掉：**

* bootstrap：需要写 resample 次数（如 1000/5000）、随机种子、以 sample 为单位还是以 base_claim 为单位。
* 多指标比较：如果你同时看 Macro-F1、CONTRADICT-F1、Synergy，最好明确哪些是 primary。

**建议补齐：**

* 主指标：Macro-F1（Full 条件、D 模式）
* 次指标：CONTRADICT-F1、Synergy、Flip-rate、Support bias
* bootstrap：paired、B=5000（或 2000），seed 固定；并报告 CI。

---

### 8. Implementation Plan（落地交付）

**可能漏掉：**

* predictions.jsonl 的字段类型与示例缺失：会导致团队实现不一致。
* error_slices：怎么抽 Top 30？按风险定义还是随机？

**建议补齐：**

* 给一条 jsonl 样例。
* “Top 高风险”定义：例如所有 CONTRADICT→SUPPORT 按模型置信/长度/某些信号排序（如果无置信，就按随机抽样 + 手工挑选），并固定抽样 seed。

---

### 9. Risks（风险）

**可能漏掉：**

* 除了训练泄漏：还可能有**提示词泄漏/基准泄漏**（模型训练时见过 MuSciClaims prompt 模板）。
* 多 panel 图缩放导致信息损失：建议加“人工可读性抽检比例”（比如随机抽 50 张看轴是否可读）。

**建议补齐：**

* 加 prompt 泄漏审计：训练数据是否包含 MuSciClaims 指令/字段名/JSON schema。
* 加抽检：随机抽样 N=50 失败样本，标注是“视觉不可读”还是“推理错误”。

---

## 3) 补齐后的完整中文版本（可直接复制使用）

> 说明：这是把原文**完整翻译成中文**，并把第 2 部分提到的“缺口”**直接补进正文**，形成一份更可执行、可审计、可复现的版本。
> 我会保留标签常量 **SUPPORT/CONTRADICT/NEUTRAL**（因为这是机读协议的一部分），中文里用括号解释其含义。

```text
MuSciClaims 评估方案（补齐版·中文）

0. 文档范围与约束（先写死，避免跑偏）
- 目标：回答一个不可回避的问题：我们的模型是否真的能“看图（figure）+读图注（caption）”来审计科学论断（claim），而不是仅靠文本模式猜测？
- 输入边界：默认只允许使用 (Figure, Caption, Claim) 三者，不允许访问论文正文、不允许外部检索、不允许调用额外知识库。
  - 例外：仅在“工具增强轨（Tool-Augmented Track）”中允许额外的图像阅读工具输出（且必须与模型能力分栏报告，避免归因错误）。
- 输出边界：所有输出必须满足严格 JSON schema（机器可解析、可批量统计、可回放审计）。
- 评估任务：三分类判断 + 诊断维度（跨模态必要性、证据定位、敏感性、可复现性）。

1. 一句话总结
我们要验证：模型是否能够基于“图像证据 + 图注”对 claim 做出三分类判断：
- SUPPORT：证据支持该 claim
- CONTRADICT：证据反驳/矛盾该 claim
- NEUTRAL：证据不相关或不足以判定

MuSciClaims 数据集用三分类 + 多维诊断来刻画这一能力边界。

2. 我们究竟在验证哪些“必要能力”（Necessary Capabilities）？
我们将“科学论断审计代理（scientific vetting agent）”拆分为最小能力公理（axioms）。
每条公理都必须对应可观察、可量化的实验验证；这些验证是不可分割的闭环：删掉任意一个环节，就会导致某条能力无法被证伪/证实。

2.1 能力公理（A1–A5）

A1 证据一致性（Evidence-Consistency）
给定 (Figure, Caption, Claim)，模型的标签必须与证据一致：
- SUPPORT：图/图注支持 claim
- CONTRADICT：图/图注与 claim 矛盾
- NEUTRAL：证据不足或不相关
补齐约束（必须写死）：
- 当图像不可读、关键轴/图例/对比信息缺失、或 caption 无法提供必要上下文时，默认应倾向 NEUTRAL（不允许用常识“补洞”）。

A2 跨模态必要性（Cross-Modal Necessity）
如果仅给 Caption 或仅给 Figure，性能应显著下降；否则说明模型可能没有真正融合视觉与文本，而是在“吃单模态红利”（例如只靠 caption 或只靠 claim 语言模式）。
补齐约束：
- 必须提供输入消融（Full / C-only / F-only / Claim-only），并用 Synergy 指标证明融合。

A3 证据定位（Evidence Localization）
审计不是黑箱：模型应能指出它主要使用了哪些 panel（例如 Panel A/B/C…）。
数据集提供 gold panel 标注，可量化评估定位能力。
补齐约束：
- panel 输出必须可规范化（normalize）到统一集合（如 “Panel A”），并对白名单之外输出计为 invalid_panels 并单独报率。

A4 认知敏感性（Epistemic Sensitivity）
对同一 base claim，支持版本 vs 人工扰动构造的矛盾版本，模型应能翻转判断（SUPPORT → CONTRADICT）。
补齐约束：
- 并非所有 base_claim_id 都一定同时包含 SUPPORT 与 CONTRADICT 变体；必须定义“可用配对”的筛选规则，并报告有效配对数量。

A5 可复现性与可审计性（Reproducibility & Auditability）
相同输入 + 相同设置下，输出应确定（deterministic），且遵循严格 schema 便于回放与统计。
补齐约束：
- 明确确定性等级与环境记录；如果做不到 100% 一致，必须报告不一致率与环境差异（GPU、CUDA、库版本、flash-attn、并行度等）。

为什么 A1–A5 不能删：
如果只做 A1（分类分数），模型可能通过文本投机（text gaming）、SUPPORT 偏置、或根本不读图而拿高分；A2/A3/A4/A5 是防“伪能力”的最小闭包。

3. 数据集与输入契约（Dataset & Input Contract）

3.1 数据集事实与规模（以实际加载为准，并做自检）
- MuSciClaims：共 1,515 个样本，三类均衡（505/类）的描述来自论文/资料；评估脚本必须在运行时做断言（len、字段存在、图片可下载率）。
- Hugging Face 数据集最小必需字段（必须存在，否则直接 fail-fast）：
  - base_claim_id：基础 claim 的组 ID（用于 A4）
  - claim_id：变体 ID（support/contra/neutral）
  - claim_text：claim 文本
  - label_3class：SUPPORT / CONTRADICT / NEUTRAL
  - （可选）label_2class：SUPPORT / NON_SUPPORT
  - associated_figure_filepath：figure 图片路径
  - associated_figure_panels：gold panel 列表（用于 A3）
  - caption：图注
  - paper_id / associated_figure_number 等元数据（用于泄漏审计）

补齐：数据版本锁定（必须）
- 评估必须记录并锁定数据版本：
  - 记录 datasets 版本、HF 数据集 revision/commit hash（或至少输出 dataset_info）
  - 任何后续复跑必须能用同一 revision 复现

3.2 数据加载与图片下载规范（必须固定）

(1) 数据集加载
- 伪代码：
  datasets.load_dataset("StonyBrookNLP/MuSciClaims", split="test", revision=<固定revision>)
- 启动自检（必须执行并写入日志）：
  - 行数、字段齐全
  - label_3class 值域是否严格为三类
  - 图片路径字段非空比例
  - 图片可成功下载/解码比例

(2) 图片下载
- 对每条样本：
  hf_hub_download(
    repo_id="StonyBrookNLP/MuSciClaims",
    filename=row["associated_figure_filepath"],
    repo_type="dataset"
  )
- 补齐：失败处理（必须写死一种口径）
  - 下载失败或解码失败：
    - 标记该样本 invalid_input_image=true
    - 仍保存该样本的预测记录（label_pred 可置为 INVALID 或强制 NEUTRAL，二选一并固定）
    - 在 summary 中报告 invalid_input_image_rate
  - 建议口径：主榜统计采用 “Valid-only + End-to-end 双口径”，避免因为丢样本导致虚高。

(3) 图像预处理（最小化且可复现）
- PIL.Image.open 后统一转换：
  - 若为 RGBA/LA：转为 RGB（丢弃 alpha）
  - 记录原始尺寸 (W,H)、mode、文件大小
- 禁止裁剪（No cropping）
  - 多 panel 图一旦裁剪，会破坏 A2/A3 的有效性
- 如模型有分辨率/patch 限制：
  - 使用保持长宽比的 resize 到 max_side
  - 记录 resize 后尺寸与缩放比例
- 任何压缩/再编码必须禁用或显式记录，否则会引入不可控信息损失

4. 评估对象与对照（Evaluation Targets & Controls）

4.1 必测模型
- Ours：HF 上发布的目标模型
  - 必须精确到：repo_id + revision(tag/commit) + 权重 dtype + 推理框架版本
- Base：对照模型
  - 必须尽量保持：架构、tokenizer、上下文长度、prompt 模板、推理设置一致
  - Base 的定义要写清：未微调版本 / 上一代版本 / 同架构随机初始化（不建议）

为什么必须要 Base：
没有 Base，只能说“模型有分数”，不能证明“训练/适配带来提升”。

4.2 输入消融基线（证明 A2，必须包含）
针对每个模型，至少跑以下输入条件：

(1) Full（F + C）：Figure + Caption + Claim
(2) C-only：Caption + Claim（image 为空）
(3) F-only：Figure + Claim（caption 为空）
(4) Claim-only（强烈建议）：仅 Claim（caption 为空 + image 为空）

补齐：输入“为空”的严格实现契约（必须固定）
- C-only：image=None（或 images=[]，二选一并全程一致），caption=原 caption
- F-only：image=原图，caption=""（空字符串）且 prompt 中明确“caption not provided”
- Claim-only：image=None，caption=""，prompt 中明确“no figure/caption provided”
说明：不同 VLM 实现对 None/空列表/占位图的处理不同，会显著影响结果；必须固定一种实现并记录。

若模型为纯文本模型（不能接图）：
- 主实验仅能跑：C-only 与 Claim-only
- 同时提供 Tool-Augmented Track（见第 6.5），并严格分栏报告（model-only vs model+tool）

5. 实验矩阵（每项映射 A1–A5，不可拆）

5.1 主任务：三分类 Claim Verification（验证 A1）
- 输入：Full / C-only / F-only / Claim-only（按模型能力）
- 输出：decision ∈ {SUPPORT, NEUTRAL, CONTRADICT}
- 指标：
  - Macro-F1（主指标）
  - per-class Precision/Recall/F1
  - confusion matrix（重点观察：CONTRADICT/NEUTRAL 被错判为 SUPPORT）

补齐：invalid_output 的统计口径（必须双口径）
- Valid-only：只在可解析 decision 的样本上算分，并报告 valid_rate
- End-to-end：把 invalid_output 视作错误预测（不丢样本），得到更接近真实系统效果的分数
两者都要给，避免“丢异常样本”导致虚高。

5.2 证据定位：Panel Prediction（验证 A3，并解释 A1 失败）
- 任务：模型输出它使用的 panel 列表，例如 ["Panel A","Panel C"]，或 []
- Gold：associated_figure_panels
- 指标：set-based F1（定义见 8.3）

补齐：panel 规范化与白名单（必须）
- normalize：统一大小写、去空格、统一前缀为 “Panel X”
- 白名单：由数据集中出现的 panel 集合构建（或 A–Z），超出则记 invalid_panels 并单独报率

5.3 跨模态聚合诊断（验证 A2）
基于 5.1 的得分计算：
- Δ_full-caption = F1(Full) - F1(C-only)
- Δ_full-figure   = F1(Full) - F1(F-only)
- Synergy = F1(Full) - max(F1(C-only), F1(F-only))
解释：
- Synergy ≈ 0：模型可能只在使用更强的单模态（没有真正融合）

5.4 认知敏感性（验证 A4）
- 按 base_claim_id 分组
- 理想：support 变体预测 SUPPORT，contradict 变体预测 CONTRADICT
- 核心指标（strict flip-rate）：
  - 对每个 base_claim_id（必须同时存在 SUPPORT 与 CONTRADICT 变体）：
    - 预测 ŷ_s（support）
    - 预测 ŷ_c（contradict）
    - 若 ŷ_s=SUPPORT 且 ŷ_c=CONTRADICT，则计为一次正确翻转
- 同时报：
  - partial flip-rate：ŷ_s=SUPPORT 且 ŷ_c∈{NEUTRAL,CONTRADICT}
  - failure breakdown：3x3 迁移统计（SUPPORT→SUPPORT 等）

5.5 可复现性（验证 A5）
- 固定解码与推理设置（temperature=0 等）
- 同一 run 重复两次：
  - 要求逐样本输出完全一致（decision agreement=100%）
- 若不一致：
  - 报告不一致率
  - 输出 run_metadata.json：GPU/驱动/CUDA/库版本/flash-attn/TF32/并行度/seed 等
  - 并列出不一致样本清单用于诊断（不要静默吞掉）

6. Prompt 与输出协议（Strict JSON，必须机读）

6.1 Schema 1：Decision-only（D）
输出必须是单 key JSON：
{"decision":"SUPPORT"}  （或 NEUTRAL/CONTRADICT）

6.2 Schema 2：Short Reasoning + Decision（R→D）
输出必须是双 key JSON：
{"reasoning":"...(1–2 sentences)...","decision":"SUPPORT"}

6.3 Schema 3：Panels + Reasoning + Decision（用于定位）
输出必须是三 key JSON：
{"figure_panels":["Panel A"],"reasoning":"...","decision":"SUPPORT"}

6.4 解析与错误处理（必须固定）
- 解析规则：
  - 只接受“第一个合法 JSON object”
  - JSON 前后不得有额外文本
  - decision 值必须严格匹配三类之一
- 若第一次 parse 失败：
  - 触发一次 format-fix retry（固定提示词，强约束“只输出 JSON，不要任何多余字符”）
- 若第二次仍失败：
  - 标记 invalid_output=true
  - 仍写入 predictions.jsonl
  - 在 summary 中单独报告 invalid_output_rate

补齐：format-fix retry 固定模板（建议）
- “Your previous output was not valid JSON. Output exactly one JSON object and nothing else. The JSON must start with { and end with }. Use only allowed labels.”

7. 推理设置（Engineering 必须固定）

7.1 解码参数（主评估）
- temperature=0（greedy）
- top_p=1.0
- max_new_tokens：
  - D：128
  - R→D / Panels：256
- 禁用 tool/function calling（除非 7.5 工具增强轨）

补齐：截断检测（必须）
- 若输出达到 max_new_tokens 上限，标记 truncated=true 并单独报率（这通常意味着 schema 风险或 prompt 过长）。

7.2 批处理与吞吐（记录，不要猜）
- 每次推理记录：
  - tokens_in / tokens_out
  - latency_ms
  - （可得则）max_memory_allocated
- 任何 timeout/OOM：
  - 记录并按相同参数重跑
  - 仍失败则标记 invalid_inference=true 并单独报率

7.3 视觉输入（VLM 场景）
- 使用 associated_figure_filepath 的整图（多 panel）
- 若必须 resize：
  - 记录原/新尺寸
  - 保证 Full 与 F-only 使用完全一致的图像预处理流水线

7.4 “投机/语言先验”检测消融（强烈建议）
- Claim-only：只给 claim
目的：
- 若 Claim-only 分数接近 Full，则表明模型未真正使用证据，这是对 A2 的强 sanity check。

7.5 工具增强轨（仅当模型纯文本，或作为扩展）
- 使用独立图像阅读器生成 figure_description（OCR + 图表结构提取/摘要）
- 输入：figure_description + caption + claim
- 输出：同 Schema 1/2
- 报告必须拆分两列：
  - Ours(text-only)
  - Ours(text-only + tool)
严格避免把“工具能力”归因给模型本身。

8. 指标与统计方法

8.1 主指标（A1）
- Macro-F1（3 类）
- per-class Precision/Recall/F1
- confusion matrix（重点错误：CONTRADICT/NEUTRAL → SUPPORT）

8.2 SUPPORT 偏置（A1 + A4）
- Pred(SUPPORT)：预测为 SUPPORT 的比例
- FN_contradict_as_support：CONTRADICT → SUPPORT 比例
- FN_neutral_as_support：NEUTRAL → SUPPORT 比例
目的：量化“倾向判 SUPPORT”的系统性偏差。

8.3 证据定位指标（A3）
设 gold panel 集为 P，预测集为 P_hat：
- Precision = |P ∩ P_hat| / |P_hat|
  - 若 |P_hat|=0，则 Precision=1 当且仅当 |P|=0
- Recall = |P ∩ P_hat| / |P|
  - 若 |P|=0，则 Recall=1
- F1 = 2PR/(P+R)

8.4 跨模态 Synergy（A2）
- Synergy = F1(Full) - max(F1(C-only), F1(F-only))
Synergy 越大，越能证明“真融合”。

8.5 认知敏感性（A4）
- strict flip-rate、partial flip-rate
- 3x3 failure breakdown（SUPPORT→SUPPORT 等）

8.6 显著性检验（避免“看起来提升”）
- 对 Macro-F1、CONTRADICT-F1 等关键差异：
  - 使用 paired bootstrap（对样本重采样）
  - resample 次数 B（建议 2000–5000）固定
  - 输出 95% CI
  - CI 不跨 0 视为显著提升
补齐：主次指标优先级
- Primary：Full + D 的 Macro-F1
- Secondary：CONTRADICT-F1、Synergy、Flip-rate、Support bias、invalid rate

9. 实施计划（到文件级，避免口号）

9.1 目录与产物规范（Evaluation = Audit）
建议结构：
- runs/{model}/{condition}/{prompt_mode}/predictions.jsonl
  每行至少包含：
  - claim_id, base_claim_id
  - label_gold, label_pred
  - panels_gold, panels_pred
  - reasoning（可选）
  - invalid_output, invalid_input_image, truncated
  - latency_ms, tokens_in, tokens_out
  - image_meta（原/新尺寸、缩放比、文件名）
  - run_id / revision 信息
- scores/{model}/summary.csv
  - Macro-F1（valid-only 与 end-to-end）
  - per-class F1
  - Synergy
  - Flip-rate
  - SupportBias
  - invalid rates
- analysis/{model}/error_slices.md
  - Top 30 高风险失败案例（例如 CONTRADICT→SUPPORT）
  - 每类错误给 3–5 个代表样本与改进建议

补齐：predictions.jsonl 示例（建议固定字段名）
{"claim_id":"...","base_claim_id":"...","label_gold":"CONTRADICT","label_pred":"SUPPORT",
 "panels_gold":["Panel B"],"panels_pred":["Panel A"],"invalid_output":false,"invalid_input_image":false,
 "truncated":false,"latency_ms":1234,"tokens_in":512,"tokens_out":45,
 "image_meta":{"orig_w":2400,"orig_h":1800,"new_w":1344,"new_h":1008,"resize_ratio":0.56},
 "model_id":"...","model_revision":"...","dataset_revision":"...","run_id":"2026-02-14T..."}

9.2 运行矩阵（必须一次写清）
对每个模型（Ours/Base）：
主任务（A1/A2）：
- Full + D
- Full + R→D
- C-only + D
- C-only + R→D
- F-only + D（仅 VLM）
- F-only + R→D（仅 VLM）
- Claim-only + D（强烈建议）
- Claim-only + R→D（可选）

定位（A3）：
- Full + Panels+R→D（或 Panels+D）

敏感性（A4）：
- 直接从主任务输出按 base_claim_id 聚合计算（无需额外推理）

可复现性（A5）：
- 至少对 Full+D 与 Full+R→D 各重复两次并检查一致性

10. 风险、有效性与泄漏审计（必须从第一性原理考虑）

10.1 数据泄漏/训练污染（Validity Threat）
- MuSciClaims 来源特定期刊/论文；若训练语料包含相同 paper/figure/caption，评估会被污染。
最小审计动作：
- 用 paper_id 与训练语料索引（若可用）交叉比对
- 若存在重合：
  - 报告“潜在污染”
  - 分别报告 overlap 子集与 overall 的分数

补齐：Prompt/基准泄漏
- 检查训练数据是否出现 MuSciClaims 的字段名、JSON schema、或同款 prompt 模板；
- 若出现，必须在报告中披露（否则 A5 的“可审计性”在对外口径上不成立）。

10.2 多 panel 图信息损失
- resize/压缩会让坐标轴/图例不可读
缓解：
- 记录尺寸与缩放比
- 必要时提高 max resolution
- 增加“人工可读性抽检”：随机抽 N 张失败样本确认是“视觉不可读”还是“推理错误”

10.3 输出格式漂移（Format Drift）
缓解：
- 严格 JSON schema
- 一次 format-fix retry
- invalid rate 单独报告（不能混进主分数里不提）

11. 交付物（Deliverables）

(1) 一页结果摘要（One-page summary）
- Ours vs Base：
  - Macro-F1（Full+D，valid-only 与 end-to-end）
  - CONTRADICT-F1
  - Synergy
  - Flip-rate（strict/partial）
  - SupportBias
  - invalid rates（invalid_output / invalid_input_image / truncated）
- 两个 confusion matrix（Ours/Base）
- 5–10 个“亮点案例”（Base 错、Ours 对）

(2) 可复现评估包
- evaluation scripts
- 环境文件（conda env / pip freeze）
- run configs
- 全量 predictions.jsonl
- scores.csv / summary.csv
- run_metadata.json（硬件/软件/开关）

(3) 失败模式报告（下一轮迭代输入）
- 失败类型：
  - localization failure
  - cross-modal fusion failure
  - visual reading failure（看不清/缩放损失）
  - language-prior gaming
- 每类 3–5 个代表样本 + 修复建议
  - 例如：提高分辨率、panel-aware prompting、对比扰动训练（perturbation-contrast training）

附录：Prompt 模板（严格 JSON，可直接用）

A. Decision-only（D）
（注意：输出 label 必须是英文大写常量，便于机读）
You are an AI model tasked with verifying claims related to visual evidence using zero-shot learning.
Your job is to analyze the given image(s) and its provided caption(s) to decide whether it SUPPORT or CONTRADICT or NEUTRAL the provided claim.
CLAIM: {claim_text}
IMAGE CAPTION(S): {caption}
After completing your analysis, output exactly one JSON object with exactly one key: "decision".
For "decision", output exactly one word — either "SUPPORT" or "CONTRADICT" or "NEUTRAL" (uppercase, no extra text).
The output must start with { and end with }.

B. Reasoning → Decision（R→D，1–2 句）
You are an AI model tasked with verifying claims related to visual evidence using zero-shot learning.
CLAIM: {claim_text}
IMAGE CAPTION(S): {caption}
After completing your analysis, output exactly one JSON object with exactly two keys in this order: "reasoning" and "decision".
- "reasoning": one- or two-sentence evidence-based explanation (refer to axes/legend/trends/panels).
- "decision": one of "SUPPORT" / "CONTRADICT" / "NEUTRAL".
No extra text.

C. Panels + Reasoning + Decision（定位 A3）
You are an AI model tasked with verifying claims related to visual evidence using zero-shot learning.
CLAIM: {claim_text}
IMAGE CAPTION(S): {caption}
Output exactly one JSON object with exactly three keys in this order: "figure_panels", "reasoning", "decision".
- "figure_panels": list ONLY panel labels needed (e.g., ["Panel A","Panel C"]), or [].
- "reasoning": one- or two-sentence explanation grounded in those panels.
- "decision": "SUPPORT" / "CONTRADICT" / "NEUTRAL".
No extra text.

D. Format-fix retry（解析失败时固定使用）
Your previous output was not valid JSON.
Output exactly ONE JSON object and NOTHING ELSE.
The output must start with { and end with }.
Use only the allowed labels: "SUPPORT", "CONTRADICT", "NEUTRAL".
```

---

如果你接下来希望我把“补齐版中文版”进一步变成：

* **可直接执行的 checklist（按工程同学一步步勾选）**，或
* **评估脚本的目录骨架 + jsonl/schema 验证器的伪代码**，
  我也可以继续在这个版本上往下落地（不需要你再重复描述需求）。

