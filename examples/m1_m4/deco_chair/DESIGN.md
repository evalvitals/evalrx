# deco_chair — 开放式描述下的 DeCo failure/success example（差异部分）

> 共同设计（场景机理、模型适配、统计纪律、protocol/Mode-2 用法、修复公式）见
> [`../deco_pope/DESIGN.md`](../deco_pope/DESIGN.md)；施工清单见本目录
> [`TODO.md`](TODO.md)。本文档只写 captioning 任务的不同之处。
> 这是论文 **Finding 2 的忠实复现**场景。定位同样是**诊断 loop 的输入**
> （cases + protocol + 容器），不向包内新增任何代码。

## 1. 任务与 case 单位

- prompt 固定为论文/OPERA 用的 `"Please help me describe the image in detail."`，
  greedy，`max_new_tokens=512`。
- **喂给 loop 的 case 单位是图像级**（与 `chair` analyzer 的约定对齐：
  `metadata["gt_objects"]` + observed caption）：
  - FAIL = caption 含 ≥1 个不在 GT 物体表中的 COCO 物体（CHAIR 匹配，含同义词表）；
  - PASS = caption 无幻觉 mention。
- **机理分析的最小单位是 mention**（manifest 里逐条记录）：
  - 幻觉 mention（FAIL 单元）vs **同一条 caption** 里被正确提到的在场物体
    （PASS 单元）——同图、同上下文，唯一差异是 mention 是否有视觉根据，
    正是论文 Fig.2 的对比方式（"people/umbrella" 18 层即激活 vs "bird/green"
    30 层才冒头）。统计 `cluster_by=image_id`。

## 2. 数据与冻结清单

- ~50 张 COCO val2014 图，**优先杂物密集的室内场景**（kitchen/office/dining——
  共现先验最强）；挑选标准写进 mine 脚本，不手工钦点。
- `data/gt_objects.json`：每图 GT 物体表，离线从 `instances_val2014.json` 提取
  （250MB 注释文件只进缓存不进库）；同义词表沿用 CHAIR 标准表
  （`data/chair_synonyms.json`，与 `analyzers/hallucination/chair.py` 的 vocab 约定对齐）。
- `mine_cases.py --model {key}`：生成 caption → CHAIR 匹配 → 每个 mention 记录
  `{surface, coco_category, mention_kind: hallucinated|grounded, token_index}`
  （`token_index` = mention 首 token 在全序列中的位置；greedy 下重喂
  `ids[:token_index]` 能精确复现产出该 token 时的分布——这是冻结清单必须
  greedy 的原因）→ 冻结 `data/cases/{model_key}.json`。
- 漂移校验：caption 是长生成，更易漂移——比对"幻觉 mention 集合"而非全文逐字。
- split：图像级 60/40 explore/validate，按是否含幻觉分层。

## 3. 机理探测规格（给 tier-(b) 探针 / GPU 侧 agent，不进包）

与 deco_pope 的两点不同：

1. **位置**：重喂 `prompt + caption[:mention)` 前缀，在 pos=-1 探测；
2. **目标 token 集**：
   - `out_token_ids` = mention 自己的首 token；
   - `gt_token_ids` = 该图在场 GT 物体的首 token 中、落在末层 top-p(0.9)
     候选集内的那些（论文 Eq.2 的前提；过滤方向对照
     [官方 repo](https://github.com/zjunlp/DeCo) 核实）；
   - grounded mention 作对照时角色互换。

预期信号与 deco_pope 同构（`activated_gt` 率、`delta_final`），另加形态断言：
幻觉 mention 的 GT 峰值层集中在 [0.55N, 0.85N]（复现"20–28 层激活"的形状）。

protocol description（M1/M5 锚定，run.py 内置）：模型在详细描述图像时提到
不存在但与场景高共现的物体；怀疑机理同 DeCo——中间层已含正确物体信息、
末层被语言先验覆盖。

## 4. 修复与验证（与 POPE 的主要差别：必须接管生成循环）

- POPE 单步重打分不够用：captioning 需要**逐步** DeCo（手写 greedy 循环 +
  每步重打分，KV cache 复用），实现放本目录 `deco_fix.py`（TODO.md Step 4）。
  超参直接沿用 deco_pope explore 上调好的 (α, 窗口)——同模型同机理，
  这本身是一次跨任务迁移检验。
- validate 图像重新生成 caption，对比：
  - **主效应**：CHAIR_S / CHAIR_I 下降（论文量级：greedy 下 LLaVA 45.0→37.8 /
    14.7→11.1；Qwen3-VL 基线低得多 → min_effect 设小、图像级配对 bootstrap）；
  - **机理一致性**：被消掉的幻觉 mention 应集中在 `activated_gt==True` 子集；
  - **"no free lunch" 护栏**（论文明示）：distinct-2、最长 n-gram 重复率、
    caption 长度、grounded mention 召回（不能把真物体也修没）——α 过大时
    这些先报警。

## 4b. Run record — 2026-06-14 (2B, opus-4-8-low judge/coder, feedback-driven fix)

The loop ran end-to-end as a **Mode-1 container input** (frozen captions +
observation-only protocol + a recall-floored CHAIR `score_fn`), with the fix
module set to **`fix_repair_rounds: 3`** — the new feedback-driven repair. 50
images (28 hallucinated / 22 clean), greedy decoding forced to match the mined
labels, zero OOM.

- **Protocol is now observation-only** — the old description named the mechanism
  (DeCo, intermediate layers, final-layer suppression, language prior); that was
  an answer leak and was removed. It now states only the observable pattern
  (the description names an absent object) plus the recall guard, and the loop
  must find any mechanism itself. The `chair` analyzer is passed via
  `analyzer_overrides` (it needs an object vocabulary, so it cannot
  auto-instantiate).
- **Detect/analyse**: M1 selected `chair` + attention analyzers; M5 verified
  **0/6** — all six hypotheses came back *inconclusive* (clustered-bootstrap on
  50 images has little power), which is honest. The hypotheses were still on
  topic: a language-prior bias toward co-occurring furniture, an
  `<|im_start|>` attention sink, and a CHAIR-gap reading (chair_s 0.28 ≫
  chair_i 0.091 → failures concentrate in a few off-rail captions, i.e. failure
  scales with the *number* of objects volunteered).
- **The no-free-lunch guard is in the scorer**: a candidate caption counts as a
  success only if it names **no absent object AND still ≥1 present object**, so a
  degenerate "stay vague / say nothing" fix loses recall on the 12 clean
  controls in the validation subset and is penalised. (Offline, the synonym
  scorer reproduces all 50/50 frozen hallucination labels.)
- **Feedback-driven repair closed to a VALIDATED, recall-preserving fix in
  round 2** (`repair_rounds=2`, `fixed=True`, `recommendation=None`), validated
  by paired McNemar + e-value on a 24-case stratified subset (12 FAIL / 12 PASS):

  | round | candidate (tier) | fixed / broke | effect | e-value | verdict |
  |---|---|---|---|---|---|
  | 1 | grounded_concise (L2) | 9 / 2 | +0.29 | 3.10 | sub-threshold |
  | 1 | verify_before_naming (L1) | 6 / 0 | +0.25 | — | sub-threshold |
  | 1 | coded_pipeline (L2) | 6 / 2 | +0.17 | — | sub-threshold |
  | **2** | **object_budget_cap (L1)** | **11 / 0** | **+0.458** | **170.7** | ✅ best |
  | 2 | caption_not_detail (L1) | 7 / 0 | +0.29 | — | sub-threshold |
  | 2 | equalize_sharpen_intersection_vote (L2) | 9 / 4 | +0.21 | — | sub-threshold |

  **Round 1 produced no fix that cleared the e-value bar** (the best,
  `grounded_concise`, was directionally right at +0.29 but e=3.10). Those
  per-candidate results — *how many each fixed/broke and its effect* — were fed
  back to the judge, which in **round 2 proposed different strategies** and found
  `object_budget_cap`: "name only objects whose identity is unmistakable; list
  at most THREE; better to omit a real object than include a doubtful one." It
  flips **11 hallucinated captions with 0 clean captions broken** (+0.458,
  e=170.7) — and it is a clean match to the loop's own CHAIR-gap diagnosis
  (cap the number of volunteered objects → kill the off-rail tail without
  touching recall). No tier escalation was needed or used.

This is the feedback-driven repair doing its job: the first round's failure
*informs* the second round instead of the run stopping at "nothing validated,
recommend escalating." It complements deco_miss (validated fix, single round)
and deco_hallu (guarded fix) by exercising the **multi-round** path on the
open-ended captioning slice.

> Note: `fix_repair_rounds` defaults to **1** (single-shot, original behaviour)
> everywhere else; only this example sets it to 3. The diagnosis (M1→M5) was
> already feedback-iterated across cycles; this adds the same idea to repair.

## 4c. Run record — 2026-06-15 (2B, fix tier raised to L3b)

Re-run with `fix_max_tier: L3b` (everything else identical) to let the loop also
reach internals read/write primitives. It closed again — `repair_rounds=3`,
`fixed=True`, `recommendation=None`, **30 candidates** over 3 feedback rounds —
and the result is the scientifically interesting part: **given the internals
levers, the loop tried them and correctly rejected them, then still converged to
the same prompt-cap mechanism as the L2 run.**

- **Internals primitives were exercised and refuted, not chosen**:
  - `visual_embedding_boost` (L3b internals-write): **inert every time** — 0
    fixed / 0 broken / +0.0 on all four attempts. Scaling image-token
    embeddings does nothing for multi-object captioning.
  - `attention_guided_crop` (L3a internals-read): noisy and sometimes
    **harmful** — across attempts it scored 6/5, 4/3, **6/9 (−0.125)**, 4/5
    (−0.042), 5/3; never validated. Cropping to one attention peak drops other
    real objects (recall loss), which the recall-floored scorer penalises.
  - the L3a coded pipeline (now given a bridged `model_attend()`) wrote a
    heavier attention-guided program that **timed out at 1200 s** (62 model
    calls) and triggered the existing execution self-repair round; the rewrite
    ran but still didn't validate (6/4, 3/0, 6/3).
- **Same winner family as L2, found in round 3**: three fixes cleared the
  e-value bar, all "limit what you volunteer" prompts —

  | candidate (tier) | fixed / broke | effect | e-value |
  |---|---|---|---|
  | **single_sentence_hard_cap (L1)** | **10 / 0** | **+0.417** | **93.1** ✅ best |
  | grounded_count_cap (L2) | 10 / 0 | +0.417 | 93.1 ✅ |
  | consensus_intersection_brief (L2) | 9 / 0 | +0.375 | 51.2 ✅ |

  Rounds 1–2 again produced only directionally-good-but-sub-threshold candidates
  (best ~+0.29, e≈3); their results were fed back and **round 3** landed the
  validated caps (e.g. "answer in ONE sentence ≤25 words, name at most three of
  the largest/central objects"). All three winners break **0** clean captions —
  the recall floor held — and they match the loop's own CHAIR-gap diagnosis
  (cap the number of volunteered objects → kill the off-rail hallucinating tail).

**L2 vs L3b takeaway**: raising the tier did not change the *answer* — the fix is
a prompt-level object cap either way — it only let the loop **demonstrate** that
the internals levers (embedding boost, attention crop) are inert or recall-harmful
on this open-ended task instead of us asserting it. Cost is real: L3b added the
primitive trials plus a 1200 s coded-pipeline timeout, and convergence took 3
rounds instead of 2. The honest reading: for captioning hallucination on
Qwen3-VL-2B, the minimum effective tier is **L1/L2 (a volunteering cap)**, and
the internals tier buys nothing here.

## 5. 预算

50 图 × 512 token 生成 ×2（基线+DeCo）+ 每 mention 1 次前缀 forward
（~3-8 mention/图）。8B 单卡几十分钟；stepwise DeCo 论文报 ~1.2× 延迟。

## 6. 文件清单

```
examples/m1_m4/deco_chair/
├── DESIGN.md          本文档
├── TODO.md            GPU 侧施工清单
├── mine_cases.py      caption 生成 + CHAIR 匹配 + mention 定位 → 冻结清单
├── run.py             冻结清单 → CaseBatch + 观察性 Protocol + CHAIR score_fn
│                       → VLDiagnoseLoop M1→M5 → run_m4 + run_fix（3 轮反馈修复）
├── config.yaml
├── Dockerfile / docker-compose.yml
└── data/
    ├── image_list.json        选图清单（含选择标准元数据）
    ├── gt_objects.json        每图 GT 物体（离线提取，提交进 git）
    ├── chair_synonyms.json    CHAIR 同义词表
    └── cases/                 冻结 manifest（每模型尺寸一份）

（运行后由 agent 产生，留在本目录：）
├── deco_probe.py      mention 级前缀重喂探针
└── deco_fix.py        stepwise DeCo 生成
```
