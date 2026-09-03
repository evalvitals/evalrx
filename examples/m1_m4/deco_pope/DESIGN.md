# DeCo 场景 example 设计 — 总纲 + deco_pope（二元判断）

> 论文：*MLLM can see? Dynamic Correction Decoding for Hallucination Mitigation*
> (DeCo, ICLR 2025, [arXiv 2410.11779](https://arxiv.org/abs/2410.11779)，
> [官方代码](https://github.com/zjunlp/DeCo))。
>
> **本 example 的定位**：给诊断 agent 的**输入**——一批同类的 failure + success
> case（冻结清单）、一份 `ExperimentProtocol`、一个可 `docker compose up` 的容器，
> 走 README 的 Mode 1（容器提交）进入 `VLDiagnoseLoop`（M1→M5）；分析与修复由
> loop 内的现有 analyzer / tier-(b) 探针生成器 / M4 SurgeryAgent 完成，
> **不向包内新增任何 analyzer**。剩余实现步骤见 [`TODO.md`](TODO.md)，
> 交给有 GPU 的机器上的 coding agent 执行。
>
> 共同设计在本文档；`../deco_chair/DESIGN.md` 只写差异。

## 1. 我们要喂给诊断 agent 的到底是什么失败

DeCo 的核心发现不是"VLM 会幻觉"，而是一个更精确的机理断言：

- **Finding 1**：对不存在的物体，逐层 probe 分类器仍有 ~80% 准确率——模型其实"看见了"；
- **Finding 2**：在幻觉 token 的生成位置，真实在场物体（GT）的 token 在中间层
  （LLaVA-1.5-7B 的 20–28 层 / 共 32 层）概率高于幻觉 token，**末几层被反超**；
- 去掉图像后 91.05% 的幻觉 token 仍留在候选集 → 压制来自**语言先验**（共现统计）。

因此 case 集的设计目标**不是收集"答错的题"**，而是让 loop 有可能区分两种机理：

| 机理 | 逐层信号 | DeCo 式修复（M4 方向） |
|---|---|---|
| H_deco：看见但被压制 | GT token 中间层激活、末层跌落 | **可翻转** |
| H_blind：真没看见 | GT token 全层无信号 | 无效（甚至有害） |

success case 是同分布、同 prompt、标签正确的对照组——M2/M5 的组间统计
（signal_label_assoc、mcnemar_evalue）必须有 PASS/FAIL 两组才能跑。

## 2. 为什么拆成两个 example

| | `deco_pope`（本目录） | `deco_chair` |
|---|---|---|
| 任务 | "Is there a {obj} in the image? Please answer Yes or No." | "Please help me describe the image in detail." |
| failure 判定 | parse_yes_no(answer) ≠ pope_label，全自动、无歧义 | CHAIR 匹配 caption 中不在 GT 列表的 COCO 物体 |
| 机理探测位置 | 固定：首个生成 token | 需重喂 caption 前缀定位每个物体 mention |
| 角色 | **主力统计载体**：信号干净、便宜 | **忠实复现论文 Finding 2**：开放生成 + within-caption 配对 |

DeCo 论文两个 setting 都测了（POPE F1 / CHAIR_S、CHAIR_I），修复效果两边都能闭环。

## 3. 模型选择与 Qwen3-VL 适配（与论文 LLaVA-1.5 的差异）

模型用注册表现成的 `qwen3-vl-2b/4b/8b-instruct`（specs.py 已有）。差异与对策：

1. **幻觉率低得多** → 必须对抗性预筛（§4），bundle ~300 probe；8B 估计 FP 率
   5–15%；2B 失败更多，适合先验证管线。manifest 记录实际 yield。
2. **层数随尺寸变化**（运行时从 config 读 `num_hidden_layers`，勿硬编码）→
   任何逐层探测的窗口都用**深度比例**表示，默认 [0.55N, 0.80N]
   （论文 [20,28]/32 ≈ [0.62, 0.88]），在 explore split 上做 4 层滑窗扫描定准。
3. **DeepStack**（specs caveat）：视觉残差注入早期文本层 → 早层轨迹与 LLaVA
   不可比，只解释中后段。
4. **final norm**：DeCo 参考实现是 `lm_head(model.norm(h_i))`；包内现有
   `logit_lens` analyzer **不做 final norm**（对 RMSNorm 系模型轨迹会失真）——
   这是 loop 现有 analyzer 覆盖不到的点，预期走 M1 tier-(b)
   （`WhiteboxProbeGenerator` 写临时探针）或由 GPU 侧 agent 在**本目录**写
   `deco_probe.py`（见 TODO.md，不进包）。
5. **答案 token 等价类**：{"Yes"," Yes","yes"," yes"} / {"No"," No","no"," no"}
   概率求和后再比较，避免 tokenizer 变体拆碎信号。

## 4. Case 设计（本 example 的核心交付）

### 4.1 probe 来源（对抗性，固定清单）

直接复用 **POPE COCO adversarial split**（DeCo 自己的 probe 训练集来源）：
absent 物体 = 与图中在场物体共现率最高的缺席物体——"语言先验最强"的构造，
天然指向 H_deco。`mine_cases.py` 离线下载该 JSON + 按 image_id 下载 COCO
val2014 图片（`http://images.cocodataset.org/val2014/COCO_val2014_{id:012d}.jpg`）。

### 4.2 同图三元组（matched triplet）

每张图固定三个 probe，使 failure/success 差异最小化：

| probe | pope_label | 作用 |
|---|---|---|
| adversarial-absent（高共现缺席物体） | no | failure 的主要来源 |
| present（在场物体） | yes | 对照①：模型看得见在场物体 |
| random-absent（低共现缺席物体） | no | 对照②：先验强度梯度——同为 absent，高/低共现失败率之差本身就是"语言先验"假设的直接证据 |

### 4.3 标签与冻结 manifest（"预先固定清单"的实现）

- `mine_cases.py --model {key}` 跑一遍 greedy（`do_sample=False`、固定 seed），
  按 `parse_yes_no(observed) == pope_label` 标 PASS/FAIL，冻结到
  `data/cases/{model_key}.json`。三个尺寸各挖一份——失败集不跨尺寸迁移。
- `run.py` 只重放冻结清单；开头做**漂移校验**（重 generate 抽样比对 observed，
  不一致 warn 并提示重挖）——transformers 升版/精度变化是已知漂移源。
- split：按 **image** 60/40 切 explore/validate（同图三 probe 永不跨 split）；
  所有调参只许碰 explore，validate 留给 M5/预注册检验一次性使用。

### 4.4 case 字段约定

```python
FailureCase(
    inputs=Inputs(prompt=POPE_TMPL.format(obj=obj), image=img_path),
    expected=pope_label,                  # "yes" / "no"
    observed=answer_text,                 # 冻结时的原始回答
    label=Label.FAIL | Label.PASS,
    tags={"hallucination", "deco"},
    metadata={
        "image_id": ..., "object": ..., "probe_type": "adversarial|random|present",
        "pope_label": ...,                # POPEAnalyzer 的约定 key（M1 tier-(a) 直接可用）
        "gt_token_ids": [...],            # 正确答案 token 等价集（供 tier-(b) 探针用）
        "out_token_ids": [...],           # 模型实际答案 token 等价集
        "split": "explore|validate",
    },
)
```

## 5. ExperimentProtocol 与 Mode 2 文本

loop 的行为由 protocol 锚定（M1 选 analyzer、M5 拒绝跑题假设）。本 example 用：

```python
ExperimentProtocol(
    description=(
        "The VLM answers 'Yes' to object-presence questions about objects that are "
        "NOT in the image, specifically objects that frequently co-occur with objects "
        "that ARE present (e.g. sees keyboard+monitor, hallucinates 'mouse'). "
        "Suspected mechanism (DeCo, arXiv 2410.11779): the model recognizes the "
        "object's absence in intermediate layers, but strong language priors "
        "suppress this in the final layers. Failure cases are wrong answers to "
        "adversarial absent-object probes; success cases are correct answers to "
        "the same probe types on the same images."
    ),
    task_domain="object hallucination",
    success_criteria="parse_yes_no(answer) matches the POPE gold label",
    failure_patterns=(
        "wrong 'Yes' concentrated on high-co-occurrence absent objects; "
        "random-absent and present probes mostly correct"
    ),
    target_modalities=frozenset({"text", "image"}),
)
```

同一段 description 也可直接走 README **Mode 2**（agent 写容器）：

```bash
python -m evalrx.eval_agent.nl_runner \
    --description "<上面的 description>" \
    --model qwen3-vl-8b-instruct \
    --out ./my_deco_experiment \
    --provider claude_code --cli-model sonnet
```

本目录相当于把 Mode 2 的产出**预先固化**：cases 来自真实挖掘而非模板占位。

## 6. 机理探测与修复的规格（给 M4 / GPU 侧 agent 的预期，不进包）

这两段是验收"loop 是否找对了原因"的参照答案，实现载体是 loop 自己
（tier-(b) 探针、M4 SurgeryAgent）或本目录内的脚本（见 TODO.md）：

- **探测信号**（每 case 一次 forward，pos=-1，`lm_head(norm(h_i))` 全词表
  softmax 后对 token 等价集求和）：
  - `s_supp = max_{i∈窗口} [p_i(gt) − p_i(out)]`，≥τ（默认 0.05）记 `activated_gt`
    （论文 Eq.2 的 activated GT token）；
  - `delta_final = max_i p_i(gt) − p_N(gt)`：GT 末层跌落量；
  - 预期：FAIL 组 `activated_gt` 率显著高于 PASS 组；adversarial-absent 失败率
    显著高于 random-absent。
  - ⚠️ 设计上**接受被拒**：若信号不成立，结论是"Qwen3-VL 残余幻觉非 DeCo 型"，
    同样是合格归因，不强行复现论文。
- **修复方向**（M4）：DeCo 单步重打分——二元任务只有首 token 重要，无需接管
  generate 循环：候选集 = 末层 top-k(20)→top-p(0.9)；anchor 层 A = 窗口内
  候选概率最大层；`logits' = logits_N + α·max_prob·logits_A`，非候选掩 -inf；
  α∈{0.1..0.6} 只在 explore 上网格。
- **验收闭环**：validate 上 FAIL→PASS 翻转率 vs PASS→FAIL 回归率
  （paired McNemar + e-value，`cluster_by=image_id`）；翻转必须集中在
  `activated_gt==True` 的 failure（机理一致性）；present(yes) 对照组 accuracy
  不降（论文 "no free lunch" 护栏）。

## 7. 运行预算

8B bf16 ≈ 17 GB 显存；mine：300 probe × 1 次 generate(≤8 tokens)；loop：每
analyzer 每 case 1 次 forward。单卡 4090/A100 分钟级。

## 8. 风险与回退

| 风险 | 回退 |
|---|---|
| failure yield 不足（模型太强） | 加 probe 数；并入 AMBER；改用 2B |
| 机理信号不成立（幻觉非 DeCo 型） | 输出占比结论；loop 自然转向其他假设（attention 等） |
| 冻结清单漂移 | run.py 启动校验 + 重挖 |
| tier-(b) 探针写不出 final-norm lens | 按 TODO.md 在本目录手写 deco_probe.py |

## 9. 文件清单

```
examples/m1_m4/deco_pope/
├── DESIGN.md            本文档（设计依据）
├── TODO.md              GPU 侧 coding agent 的施工清单（验收标准在内）
├── mine_cases.py        离线挖掘 → data/cases/{model_key}.json（每尺寸一份）
├── run.py               冻结清单 → CaseBatch + Protocol → VLDiagnoseLoop → run_m4
├── config.yaml          模型 key、窗口比例、α 网格、τ、split 比例
├── Dockerfile / docker-compose.yml   GPU 模式（沿用 qwen_loop_agy 约定）
└── data/
    ├── probes_source.sample.json     probe 清单 schema 示例
    └── cases/                        冻结 manifest（挖掘后生成，提交进 git）

（运行后由 agent 产生，均留在本目录，不进包：）
├── deco_probe.py        final-norm 逐层探针（若 tier-(b) 不能自动生成）
└── deco_fix.py          DeCo 单步重打分（M4 的 verify_fn 载体）
```
