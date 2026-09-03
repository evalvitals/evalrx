# LAMBDA 探索 × M2/M5 确证 集成设计

> 状态：**Phase A–D 已实现**（2026-06-27）。如何把 LAMBDA 式探索层（`evalrx/analysis/`）接入 M1→M5 诊断-修复链路。
> 新增模块：[adjudicate.py](adjudicate.py)（宿主裁决）· [operationalize.py](operationalize.py)（recipe→冻结信号 + in-loop 桥接）· [fused_pipeline.py](fused_pipeline.py)（explore→confirm 编排）；in-loop 注入在 [loop.py](../eval_agent/loop.py) `VLDiagnoseLoop._bridge_signals`。测试 `tests/test_analysis/test_{adjudicate,operationalize,fused_pipeline,bridge_inloop,integration_fused}.py`（74 项全过）。
>
> **延伸（单轮 viz 管线：图表/观察→M3 + dashboard，含 chat 退役）**：本文档接的是**信号→M2 确证**；让 M3 消费/输出 explorer **图表+观察(机制语言)**、把 viz 提成共享核、退役交互 REPL（只留单轮管线）的设计见 [DESIGN_m3_charts.md](DESIGN_m3_charts.md)（设计提案，未实现）。
> 一句话：**LAMBDA 提议、M2 扩族确证、M5（gated on M2）验证、FixAgent 修复——永不用一张图来确证。** 核心新部件是 **operationalization bridge**，把 LAMBDA 发现的复合/切片信号编译成冻结 per-case 信号，使其进入 M2 的 e-BH family。
> 配套：`/tealab-data/jiaqiliu/evalsmith/LAMBDA/LAMBDA_架构与设计原理.md`。

**原则：explore freely, confirm rigorously。**
- 探索侧（EXPLORE split）：LLM 自由写代码/出图/提候选——**只提议，不裁决**。
- 确证侧（CONFIRM split，留出）：一切"是否显著/SUPPORTED/修好"只走已验证引擎（`compare()`→e-value→e-BH），**且在提案没碰过的数据上算**。

---

## 1. 现有链路与三条载荷依赖（已核实，file:line）

```
M1 ProbeAgent      analyzers → probe_results（Result.findings["per_case"] 是关键载体）
  → M2 StatsAnalysisAgent   在「全部 per_case 信号」上跑已验证工具族
                            → e-BH → corrected_rejections.rejected_tools + conclusion/evidence
  → M3 DiagnosisAgent       读 M2 conclusion/evidence + findings → 提可证伪假设
  → M5 HypothesisTester     检验每个假设
  → M4 SurgeryAgent + FixAgent  设计实验、修复；配对 e-value、候选族 e-BH、留出 CONFIRM
```

为什么不能删 M2、为什么 LAMBDA 必须在 M2 **上游**：

1. **M2 的信号族只从 `per_case` 构造**（`build_stats_input`，[stats_tools.py:175](../eval_agent/stages/stats_tools.py)，:202-212）。**一个信号若不在某 analyzer 的 `per_case` 里，永远进不了 M2 family、拿不到 e-value、永不出现在 `rejected_tools`。**
2. **M5 不自洽，gated on M2**（`_decisive`，[hypothesis_tester.py:324-343](../eval_agent/stages/hypothesis_tester.py)）：带 e-value 的结果只有 `tool ∈ rejected_tools` 才算 decisive。
3. **M3 优先读 M2 的 conclusion/evidence**（[diagnosis.py:402-411](../eval_agent/stages/diagnosis.py)）。

→ 推论：若 LAMBDA 只在 M2 **之后**提假设，它发现的任何新信号永远拿不到 e-value，M5 判非 decisive，假设全塌成 INCONCLUSIVE。**必须在 M2 跑之前把信号桥接进它的族。**

**已具备的护栏（漏洞已修）**：充分统计量+宿主重算决策（`_reconstruct_decision`，[stats_tool_generator.py:328](../eval_agent/stages/stats_tool_generator.py)）；候选族 e-BH（`_ebh_survivors`，[fix_agent.py:560](../eval_agent/stages/fix_agent.py)）；留出 CONFIRM split（`_split_explore_confirm`，[loop.py:760](../eval_agent/loop.py)，`(label, probe_type)` 分层、按 id 不相交）。

---

## 2. 目标架构

```
            ┌──────────── EXPLORE split（选/提议）────────────┐
输入 data ─┤  M1 analyzers ──────────────────────────────────┐│
 _split_   │  LAMBDA explorer（自由 EDA / charts /           ││  并集 + 去重
 explore_  │    candidate_signals = 信号配方 + suggested_test ││  （按 estimand）
 confirm   │    ；无 e-value）────────────────────────────────┘│
   │       └──────────────────────────────────────────────────┘
   │            ② OPERATIONALIZATION BRIDGE（核心新组件）
   │               每个候选的「配方」→「冻结 per-case 提取器」→ 落进 inp.per_case
   ▼                                  ▼
CONFIRM split ── ③ M2 = 防火墙（保留，信号族扩大）：validated catalog 跑
                    「catalog 信号 + 桥接来的 LAMBDA 信号」→ e-BH FDR
                    → rejected_tools + conclusion/evidence  【盲于、先于 M3，在 CONFIRM 上】
                                       ▼
                 ④ M3 = 提议者（增强）：读 (a) M2 survivor+evidence（什么真）
                    + (b) LAMBDA observations/charts（机制语言）→ 提可证伪假设
                                       ▼
                 ⑤ M5 = gate（契约不变）：假设 SUPPORTED ⟺ 信号过了 CONFIRM 上的 e-BH
                                       ▼
                 ⑥ M4 surgery + FixAgent：配对 e-value、候选族 e-BH、留出 CONFIRM 验证修复
```

**角色**：谁提议=LAMBDA→M3；谁验证=M5（gated on M2）；图表=只在探索侧；family-FDR 防火墙=M2（独立、提案前、CONFIRM split）。

---

## 3. Operationalization Bridge（核心新部件）

把 LAMBDA 的丰富性接进验证引擎的**唯一通道**：一个假设必须先被编译成**一个冻结、可复现的 per-case 信号函数**（case→value），作为 analyzer-grade finding 落进 `inp.per_case`，才可能被 M2 测。

```python
@dataclass
class SignalRecipe:
    name: str               # 新 per-case 信号 key，如 "explored.small_and_peripheral"
    description: str         # 机制语言（给 M3 读）
    kind: str               # "expr" | "code"
    expr: str = ""          # kind=="expr": 受限 DSL，仅对【已有 per_case 列】做交互/阈值/复合
                            #   例: "(obj_size < 40) and (attention_focus_share < 0.3)"
    code: str = ""          # kind=="code": 沙箱 codegen，对每 record 算一个标量（全新信号）
    suggested_test: str = ""  # 建议路由的具名工具，如 "signal_label_assoc"

def compile_recipe(recipe, records) -> dict[str, float]:
    """配方 → {case_id -> value}，即一个 per_case finding 条目。
    - expr（首选）：在已有 per_case 列上求值，无 codegen、确定性、可审计。
                   交互/阈值/切片本质是把复合谓词归约成一列已测信号 → signal_label_assoc 即可测。
    - code（兜底）：信号尚不是任何已测列时（如"答案全空白"），沙箱跑，复用 _reconstruct_decision
                   的执行/校验路径，只发标量、宿主按内容 hash 冻结、禁网/禁写仓库。
    产物进 Result.findings["per_case"] → build_stats_input 自动收进 inp.per_case。"""
```

**注册进 M2**：编译出的信号塞进合成 analyzer 的 per_case finding → 扩展 `default_plan` 为新信号加一条已验证工具 → 在 CONFIRM split 上跑 `fdr_correct`，把【catalog 信号 + 桥接信号】放进**同一个 e-BH family**（"多提候选→多付多重性"自动成立）→ M5 `_decisive` 不变。

**关键不变量**：信号在 **EXPLORE split** 被发现/选中，在 **CONFIRM split** 被编译求值 + 确证；case id 不相交。

---

## 4. 不变量 / 护栏（不可协商）

| 护栏 | 规则 | 支撑 |
|---|---|---|
| **无 double-dip** | 选在 EXPLORE、每个 e-value 算在冻结 CONFIRM | `_split_explore_confirm`（loop.py:760） |
| **宿主裁决** | LLM 只发充分统计量/标量；reject/e_value/p_value 一律宿主重算 | `_reconstruct_decision`（stats_tool_generator.py:328） |
| **统一 e-BH family** | catalog + 桥接信号进**同一个** `fdr_correct` 族 | `fdr_correct`/`ebh`；**注**：边际 `signal_label_assoc` 是 bootstrap-CI、无 e-value，故按 alpha 出 CI-reject 但**不进 e-BH**,如实标 `fdr_corrected=False`(见 §6 catalog 表达力) |
| **图表无权威** | charts/observations 只进 M3 prompt 与人工引导，绝不进确证/修复门 | explorer schema 无 reject |
| **M2 盲于假设** | M2 在 M3 提假设前、独立跑完整信号族 | 现有 M2→M3 顺序 |
| **e-value 不乱合并** | 撞同一 estimand 先去重，仅**算术平均**（product/sum 非法）；不同 null 各作独立成员进 e-BH | 新增去重步 |

---

## 5. 落地计划（分阶段，file-level）—— ✅ 全部已实现

**Phase A ✅ — 把 chat 每个候选接上已有宿主裁决**
- [explorer.py](explorer.py)：`CandidateSignal`/`ExploratoryAnalysisReport` 加可选 `sufficient`/`recipe`/host 裁决字段（`to_dict` 向后兼容）；`_GENERATE_PROMPT` 允许附 `sufficient`、明确自报 reject/e_value 被忽略；parser 透传。
- [adjudicate.py](adjudicate.py)：`adjudicate_signals/adjudicate_report` 复用 `_reconstruct_decision` + `fdr_correct` 做宿主裁决（`paired_binary`→e-BH；`two_group`→CI、标 `fdr_corrected=False`；无 sufficient→`descriptive_only`）。[chat.py](chat.py) 串联，标 `in_sample`。

**Phase B ✅ — bridge + 双发现源（核心）**
- [operationalize.py](operationalize.py)：`SignalRecipe` + `compile_recipe`（`kind="expr"` 受限 DSL + safe AST eval，拒 import/attr/subscript/lambda；`kind="code"` 暂 NotImplemented）；`compile_recipes`/`per_case_finding`。
- [fused_pipeline.py](fused_pipeline.py)：`run_fused_analysis` —— `_split_records` 分层切分；EXPLORE 跑 explorer + catalog 两发现源、按名去重；CONFIRM 上编译 recipe + `StatsAnalysisAgent.analyze_input` 跑 M2；输出组装 + 无法归约者降级 `recommended_confirmatory_tests`。

**Phase C ✅ — 桥接信号接回 in-loop M2（合成-Result 注入）**
- [operationalize.py](operationalize.py)：`safe_ident`/`per_case_to_records`（转置 + 把 `analyzer.metric` 消歧成 DSL 标识符）/`bridge_recipes_to_result`（recipe 在已有 analyzer 信号上编译 → 合成 `explored` analyzer Result）。
- [loop.py](../eval_agent/loop.py)：`VLDiagnoseLoop` 加 `signal_recipes` 参数 + `_bridge_signals`（M1→M2 之间注入，no-op 默认）。**leak-free by construction**：recipe 必须 out-of-band 预注册（B2 留出集或手写），非窥探本轮 label，故 in-loop 测它等同测预注册 analyzer。

**Phase D ✅ — 测试**：双盲守卫（explorer 只见 explore、verdict 算在不相交 confirm）；宿主裁决（自报 `reject=true`/`e_value=1e9` 被结构性丢弃重算）；**不合并**（不同 estimand → 两个独立 e-BH 成员，从不合并 e-value）；确定性复现（同 seed 同输出）；`expr` 编译/安全；真实 explorer codegen→bridge→M2 端到端。

---

## 6. 风险

- **数据税**：留出 CONFIRM 缩小两边；`_split_explore_confirm` 在 ~4 case 以下 no-op，小 confirm 产更多 underpowered/descriptive，应路由到 `recommended_confirmatory_tests` 而非过度宣称。
- **`expr` 表达力上限**：交互/阈值/复合够用，真正全新连续估计量需 `kind="code"`（沙箱）；先做 expr。
- **estimand 漂移**：chart 与 test 可能指不同的 X；优先单脚本（chart 与 recipe 同段代码、同批 confirm 行）或加 estimand-identity 检查再合并。
- **catalog 表达力**：M2 的 ~6 边际工具是 e-BH 防火墙上限；bridge 的价值正是把交互/切片**归约成 catalog 能测的一列**，扩大防火墙覆盖面。

---

*依据：stats_tools.py / stats_agent.py / stats_tool_generator.py / hypothesis_tester.py / diagnosis.py / surgery.py / fix_agent.py / loop.py / analysis/explorer.py（file:line 见正文）。*
