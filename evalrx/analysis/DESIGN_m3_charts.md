# 单轮可视化管线：LAMBDA 图表/观察 → M3 + Dashboard（含 chat 退役）

> 状态：**已实现（Phase 0+A+B+C+D）**。chat REPL 已退役；`evalrx explore` 单轮入口、`render_chart_specs` 渲染核、`load_run`/`load_loop_story` dashboard、`ExploreContext`→M3、`run_m2-5 --explore-report` 全部落地，附 30+ 测试。经多智能体对抗审查（9 个发现全部修复：CSV 解码/超长字段不再 crash、schema builder 同步、loop story 跨目录定位 Step1 报告、provenance 仅记实际存在的图、`_extract_referenced` 词边界匹配、`--model` 别名对齐等）。全量测试 958 passed。
>
> 本文档定义如何在**只有单轮管线、无交互式 REPL**的前提下，让 explorer 的**图表/观察(机制语言)**进入 in-loop **M3** 并渲染成 **dashboard**，同时把 `chat` 的能力收敛进单轮入口、退役交互 REPL，使**全系统只有一套探索引擎、一套 viz 核**。
> 配套：[DESIGN.md](DESIGN.md)（LAMBDA×M2 信号集成，Phase A–D 已实现）。前者接「信号→M2 确证」；本设计接「机制语言→M3 提假设」+「可视化共享核」。
> 一句话：**砍掉交互 REPL，三条单轮入口（explore / fused / loop）共用一个 `render_chart_specs` + 一个 dashboard loader；图表只进 M3 prompt 与 dashboard，永不进 M2/M5/修复门。**

---

## 1. 目标与原则

**目标**：
1. **单轮 only**：不要交互式多轮 REPL。一次调用 = 一条管线 = 一份产物（可选 dashboard）。
2. **M3 看得到图**：循环里 M3 提假设时消费 explorer 的图表/观察（机制语言），提出更有机制依据的假设。
3. **可视化共享核**：图表渲染 + dashboard 是**一套代码**，被 explore / fused / loop 三入口共用——不再有 chat 的第二套。
4. **chat 退役**：删交互 REPL，其非交互用法由 `evalrx explore`（单轮）顶替。

**核心原则（不变量）**：
- **图表/观察 = 描述性、未认证。** 只进 **M3 的提假设 prompt** 与 **dashboard**；**绝不进 M2(确证)、M5(检验)、Fix(修复门)**（[DESIGN.md](DESIGN.md) 护栏「图表无权威」）。
- **机制语言 informs *哪个* 假设，不 informs *是否* 为真**：chart 启发的假设照走 M5(gated on M2)+M4+e-BH。
- **explorer 不在循环内重跑**：循环消费 Step 1 的 explore 报告（守 split：explorer 只见 EXPLORE 留出集）。
- **图表宿主确定性渲染**：从 chart spec + CSV 渲染，不执行 LLM 写的绘图代码。
- **一引擎一 viz 核**：`ExploratoryAnalysisAgent`（探索）+ `render_chart_specs`（渲染）+ `dashboard.load_run`（加载）三者唯一，三入口共用。

---

## 2. 现状盘点（已核实，file:line）

| 现有件 | 是什么 | 本设计的处置 |
|---|---|---|
| `cli.py:main`（`evalrx-m2-explore`，:13） | **单轮**探索：`ExploratoryAnalysisAgent.explore_path` 跑一次写产物 | **保留+增强** → `evalrx explore`，加 adjudicate+render+`--dashboard` |
| `cli.py:chat_main`（:90）+ `chat.py:M2ChatShell` | **交互 REPL**：`input()` 多轮循环 | **退役删除**（REPL 是 chat 唯一独有；非交互已被 explore 覆盖） |
| `run_fused.py`（examples） | **单轮** explore→bridge→confirm | 保留；加 `--dashboard` |
| `dashboard.py:load_session` | 只读 chat **会话**（`turn_*/exploratory_report.json`） | **泛化** → `load_run`（读单轮产物：explore 输出 / loop run，去掉 turn_* 会话） |
| `DiagnosisAgent.diagnose`（[diagnosis.py:370](../eval_agent/stages/diagnosis.py)） | M3，**无 explorer 入口**；prompt(:447) 无 explore 段 | 加 `explore_context=` 参数 + `{explore_section}` |
| 图表渲染 | explorer 只产 chart **spec**；venv 无 matplotlib → 未渲染 | 新 `render_chart_specs`（宿主确定性渲染） |

**已具备、可复用**：M3 已会把图当 image 传 judge（[diagnosis.py:456](../eval_agent/stages/diagnosis.py)，`if "images" in sig: judge.generate(prompt, images=_figs)`；`ClaudeModel.generate(inputs, images, ...)` 接受 images → **M3 能看见图**）；`write_report_artifacts`/`_verdict_suffix` 在 [chat.py](chat.py)（退役时移入共享单轮模块）；`adjudicate_report` Phase A 宿主裁决已在。

---

## 3. 目标架构

```
共享核（唯一一套）
  ExploratoryAnalysisAgent                         探索引擎（不变）
  viz/renderer.py: render_chart_specs  spec+CSV → PNG，宿主确定性
  analysis/dashboard.py: load_run(dir)    读单轮产物(explore 输出 / loop run) → 渲染

三条单轮入口（都用上面的核）
  ① evalrx explore <results> [--dashboard]      ← 替代 chat（非交互）
       ExploratoryAnalysisAgent.explore_path → adjudicate(有label) → render_chart_specs
       → 写产物 explore_report.json + figures/ → 可选 launch_dashboard
  ② run_fused.py [--dashboard]                       ← 单轮 explore→bridge→confirm
       run_fused_analysis → render_chart_specs → fused_report.json + figures/
  ③ VLDiagnoseLoop(explore_report=ctx)               ← 单轮诊断管线（A+B+C 核心）
       M1(replay) → 桥接信号 → M2(确证，不见图表)
       → M3.diagnose(analysis, explore_context=ctx)  ← 图表/观察进 M3
            prompt += explore_section(标 UNCONFIRMED) + images=[explore PNG]
            → DiagnosisResult.referenced_charts
       → M5 → M4 → Fix(e-BH)                          ← explore_ctx 全程不进这些门
       → run_logger 落 explore 图 + M3 引用 → load_run 渲染 loop dashboard

退役删除
  ✗ chat.py: M2ChatShell（REPL）  ✗ cli.py: chat_main  ✗ dashboard.load_session（turn_* 会话）
```

**角色一句话**：探索=ExploratoryAnalysisAgent(唯一)；渲染=render_chart_specs(宿主)；看图提假设=M3；渲染面板=load_run；谁**绝不**碰图表=M2/M5/Fix；交互 REPL=**没有了**。

---

## 4. 接口设计

### 共享核 A — 渲染 + dashboard loader

```python
# evalrx/viz/renderer.py（新）
def render_chart_specs(charts, tables_dir, out_dir) -> list[dict]:
    """每个 chart spec({name,kind,data(CSV),x,y,title}) 从其 CSV 表确定性渲染 PNG
    (bar/line/scatter)。返回带 figure_path 的 chart dict。matplotlib 缺失 → 跳过渲染、
    回退纯文字描述。宿主侧、确定性、可审计 —— 不执行 LLM 绘图代码。"""

# evalrx/analysis/dashboard.py（改）
def load_run(path) -> dict:
    """统一单轮加载器，自动识别两种产物目录：
      - explore 输出：exploratory_report.json / fused_report.json（+ figures/ tables/）
      - loop run：    logs_*/run_log.jsonl（M2 统计 / M3 假设+引用 / M5 / Fix）+ fused_report.json
    组装成统一"诊断故事"视图供静态 HTML 报告渲染。取代 load_session(turn_* 会话)。"""
```
- env：venv 装 `matplotlib`；`pyproject.toml` 加 `viz=["matplotlib>=3.5"]`（可选 extra，缺则纯文字 explore_section / 报告图表降级）。

### 入口 ① — `evalrx explore`（替代 chat，单轮）

```python
# evalrx/analysis/explore_run.py（新；吸收 chat.py 的 write_report_artifacts/_verdict_suffix）
def run_explore(path, *, question, out, backend, dashboard=False, ...) -> int:
    report = ExploratoryAnalysisAgent(cli_config=...).explore_path(path, question=question)
    adjudicate_report(report, split_label="in_sample")          # Phase A 宿主裁决(有label时)
    report.charts = render_chart_specs(report.charts, out/"tables", out/"figures")  # A
    write_report_artifacts(report, out)
    if dashboard: launch_dashboard(out)
    return 0 if report.ok else 1
# cli.py: 删 chat_main；main 改名/暴露为 `evalrx explore`，加 --dashboard / --backend
```

### B — 喂 M3（核心）

```python
# evalrx/eval_agent/stages/diagnosis.py
@dataclass
class ExploreContext:
    """explorer 的描述性机制语言 —— 永不权威，只读，只进 M3 prompt。"""
    observations: list[str] = field(default_factory=list)
    charts: list[dict] = field(default_factory=list)   # {title, kind, description, figure_path}
    caveats: list[str] = field(default_factory=list)
    source: str = "lambda_explorer"
    @classmethod
    def from_report(cls, data: dict) -> "ExploreContext": ...   # 从 fused_report.json 装

class DiagnosisAgent:
    def diagnose(self, analysis, model_name="", prior_cycles=None,
                 explore_context: "ExploreContext | None" = None) -> DiagnosisResult:
        explore_section = _format_explore_section(explore_context)   # 标 UNCONFIRMED
        prompt = _DIAGNOSE_PROMPT.format(..., explore_section=explore_section)
        _figs = [M2 figures] + [c["figure_path"] for c in (explore_context.charts if explore_context else []) if c.get("figure_path")]
        raw = self.judge.generate(prompt, images=_figs) if "images" in sig else self.judge.generate(prompt)
        result.referenced_charts = _extract_referenced(raw, explore_context)   # C
        return result
```

`_DIAGNOSE_PROMPT` 新增 `{explore_section}`，渲染为强标注块：
```
EXPLORATORY MECHANISM NOTES (free-form EDA on a HELD-OUT explore split — DESCRIPTIVE,
UNCONFIRMED; use ONLY to decide WHICH hypotheses to propose, NEVER as evidence; every
claim must still be tested downstream):
  observations: - ...
  charts (attached as images): - [<title>] <description>
  caveats (explorer's own warnings): - ...
```

透传（不重跑 explorer）：
```python
# loop.py: VLDiagnoseLoop.__init__(..., explore_report: ExploreContext|None=None) → self._explore_context
#          run() 调 M3 处: self.diagnosis_agent.diagnose(stats_report, explore_context=self._explore_context)
# run_m2-5.py: --explore-report fused_report.json → ExploreContext.from_report(json) → VLDiagnoseLoop(explore_report=ctx)
```

### C — M3 出图 + loop dashboard

```python
# DiagnosisResult 加: referenced_charts: list[str]; explore_context_used: bool
# run_logger.log_diagnosis 落 explore 图(路径) + M3 引用 → run 目录
# HTML 报告加 loop-run 视图: explore 图 → M2 forest/统计 → M3 假设(每条标引用的图/观察) → M5 → Fix+e-BH
```

---

## 5. 护栏（不可协商）

| 护栏 | 规则 | 支撑 |
|---|---|---|
| **图表无权威** | explore_context 只进 M3 prompt + dashboard；M2/M5/Fix 永不接收 | 仅是 `DiagnosisAgent` 参数，M2/M5/Fix 签名不含它 |
| **机制语言≠证据** | prompt 强标 UNCONFIRMED；假设照走 M5+M4+e-BH | 现有链路不变 |
| **不重跑 explorer** | 循环消费 Step1 报告；explorer 只见 EXPLORE | `--explore-report` 透传 |
| **宿主渲染** | 图从 spec+CSV 确定性渲染，不执行 LLM 绘图代码 | `render_chart_specs` |
| **一引擎一 viz 核** | 探索/渲染/dashboard 各唯一，三入口共用，无重复 | 退役 chat 第二套 + 共享核 |
| **单轮** | 无交互状态、无多轮会话；一次调用一份产物 | 删 REPL；dashboard 读单 run |

---

## 6. 落地计划（分阶段，file-level）

**Phase 0 — chat 退役 + explore 单轮入口（S/M）**
- 0a 抽 `analysis/explore_run.py`：`run_explore`（吸收 chat.py 的 `write_report_artifacts`/`_verdict_suffix` + adjudicate）。
- 0b `cli.py`：删 `chat_main`；`main` 暴露为 `evalrx explore`，加 `--dashboard`/`--backend`。删 `chat.py:M2ChatShell`（REPL）。
- 0c `dashboard.py`：`load_session` → `load_run`（读单轮产物，去 turn_*）。

**Phase A — 渲染共享核（M）**
- `viz/renderer.py: render_chart_specs`（bar/line/scatter，CSV→PNG，缺 matplotlib 优雅回退）；`run_explore`/`run_fused.py` 调用 + `--dashboard`；pyproject extra。

**Phase B — 喂 M3（M，核心）**
- `ExploreContext` + `_format_explore_section` + `_DIAGNOSE_PROMPT` 加 `{explore_section}`；`diagnose(..., explore_context=)`；figures 合并传 `images=`。
- `VLDiagnoseLoop(explore_report=)` + run 调 M3 透传；`run_m2-5.py --explore-report`。

**Phase C — M3 出图 + loop dashboard（M）**
- `DiagnosisResult.referenced_charts`/`explore_context_used`；`run_logger.log_diagnosis` 落 explore 图+引用；HTML 报告的 loop-run 视图。

**Phase D — 测试**
- 双盲守卫：带 explore_context 的 M3 调用，断言它**从不**进 M2/M5/Fix（签名+运行期）。
- 渲染确定性：spec+CSV→PNG 同输入同图；matplotlib 缺失优雅回退。
- M3 读取：explore_section 进 prompt、figures 进 `images=`、`referenced_charts` 抽取正确。
- chat 退役：`evalrx explore` 覆盖原 chat 非交互用例（单轮 explore_path + 产物 + dashboard）；无 REPL 残留。
- 端到端(轻)：run_fused → fused_report.json → run_m2-5 --explore-report → M3 prompt 含 explore_section（ScriptedJudge，无 GPU）。

---

## 7. 明确不要做

- ❌ **保留/新增交互 REPL** —— 只要单轮。
- ❌ **第二套图表/dashboard 代码** —— 全收敛到 `render_chart_specs` + `load_run`。
- ❌ **把图表/观察当确证** —— 描述性，只进 M3 prompt 与 dashboard。
- ❌ **循环内重跑 explorer** —— 消费 Step1 留出集报告（防 double-dip + 省钱）。
- ❌ **执行 LLM 绘图代码** —— 宿主从 spec+CSV 确定性渲染。
- ❌ **让 explore_context 影响 M5/Fix 门** —— 它只选**哪个**假设。

---

## 8. 开放问题 / 风险 / 迁移

- **chat 退役清单**（删/移）：删 `chat.py:M2ChatShell` + `cli.py:chat_main` + `dashboard.load_session`；移 `write_report_artifacts`/`_verdict_suffix` → `explore_run.py`；留 `ExploratoryAnalysisAgent`/`run_fused_analysis`/`adjudicate_report`。**core 分析环节(M1/M2/M3/M5)零改动。**
- **explore 是否覆盖所有 chat 用例**：chat 唯一独有的是多轮对话；其余(单 question 探索、写产物、dashboard)`evalrx explore` 全覆盖。需确认无脚本依赖 `evalrx chat` 入口名（可留一个 `chat`→`explore` 的弃用别名一个版本周期）。
- **循环信号的图误导 M3**：会启发(如 deco_hallu probe1≈label)，但 M5/M4/e-BH 兜底（实测：语言先验假设被 M4 确证、best-of-N 修复被 e-BH 拒）。把 explorer 自己的循环 caveat 一并呈现给 M3。
- **多模态 judge 图像支持**：`ClaudeModel.generate` 接受 `images=`，但 claude CLI 实际是否消费图像需实测；不支持则 M3 退化为读图的文字描述（仍有用）。
- **依赖**：matplotlib(渲染) 为可选 extra；缺则纯文字 explore_section / 报告无图表但 run 正常。
- **dashboard 单轮 vs 历史**：`load_run` 读单个 run/输出目录；若要跨 run 对比，另起一个 `load_runs(dirs)`（非本设计范围）。

---

*依据：cli.py(main/chat_main)、chat.py(M2ChatShell/write_report_artifacts/_verdict_suffix)、dashboard.py(load_session/launch_dashboard)、diagnosis.py(diagnose/_DIAGNOSE_PROMPT/images 传参)、cli_agent.py(ClaudeModel.generate 签名)、explorer/fused_pipeline.py。本设计是 [DESIGN.md](DESIGN.md) 的延伸（信号→M2 之外，补 机制语言→M3 + 单轮 viz 共享核 + chat 退役）。*
