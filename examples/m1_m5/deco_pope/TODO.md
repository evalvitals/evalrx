# TODO — deco_pope 施工清单（在有 GPU 的机器上由 coding agent 执行）

> 设计依据全部在 [`DESIGN.md`](DESIGN.md)，本清单只列"做什么、怎么验收"。
> 原则：**不修改 `evalrx/` 包内任何代码**；所有新逻辑留在本目录。
> 建议先用 `qwen3-vl-2b-instruct` 走通全流程（失败多、显存小），再跑 4b/8b。

## Step 0 — 环境自检

- [ ] GPU 可见（`nvidia-smi`），bf16 显存：2B ≈ 5GB / 4B ≈ 9GB / 8B ≈ 17GB
- [ ] `pip install -e "/path/to/evalrx[local,data]"`，transformers ≥ 4.57
- [ ] `pytest`（包自带 fast tests）通过，确认环境没坏
- [x] 判别模型（M2/M3/M4 的 judge）可用：默认 `ClaudeModel(model="claude-fable-5",
      effort="low")`（agy 配额已耗尽，2026-06-12 切换；测试可
      `--judge-model sonnet|haiku`），容器挂载见 docker-compose.yml

## Step 1 — 完成 mine_cases.py 并挖出冻结清单

- [ ] 实现 `fetch_pope_probes()`：下载 POPE 两个 split（URL 在文件顶部，
      **先核实路径是否仍有效**，并固定 commit hash），解析 JSON-lines
      （`{"question_id","image","text","label"}`，从 "Is there a X" 抽物体名），
      按 DESIGN §4.2 组装同图三元组
- [ ] 核实 greedy 参数透传方式（`do_sample=False, max_new_tokens=8`——查
      `models/backends/hf_local.py` 的 generate kwargs / RuntimeConfig）
- [ ] 核实 tokenizer 取法（`model.tokenizer` 还是 processor.tokenizer）
- [ ] 跑 `python mine_cases.py --model qwen3-vl-2b-instruct --n-images 100`
- [ ] manifest 里补记 transformers/torch 版本（漂移校验用）

**验收**：`data/cases/qwen3-vl-2b-instruct.json` 存在；yields 表里
explore 的 fail ≥ 15。不足→提高 `--n-images`；8B 仍不足→并入 AMBER
（DESIGN §8）。三元组完整率 > 90%（个别图缺 random probe 可丢弃整图）。

## Step 2 — 完成 run.py 的 loop 接线并跑通 M1→M4

- [ ] 按 `examples/m1_m3/qwen_loop_agy/run.py` 的样式补全 judge 构造（agy 或 API）
- [ ] `python run.py --model qwen3-vl-2b-instruct --smoke-test` 先过 schema 检查
- [ ] 真跑：确认漂移校验通过；`outputs/logs/run_log.jsonl` 有 M1/M2/M3/M4 事件；
      M1 tier-(a) 应自动选中 `pope`（cases 带 `pope_label`）等 analyzer

**验收**：loop 正常退出（resolved 或 max_cycles）；`report.final_hypotheses`
非空且至少一条假设谈及 absent-object / co-occurrence / prior 方向
（protocol 一致性检验在 M4 内部完成）。

## Step 3 — 机理探针（预期 tier-(a) 不够用时）

包内 `logit_lens` 不做 final norm 且只读单 case（DESIGN §3.4），逐层压制信号
大概率要靠：

- 优先：给 `ProbeAgent` 配 `WhiteboxProbeGenerator`（tier-(b)），让 loop 自己写探针；
- 否则：在本目录写 `deco_probe.py`，规格照 DESIGN §6 第一条逐条实现
  （final norm 后 unembed、全词表 softmax、token 等价集求和、窗口用深度比例、
  输出 per-case `s_supp / activated_gt / delta_final`，schema 对齐
  `findings["per_case"]` 以便 M2 收割）

**验收**：explore 上能产出每个 case 的 `activated_gt`；
报告 `activated_rate_fail` vs `activated_rate_pass`（**两者差距即本实验主结论**，
差距不显著 = 幻觉非 DeCo 型，照实记录，不算失败）。

## Step 4 — M5 修复（DeCo 单步重打分）

- [ ] 在本目录写 `deco_fix.py`：`deco_rescore_answer(model, inputs, alpha, window)`，
      公式照 DESIGN §6 第二条（候选 top-k20→top-p0.9 的**边界方向对照
      [官方 repo](https://github.com/zjunlp/DeCo) 核实**；二元任务单步即可，
      不要接管 generate）
- [ ] α 与窗口只在 explore 网格（config.yaml 给了网格）；
- [ ] 通过 `loop.run_m5(report, cases)` 的 `verify_fn` 注入，或独立脚本跑
      validate 一次性验证

**验收**（validate，一次性）：
1. FAIL→PASS 翻转率 > PASS→FAIL 回归率，`stats.compare(..., paired=True,
   cluster_by=image_ids, min_effect=config)` 给出 REJECT H0；
2. 翻转集中在 `activated_gt==True` 的 failure（机理一致性）；
3. present(yes) 对照组 accuracy 不降。

## Step 5 — 固化产物

- [ ] `data/cases/*.json` 提交进 git；`outputs/` 不提交（.gitignore）
- [ ] 在本文件末尾追加一节"运行记录"：模型、yields、activated_rate、
      修复前后指标、最终采纳/拒绝的假设
- [ ] 4b / 8b 重复 Step 1–4（每尺寸独立 manifest 与调参）

## 已知坑（提前避雷）

- POPE 原始 JSON 是 **JSON-lines** 不是数组；`image` 字段带 `COCO_val2014_` 前缀
- 同一物体 "Yes/No" 的 token 等价集只保留**单 token** 变体（多 token 变体在
  pos=-1 不可比）
- Qwen3-VL 是 DeepStack：早层（< 0.3N）轨迹不要拿去解读
- 窗口/层号一律从 config 读层数换算，2B 与 4B/8B 层数不同
- 漂移校验失败 ≠ 代码错：先查 transformers 版本是否与 manifest 记录一致
- transformers 4.57 的 processor **不接受 str 图片路径**（`make_list_of_images`
  无 str 分支）——`Inputs.image` 必须传 PIL（mine/run 均已如此）
- PyPI 默认 torch 2.12 是 cu130，driver 550（CUDA 12.4）带不动 → 装
  cu124 专用 wheel（torch 2.6.0，同 Dockerfile）

## 运行记录

### 2026-06-12 挖掘（Step 1 完成）

- 环境：RTX A6000、torch 2.6.0+cu124、transformers 4.57.6、bf16、greedy
  （`do_sample=False, max_new_tokens=8`）、seed 42
- 数据：POPE COCO adversarial/random 全部 **500 图**（三元组完整率 100%），
  commit 固定 `08d957b9`；三元组规则：各类型取文件序第一个可用 probe，
  random-absent 强制 ≠ adversarial-absent
- yields（cases = 1500/尺寸；FAIL 按 probe 类型细分）：

| model | adversarial | present | random | explore F/P | validate F/P |
|---|---|---|---|---|---|
| 2B | **41F**/459P | 59F/441P | 5F/495P | 55/845 | 50/550 |
| 4B | **35F**/465P | 70F/430P | 4F/496P | 61/839 | 48/552 |
| 8B | **50F**/450P | 68F/432P | 5F/495P | 71/829 | 52/548 |

- 全部尺寸 `parse_yes_no` 无法解析的回答 = 0
- **共现先验梯度成立**：同为 absent，adversarial 失败率是 random 的 ~8–10×
  （2B 8.2×、4B 8.8×、8B 10×）——与 DESIGN §4.2 的 H_deco 预期一致；
  present 假 No（12–14%）是并存的另一失败模式（漏检），留给 loop 区分
- explore fail 全部 ≥ 15 验收线（55/61/71）
- judge：agy 配额耗尽，2026-06-12 起默认
  `ClaudeModel(model="claude-fable-5", effort="low")`（探活通过）；
  测试可 `--judge-model sonnet|haiku`

### 2026-06-12 loop（Step 2 完成）

- 2B、max_cycles=2、fable-low judge、--skip-m5；漂移校验 10/10 复现
- 静态 `StrategyProbe` 选了 attention/attention_rollout（非 pope）→ 改为
  `ProbeAgent(judge=...)` LLM 引导选择：选中 logit_lens / prompt_contrast /
  relative_attention，理由直指 DeCo 机理 ✓
- 包内 `logit_lens` 在本环境 cpu/cuda device mismatch 崩溃（DESIGN §3.4
  预言的缺口）；analyzer 子采样 n=32 → M4 全部 inconclusive（诚实），
  4 条假设均切题（late-layer suppression / 先验主导 / 仪器偏差 / 标签疑误）
- 结论：tier-(a) 功效不足，逐层证据需走 tier-(b)（WhiteboxProbeGenerator，
  由 loop 自主生成探针；2026-06-12 起 run.py 已接线）

### 2026-06-12 全自主链路 run#1（2B，opus-4-8-low coder）— 暴露缺陷 7

- 链路修复 1–6 实跑确认生效：缺陷 3（pope 失败→记录→触发自写
  `generated_wb:probe1`）、缺陷 5（fix outcome 带 "5 candidate(s) never
  executed" caveat）、缺陷 6（M4 输出 "No discriminating M2 result" 而非
  误导的 vs p0=0.5 → reject）
- 但 OOM 污染结论：run 自身占 48GB（2B 模型仅 5GB），`pope` 两轮 OOM、
  fix 候选大量 OOM，M1 退回静态选择（fable-low 的 JSON 没解析成功）
- 诊断（隔离复现）：pope 顺序 generate 300 图平稳 4.9GB；白盒注意力捕获
  32 图平稳 4.6GB；**单独都不泄漏**。根因 = `compose` 用了默认
  `device_map="auto"`（accelerate 钩子 + meta tensor）+ ProbeAgent 把白盒
  分析器丢进 8 线程 ThreadPoolExecutor 并发 forward → 钩子非线程安全
  （meta-tensor/dtype 报错）+ 激活堆叠 → 46GB OOM
- 缺陷 7 修复（两处）：
  1. run.py 显式 `RuntimeConfig(device="cuda", dtype="bfloat16")`（对齐
     qwen_loop 例子，去掉 accelerate 自动分发）
  2. ProbeAgent 把**白盒分析器串行执行**，仅黑盒（GENERATE/LOGPROBS）保留
     并行——并发 GPU forward 共享同一模型本就不安全
  - 复现验证：device=cuda + 串行后，pope/attention/attention_rollout 三个
    全部成功，峰值 **4.88GB**（原 46GB）

### 2026-06-12 全自主链路 run#2（2B，opus-4-8-low judge+coder，全修复后）

- judge 改 opus-4-8-low（fable-5 CLI 当前不可用 "currently unavailable"）；
  coder 同；device=cuda；297 图/891 case；零 OOM（grep=0）
- **链路定性上完全成功**——M1 两轮都解析 judge JSON 并选中
  **logit_lens + linear_probe + prompt_contrast**（不再退静态），白盒分析器
  全部跑出（logit_lens n_layers=29 final_norm_applied=1；linear_probe
  best_layer=26≈0.9N）
- **M3 自主重现了参照答案的核心结论**（无人工、未读参照）：
  1. late_layer_suppression：「27–28 层注入 'No'(id 2753)，把中层置信 'Yes'
     峰值~0.94 压到末层 0.51–0.65」= 教科书 DeCo 末层压制
  2. heterogeneous_failure：「失败至少两个群体——早决定无跌落(内容错误)
     vs 末层压制」= 参照答案的漏检/幻觉二分
  3. hallucination：「over-assertion，final_top1≈1.0、late_drop=0.0，稳定
     幻觉而非压制」= 参照答案"假Yes非DeCo型"
  4. 还自我批判 linear_probe 是否混淆（峰值 acc 0.804 < 多数类基线 0.875）
- **但 M4 全部 inconclusive**（logit_lens.decision_layer vs FAIL：
  effect=+0.206 方向对，CI[-0.12,0.59] 跨零）→ m5 None → fix 无 verified
  假设可用。根因 = 白盒分析器子采样 32/64 取 batch 头部，富集批次里只落
  ~4–8 fail，功效不足（**缺陷 8**：子采样需按标签分层）
- fix 阶段：缺陷 4 修复生效——coded pipeline 超时被**明确命名**
  「timed out after 600s, bridge served 4418 model calls」（不再静默/
  unknown tool），触发一轮 repair 仍超时 → recommend L4。根因 =
  `max_validation_cases` 默认 0、run.py 未接线（**缺陷 9**：fix 验证需限流
  + 提高 exec_timeout）
- 结论：**链路工程链路已通**（选器/白盒探针/统计/诚实判定/超时命名/自修
  全部按设计工作），剩两个收尾让其闭环到统计确认 + validated fix：
  缺陷 8（分层子采样）+ 缺陷 9（fix 验证限流接线）

### 2026-06-12 全自主链路 run#3（缺陷 8/9 修复后，闭环）

- 同配置（2B、opus-4-8-low judge+coder、device=cuda、891 case）；零 OOM
- **缺陷 8 生效**：linear_probe 子采样从 8fail/56pass → **32fail/32pass**
  （均衡），M4 首次达到统计显著
- **M4 显著判定**（足量 fail 后 CI 不跨零）：
  - 「失败是低置信案例」→ **REFUTED**（final_top1_prob vs FAIL
    effect=−0.75 CI[−0.94,−0.50] REJECT H0, conf 0.63）= 确认失败是
    **高置信的错误输出**，不是犹豫
  - 「absence 中层可解码但未写入 No logit(unreadout)」→ REFUTED（conf 0.63）
  - hallucination 先验 / no_systematic_defect / DeCo-invisible /
    attention_misplacement → inconclusive
- **缺陷 9 生效**：fix 阶段 coded pipeline 在 60-case 分层子集上执行
  （无 "timed out"、无 "never executed" caveat，对比 run#2 的 4418 调用/
  600s 超时）→ 候选全部执行但无一显著修复 → recommend L4
- **m5 None / verified=0/6 是正确结果**：2B POPE 这个切片的主导失败模式
  是高置信假 Yes 幻觉（非 DeCo 型），DeCo 末层压制只是少数；prompt/解码层
  修复对高置信幻觉无效（参照答案的"修不动幻觉 / no free lunch"）——链路
  诚实地没有强行 support 一个 DeCo 假设、没有谎报一个 validated fix，并
  正确地把唯一剩余手段（L4 微调）作为建议
- **与参照答案对照**：链路自主结论 = 参照（2B 假Yes=非DeCo型、高置信、
  DeCo修不动；DeCo压制为少数）。机理判断方向完全一致

## 链路缺陷修复总览（本 example 的真正交付）

诊断 loop 自身在自主跑 DeCo 场景时暴露并已修复 9 个缺陷（均带回归测试，
全量 767 测试通过）：

| # | 缺陷 | 修复 |
|---|---|---|
| 1 | logit_lens cpu/cuda + 无 final RMSNorm | device 对齐 + `model.final_norm()` + per-case 信号 |
| 2 | linear_probe 未实现 stub | 逐层 logistic + 2-fold CV + 小样本降级 |
| 3 | M1 工具失败不回流、tier-(b) 触发过严 | 失败记录进选择 prompt + 失败即触发自写探针 |
| 4 | fix 沙箱静默跳过未知工具 | 严格契约报错回流 + coder 自修一轮 + 超时命名 |
| 5 | 执行失败误判为档位无效 → 过早升 L4 | 区分 never-executed vs 试过无效 |
| 6 | M4 拿 rate vs p0=0.5 当头条判定 | 描述性工具不当头条 + p0 诚实化 |
| 7 | GPU 46GB OOM | device=cuda（去 accelerate 钩子）+ 白盒分析器串行 |
| 8 | M4 功效不足（子采样取头部多为 PASS） | `CaseBatch.stratified_head` 分层子采样 |
| 9 | fix 验证全批超时 | run.py 接线 max_validation_cases + 提高 timeout |

### 2026-06-12 全自主链路（检测→分析→修复，零人工干预）

- 2B、`--max-clean-images 200`（297 图/891 case，105 FAIL 全保留）、
  judge=fable-low、coder=sonnet、fix 上限 L3b、漂移校验 9/10
- 主循环 174s / judge ~1000 tokens / 修复模块 ~40min
- M1：judge 两轮都选中 logit_lens + linear_probe + prompt_contrast
  （方向正确），但前两者运行即坏，仅 prompt_contrast 存活（n=32 子采样，
  三种提示策略 0.875/0.875/0.875 零差异）
- M3 共 4 条假设（case-intrinsic / 高共现 / describe-first 中介 /
  decoding-deterministic logit margin），M4 全部 inconclusive，0 verified
- 修复：假设#4 含 "logit" 解锁 L3a 路由；coder 产出 L2 三票投票脚手架
  （怀疑提示 + 幻觉警告 + attention 裁剪重问），但**沙箱执行 0 票**——
  生成代码调用了契约未定义的 `model_attend` 与自定义工具名，dispatcher
  全部 "unknown tool ''" 跳过 → 无 FIX_PIPELINE_RESULT_JSON → 候选记为
  无效 → FixOutcome 建议升级 L4（微调）
- 验证机制诚实：无假阳性修复通过 McNemar 把关

**暴露的链路缺陷清单（本实验主产出）**：
1. `logit_lens`：cpu/cuda device mismatch + 不做 final RMSNorm
2. `linear_probe`：未实现 stub（运行即抛自身 docstring）
3. M1 选择器：analyzer 运行时失败不回流 judge（下一轮还在等坏工具的
   证据）；`need_custom` 仅当**全部**候选不可用才触发 → 只要有一个
   analyzer 活着，tier-(b) 自主探针永不激活
4. fix 沙箱：codegen 契约与 dispatcher schema 不匹配（"unknown tool ''"
   静默跳过而非报错回流），coder 也会调用契约外函数
5. FixAgent：把"候选执行失败"与"档位内试过且无效"混为一谈 →
   过早建议 L4
6. M4 默认检验（FAIL 率 vs p0=0.5）对富集批次无意义
