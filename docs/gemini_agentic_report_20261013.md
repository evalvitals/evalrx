# Gemini agentic 修复与实测汇报

日期：2026-10-13。目标模型：Gemini 3.6 Flash；诊断及代码生成：Claude Opus 5/high。

本次完成的是框架自动化修复，以及三个任务上的自动修复搜索与独立确认。ChartQA 的修复通过验证；MMAU 有改善但证据不足；BBH 修复退化并被拒绝。没有修改 Gemini 模型权重。

## 框架如何修复

1. **让 agent 决定诊断下一步。** 默认使用 agentic 模式，由 controller 根据已有证据选择探针、统计分析、数据探索、失败聚类、假设生成及检验；可以追加探针和重复分析。无需用户逐阶段开启功能。保留行动预算和能力约束，避免无限循环。
2. **减少需要用户理解的 flags。** 移除新增的 Gemini logprobs、top-k、诊断重试、M1 codegen、floor candidate 等细分开关。概率支持自动探测，诊断异常自动允许一次重试，代码生成沿用已有配置；历史 fixed 模式保留隐藏兼容入口。
3. **不强塞预设修复候选。** Agentic 模式不注入内置 floor candidates；修复 agent 可以提出已知方法、组合策略或生成代码。框架仍负责候选执行、数据隔离和最终判定。
4. **修复真实运行故障。** LLM runner 曾重复传入 `floor_candidates`，导致构造失败；删除重复传参后完整重跑 BBH。M3 将异常与“正常返回但没有假设”区分开，保留错误及重试记录；MMAU 实测遇到超时并通过重试恢复。
5. **接通探索结果与诊断。** Agentic `explore_data` 原先直接调用 explorer，绕过统一的证据判定、文件保存和 M3 上下文发布。改为复用 `loop._do_explore(..., question=question)`，同时保留 agent 自拟问题。
6. **修正评分与能力记录。** 单选答案解析兼容 `**A**` 等 Markdown 包裹；这属于评分器修复，不作为模型能力改善的证据。Gemini 自动探测实测返回 `Logprobs is not enabled for this model`；本次没有可用 logits/logprobs，也没有模拟这些数据。

## Agent 实际提出了什么修复

以下依据各运行的 `logs/contract/m5_fix.json` 中最终确认候选记录。

### ChartQA：先枚举图表元素，再投票

最终候选为 `enumerate_marks_then_vote`（L2，agent 提出的 spec）。将图像放大 2 倍，让模型先逐行列出问题涉及的柱、点、标签、颜色或数值，再完成计数、比较或运算，最后输出短答案；重复五次，取最常见答案。

这改变的是输入处理和推理流程。确认集从 45/64 提升到 57/64，修好 13 条、破坏 1 条。该结果支持这个组合流程有效，但本次未逐项消融，不能单独归因于放大、枚举或投票中的某一项。

### MMAU：先描述听到的内容，再逐项检查选项

最终候选为 `describe_check_vote`（L2，agent 提出的 spec）。先描述音频内容，再将各选项与听觉证据逐一核对后选择；执行五次并投票。

确认集从 44/64 提升到 50/64，修好 6 条、破坏 0 条。但统计证据未达到预设门槛，因此记为 partial，没有宣称修复成功。

### BBH：多种措辞复核，只有一致时替换原答案

最终候选为 `coded_pipeline`（L2，代码生成）。对同一问题进行四次复核，强调故事原文、避免依赖记忆中的研究结论；其中两次以相反顺序讨论正反理由。只有四次一致给出不同答案时才替换初始回答。

确认集从 68/94 降到 66/94，修好 4 条、破坏 6 条。系统将该候选判为 unsafe 并拒绝接受；自动生成修复不等于修复有效。

## 如何避免把探索收益当成修复成功

Controller 仅在 EXPLORE 上调查。Host 管理修复搜索，在 EXPLORE 中选定并冻结一个候选，然后在未参与选择的 CONFIRM 上验证。Controller 的工具列表不开放最终修复确认，不能看到确认反馈后继续调优同一确认集。

| 数据集 | 全部样本 | EXPLORE / CONFIRM | 搜索候选次数 | 确认前 → 后 | 修好 / 破坏 | e-value，门槛 20 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BBH causal judgement | 187 | 93 / 94 | 13 | 68/94 → 66/94 | 4 / 6 | 0.4433 | 退化，拒绝 |
| ChartQA | 128 | 64 / 64 | 10 | 45/64 → 57/64 | 13 / 1 | 78.0190 | 通过验证 |
| MMAU | 128 | 64 / 64 | 15 | 44/64 → 50/64 | 6 / 0 | 9.1429 | 改善，证据不足 |

三组最终运行均完成 M1–M5 并退出 0；退出成功只代表流程执行完成。离线重评分复现了上述计数，EXPLORE/CONFIRM ID 无交叉，冻结 manifest 和运行源码快照校验通过。探索集大小不代表每个候选都使用全部探索样本。

## 版本与验证边界

| 提交 | 内容 | 验证范围 |
| --- | --- | --- |
| `4bbb60d` | Gemini 能力探测、评分解析、诊断错误处理等前置修复 | 后续实测包含 |
| `013fb43` | 默认 agentic，简化 CLI | ChartQA、MMAU 完整实测源码 |
| `22d4f85` | LLM 重复参数修复及三模态 runner 构造回归 | BBH 完整重跑源码 |
| `7285788` | 探索文件与 M3 上下文发布修复 | 回归测试；未再次全量实测 |
| `7709479` | 发布完整验证结果及版本边界说明 | 上次已推送到 `ruinan` 的汇报基点 |

最终相关回归套件为 **426 passed**。真实运行使用已有冻结样本清单，但默认 agentic 已改变诊断协议，因此不能将其与历史 Gemma/fixed 结果视为严格“只换模型”的受控对比。此次也没有等调用预算的消融对比；五次投票增加了推理成本。修复效果不等于相关诊断机制已获因果证明。

## 证据位置

以下均为相对仓库根目录的本地路径；原始运行文件没有发布到文档站点。
保留规则和清单生成方式见[本地产物索引](local_artifacts.md)。

- 完整机器审计：`examples/benchmark/smoke_checks/agentic-default-20261012/full_chain_audit.json`
- 命令、哈希和运行说明：`examples/benchmark/smoke_checks/agentic-default-20261012/README.md`
- ChartQA 修复记录：`examples/benchmark/smoke_checks/agentic-default-20261012/runs/vlm/gemini-3.6-flash/chartqa/logs/contract/m5_fix.json`
- MMAU 修复记录：`examples/benchmark/smoke_checks/agentic-default-20261012/runs/alm/gemini-3.6-flash/mmau/logs/contract/m5_fix.json`
- BBH 修复记录：`examples/benchmark/smoke_checks/agentic-default-20261012/runs/llm-retry/gemini-3.6-flash/bbh_causal_judgement/logs/contract/m5_fix.json`

本汇报、运行说明和审计脚本作为源码维护；原始数据、JSON、日志及恢复出的工作区在原路径保留并由 Git 忽略。公开汇报的指标和验证边界与原始记录一致。
