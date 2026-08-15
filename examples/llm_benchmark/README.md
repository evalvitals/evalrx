# llm_benchmark — 文本 LLM 的全链路探测与修复

在**实测确认可诊断**的八个数据集上,对 Qwen3.5 **2B / 4B / 9B** 跑完整的
M1 → M2 → M3 → M5 → M4 链路,结果落到 `outputs/<model>/<dataset>/`。

---

## 0. 为什么是这八个

M2 用配对统计对比 PASS 与 FAIL,所以一个数据集**只有在两类都有质量时才可诊断**。
太简单(饱和)则没有 FAIL 可归因,太难(地板)则没有 PASS 做对照。
可用区间是准确率 ∈ **[30%, 70%]**。

这八个是在 **47 个候选切分**上逐个实测筛出来的,不是从榜单上抄的。
筛选过程另见 `evalsmith/DATASETS_qwen35_9b.md`(本目录的 `../../../`)。

> **⚠️ 带位是(模型, 数据集)这个「配对」的属性,不是数据集的属性。**
> 下表全部测于 **Qwen3.5-9B**。同一个切片在 2B/4B 上可能落到地板区。
> `build_cases.py` 会打印实际命中的准确率,并在落到 [0.15, 0.85] 之外时**拒绝写文件**。
> 这不是 bug,是它该做的事。

---

## 1. 八个数据集:来源与划分标准

| 数据集 | 章节 | 题数 | 9B 准确率 | 95% CI | 预算信号 |
|---|---|---|---|---|---|
| `cruxeval_output` | 代码推理 | 800 | 0.700 | [0.56, 0.81] | 6% |
| `bbh_causal_judgement` | 原子推理 | 187 | 0.600 | [0.46, 0.72] | 10% |
| `supergpqa_economics` | 原子推理 | 873 | 0.580 | [0.44, 0.71] | 6% |
| `bbh_tracking7` | 原子推理 | 250 | 0.540 | [0.40, 0.67] | 2% |
| `bamboogle` | 多跳 QA | 125 | 0.520 | [0.39, 0.65] | 8% |
| `minervamath` | 数学推理 | 272 | 0.500 | [0.37, 0.63] | 10% |
| `supergpqa_law` | 原子推理 | 656 | 0.460 | [0.33, 0.60] | 10% |
| `supergpqa_medicine_hard` | 原子推理 | 217 | 0.360 | [0.24, 0.50] | 10% |

`预算信号 = max(截断率, 无答案标签率)`。**超过 10% 时准确率量到的是 token 预算而不是模型能力**,
因为那些题模型根本没走到答案。

机器可读的同一份数据在 [`datasets.py`](datasets.py) 里,它直接从
`../llm_band_probe/band_locate.py` 取 spec,所以两边不会漂移:

```bash
python datasets.py        # 打印目录并校验每个条目都能解析到 spec
```

### 逐个说明

#### `cruxeval_output` — 800 题
- **来源**:`cruxeval-org/cruxeval` · `split=test`
- **出处**:CRUXEval,Gu et al.,**ICML 2024**
- **划分**:整个 test split,只取 **output 方向**(预测函数返回值),不取 input 方向
- **判分**:抽取答案后精确字符串匹配。**不需要沙箱,不执行任何代码**
- **注意**:正好压在带的上边界 0.700,真值再高一点就滑出去变成饱和

#### `bbh_causal_judgement` — 187 题
- **来源**:`lukaemon/bbh` · `config=causal_judgement` · `split=test`
- **出处**:BIG-Bench Hard,Suzgun et al.,**ACL Findings 2023**
- **划分**:BBH 的一个具名子任务,因果归因判断,Yes/No
- **判分**:归一化精确匹配,二元答案空间,无歧义
- **注意**:**边缘通过**。预算信号正好 0.10 压在否决阈值上,CI 上界 0.724 越过 0.70,
  而且它跑在 8k 预算而非 40k。**当作头条数字前请用 n=100 / 24k 复测**

#### `supergpqa_economics` — 873 题
- **来源**:`m-a-p/SuperGPQA` · `split=train` · `where discipline='Economics'`
- **出处**:SuperGPQA,M-A-P,2025(285 个研究生学科)
- **划分**:对数据集**自带的 `discipline` 列**做服务端过滤 —— 是具名子集,不是随机抽样。
  难度构成 65% `middle`
- **判分**:10 选项单选。随机基线约 10%,可用带宽远比 4 选项宽
- **注意**:CI 上界 0.706 压着 0.70 边界

#### `bbh_tracking7` — 250 题
- **来源**:`lukaemon/bbh` · `config=tracking_shuffled_objects_seven_objects`
- **出处**:同 BBH
- **划分**:BBH 的一个具名子任务,跟踪七个物体经过一串交换后的归属
- **判分**:归一化精确匹配
- **注意**:**全表最干净的一次测量**(预算信号 2%)。也是"外推不可信"最直接的证据:
  调研 agent 外推它是 0.93,实测 0.540,**差 39 个点**

#### `bamboogle` — 125 题
- **来源**:`chiayewken/bamboogle` · `split=test`
- **出处**:Press et al.,**EMNLP Findings 2023**(self-ask)
- **划分**:整个 test split。**闭卷**两跳组合问题
- **判分**:短答案归一化精确匹配
- **注意**:题数最小,全量扫一遍最便宜;但 125 题会让任何子组分析的区间都很宽

#### `minervamath` — 272 题
- **来源**:`math-ai/minervamath` · `split=test`
- **出处**:Minerva,Lewkowycz et al.,**NeurIPS 2022**
- **划分**:整个 test split。大学物理/天文定量题,自由作答
- **判分**:`_grade_latex`,数值优先,再比归一化 LaTeX 表面形式。**不是 CAS**。
  科学计数法在程序写法(`4.5e33`)与 LaTeX 写法(`4.5 \times 10^{33}`)之间归一,
  **1% 相对容差只对含指数的答案开放**,普通整数仍要求精确相等
- **注意**:这是 **65k 预算**下的数字。40k 时它读作 0.320 / `budget_limited`,
  而差异主要来自一个 grader bug(23.5% 的金标是程序形式科学计数法),不是预算。
  **⚠️ 影响力证据未确认**:`lm_eval/tasks/minerva_math` 指向的是
  `EleutherAI/hendrycks_math`,**不是**这个数据集

#### `supergpqa_law` — 656 题
- **来源**:`m-a-p/SuperGPQA` · `split=train` · `where discipline='Law'`
- **出处**:同 SuperGPQA
- **划分**:对 `discipline` 列服务端过滤。难度构成 52% `middle` / 39% `easy` / 9% `hard`
- **判分**:10 选项单选
- **注意**:**推荐默认选它**。全表唯一一个 **95% CI 两端都在带内**的切片,
  点估计和区间都无争议

#### `supergpqa_medicine_hard` — 217 题
- **来源**:`m-a-p/SuperGPQA` · `split=train` ·
  `where discipline='Medicine' AND difficulty='hard'`
- **出处**:同 SuperGPQA
- **划分**:两列联合过滤。**唯一一个 <1000 且全为 `hard` 的切片** ——
  其余学科的 hard 档都碎到无法抽样(History 3 题、Education 1 题、Sociology 1 题)
- **判分**:10 选项单选
- **注意**:CI 下界 0.241 跌破 0.30。这里值得用 n=100 —— 217 题里抽 100 已接近半数普查

### 关于"子采样"的口径

三个 SuperGPQA 切片用的是 `Spec.where`(datasets-server 的 `/filter` 端点,**服务端过滤**),
寻址的是数据集 schema 自带的 `discipline` / `difficulty` 列。

**具名子集算数,随机抽 500 题不算。** 前者可被他人精确复现和引用,后者不能。

### ⚠️ `difficulty` 标签跨学科不可比

一条实测得出的教训。测之前的预判是"小学科偏 easy → 大概率饱和",**结果相反**:

```
全集(67% 是 Engineering+Science)   0.520
Economics  (hard 仅 5%)            0.580
Law        (hard 仅 9%)            0.460   ← hard 占比不到全集一半,分数反而更低
Medicine/hard (100% hard)          0.360
```

Law 的 hard 占比不到全集的一半,准确率却更低。**`difficulty` 只在学科内部可比。**
用难度构成去外推带位是错的,只能实测。

---

## 2. 环境准备

```bash
# 依赖(vLLM 环境用来起服务,评测脚本用主环境)
VLLM_PY=/tealab-data/jiaqiliu/venvs/vllm35/bin/python   # vLLM 0.27.1
EVAL_PY=/tealab-data/jiaqiliu/venvs/vllm/bin/python     # 跑本目录脚本

# judge / coder 用的是 claude CLI,不是被测模型
claude --version
```

### 起 vLLM 服务

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID    # 必须!否则 CUDA_VISIBLE_DEVICES 按算力排序
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_FLASHINFER_SAMPLER=0

/tealab-data/jiaqiliu/venvs/vllm35/bin/vllm serve Qwen/Qwen3.5-9B \
  --served-model-name qwen3.5-9b --port 8020 \
  --max-model-len 32768 --max-num-seqs 32 --gpu-memory-utilization 0.92
```

三个尺寸各自的建议参数:

| 模型 | `--served-model-name` | 端口 | `--max-model-len` | 显存 |
|---|---|---|---|---|
| `Qwen/Qwen3.5-2B` | `qwen3.5-2b` | 8020 | 32768 | 单卡 A6000 富余 |
| `Qwen/Qwen3.5-4B` | `qwen3.5-4b` | 8020 | 32768 | 单卡 A6000 富余 |
| `Qwen/Qwen3.5-9B` | `qwen3.5-9b` | 8020 | 32768 | 单卡 A6000 约 40GB |

> **`CUDA_DEVICE_ORDER=PCI_BUS_ID` 不能省。** torch 默认按算力排序,
> 在混合卡机器上 `CUDA_VISIBLE_DEVICES=0` 会解析到 A100 而不是你以为的那张卡,
> 表现为莫名其妙的 OOM。

> **不要用贪婪解码。** `temperature=0` 会让 Qwen thinking 模型陷入逐字重复的
> 自检死循环直到烧完 token 预算。同一道已解出的题:`T=0` 烧满 16,384 token 从不停止,
> `T=0.6/top_p=0.95/top_k=20` 用 1,352 token 就正常结束。`config.yaml` 里已经钉死了这组参数。

---

## 3. 跑全链路

### Stage 0 — 冻结带标签的 CaseBatch(唯一的 GPU 生成步骤)

```bash
cd /tealab-data/jiaqiliu/evalsmith/evalvitals/examples/llm_benchmark

$EVAL_PY build_cases.py --model qwen3.5-9b --dataset supergpqa_law --n 120
```

输出:

```
[build_cases] qwen3.5-9b x supergpqa_law n=120
  accuracy 55/120 = 0.458 (9B reference 0.460)
  truncated 9%   errors 0%   1180s
  wrote outputs/qwen3.5-9b/supergpqa_law/cases.json
```

**为什么单独一步**:M2/M3/M5 可以在冻结的 batch 上反复重跑、改 prompt、调试,
不必重新付生成的钱;而且两次运行的 judge 看到的是**字面相同**的 PASS/FAIL 标签。

它会在两种情况下叫停或警告:

- 准确率落在 **[0.15, 0.85] 之外 → 拒绝写文件**。一类样本太少,M2 无从对比。
  换个数据集,或者你确实知道自己要干什么时加 `--force`
- 截断率 **> 10% → 警告**。此时部分 FAIL 标签是预算产物而非能力信号,
  下游会把"没写完"当成"不会做"去归因。先加 `--max-tokens`

### Stage 1 — M1 → M2 → M3(不确认,先看分析)

```bash
$EVAL_PY run_pipeline.py --model qwen3.5-9b --dataset supergpqa_law --analysis-only

# dashboard 要在 evalvitals/ 下跑(`evalvitals` 包才在 import path 上)
cd /tealab-data/jiaqiliu/evalsmith/evalvitals
$EVAL_PY -m evalvitals.cli dashboard examples/llm_benchmark/outputs/qwen3.5-9b/supergpqa_law
```

产出提出的假设,但**不做 M5 确认、不做修复**。先把分析故事看明白再决定要不要往下走。

### Stage 2 — 全链路 M1 → M5 → M4

```bash
$EVAL_PY run_pipeline.py --model qwen3.5-9b --dataset supergpqa_law
```

各阶段:

| 阶段 | 类 | 做什么 |
|---|---|---|
| **M1** | `ProbeAgent` | 按 protocol 选择并运行 analyzer |
| **M2** | `StatsAnalysisAgent` | 对 M1 的 per-case 信号做统计分析 |
| **M3** | `DiagnosisAgent` | 从统计结果提出假设 |
| **M5** | `HypothesisTester` | 统计检验 + protocol 一致性检查 |
| **M4** | `SurgeryAgent` / `FixAgent` | 对已确认假设提出并验证修复 |

> **M4 跑在循环之外**,且跑在 `confirm_split` 留出的**留出集**上,
> 所以修复是在循环从未挖过的数据上验证的。
> `confirm_split: 0.0` 会让修复在产生假设的同一批数据上打分 —— 那是诊断循环自我恭维的标准做法。
> 默认给的是 **0.3**。

### 三个尺寸都跑

```bash
for M in qwen3.5-2b qwen3.5-4b qwen3.5-9b; do
  # 先把对应尺寸的 vLLM 起起来,再:
  $EVAL_PY build_cases.py  --model $M --dataset supergpqa_law --n 120
  $EVAL_PY run_pipeline.py --model $M --dataset supergpqa_law
done
```

**2B/4B 上预期会有数据集掉出带外。** 这正是要测的东西:哪个失效机制随规模变化。
`build_cases.py` 拒绝写文件时,换一个 9B 上分数更高的数据集
(如 `cruxeval_output` 0.700 或 `bbh_causal_judgement` 0.600)——
它们在小模型上更可能落进带内。

---

## 4. 输出布局

```
outputs/
└── <model>/                       # qwen3.5-2b / -4b / -9b
    └── <dataset>/                 # supergpqa_law / minervamath / ...
        ├── cases.json             # Stage 0:冻结的带标签 batch + 生成统计
        ├── summary.json           # 一行式结果:假设数、确认数、调用数
        └── logs/                  # RunLogger:M1-M5 逐阶段轨迹
            └── run_log.jsonl
```

`cases.json` 顶层字段:

| 字段 | 含义 |
|---|---|
| `accuracy` / `n_pass` / `n_fail` | 实际命中的带位 |
| `truncated_rate` | > 0.10 时下游的归因不可信 |
| `error_rate` | 死请求,与答错分开计;不为 0 说明端点有问题 |
| `reference_9b_accuracy` | 9B 上的实测值,用于对比,**不是预测** |
| `cases[]` | `prompt` / `gold` / `output` / `label` / `finish_reason` |

---

## 5. 常见问题

**vLLM 起不来,报 free memory 不足**
→ 漏了 `export CUDA_DEVICE_ORDER=PCI_BUS_ID`,`CUDA_VISIBLE_DEVICES=0` 解析到了别的卡。

**`build_cases.py` 拒绝写文件**
→ 这个(模型, 数据集)配对不在带内。换数据集,别加 `--force`,除非你明确知道理由。

**截断率很高**
→ 加 `--max-tokens`。但要注意截断有**两种病因**:一种是"推导本身长,空间不够",
加预算能转化成正确率;另一种是"模型不知道答案,在候选之间反复猜",加预算只会让它猜更久。
看 `cases.json` 里被截断样本的 `output` 结尾就能分辨。

**judge 探针返回空**
→ claude CLI 被限流。换 `--judge-model sonnet`。

**M4 报 `no verified hypothesis to fix`**
→ M5 没有确认任何假设。这是合法结果,不是故障。先看 `--analysis-only` 的产出,
确认 M3 提的假设是否本来就站不住。

---

## 6. 相关文件

| 路径 | 作用 |
|---|---|
| [`datasets.py`](datasets.py) | 八个数据集的机器可读目录 |
| [`build_cases.py`](build_cases.py) | Stage 0:生成 + 判分 + 冻结 batch |
| [`run_pipeline.py`](run_pipeline.py) | M1→M2→M3→M5→M4 驱动 |
| [`config.yaml`](config.yaml) | 全部默认参数 |
| `../llm_band_probe/` | 带位定位工具(spec / grader / fetch 的真正来源) |
| `../../../DATASETS_qwen35_9b.md` | 47 个切分的完整筛选记录 |
| `../diagnosis_loops/deco_hallu/` | VLM 版参考实现,阶段划分同构 |

---

## 附:这套东西是怎么被验证的

写这份文档时,下列内容是**逐条核对过的**,不是照着记忆写的:

- 八个条目全部能解析到 `band_locate.SPECS` 里的实际 spec(`python datasets.py` 会校验)
- `run_pipeline.py` 里用到的每一个构造函数关键字参数,都对着实际签名验过
- `band_locate.generate` 在调用时读 `MODEL_ID` / `BASE_URL` 全局,所以覆盖它们是有效的

**尚未验证的**:全链路本身还没在这三个尺寸上端到端跑过一次。
上面的命令是照着已验证的接口和 `deco_hallu` 的参考实现写的,
但第一次跑仍可能碰到运行时问题 —— 建议先用 `--analysis-only` 加一个小 `--n` 试通。
