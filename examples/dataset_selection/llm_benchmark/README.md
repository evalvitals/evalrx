# llm_benchmark — 文本 LLM 的全链路探测与修复

在**实测确认可诊断**的八个数据集上,对 Qwen3.5 **2B / 4B / 9B** 跑完整的
M1 → M2 → M3 → M5 → M4 链路,结果落到 `outputs/<model>/<dataset>/`。

---

## 🔧 在新服务器上从零搭建

> 本节假设你刚 `git pull` 下来,机器上什么都没有。
> **全部路径都是相对仓库的**,没有任何一条钉死在某台机器上。

### 硬件与网络前提

| 项 | 要求 |
|---|---|
| GPU | CUDA 卡。9B 在默认 `--max-model-len 32768` 下需 **约 40GB 显存**;4B 约 20GB;2B 约 12GB |
| 磁盘 | 权重 2B≈5GB / 4B≈9GB / 9B≈**21GB**,加上 outputs |
| 网络 | 必须能访问 **HuggingFace**(`huggingface.co` 取权重 + `datasets-server.huggingface.co` 取题目)。数据是**运行时现取**的,不随仓库分发 |

> 显存不够就调低 `--max-model-len`(改 `run_all.sh` 里的 serve 行),
> 别硬上——OOM 的报错不会告诉你是参数问题。

### 1. 两个 venv

vLLM 会钉死一批版本,和评测侧的依赖冲突,所以**分开装**:

```bash
git clone <你的远端地址> evalsmith
cd evalsmith/evalvitals          # 有 pyproject.toml 的那一层

# (a) 评测环境 —— 跑本目录的脚本
python3 -m venv .venv            # 需要 python >= 3.10
.venv/bin/pip install -U pip
.venv/bin/pip install -e ".[stats,viz,dashboard]"
.venv/bin/pip install requests

# (b) 服务环境 —— 只用来起 vLLM
python3 -m venv .venv-vllm
.venv-vllm/bin/pip install -U pip
.venv-vllm/bin/pip install vllm==0.27.1
```

`run_all.sh` 会**自动找到** `.venv/bin/python` 和 `.venv-vllm/bin/vllm`
(在 `evalvitals/` 或其上一层),不需要你导出任何变量。
装在别处就用环境变量覆盖:

```bash
export EVALVITALS_PYTHON=/your/python
export VLLM_BIN=/your/vllm
```

**extras 各自的用途**:`stats`(statsmodels + scikit-learn)是 **M2 统计的硬依赖**,
缺了整条链走不到 M3;`viz` 是 M2 出图;`dashboard` 只影响 dashboard 命令。

### 2. claude CLI(judge,不可省)

M1 选 analyzer、M2 写统计、M3 提假设、M5 判定 —— **全部**靠它。
它不是被测模型,被测模型是 vLLM 那个。

```bash
# 安装:https://claude.com/claude-code
claude --version          # 必须有输出
claude                    # 首次需要交互登录一次完成认证
```

**默认 judge 与 coder 都是 `claude-opus-5`,effort `high`**(`config.yaml`)。
模型名是**钉死的全名而不是 `opus` 别名** —— 别名会随最新 opus 漂移,
judge 在两次运行之间变了,结果就不可比。

`--effort` 合法值:`low | medium | high | xhigh | max`。
`high` 想得更久也更贵,所以 `codegen_budget_usd`(25.0)和
`codegen_timeout_sec`(1800)是配合它一起调高的 ——
预算或超时不跟着提,M2 的 codegen 会在中途被砍断,
而那个阶段会把**超时报成失败**,看起来像统计做不出来。

`preflight.py` 会用配置里的 model + effort **真发一次请求**,
所以模型名或 effort 写错在上机前就会暴露,不会等到 M1。

### 3. 模型权重

首次 `vllm serve` 会自动下载(9B 约 21GB,视网速十几分钟到一小时)。
想预先下好:

```bash
.venv/bin/pip install huggingface_hub
.venv/bin/huggingface-cli download Qwen/Qwen3.5-9B
# 缓存位置默认 ~/.cache/huggingface,可用 HF_HOME 改到大盘
export HF_HOME=/data/hf-cache
```

### 4. 上机前自检 —— **先跑这个**

```bash
cd evalsmith/evalvitals/examples/dataset_selection/llm_benchmark
../../.venv/bin/python preflight.py --model qwen3.5-9b --dataset supergpqa_law
```

它会逐条检查 python 版本、每个必需模块、claude CLI 能否响应、vllm 在哪、
GPU 空闲显存够不够这个尺寸、磁盘够不够放权重、datasets-server 通不通、
以及数据集名能否解析。**每条 FAIL 都直接给出该敲的安装命令。**

```
  OK    python 3.12.13 (>=3.10)
  FAIL  missing module statsmodels — needed for M2 statistics
        -> pip install -e ".[stats]"
  OK    claude CLI 2.1.181 (Claude Code)
  OK    GPU 0 (NVIDIA RTX A6000) has 48 GB free, qwen3.5-9b needs ~40
  ...
```

退出 0 才往下走。`run_all.sh` 每次也会自动先跑一遍(`SKIP_PREFLIGHT=1` 可跳过)。

> **为什么值得先跑**:整条链最贵的是 Stage 0 的 GPU 生成(20–60 分钟)。
> 如果 claude CLI 没装,要到那之后才暴露,那一小时就白烧了。

---

## ⚡ 给 agent / 自动化执行者:一条命令

**如果你是自动化执行者,只读这一节就够了。** 下面的手动分步说明是给人看的。

```bash
cd <repo>/evalvitals/examples/dataset_selection/llm_benchmark
./run_all.sh qwen3.5-9b supergpqa_law
```

**首次在一台新机器上跑,先看上面的「从零搭建」并跑 `preflight.py`。**

它会依次:挑一张空闲 GPU → 起 vLLM → **等就绪** → Stage 0 生成并冻结 batch →
M1→M2→M3→M5→M4 → **无论成败都关掉 vLLM 释放显存**(EXIT trap)。

```bash
./run_all.sh <model> <dataset> [n_cases]   # n_cases 省略 = 全量

# model:   qwen3.5-2b | qwen3.5-4b | qwen3.5-9b
# dataset: 见第 1 节的八个
# 环境变量:
#   ANALYSIS_ONLY=1   只跑 M1→M2→M3,不做 M5 确认和 M4 修复
#   GPU=3             指定显卡(默认自动挑第一张显存占用 <1GB 的)
#   PORT=8021         换端口(默认 8020)
#   WHITEBOX_PYTHON=  设了才会在主链路之后跑 Stage W(白盒 attention),
#                     必须是 transformers>=5.15 的解释器,见第 4.2 节
#   WHITEBOX_N=24     Stage W 取多少 case(按 PASS/FAIL 均衡)
```

### ⏱ 耗时预期 —— 不要把工具调用超时设短

| 步骤 | 耗时 |
|---|---|
| vLLM 加载权重 | 3–6 分钟(冷 page cache 更久) |
| Stage 0 `build_cases.py` **全量** | **20 分钟 – 6.5 小时**,见下方逐数据集表 |
| M1→M5 | **30–90 分钟**(judge 是 `claude-opus-5 --effort high`,想得久) |
| M4 修复 | **20–60 分钟** |

**总计 1.5–8 小时**,几乎全部取决于切片大小(judge 用 high effort;调低 `judge_effort` 会快很多)。 建议 `setsid nohup ./run_all.sh ... > run.log 2>&1 &` 后台跑再轮询日志,
不要在一次前台工具调用里等它。

### 退出码

| 码 | 含义 | 该怎么办 |
|---|---|---|
| 0 | 成功 | 读 `outputs/<model>/<dataset>/summary.json` |
| 1 | `build_cases.py` **拒绝写文件** | **不是崩溃**。这个(模型, 数据集)配对不在带内。换数据集,见下 |
| 2 | 模型名不认识 | 只能是 `qwen3.5-2b` / `-4b` / `-9b`,**不是** HF repo id |
| 3 | 没有空闲 GPU | 等,或 `GPU=<idx>` 指定 |
| 4 / 5 | vLLM 启动失败 / 超时 | 看 `outputs/<model>/<dataset>/vllm.log` |
| 6 | 找不到 python 或 vllm | 环境没装好。见「从零搭建」,或 `export EVALVITALS_PYTHON=` / `VLLM_BIN=` |
| 7 | **preflight 未通过** | 照它每条 FAIL 后面给的命令装。`SKIP_PREFLIGHT=1` 可强行跳过(不建议) |

### 退出码 1 时怎么换数据集

`build_cases.py` 在准确率落到 [0.15, 0.85] 之外时**故意拒绝写文件**——
一类样本太少,M2 无从对比,继续跑只会产出无意义的归因。

带位是**(模型, 数据集)配对**的属性。下表按 9B 分数排序;
**模型越小,越该往表的上方选**:

```
cruxeval_output          0.700   ← 2B/4B 优先从这里试
bbh_causal_judgement     0.600
supergpqa_economics      0.580
bbh_tracking7            0.540
bamboogle                0.520
minervamath              0.500
supergpqa_law            0.460
supergpqa_medicine_hard  0.360   ← 9B 上就已经偏难,小模型大概率地板
```

**不要用 `--force` 硬闯**,除非你明确知道为什么要一个带外的 batch。

### 前置条件(脚本不会替你装)

```bash
./preflight.py        # 或 <python> preflight.py —— 一次查完所有前置条件
```

它覆盖:python 版本、必需模块、claude CLI、vllm、GPU 显存、磁盘、网络、数据集解析。

按上面「从零搭建」装好后,`evalvitals` 是 **`pip install -e` 进 venv 的**,
任何 cwd 都能 import。若你跳过了安装、直接用系统 python 跑,
本目录的脚本仍能工作(它们自己插 `sys.path`),但 `python -m evalvitals.cli dashboard`
必须先 `cd` 到 `evalvitals/`。

---

## 0. 为什么是这八个

M2 用配对统计对比 PASS 与 FAIL,所以一个数据集**只有在两类都有质量时才可诊断**。
太简单(饱和)则没有 FAIL 可归因,太难(地板)则没有 PASS 做对照。
可用区间是准确率 ∈ **[30%, 70%]**。

这八个是在 47 个候选切分上逐个实测筛出来的,不是从榜单上抄的。
**本文档自足**:下面第 1 节包含跑实验所需的全部信息,不依赖任何外部文件。

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
$PY datasets.py               # 目录 + 校验每个条目都能解析到 spec
$PY datasets.py --acquisition # 每个切片的 dataset/config/split/where
$PY datasets.py --probe       # 实时请求 datasets-server 核对题数
```

### 数据怎么拿到:运行时现取,不需要手工下载

**题目不随仓库分发**,全部在运行时从 HuggingFace datasets-server 取。
下面这四个字段就是数据集的完整身份 —— `band_locate.py` 拿它们去请求
`/rows`(无 `where`)或 `/filter`(有 `where`):

| 数据集 | HF dataset | config | split | where |
|---|---|---|---|---|
| `cruxeval_output` | `cruxeval-org/cruxeval` | `default` | `test` | — |
| `bbh_causal_judgement` | `lukaemon/bbh` | `causal_judgement` | `test` | — |
| `bbh_tracking7` | `lukaemon/bbh` | `tracking_shuffled_objects_seven_objects` | `test` | — |
| `bamboogle` | `chiayewken/bamboogle` | `default` | `test` | — |
| `minervamath` | `math-ai/minervamath` | `default` | `test` | — |
| `supergpqa_economics` | `m-a-p/SuperGPQA` | `default` | `train` | `"discipline"='Economics'` |
| `supergpqa_law` | `m-a-p/SuperGPQA` | `default` | `train` | `"discipline"='Law'` |
| `supergpqa_medicine_hard` | `m-a-p/SuperGPQA` | `default` | `train` | `"discipline"='Medicine' AND "difficulty"='hard'` |

`where` 是**服务端过滤**,过滤的是数据集自己的列。这正是那三个 SuperGPQA
条目算"可引用的具名切分"而不是"私有抽样"的原因 —— 别人照这个字符串能拿到一模一样的题。

**自己核一遍**(不需要 GPU,约 20 秒):

```bash
$PY datasets.py --acquisition   # 打印上面这张表(从代码里读,不会和文档漂移)
$PY datasets.py --probe         # 真的去请求一次,核对每个切片的题数
```

`--probe` 的输出:

```
dataset                    rows in slice  status
--------------------------------------------------------------------------
cruxeval_output                      800  OK
bbh_causal_judgement                 187  OK
supergpqa_economics                  873  OK
bbh_tracking7                        250  OK
bamboogle                            125  OK
minervamath                          272  OK
supergpqa_law                        656  OK
supergpqa_medicine_hard              217  OK
```

任何一行显示 `SIZE CHANGED` 就说明上游数据集变了,
**这里记录的带位不再描述你实际会拿到的那批题** —— 别直接沿用,重新定位带位。
`--probe` 有非零退出码时不要继续。

**抽样方式**:`band_locate.fetch_rows` 不是取前 N 条,而是在切片上铺 12 个窗口做
**分层整群抽样**(很多 split 按子集或难度排序,取头部会测到一个子集却声称测了全集)。
代价是 Wilson 区间只是近似 —— 把边界当参考,不要当检验。

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

### n_cases 默认全量:批次就是切片本身

`n_cases: 0`(默认)= **取切片里的每一条**。

设上限是没意义的:切片大小本来就是硬天花板(对 125 题的 bamboogle 要 240 只会拿到 125),
所以限制只会砍掉那些**本来撑得起更多**的数据集,对撑不起的毫无帮助。

全量的好处是**批次里不再有抽样波动**——PASS/FAIL 的划分**就是**这个切片,
而不是从中抽的一把。同一个模型跑两次看到的是同一批题,可直接比较。
区间的含义随之变化:它说的是"推广到这类任务",而不是"抽到哪些题"。

已实测两个最大切片能 100% 取回(`supergpqa_economics` 873/873、`cruxeval_output` 800/800,
0 个失败窗口)。

#### ⏱ 代价:逐数据集的 Stage 0 耗时

按实测吞吐(并发 16)推算,**单个模型尺寸**:

| 数据集 | 题数 | 秒/题 | Stage 0 全量耗时 |
|---|---|---|---|
| `supergpqa_economics` | 873 | 27.1 | **6h33m** |
| `minervamath` | 272 | 72.3 | **5h27m** |
| `supergpqa_law` | 656 | 24.7 | **4h30m** |
| `cruxeval_output` | 800 | 15.4 | **3h25m** |
| `supergpqa_medicine_hard` | 217 | 32.6 | 1h57m |
| `bamboogle` | 125 | 36.6 | 1h16m |
| `bbh_causal_judgement` | 187 | 12.7 | 0h39m |
| `bbh_tracking7` | 250 | 4.7 | **0h19m** |
| **八个合计** | 3,380 | | **约 24 小时** |

三个尺寸全跑完 = **约 3 天 GPU**。

> **一次跑一个数据集,不要无脑排完八个。**
> 想先看链路通不通,用 `bbh_tracking7`(19 分钟)或 `bbh_causal_judgement`(39 分钟),
> 它们题数不小但推理链短。
>
> `minervamath` 题数只有 272 却要 5.5 小时 —— 它是自由作答的物理题,
> 单题推理链最长(72 秒/题,是 `bbh_tracking7` 的 15 倍)。**题数不代表耗时。**

需要更快时显式传 n 封顶:

```bash
./run_all.sh qwen3.5-9b supergpqa_economics 200    # 抽 200 条而不是全部 873
```

#### 全量 + 1:1 之后每边还剩多少

```bash
$PY datasets.py --plan          # 读 config.yaml 的当前设置
```

| 数据集 | 题数 | 每半 | PASS/FAIL | 较薄一侧 |
|---|---|---|---|---|
| `supergpqa_economics` | 873 | 436 | 253/183 | **183** |
| `supergpqa_law` | 656 | 328 | 151/177 | 151 |
| `cruxeval_output` | 800 | 400 | 280/120 | 120 |
| `minervamath` | 272 | 136 | 68/68 | 68 |
| `bbh_tracking7` | 250 | 125 | 68/57 | 57 |
| `supergpqa_medicine_hard` | 217 | 108 | 39/69 | 39 |
| `bbh_causal_judgement` | 187 | 93 | 56/37 | 37 |
| `bamboogle` | 125 | 62 | 32/30 | 30 |

决定 M2 能否下结论的是**较薄的那一类**,配对检验受制于薄的那一侧。
注意 `cruxeval_output` 有 800 题,较薄一侧却只有 120 —— 准确率 0.70 让 FAIL 稀少,
**切片大不等于功效高**。

PASS/FAIL 比例用的是 9B 准确率;换 2B/4B 会移动,届时重跑 `--plan`。

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

## 2. 手动分步(不用 run_all.sh 时)

`run_all.sh` 已经把下面这些串起来了。只有需要单独调试某一步时才手动跑。
以下用 `$PY` 代表你的评测解释器(`<repo>/evalvitals/.venv/bin/python`),
`$VLLM` 代表 `<repo>/evalvitals/.venv-vllm/bin/vllm`
—— **每次新开 shell 都要重新设**,否则会静默变成空串。

### 起 vLLM 服务

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID    # 必须!见下
export CUDA_VISIBLE_DEVICES=0          # 换成一张空闲卡
export VLLM_USE_FLASHINFER_SAMPLER=0

$VLLM serve Qwen/Qwen3.5-9B \
  --served-model-name qwen3.5-9b --port 8020 \
  --max-model-len 32768 --max-num-seqs 32 --gpu-memory-utilization 0.92
```

它**前台阻塞**,而且加载权重要 3–6 分钟。手动跑时另开一个终端,
并等 `curl -sf http://127.0.0.1:8020/v1/models` 有返回再往下走。

| 模型 | `--served-model-name` | 需要显存 |
|---|---|---|
| `Qwen/Qwen3.5-2B` | `qwen3.5-2b` | ~12 GB |
| `Qwen/Qwen3.5-4B` | `qwen3.5-4b` | ~20 GB |
| `Qwen/Qwen3.5-9B` | `qwen3.5-9b` | ~40 GB |

> **`--served-model-name` 是后面 `--model` 要传的值**,不是 HF repo id。

> **`CUDA_DEVICE_ORDER=PCI_BUS_ID` 不能省。** torch 默认按算力排序,
> 在混合卡机器上 `CUDA_VISIBLE_DEVICES=0` 会解析到你没预期的那张卡,
> 表现为莫名其妙的显存不足。

> **不要用贪婪解码。** `temperature=0` 会让 Qwen thinking 模型陷入逐字重复的
> 自检死循环直到烧完 token 预算。同一道已解出的题:`T=0` 烧满 16,384 token 从不停止,
> `T=0.6/top_p=0.95/top_k=20` 用 1,352 token 就正常结束。`config.yaml` 已钉死这组参数。

---

## 3. 跑全链路

### Stage 0 — 冻结带标签的 CaseBatch(唯一的 GPU 生成步骤)

```bash
cd <repo>/evalvitals/examples/dataset_selection/llm_benchmark

$PY build_cases.py --model qwen3.5-9b --dataset supergpqa_law
```

输出:

```
[build_cases] qwen3.5-9b x supergpqa_law n=ALL
  accuracy 301/656 = 0.459 (CENSUS of the slice) (9B reference 0.460)
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
$PY run_pipeline.py --model qwen3.5-9b --dataset supergpqa_law --analysis-only

# dashboard 要在 evalvitals/ 下跑(`evalvitals` 包才在 import path 上)
cd <repo>/evalvitals
$PY -m evalvitals.cli dashboard examples/dataset_selection/llm_benchmark/outputs/qwen3.5-9b/supergpqa_law
```

产出提出的假设,但**不做 M5 确认、不做修复**。先把分析故事看明白再决定要不要往下走。

### Stage 2 — 全链路 M1 → M5 → M4

```bash
$PY run_pipeline.py --model qwen3.5-9b --dataset supergpqa_law
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
>
> **默认 `confirm_split: 0.5`,即 1:1** —— 循环挖一半,修复在它从没见过的另一半上打分。
> 划分是**确定性的**(按 label + probe_type 分层、固定 seed),所以 `run()` 和
> `run_m4()` 从同一批输入推出完全相同的划分。
>
> ⚠️ **代价是两边的统计功效都减半**,这正是 `n_cases` 默认取全量的原因。
> 每个数据集实际剩多少,用 `$PY datasets.py --plan` 算,别估。

### 三个尺寸都跑

`run_all.sh` 每次都自己起停 vLLM,所以串行跑三个尺寸不会撞车:

```bash
cd <repo>/evalvitals/examples/dataset_selection/llm_benchmark
for M in qwen3.5-2b qwen3.5-4b qwen3.5-9b; do
  ./run_all.sh "$M" supergpqa_law
done
```

后台跑(**推荐**,全程 3–6 小时):

```bash
setsid nohup bash -c 'for M in qwen3.5-2b qwen3.5-4b qwen3.5-9b; do
  ./run_all.sh "$M" supergpqa_law; done' > sweep.log 2>&1 &
```

某个尺寸退出码为 1 是**正常的**——那个配对不在带内,循环会继续跑下一个。

**2B/4B 上预期会有数据集掉出带外。** 这正是要测的东西:哪个失效机制随规模变化。
`build_cases.py` 拒绝写文件时,换一个 9B 上分数更高的数据集
(如 `cruxeval_output` 0.700 或 `bbh_causal_judgement` 0.600)——
它们在小模型上更可能落进带内。

---

## 4. 模型能力边界:endpoint 给什么、不给什么

vLLM 的 OpenAI 接口返回**文本**,不返回内部状态。框架按"模型声明了哪些
capability"来匹配 analyzer,**不匹配的会被静默跳过**,不报错。所以先搞清楚
每一层能解锁多少 analyzer(实测数字,文本模态):

| 模型形态 | capability | 可用 analyzer | 增量 |
|---|---|---|---|
| endpoint(旧) | `GENERATE` | **23** | — |
| endpoint(现在) | `+ LOGPROBS` | **26** | `calibration` `logprob_entropy` `mm_shap` |
| Stage W(transformers) | `+ ATTENTION` `HIDDEN_STATES` `LOGITS` | **37** | `attention` `attention_sink` `attention_rollout` `causal_trace` `logit_lens` `tuned_lens` `layer_contrast` `linear_probe` `cka` `token_entropy` `counterfactual` |

### 4.1 logprobs —— 白捡的,不用额外 GPU

vLLM 一直支持 `logprobs`,只是以前的 wrapper 把它扔了。现在 `EndpointModel`
声明 `LOGPROBS`,每个 case 多花一次 ~64 token 的续写。

`config.yaml` 里的三个键:

```yaml
logprobs_mode: answer     # answer | chain
logprobs_max_tokens: 64
logprobs_top_k: 5
```

**`mode` 决定给哪段文本打分,对 thinking 模型这就是全部问题。** 开着 thinking
时,前 64 个 token 永远是思维链的开场白,它的概率和答对答错几乎无关。实测
(5 个难度递增的问题,`exp(mean logprob)`):

| mode | 置信度范围 | 标准差 | 排序 |
|---|---|---|---|
| `answer` | 0.758 – 0.9998 | **0.1134** | 常识 > 医学 > 数学 > 不可知,合理 |
| `chain` | 0.931 – 0.960 | 0.0108 | 最难的数学题反而最高,无意义 |

`answer` 模式发 `enable_thinking=False`,打分的就是答案本身。**代价要说清楚**:
PASS/FAIL 标签来自完整思考的那次生成,所以这是"不思考时的置信度"对
"思考后的正确性",是个代理量——它问的是"不动脑子它知不知道"。

**停止符不计分。** `<|im_end|>` 的概率接近 1,而答案常常只有 1–3 个 token,
算进去会把所有短答案往 1 拉。实测:一个**答错**的单 token 答案,含停止符
0.787,不含 0.629;答对的那个两种算法都是 0.996 —— 也就是说它吃掉的正好是
`calibration` 要测的那个差。(这一点与 hf_local 后端不同,后者全算,两边的
置信度数值不可直接互比。)

> ⚠️ **别默认 `calibration` 在你的切片上有信号。** 我们各跑了 n=24 实测:
>
> | 切片 | 答案形态 | PASS 均值 | FAIL 均值 | AUC | 95% CI |
> |---|---|---|---|---|---|
> | `bbh_causal_judgement` | 是/否二选一 | 0.820±0.060 | 0.833±0.062 | **0.415** | [0.17, 0.66] |
> | `supergpqa_law` | 十选一 | 0.852±0.090 | 0.820±0.099 | **0.607** | [0.37, 0.84] |
>
> 十选一方向是对的(答对更自信),二选一方向是反的。但**两个区间都包含 0.5,
> 两者之间的差也完全落在噪声里** —— 也就是说:目前既没有证据说它有用,也没有
> 证据说它没用,n=24 根本判不了。
>
> 一个合理但**未经证实**的解释是:二选一任务里答错也能答得很流利,置信度测到的
> 是措辞而不是对错;选项越多,置信度才越有腾挪空间。
> **实践建议**:在你自己的切片上先用几十个 case 看一眼 AUC 再决定要不要信它,
> 自由作答(`minervamath`)最值得试。

### 4.2 Stage W —— 白盒 attention,只跑一小撮 case

**为什么单独一步、还要单独一个解释器**:`qwen3_5` 这个架构 transformers
4.57.6 **不认识**(评测 venv 里就是这个版本),5.15.0 认识(vLLM venv 里的版本)。
所以 Stage W 跑在 `WHITEBOX_PYTHON` 下,和 `VLLM_BIN` 是同一个道理。

```bash
# 接在 run_all.sh 后面自动跑(仅当设了这个变量)
WHITEBOX_PYTHON=/path/to/vllm-venv/bin/python ./run_all.sh qwen3.5-9b supergpqa_law

# 或者单独跑(cases.json 已存在即可)
$WHITEBOX_PYTHON run_whitebox.py --model qwen3.5-9b --dataset supergpqa_law --n 24
```

它在 vLLM **停掉之后**才启动:服务占着 92% 显存,transformers 要把这块拿回来。

#### ⚠️ Qwen3.5 是混合注意力栈 —— 32 层里只有 8 层有注意力矩阵

这是实测出来的,不是推测。`config.text_config.layer_types` 是
`[linear, linear, linear, full] × 8`(`full_attention_interval: 4`):

```
attention_layers() = [3, 7, 11, 15, 19, 23, 27, 31]
forward(...) 返回 8 个 tensor,每个 (16 heads, seq, seq)
```

其余 24 层是线性注意力(SSM 类),**根本不存在 QK 矩阵**。三个后果:

1. **返回列表的下标不是层号。** 第 `i` 个 tensor 是模型第 `4i+3` 层。报告里说
   "第 2 层注意力高"会指向一个压根没有注意力的层。`whitebox.json` 里存了
   `capturable_layer_indices` 做映射。
2. **`attention_rollout` 在这个架构上不成立**,它要连乘穿过整个栈才叫 rollout,
   这里只穿过 8/32 层。所以它**不在默认 analyzer 里**,要用得显式指定,并且
   程序会打 WARNING。
3. spec 里标成了新的 `AttnSemantics.HYBRID_SPARSE`(不是 `STANDARD`),
   这样任何将来读这个字段的代码都能知道该谨慎。

#### 显存:attention 是 O(seq²),会拒绝而不是 OOM

analyzer 调 `forward(capture={ATTENTION})` 时**不带 CaptureSpec**,即"全都要"。
`BoundedWhitebox` 拦下来先估算:

| prompt 长度 | 8 层 × 16 头 | 默认 8 GB 预算 |
|---|---|---|
| 512 | 0.13 GB | ok |
| 1024 | 0.52 GB | ok |
| 2048 | 2.1 GB | ok |
| 4096 | 8.4 GB | 拒绝 |

超预算时抛 `MemoryError` 并给出三条出路(提预算 / `--max-prompt-tokens` 筛掉长题 /
显式 `--layers`),**不会偷偷只抓几层** —— 那会让 rollout 之类的结果悄悄变味。

#### 子集是按标签均衡取的,不是按准确率

`--n 24` 是 12 PASS + 12 FAIL。按原分布取的话,一个 0.70 准确率的切片只会给
FAIL 三分之一的样本量,而报告出来的还是同一个 n。

#### 输出:自己做 PASS/FAIL 对比

`attention_sink`、`attention_rollout`、`attention` 这三个 analyzer **只分析
`cases[0]`**(它们源码里自己写了 "Stage 1: single-case ergonomics")。直接把
24 个 case 的 batch 丢进去,拿到的是一个 case 的数字、却长着 batch 的样子。
所以 `run_whitebox.py` 逐 case 跑,自己聚合:

```json
"contrasts": {
  "attention_sink.mean_sink_mass": {
    "n_pass": 12, "n_fail": 12,
    "pass_mean": 0.198, "fail_mean": 0.241,
    "gap": 0.043, "cohens_d": 0.62
  }
}
```

单看 0.198 没有意义;同一切片上 FAIL 0.241 对 PASS 0.198 才有意义。
报的是 Cohen's d 而不是 p 值:每边 12 个样本,给 p 值等于鼓励一个样本量
撑不起的结论。

#### ⚠️ 长度混淆 —— 第一次真跑就撞上了

在 `bbh_tracking7` 上跑通的第一次结果:

```
attention_sink.mean_sink_mass   pass=0.0596 fail=0.0582 gap=-0.0014 d=-1.15
prompt_tokens                   pass=225.8  fail=234.6  gap=+8.8    d=1.323
```

`d=-1.15` 按惯例算"大效应",但差值只有 **0.0014**——组内方差极小才撑出这个 d。
而同一批里 **prompt 长度本身就区分了 PASS/FAIL**(d=1.32)。sink mass 是
token 0 上的注意力在所有 query 位置上的平均,**序列越长它机械地越低**,
所以这个"效应"很可能就是长度。

`run_whitebox.py` 现在会自动检查:`prompt_tokens` 的 |d| ≥ 0.5 时打警告,
并在 `whitebox.json` 里写 `"length_confounded": true`。
**看到这个警告时,上面所有 gap 都要当作被长度污染,直到你在长度匹配的子集上重跑。
"很小的差 + 很大的 d"就是典型信号。**

---

## 5. 输出布局

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

## 6. 常见问题

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

## 7. 相关文件

| 路径 | 作用 |
|---|---|
| [`datasets.py`](datasets.py) | 八个数据集的机器可读目录 |
| [`build_cases.py`](build_cases.py) | Stage 0:生成 + 判分 + 冻结 batch |
| [`run_pipeline.py`](run_pipeline.py) | M1→M2→M3→M5→M4 驱动;`EndpointModel` 含 logprobs |
| [`whitebox.py`](whitebox.py) | Stage W:hf_local 加载 + 显存护栏 + 混合栈层映射 |
| [`run_whitebox.py`](run_whitebox.py) | Stage W 驱动:逐 case 跑 attention analyzer 并做 PASS/FAIL 对比 |
| [`preflight.py`](preflight.py) | 上机前自检,每条 FAIL 附安装命令 |
| [`run_all.sh`](run_all.sh) | **一条命令跑完全链路**(agent 用这个) |
| [`config.yaml`](config.yaml) | 全部默认参数 |
| `../llm_band_probe/` | 带位定位工具(spec / grader / fetch 的真正来源) |
| `../diagnosis_loops/deco_hallu/` | VLM 版参考实现,阶段划分同构 |

---

## 附:这套东西是怎么被验证的

写这份文档时,下列内容是**逐条核对过的**,不是照着记忆写的:

- 八个条目全部能解析到 `band_locate.SPECS` 里的实际 spec(`python datasets.py` 会校验)
- `run_pipeline.py` 里用到的每一个构造函数关键字参数,都对着实际签名验过
- `band_locate.generate` 在调用时读 `MODEL_ID` / `BASE_URL` 全局,所以覆盖它们是有效的

**第 4 节的内容是在跑起来的 9B 上实测的**(不是推的):

- thinking 默认开:chat template 里 `add_generation_prompt` 会追加 `<think>\n`,
  除非 `enable_thinking=False`。补充一个此前说法的更正——**完成文本里没有开头的
  `<think>`**(它在 prompt 里),但**有结尾的 `</think>`**,这是切分答案的可靠锚点
- vLLM 的 `/chat/completions` 确实返回 `logprobs` / `top_logprobs`
- `enable_thinking=False` 在 Qwen3.5 上有效(答案直出,`finish_reason=stop`)
- 停止符对短答案的影响、answer 与 chain 两种模式的方差差异,都是量出来的数字
- Qwen3.5 是混合栈:`layer_types` 实测 8 个 `full_attention`,位于 3/7/…/31 层
- `BoundedWhitebox` 的显存护栏在过小预算下确实抛 `MemoryError`,正常预算下放行
- `attention_sink` 在真实 case 上跑通(0.2s,8 层的 per-layer 数值)
- **Stage 0 → Stage W 端到端真跑过一次**:`bbh_tracking7` n=16 生成(174s,
  准确率 11/16)→ `run_whitebox.py --n 10`(5 PASS / 5 FAIL)→ 写出
  `whitebox.json`,长度混淆警告正确触发
- 新增 25 个单测;仓库全量 **1814 passed / 15 skipped**

**尚未验证的**:M1→M5→M4 主链路本身还没在这三个尺寸上端到端跑过一次
(第 4 节的 logprobs 与 Stage W 已验证,Stage 0 也已验证)。
上面的命令是照着已验证的接口和 `deco_hallu` 的参考实现写的,
但第一次跑仍可能碰到运行时问题 —— 建议先用 `--analysis-only` 加一个小 `--n` 试通。
