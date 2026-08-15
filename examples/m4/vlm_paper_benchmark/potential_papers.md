# Potential paper-method candidates (2023–2026 survey)

This file records the repair-method papers from a 2026-08 literature survey
that were **not** yet folded into [`literature_matrix.json`](literature_matrix.json)
/ [`paper_casebook.json`](paper_casebook.json). Survey scope: main-conference
papers at ICML / NeurIPS / ICLR / CVPR / ICCV / ECCV / AAAI, conference edition
between October 2023 and August 2026, proposing a fix for a VLM failure mode,
with official author code. Every venue claim was verified against a
proceedings / OpenReview-decision / CVF / PMLR / AAAI-OJS page, and every code
link was fetched; workshops, ACL-family venues, journals and arXiv-only papers
were dropped. The five papers that already met the "black-box runnable on an
existing data surface" bar (DyFo, DC², RAP, API, CCoT) live in the casebook,
not here.

A row is promoted from this file to the casebook by satisfying the casebook
admission rule: name the failure mechanism, a public data source, the proposed
repair, and the model access needed — and be honest about fidelity: a method
whose exact form needs logits / attention / hidden states must be recorded as
`adapted` or `unavailable` when run through a weaker access level, never
silently relabelled.

The "≈ casebook row" column is the mechanistically closest existing case
(useful for FixAgent keyword gating and for picking which diagnosis slice
should route to the method); "—" means the mechanism has no casebook sibling
yet, which is exactly what makes some tier-B rows interesting.

## Tier A — runnable on the example's `hf_local` specs (14 papers)

Training-free decoding- or attention-level methods demonstrated on LLaVA-1.5
and/or InstructBLIP — the two architectures this repo already pins as
`llava-1.5-7b-hf` / `instructblip-vicuna-7b` specs — so a faithful (`exact` or
`native_*`) implementation is possible through the same `run_hf_autofix.py`
route as VCD/ICD/PAI/ViCrop. Notes: SID and AGLA need attention access (L3a);
MemVR and ONLY are single-forward-pass interventions (L3b patches); DeGF and
ConVis additionally require a text-to-image model at inference, which makes
them the most expensive rows here; DeCo is the same method family as the
`deco_hallu` example elsewhere in this repo; Devils-in-Mid-Layers also ships a
middle-layer visual-attention-ratio (VAR) hallucination *detector* that could
be ported as an M1 probe signal, independent of its repair.

| Paper | Venue | ≈ casebook row | Access needed | Mechanism | Code |
| --- | --- | --- | --- | --- | --- |
| [AGLA](https://arxiv.org/abs/2406.12718) | CVPR 2025 | VCD | attention/gradients | Two-pass logit assembly: a global pass on the original image plus a local pass on an augmented image where prompt-irrelevant regions are masked out via gradient-based image-prompt… | [code](https://github.com/Lackel/AGLA) |
| [CausalMM](https://arxiv.org/abs/2410.04780) | ICLR 2025 | PAI | attention/gradients | Treats modality priors as confounders in a structural causal model. | [code](https://github.com/The-Martyr/CausalMM) |
| [ConVis](https://arxiv.org/abs/2408.13906) | AAAI 2025 | VCD | decoding loop | Hallucination visualization contrast: a text-to-image model reconstructs the image from the model's (possibly hallucinated) caption; token distributions from the original vs recons… | [code](https://github.com/yejipark-m/ConVis) |
| [DeCo](https://arxiv.org/abs/2410.11779) | ICLR 2025 | VCD | hidden states | Empirical finding: hallucinated objects are correctly recognized in preceding layers but suppressed by language priors at the top. | [code](https://github.com/zjunlp/DeCo) |
| [DeGF](https://arxiv.org/abs/2502.06130) | ICLR 2025 | VCD | decoding loop | Generative feedback: a text-to-image diffusion model renders an image from the LVLM's initial response; a second forward pass conditioned on this generated image yields an auxiliar… | [code](https://github.com/zhangce01/DeGF) |
| [Devils-in-Mid-Layers](https://arxiv.org/abs/2411.16724) | CVPR 2025 | PAI | attention/gradients | Attention-lens analysis shows middle layers' image attention has a two-stage (broad then focused) pattern and real-object tokens attend more strongly to relevant image regions. | [code](https://github.com/ZhangqiJiang07/middle_layers_indicating_hallucinations) |
| [EAZY](https://arxiv.org/abs/2503.07772) | ICCV 2025 | PAI | hidden states | Training-free: identifies the small subset (~1. | [code](https://github.com/pseudoc18/EAZY) |
| [MARINE](https://arxiv.org/abs/2402.08680) | ICML 2025 | VCD | decoding loop | Classifier-free guidance in logit space: external open-vocabulary vision models (DETR, RAM++) extract object-level evidence that is injected as an additional grounded condition; th… | [code](https://github.com/Linxi-ZHAO/MARINE) |
| [MemVR](https://arxiv.org/abs/2410.03577) | ICML 2025 | PAI | hidden states | When decoding uncertainty is high, re-injects the projected visual tokens into a middle-layer FFN as extra key-value memory (a 'look twice' retrace), replenishing visual evidence t… | [code](https://github.com/1zhou-Wang/MemVR) |
| [ONLY](https://arxiv.org/abs/2507.00898) | ICCV 2025 | PAI | hidden states | Single-query, one-layer decoding intervention: computes a text-to-visual entropy ratio per token and selectively amplifies crucial textual/visual information in one decoder layer,… | [code](https://github.com/zifuwan/ONLY) |
| [SID](https://arxiv.org/abs/2408.02032) | ICLR 2025 | VCD | attention/gradients | Context and Text-aware Token Selection (CT2S): uses early-decoder-layer attention to keep only the least-important vision tokens in a second introspective pass, adaptively amplifyi… | [code](https://github.com/huofushuo/SID) |
| [VDGD](https://arxiv.org/abs/2405.15683) | ICLR 2025 | ICD | decoding loop | Two-part scaffold: first elicit a detailed image description and prepend it to the instruction (prompt-level grounding), then during generation prefer candidate tokens with low KL… | [code](https://github.com/Sreyan88/VDGD) |
| [CODE](https://arxiv.org/abs/2406.01920) | NeurIPS 2024 | VCD | decoding loop | Uses the model's own self-generated comprehensive image description as the contrasting reference: contrasts logits conditioned on the image vs conditioned on the self-description,… | [code](https://github.com/IVY-LVLM/CODE) |
| [HALC](https://arxiv.org/abs/2403.00425) | ICML 2024 | VCD | decoding loop | Adaptive focal-contrast decoding: each candidate token is grounded to an image region via an off-the-shelf detector (Grounding DINO); the method samples different fields of view ar… | [code](https://github.com/BillChan226/HALC) |

## Tier B — mechanism or task-axis gaps (21 papers)

Methods whose mechanism class or failure axis has **no row in the current
casebook**: prefill-stage KV steering (PTI — every current row acts at decode
time), encoder-side defense (SHIELD), weight/feature-space erasure (Nullu,
ProjectAway), scene-text semantic hallucination (ZoomText+GLC — an OCR axis
the dataset table lacks; its TextHalu-Bench is public), spatial reasoning
(AdaptVis, VADAR — a natural bridge to the VTC `spatial` subtask), hidden-state
hallucination probes (TruthPrInt), steering-vector calibration (VTI, VISTA,
ICT), and black-box focus tools that predate chat-VLMs (FGVP, IVM,
Visual Sketchpad). Promoting one of these means writing a **new** casebook row
(new mechanism keywords for FixAgent gating), not attaching to an existing one.
M3ID is listed here despite CVPR 2024 acceptance because Amazon never released
official code — only third-party re-implementations inside the HALC/SID zoos —
so its fidelity ceiling is `adapted`.

| Paper | Venue | ≈ casebook row | Access needed | Mechanism | Code |
| --- | --- | --- | --- | --- | --- |
| **OWL** | AAAI 2026 | PAI | attention/gradients | Training-free causal intervention on visual vs. | [code](https://github.com/CikZ2023/OWL) |
| [PTI](https://arxiv.org/abs/2604.25642) | CVPR 2026 | PAI | hidden states | Training-free one-shot intervention at the PREFILL stage: modality-aware steering of the initial KV cache before any token is generated — keys steered toward visually-grounded obje… | [code](https://github.com/huaiyi66/PTI) |
| [SHIELD](https://arxiv.org/abs/2510.16596) | ICLR 2026 | VCD | hidden states | Training-free, non-invasive wrapper targeting the visual ENCODER as the hallucination source: (1) re-weights visual tokens to fix statistical bias, (2) injects noise-derived tokens… | [code](https://github.com/hukcc/SHIELD) |
| [AdaptVis](https://arxiv.org/abs/2503.01773) | ICML 2025 | PAI | attention/gradients | Diagnoses that spatial-reasoning errors coincide with image attention misdirected to irrelevant objects. | [code](https://github.com/shiqichen17/AdaptVis) |
| **DAMO** | ICLR 2025 | — | hidden states | Layer-momentum decoding: observes that VLMs 'overthink' in the last few layers (late prediction shifts toward hallucinations) and counters it by accumulating early-exit activations… | [code](https://github.com/tunantu/DAMO) |
| [ICoT](https://arxiv.org/abs/2411.19488) | CVPR 2025 | ViCrop | attention/gradients | Attention-driven Selection (ADS): during CoT generation, uses the model's attention maps to pick the image patches most relevant to the current reasoning step and interleaves those… | [code](https://github.com/jungao1106/ICoT) |
| [ICT](https://arxiv.org/abs/2411.15268) | CVPR 2025 | PAI | hidden states | Computes two intervention directions from paired trusted/untrusted activations and adds them to selected attention-head outputs during the forward pass: an image-level vector shift… | [code](https://github.com/THU-BPM/ICT) |
| [Nullu](https://arxiv.org/abs/2412.13817) | CVPR 2025 | — | hidden states | Extracts a 'HalluSpace' — top SVD directions of the difference between hidden features of paired truthful vs hallucinated samples (dominated by LLM language priors) — then orthogon… | [code](https://github.com/Ziwei-Zheng/Nullu) |
| [ProjectAway](https://arxiv.org/abs/2410.02762) | ICLR 2025 | — | hidden states | Projects internal image-token representations onto the language vocabulary (logit lens); hallucinated objects show low internal confidence. | [code](https://github.com/nickjiang2378/vl-interp) |
| [TruthPrInt](https://arxiv.org/abs/2503.10602) | ICCV 2025 | OPERA | hidden states | Finds that decoding-time hidden states are high-specificity per-token hallucination indicators and that different LVLMs share a common hallucination subspace (ComnHallu enables cro… | [code](https://github.com/jinhaoduan/TruthPrInt) |
| **VADAR** | CVPR 2025 | V* | tools/crops | Agentic visual program synthesis: Signature and Implementation LLM agents dynamically author a Pythonic API (rather than ViperGPT's fixed human-defined API), then a Program agent c… | [code](https://github.com/damianomarsili/VADAR) |
| [VASparse](https://arxiv.org/abs/2501.06553) | CVPR 2025 | OPERA | attention/gradients | Plug-and-play decoding: visual-aware token selection sparsifies the KV/token set while preserving visually grounded tokens (countering the visual-agnostic sparsity that worsens hal… | [code](https://github.com/mengchuang123/VASparse-github) |
| [VISTA](https://arxiv.org/abs/2502.03628) | ICML 2025 | PAI | hidden states | Two inference-time modules: a Visual Steering Vector reinforces visual information in activation space to counter gradual visual-information loss across generation, and Self-Logits… | [code](https://github.com/LzVv123456/VISTA) |
| [VTI](https://arxiv.org/abs/2410.15778) | ICLR 2025 | PAI | hidden states | Visual-and-Textual Intervention: pre-computes steering vectors from ~a few dozen example pairs (captions with/without hallucination) by recording hidden states, then adds these shi… | [code](https://github.com/shengliu66/VTI) |
| [ZoomText+GLC](https://arxiv.org/abs/2506.05551) | NeurIPS 2025 | ViCrop | hidden states | Training-free two-stage fix for scene-text 'semantic hallucination': (1) ZoomText — coarse-to-fine attention-guided localization of text regions with no external detector (ViCrop-s… | [code](https://github.com/shuyansy/MLLM-Semantic-Hallucination) |
| [ControlMLLM](https://arxiv.org/abs/2407.21534) | NeurIPS 2024 | PAI | attention/gradients | Test-time optimization of a learnable latent added to visual tokens: an energy function over the text-to-image attention map is optimized so the model attends to a user-referred re… | [code](https://github.com/mrwu-mac/ControlMLLM) |
| [IVM](https://arxiv.org/abs/2405.19783) | NeurIPS 2024 | ViCrop | tools/crops | Trains a standalone visual-grounding 'masker' (on IVM-Mix-1M with Discriminator-Weighted Supervised Learning) that blacks out instruction-irrelevant image regions; the masked image… | [code](https://github.com/2toinf/IVM) |
| [M3ID](https://arxiv.org/abs/2403.14003) | CVPR 2024 | VCD | decoding loop | Multi-Modal Mutual-Information Decoding: at each step contrasts the image-conditioned prediction with the unconditioned (text-only) prediction, boosting tokens with high pointwise… | *(no official code)* |
| [Visual Sketchpad](https://arxiv.org/abs/2406.09403) | NeurIPS 2024 | V* | tools/crops | Agentic scaffold that gives the (black-box) LMM a sketchpad: it plans, calls drawing/vision tools (detection, segmentation, zoom/crop, mark overlay, matplotlib) to create visual ar… | [code](https://github.com/Yushi-Hu/VisualSketchpad) |
| [DDCoT](https://arxiv.org/abs/2310.16436) | NeurIPS 2023 | — | tools/crops | Duty-split prompt scaffold: the LLM decomposes the question into reasoning steps and explicitly flags sub-questions it cannot answer from text (negative-space prompting to keep it… | [code](https://github.com/SooLab/DDCOT) |
| [FGVP](https://arxiv.org/abs/2306.04356) | NeurIPS 2023 | ViCrop | tools/crops | Pixel-level visual prompts from SAM masks — the key finding is the Blur Reverse Mask (blur everything outside the candidate mask), which suppresses weakly related regions while kee… | [code](https://github.com/ylingfeng/FGVP) |

## Tier C — training-based upper bounds (21 papers)

Preference-optimization / finetuning repairs. For a black-box diagnosis system
these are `unavailable` casebook rows by construction, but they matter as
upper-bound comparators: if a paired, e-value-validated inference-time fix
recovers a meaningful fraction of what finetuning buys, that is a strong
result — and if it does not, that bounds honest expectations. REVERSE,
SENTINEL and ZwZ released checkpoints/LoRA weights, so they can be *evaluated*
(not reproduced) without any training run.

| Paper | Venue | ≈ casebook row | Access needed | Mechanism | Code |
| --- | --- | --- | --- | --- | --- |
| [ZwZ](https://arxiv.org/abs/2602.11858) | ICML 2026 | ViCrop | training | Region-to-Image (R2I) distillation turns zooming from an inference-time tool into a training-time primitive: strong teachers answer over micro-crops to synthesize region-grounded V… | [code](https://github.com/inclusionAI/Zooming-without-Zooming) |
| **ChartMoE** | ICLR 2025 | — | training | Replaces the linear vision-language connector of InternLM-XComposer2 with a Mixture-of-Experts connector whose experts are separately initialized via chart-to-table, chart-to-JSON,… | [code](https://github.com/DataArcTech/ChartMoE) |
| [CogCoM](https://arxiv.org/abs/2402.04236) | ICLR 2025 | V* | training | Finetunes a VLM (CogVLM-17B base) on 70K chain-of-manipulations traces so the model itself emits and consumes manipulations mid-reasoning — GROUNDING (boxes), CROP_AND_ZOOMIN (new… | [code](https://github.com/THUDM/CogCoM) |
| [DCD](https://arxiv.org/abs/2504.08809) | NeurIPS 2025 | VCD | training | Trains SEPARATE positive and negative image projections inside the MLLM (decoupled from pairwise DPO, avoiding likelihood displacement); the negative projector implicitly models re… | [code](https://github.com/HKUST-LongGroup/DCD) |
| [HALVA](https://arxiv.org/abs/2405.18654) | ICLR 2025 | — | training | Generative data augmentation selectively corrupts ground-truth phrases to create correct/hallucinated response pairs; a Data-augmented Phrase-level Alignment (DPA) loss lowers the… | [code](https://github.com/pritamqu/HALVA) |
| [Octopus](https://arxiv.org/abs/2503.00361) | CVPR 2025 | VCD | training | Dynamic contrastive-decoding router: a lightweight learned 'decision token' (the eye) classifies which hallucination type each generation step faces and routes to the matching CD s… | [code](https://github.com/LijunZhang01/Octopus) |
| [OPA-DPO](https://arxiv.org/abs/2501.09695) | CVPR 2025 | — | training | Shows DPO's anti-hallucination gains hinge on data being on-policy w.r.t. the reference policy. | [code](https://github.com/zhyang2226/OPA-DPO) |
| [REVERSE](https://arxiv.org/abs/2504.13169) | NeurIPS 2025 | OPERA | training | Hallucination-aware SFT on 1.3M semi-synthetic samples teaches the model to emit confidence/hallucination marker tokens, plus inference-time retrospective resampling: when a halluc… | [code](https://github.com/tsunghan-wu/reverse_vlm) |
| [RLAIF-V](https://arxiv.org/abs/2405.17220) | CVPR 2025 | — | training | Fully open-source AI feedback: a divide-and-conquer strategy splits responses into atomic claims scored by open-source MLLM labelers (OmniLMM-12B, MiniCPM-Llama3-V 2. | [code](https://github.com/RLHF-V/RLAIF-V) |
| [SENTINEL](https://arxiv.org/abs/2507.12455) | ICCV 2025 | — | training | Human-annotation-free preference training: samples multiple in-domain responses, cross-validates mentioned objects with open-vocabulary detectors to find the FIRST hallucinated sen… | [code](https://github.com/pspdada/SENTINEL) |
| [AMP](https://arxiv.org/abs/2405.11165) | NeurIPS 2024 | — | training | Annotator-free pipeline builds multi-level (superior/medium/inferior) preference datasets, then a multi-level DPO (MDPO) objective with cross-level comparisons teaches the model to… | [code](https://github.com/takomc/amp) |
| [CCA-LLaVA](https://arxiv.org/abs/2410.15926) | NeurIPS 2024 | PAI | training | Diagnoses object hallucination as RoPE long-term decay / recency bias: visual cues far from instruction tokens get under-attended. | [code](https://github.com/xing0047/cca-llava) |
| [CSR](https://arxiv.org/abs/2405.14622) | NeurIPS 2024 | VCD | training | Iterative self-rewarding loop: the LVLM generates candidate responses, scores them with a step-wise self-reward calibrated by a visual constraint (image-conditioned likelihood cont… | [code](https://github.com/YiyangZhou/CSR) |
| [HACL](https://arxiv.org/abs/2312.06968) | CVPR 2024 | ICD | training | Observes that representations of hallucinated and faithful text are entangled in MLLM embedding space; adds cross-modal contrastive learning during training that uses GPT-generated… | [code](https://github.com/X-PLUG/mPLUG-HalOwl/tree/main/hacl) |
| [HalluciDoctor](https://arxiv.org/abs/2311.13614) | CVPR 2024 | — | training | Data-centric repair: cross-checking pipeline decomposes machine-generated instruction answers into chunks, generates probing questions, and votes across multiple expert VQA models… | [code](https://github.com/Yuqifan1117/HalluciDoctor) |
| [LRV-Instruction](https://arxiv.org/abs/2306.14565) | ICLR 2024 | — | training | Robust visual instruction tuning on ~400k GPT-4-generated positive AND negative instructions (nonexistent-object, existent-object, and knowledge manipulation) so the model learns t… | [code](https://github.com/FuxiaoLiu/LRV-Instruction) |
| [LURE](https://arxiv.org/abs/2310.00754) | ICLR 2024 | — | logits | Post-hoc hallucination revisor: statistical analysis pins hallucination on co-occurrence priors, decoding uncertainty, and late-sentence position; at test time uncertain/late objec… | [code](https://github.com/YiyangZhou/LURE) |
| [RLHF-V](https://arxiv.org/abs/2312.00849) | CVPR 2024 | — | training | Collects segment-level human corrections of hallucinated spans in model responses (1. | [code](https://github.com/RLHF-V/RLHF-V) |
| **SpatialRGPT** | NeurIPS 2024 | — | training | Data pipeline lifts single images to 3D scene graphs to build the Open Spatial Dataset; adds region-prompt (box/mask) inputs and a plug-in depth-map module to the visual encoder; V… | [code](https://github.com/AnjieCheng/SpatialRGPT) |
| [STIC](https://arxiv.org/abs/2405.19716) | NeurIPS 2024 | — | training | Two-stage self-training with no human or teacher labels: stage 1 self-constructs an image-description preference dataset (preferred = well-prompted self-descriptions; dispreferred… | [code](https://github.com/yihedeng9/STIC) |
| [Visual-CoT](https://arxiv.org/abs/2403.16999) | NeurIPS 2024 | V* | training | Trains VisCoT on 438k QA pairs annotated with intermediate bounding boxes (98k with reasoning steps); multi-turn inference pipeline: model predicts the key region bbox, the region… | [code](https://github.com/deepcs233/Visual-CoT) |

## Audit notes from verification

- **Chameleon** (NeurIPS 2023) was collected and then excluded: venue and code
  are genuine, but it is an LLM tool-orchestration framework, not a repair of a
  native VLM failure mode.
- **RAP** is an ICML 2025 *Spotlight Poster*, not the Oral its repo claims.
- **ProjectAway**'s repo `nickjiang2378/vl-interp` was renamed to
  `nickjiang2378/vlm-hallucinations`; the table links the new location.
- **OWL** (AAAI 2026) is filed under AAAI's "Philosophy and Ethics of AI"
  technical track — odd routing, but a main-conference technical track.
- Dropped at verification for failing the venue rule: Pensieve, AID
  ("attention hijackers"), TextMonkey (arXiv-only); Woodpecker (journal, SCIS);
  VOLCANO, LogicCheckGPT, Scaffold and most prompting papers (ACL family).

Full per-paper verification records (venue-evidence URLs, verifier verdicts,
benchmarks, architectures) are in the survey JSON kept outside the repo:
`/tealab-data/jiaqiliu/evalsmith/vlm_repair_survey_2026-08.json`.
