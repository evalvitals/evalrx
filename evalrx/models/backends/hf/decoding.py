"""L3a decoding repairs for the ``hf_local`` backend: contrastive decoding (VCD,
instruction CD, IFCD, TCD, AAD), attention interventions (PAI, OPERA) and
visual cropping (ViCrop), plus the fidelity report for each paper method.

:class:`ContrastiveDecodingMixin` is mixed into
:class:`~evalrx.models.backends.hf.model.HFLocalModel`; its methods use the
model's loaded state (``self._model``, ``self._processor``, ``self.spec``,
``self.runtime``) and its encoding helpers (``_encode``, ``_encode_vlm``,
``_render_text_prompt``). The algorithms themselves live in
:mod:`evalrx.models.paper_methods`.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from evalrx.core.capability import Capability
from evalrx.core.case import Inputs
from evalrx.models._media import AUDIO_SAMPLE_RATE
from evalrx.models._media import resolve_audio as _resolve_audio
from evalrx.models.backends.hf._util import (  # noqa: F401
    _check_audio_duration,
    _collect_message_images,
    _new_tokens,
    _populate_audio_extras,
    _populate_vision_extras,
    _read_nested_attr,
)

logger = logging.getLogger(__name__)


class ContrastiveDecodingMixin:
    """L3a decoding-time repairs (see the module docstring)."""

    def paper_method_fidelity(self, method: str) -> str:
        """Declare whether a paper-method adapter is exact or architecture-adapted.

        ``FixAgent`` admits architecture-native paper routes by default.  An
        experiment may explicitly opt into ``"adapted"`` methods, but reports
        must retain that distinction rather than treating a same-formula port
        to a different model architecture as a paper reproduction.
        """
        fidelity = self._paper_method_fidelity_impl(method)
        logger.debug("%s: paper_method_fidelity(%r) -> %r", self.spec.key, method, fidelity)
        return fidelity


    def _paper_method_fidelity_impl(self, method: str) -> str:
        if method == "vcd":
            # The executor uses the released corruption, plausibility cutoff,
            # and per-token sampler. Image-specific seeding keeps paired
            # evaluation independent of iteration order.
            return "per_item_seeded_sampler_specialization"
        if method == "icd":
            return (
                "native_binary_specialization"
                if self.spec.model_type == "instructblip"
                else "adapted"
            )
        if method == "vicrop":
            return (
                "native_selector_specialization" if self.spec.model_type == "llava" else "adapted"
            )
        if method == "opera":
            # This preserves OPERA's first-token over-trust penalty for POPE's
            # binary task. Its beam rollback is not meaningful when exactly
            # one output token is evaluated.
            return (
                "native_binary_specialization" if self.spec.model_type == "llava" else "unavailable"
            )
        if method == "ifcd":
            checkpoint = self.runtime.engine_kwargs.get("ifcd_checkpoint")
            if self.spec.model_type == "llava" and checkpoint and Path(str(checkpoint)).is_file():
                # The public Vicuna TruthX artifact is compatible with
                # LLaVA-1.5's decoder but is not the paper's MSCOCO-trained
                # editor, and modern HF hooks after ``o_proj``. Never promote
                # this to an exact IFCD reproduction.
                return "adapted_truthx_artifact"
            return "unavailable"
        if method == "pai":
            return (
                "native_attention_cfg_specialization"
                if self.spec.model_type == "llava"
                else "unavailable"
            )
        if method == "tcd":
            if self.spec.audio is None:
                return "unavailable"
            # Eq. 4 zips per-layer encoder stability with per-layer decoder
            # audio-attention ratio index-for-index — faithful only when both
            # towers have the same layer count. Qwen2-Audio-Instruct's
            # Whisper-style encoder and Qwen2 decoder both have 32 layers
            # (verified against the live config, not assumed); the paper's
            # own hyperparameters (Appendix A) are anchored on this
            # checkpoint. Every other registered audio spec has a depth
            # mismatch (e.g. Qwen2.5-Omni: 32 encoder / 28 decoder), so
            # generate_tcd truncates both to min(...) there instead of
            # raising -- report that truncation as an adaptation.
            return (
                "native_layer_matched_stability"
                if self.spec.key == "qwen2-audio-7b-instruct"
                else "adapted_truncated_layer_stability"
            )
        if method == "aad":
            # Unlike TCD, AAD's contrast (real audio vs. the same prompt with
            # the waveform silenced) never reads architecture internals -- no
            # layer counts, no attention weights, no audio-token span. The
            # released repo's own claim is that this generalises across
            # audio LALMs (Qwen2-Audio and SALMONN both evaluated); this
            # codebase's registered audio specs are exactly the checkpoints
            # that claim covers, so any of them is native, not adapted.
            return "native_silence_contrast" if self.spec.audio is not None else "unavailable"
        return "unavailable"


    def _vcd_encodings(
        self,
        inputs: Any,
        *,
        noise_step: int,
        noise_seed: int,
    ) -> tuple[Any, dict[str, Any]]:
        """Build clean/noisy VCD inputs at the published tensor boundary.

        The VCD release calls ``add_diffusion_noise`` *after* its image
        processor: it perturbs the model's normalized image tensor, not an RGB
        image that will subsequently be normalized again.  Matching that
        boundary is material; image-space noise has a different distribution.
        A content-derived seed preserves paired-test reproducibility without
        coupling one example's corruption to the iteration order of another.
        """
        import hashlib

        import torch

        model, processor = self._loaded
        enc, _, _, _ = self._encode_vlm(inputs, model, processor)
        enc.pop("token_type_ids", None)
        pixels = enc.get("pixel_values")
        if pixels is None:
            raise ValueError(f"{self.spec.key}: VCD requires processor pixel_values")
        step = max(0, min(999, int(noise_step)))
        # This is vcd_utils/vcd_add_noise.py verbatim in numerical form.
        betas = torch.sigmoid(torch.linspace(-6, 6, 1000, device=pixels.device))
        betas = betas * (0.5e-2 - 1e-5) + 1e-5
        alpha_bar = torch.cumprod(1 - betas, dim=0)[step].to(dtype=pixels.dtype)
        digest = hashlib.sha256(pixels.detach().float().cpu().numpy().tobytes()).digest()
        item_seed = (int(noise_seed) + int.from_bytes(digest[:8], "little")) % (2**63 - 1)
        generator = torch.Generator(device=pixels.device).manual_seed(item_seed)
        noise = torch.randn(
            pixels.shape, device=pixels.device, dtype=pixels.dtype, generator=generator
        )
        noisy_enc = dict(enc)
        noisy_enc["pixel_values"] = alpha_bar.sqrt() * pixels + (1 - alpha_bar).sqrt() * noise
        return enc, noisy_enc


    def _aad_encodings(self, inputs: Any) -> tuple[Any, dict[str, Any]]:
        """Build real-audio/silent-audio AAD inputs at the published boundary.

        AAD's release zeroes the raw WAVEFORM (``np.zeros_like(audio)``) and
        re-runs it through the same feature extractor, not the post-extraction
        feature tensor directly -- a zeroed waveform's log-mel features are
        not literally zero, so matching that boundary (not skipping straight
        to zeroed ``input_features``) is material, same reasoning as VCD's
        noise-after-processor boundary above.
        """
        import numpy as np

        from evalrx.core.case import Inputs

        model, processor = self._loaded
        audio = getattr(inputs, "audio", None)
        if audio is None or isinstance(audio, (list, tuple)):
            raise ValueError("AAD requires exactly one audio clip")
        waveform = _resolve_audio(audio)
        real_inputs = Inputs(
            prompt=self._as_prompt(inputs),
            image=getattr(inputs, "image", None),
            audio=waveform,
            video=getattr(inputs, "video", None),
        )
        silent_inputs = Inputs(
            prompt=self._as_prompt(inputs),
            image=getattr(inputs, "image", None),
            audio=np.zeros_like(waveform),
            video=getattr(inputs, "video", None),
        )
        enc, _ids, _tokens, _ttm = self._encode_vlm(real_inputs, model, processor)
        enc.pop("token_type_ids", None)
        silent_enc, _ids2, _tokens2, _ttm2 = self._encode_vlm(silent_inputs, model, processor)
        silent_enc.pop("token_type_ids", None)
        return enc, silent_enc


    def generate_aad(self, inputs: Any, *, alpha: float = 0.5) -> str:
        """Run AAD (Hsu et al. 2025, arXiv:2506.07233): contrast real-audio
        decoding against the same prompt with the audio waveform silenced,
        at every step. See :mod:`evalrx.models.paper_methods.aad`.
        """
        logger.debug("%s: generate_aad(alpha=%s)", self.spec.key, alpha)
        if self.spec.audio is None:
            raise ValueError(f"{self.spec.key}: AAD requires an audio-capable spec")
        import torch
        from transformers.generation.logits_process import LogitsProcessorList

        from evalrx.models.paper_methods.aad import AADLogitsProcessor

        model, processor = self._loaded
        tok = getattr(processor, "tokenizer", processor)
        enc, silent_enc = self._aad_encodings(inputs)
        processor_list = LogitsProcessorList(
            [AADLogitsProcessor(model, silent_enc, alpha=alpha)]
        )
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=self.runtime.max_new_tokens,
                do_sample=False,
                use_cache=True,
                logits_processor=processor_list,
            )
        return tok.decode(_new_tokens(out[0], enc["input_ids"][0]), skip_special_tokens=True)


    def generate_aad_baseline(self, inputs: Any) -> str:
        """Greedy-decode the real-audio arm alone -- the paired-comparison
        control AAD's own eligibility gate looks for (mirrors
        generate_vcd_baseline/generate_tcd_baseline's role)."""
        logger.debug("%s: generate_aad_baseline()", self.spec.key)
        if self.spec.audio is None:
            raise ValueError(f"{self.spec.key}: AAD requires an audio-capable spec")
        import torch

        model, processor = self._loaded
        tok = getattr(processor, "tokenizer", processor)
        enc, _silent_enc = self._aad_encodings(inputs)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=self.runtime.max_new_tokens, do_sample=False, use_cache=True,
            )
        return tok.decode(_new_tokens(out[0], enc["input_ids"][0]), skip_special_tokens=True)


    def _vcd_next_token_logits(
        self,
        inputs: Any,
        *,
        noise_step: int,
        noise_seed: int,
    ) -> tuple[Any, Any]:
        """Return clean/noisy next-token logits for the legacy binary route."""
        import torch

        model, _ = self._loaded
        enc, noisy_enc = self._vcd_encodings(
            inputs, noise_step=noise_step, noise_seed=noise_seed
        )
        with torch.no_grad():
            clean = model(**enc).logits[0, -1]
            noisy = model(**noisy_enc).logits[0, -1]
        return clean, noisy


    def generate_vcd(
        self,
        inputs: Any,
        *,
        alpha: float = 0.5,
        beta: float = 0.1,
        noise_step: int = 500,
        noise_seed: int = 55,
    ) -> str:
        """Run VCD's released contrastive formula through every output token.

        The source sampler uses temperature-one multinomial sampling.  Here the
        diffusion noise is seeded per image content, rather than a global loop
        RNG, so a frozen selection/confirmation split remains reproducible if
        case order changes.  That seed policy is recorded as a specialization.
        """
        logger.debug(
            "%s: generate_vcd(alpha=%s, beta=%s, noise_step=%s, noise_seed=%s)",
            self.spec.key, alpha, beta, noise_step, noise_seed,
        )
        if not self.spec.is_vlm:
            raise ValueError("VCD visual contrast requires a VLM")
        image = getattr(inputs, "image", None)
        if image is None or isinstance(image, (list, tuple)):
            raise ValueError("VCD requires exactly one image")
        import hashlib

        import torch
        from transformers.generation.logits_process import LogitsProcessorList

        model, processor = self._loaded
        tok = getattr(processor, "tokenizer", processor)
        clean_enc, noisy_enc = self._vcd_encodings(
            inputs, noise_step=noise_step, noise_seed=noise_seed
        )
        from evalrx.models.paper_methods.vcd import VCDLogitsProcessor

        # Match the release's temperature=1, top_p=1, no-top-k multinomial
        # branch while preventing evaluation-order-dependent randomness.
        fingerprint = hashlib.sha256(
            clean_enc["pixel_values"].detach().float().cpu().numpy().tobytes()
        ).digest()
        item_seed = (int(noise_seed) + int.from_bytes(fingerprint[:8], "little")) % (2**63 - 1)
        processor_list = LogitsProcessorList(
            [VCDLogitsProcessor(model, noisy_enc, alpha=alpha, beta=beta)]
        )
        cuda_devices = [clean_enc["input_ids"].device.index] if clean_enc["input_ids"].is_cuda else []
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            torch.manual_seed(item_seed)
            if clean_enc["input_ids"].is_cuda:
                torch.cuda.manual_seed(item_seed)
            out = model.generate(
                **clean_enc,
                max_new_tokens=self.runtime.max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                use_cache=True,
                logits_processor=processor_list,
            )
        return tok.decode(_new_tokens(out[0], clean_enc["input_ids"][0]), skip_special_tokens=True)


    def generate_vcd_baseline(self, inputs: Any, *, noise_seed: int = 55) -> str:
        """Sample the clean VCD control with the candidate's per-image RNG.

        VCD evaluates both arms with temperature-one multinomial sampling. A
        greedy baseline paired with a sampled contrastive arm is not a valid
        paper-method comparison, so the white-box runner calls this method
        whenever it freezes the VCD candidate.
        """
        logger.debug("%s: generate_vcd_baseline(noise_seed=%s)", self.spec.key, noise_seed)
        if not self.spec.is_vlm:
            raise ValueError("VCD clean control requires a VLM")
        image = getattr(inputs, "image", None)
        if image is None or isinstance(image, (list, tuple)):
            raise ValueError("VCD clean control requires one image")
        import hashlib

        import torch

        model, processor = self._loaded
        clean_enc, _ = self._vcd_encodings(inputs, noise_step=999, noise_seed=noise_seed)
        fingerprint = hashlib.sha256(
            clean_enc["pixel_values"].detach().float().cpu().numpy().tobytes()
        ).digest()
        item_seed = (int(noise_seed) + int.from_bytes(fingerprint[:8], "little")) % (2**63 - 1)
        cuda_devices = [clean_enc["input_ids"].device.index] if clean_enc["input_ids"].is_cuda else []
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            torch.manual_seed(item_seed)
            if clean_enc["input_ids"].is_cuda:
                torch.cuda.manual_seed(item_seed)
            out = model.generate(
                **clean_enc,
                max_new_tokens=self.runtime.max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                use_cache=True,
            )
        tok = getattr(processor, "tokenizer", processor)
        return tok.decode(_new_tokens(out[0], clean_enc["input_ids"][0]), skip_special_tokens=True)


    def generate_instruction_cd(
        self,
        inputs: Any,
        *,
        alpha: float = 1.0,
        beta: float = 0.1,
        qformer_mode: str = "normal",
        disturbance: str = (
            "You are a confused objects detector to provide a fuzzy overview "
            "or impression of the image."
        ),
    ) -> str:
        """ICD for a binary visual-grounding decision.

        ICD (Wang et al., ACL 2024) contrasts a normal forward pass with a
        *disturbance-instruction* pass.  For InstructBLIP, the disturbance
        replaces only ``qformer_input_ids`` while the decoder prompt remains
        intact, matching the released ``normal.json`` route; ``qformer_mode``
        can also run the released disturbed-question variant.  Decoder-only
        VLMs such as Qwen have no Q-Former, so their text-prefix fallback is
        explicitly architecture-adapted.

        As with :meth:`generate_vcd`, this intentionally supports only a
        one-token Yes/No task.  Applying a first-token shortcut to free-form
        generation would not implement ICD's token-by-token sampler.
        """
        logger.debug(
            "%s: generate_instruction_cd(alpha=%s, beta=%s, qformer_mode=%r)",
            self.spec.key, alpha, beta, qformer_mode,
        )
        if not self.spec.is_vlm:
            raise ValueError("instruction contrast requires a VLM")
        image = getattr(inputs, "image", None)
        if image is None or isinstance(image, (list, tuple)):
            raise ValueError("instruction contrast requires one image and a binary answer task")
        model, processor = self._loaded
        tok = getattr(processor, "tokenizer", processor)

        def token_id(answer: str) -> int:
            for spelling in (" " + answer, answer):
                encoded = tok(spelling, add_special_tokens=False)["input_ids"]
                if len(encoded) == 1:
                    return int(encoded[0])
            raise ValueError(
                f"{self.spec.key}: {answer!r} is not one token; ICD binary mode unavailable"
            )

        if self.spec.model_type == "instructblip":
            import torch

            if qformer_mode not in {"normal", "question"}:
                raise ValueError("qformer_mode must be 'normal' or 'question'")
            enc, _, _, _ = self._encode_vlm(inputs, model, processor)
            clean_enc = dict(enc)
            dirty_enc = dict(enc)
            disturbed_qformer_prompt = str(disturbance)
            if qformer_mode == "question":
                disturbed_qformer_prompt += self._as_prompt(inputs)
            qformer = processor.qformer_tokenizer(
                disturbed_qformer_prompt, return_tensors="pt", padding="longest", truncation=True
            ).to(next(model.parameters()).device)
            dirty_enc["qformer_input_ids"] = qformer["input_ids"]
            dirty_enc["qformer_attention_mask"] = qformer["attention_mask"]
            with torch.no_grad():
                clean = model(**clean_enc).logits[0, -1]
                disturbed = model(**dirty_enc).logits[0, -1]
        else:
            clean = self.forward(inputs, capture={Capability.LOGITS}).require(Capability.LOGITS)[-1]
            disturbed_inputs = Inputs(
                prompt=str(disturbance) + self._as_prompt(inputs), image=image
            )
            disturbed = self.forward(disturbed_inputs, capture={Capability.LOGITS}).require(
                Capability.LOGITS
            )[-1]
        ids = {answer: token_id(answer) for answer in ("Yes", "No")}
        clean_scores = {answer: float(clean[token].float()) for answer, token in ids.items()}
        disturbed_scores = {
            answer: float(disturbed[token].float()) for answer, token in ids.items()
        }
        cutoff = max(clean_scores.values()) + math.log(float(beta))
        scores = {
            answer: (1.0 + float(alpha)) * clean_scores[answer]
            - float(alpha) * disturbed_scores[answer]
            for answer in ids
            if clean_scores[answer] >= cutoff
        }
        return max(scores or clean_scores, key=(scores or clean_scores).get)


    def generate_vicrop(self, inputs: Any, *, layer: int | float = 14) -> str:
        """Run the architecture-native LLaVA ViCrop paper executor."""
        logger.debug("%s: generate_vicrop(layer=%s)", self.spec.key, layer)
        if self.spec.model_type != "llava":
            raise ValueError(f"{self.spec.key}: ViCrop is only native on the LLaVA executor")
        from evalrx.models.paper_methods.vicrop import generate

        return generate(self, inputs, layer=layer)


    def generate_opera_binary(
        self,
        inputs: Any,
        *,
        num_attn_candidates: int = 5,
        penalty_weight: float = 1.0,
    ) -> str:
        """Run OPERA's first-token over-trust penalty for a binary VQA task.

        OPERA scores each likely continuation by the image attention of the
        *candidate token* and subtracts ``-image_attention`` from its logit.
        This is its published early-response penalty verbatim. POPE evaluates
        a single Yes/No token, therefore the later multi-token rollback branch
        is intentionally out of scope and this method must stay labelled a
        binary specialization.
        """
        import torch

        logger.debug(
            "%s: generate_opera_binary(num_attn_candidates=%s, penalty_weight=%s)",
            self.spec.key, num_attn_candidates, penalty_weight,
        )
        if self.spec.model_type != "llava":
            raise ValueError(f"{self.spec.key}: OPERA binary route is only native on LLaVA")
        if int(num_attn_candidates) < 1:
            raise ValueError("num_attn_candidates must be positive")
        model, processor = self._loaded
        enc, _, _, _ = self._encode_vlm(inputs, model, processor)
        enc.pop("token_type_ids", None)
        image_id = getattr(model.config, "image_token_index", None)
        if image_id is None:
            raise ValueError(f"{self.spec.key}: OPERA requires config.image_token_index")
        image_positions = (enc["input_ids"][0] == int(image_id)).nonzero().flatten()
        if image_positions.numel() == 0 or not bool(
            (image_positions[1:] == image_positions[:-1] + 1).all()
        ):
            raise ValueError(f"{self.spec.key}: OPERA requires one contiguous image-token span")

        with torch.no_grad():
            prefill = model(**enc, return_dict=True, output_attentions=True, use_cache=False)
        if not getattr(prefill, "attentions", None):
            raise ValueError(f"{self.spec.key}: OPERA requires eager self-attention outputs")
        raw_logits = prefill.logits[:, -1, :]
        k = min(int(num_attn_candidates), int(raw_logits.shape[-1]))
        candidate_scores, candidate_tokens = torch.topk(raw_logits, k, dim=-1, largest=True, sorted=True)
        adjusted_scores = candidate_scores.clone()
        for candidate_index in range(k):
            candidate_enc = dict(enc)
            candidate_enc["input_ids"] = torch.cat(
                (enc["input_ids"], candidate_tokens[:, candidate_index : candidate_index + 1]), dim=1
            )
            if "attention_mask" in candidate_enc:
                candidate_enc["attention_mask"] = torch.cat(
                    (enc["attention_mask"], torch.ones_like(enc["attention_mask"][:, :1])), dim=1
                )
            with torch.no_grad():
                candidate_output = model(
                    **candidate_enc, return_dict=True, output_attentions=True, use_cache=False
                )
            attentions = getattr(candidate_output, "attentions", None)
            if not attentions:
                raise ValueError(f"{self.spec.key}: OPERA candidate forward returned no attentions")
            # Reference OPERA maximises heads then sums the candidate's image
            # attention. With one beam, selecting the adjusted top candidate
            # is identical to its first beam-search step.
            last_attention = attentions[-1].amax(dim=1)[:, -1, :]
            image_attention = last_attention[:, image_positions].sum(dim=-1)
            adjusted_scores[:, candidate_index] += float(penalty_weight) * image_attention
        selected = candidate_tokens.gather(1, adjusted_scores.argmax(dim=-1, keepdim=True)).squeeze(1)
        tok = getattr(processor, "tokenizer", processor)
        return tok.decode([int(selected[0])], skip_special_tokens=True)


    def generate_ifcd(
        self,
        inputs: Any,
        *,
        alpha: float = 0.1,
        beta: float = 0.1,
        edit_strength: float = 0.5,
        top_layers: int = 15,
        max_new_tokens: int | None = None,
    ) -> str:
        """Run an explicitly adapted TruthX-backed IFCD decoder on LLaVA.

        The checkpoint path is a required runtime artifact, rather than an
        implicit download: its provenance controls whether an experiment can
        compare itself with IFCD's MSCOCO-trained editor.  This adapter uses
        modern HF output hooks, so even a matching artifact remains labelled
        adapted until its pre-``o_proj`` boundary is ported.
        """
        import torch
        from transformers.generation.logits_process import LogitsProcessorList

        logger.debug(
            "%s: generate_ifcd(alpha=%s, beta=%s, edit_strength=%s, top_layers=%s)",
            self.spec.key, alpha, beta, edit_strength, top_layers,
        )
        if self.spec.model_type != "llava":
            raise ValueError(f"{self.spec.key}: IFCD is only wired for LLaVA's Vicuna decoder")
        checkpoint = self.runtime.engine_kwargs.get("ifcd_checkpoint")
        if not checkpoint or not Path(str(checkpoint)).is_file():
            raise ValueError("IFCD requires runtime.engine_kwargs['ifcd_checkpoint']")
        model, processor = self._loaded
        enc, _, _, _ = self._encode_vlm(inputs, model, processor)
        enc.pop("token_type_ids", None)
        key = f"{Path(str(checkpoint)).resolve()}:{int(top_layers)}"
        editor = self._ifcd_editors.get(key)
        if editor is None:
            from evalrx.models.paper_methods.ifcd import TruthXEditor

            hidden_size = int(getattr(model.config.text_config, "hidden_size", 0) or model.config.hidden_size)
            editor = TruthXEditor(checkpoint, hidden_size=hidden_size, top_layers=top_layers)
            self._ifcd_editors[key] = editor
        from evalrx.models.paper_methods.ifcd import IFCDLogitsProcessor, truthx_editing

        language_model = getattr(model, "language_model", model)
        editor.strength = float(edit_strength)
        processor_list = LogitsProcessorList(
            [
                IFCDLogitsProcessor(
                    model,
                    dict(enc),
                    editor,
                    alpha=alpha,
                    beta=beta,
                    edit_strength=edit_strength,
                )
            ]
        )
        with truthx_editing(language_model, editor), torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens or self.runtime.max_new_tokens,
                do_sample=False,
                logits_processor=processor_list,
            )
        tok = getattr(processor, "tokenizer", processor)
        return tok.decode(_new_tokens(out[0], enc["input_ids"][0]), skip_special_tokens=True)


    def generate_tcd(
        self,
        inputs: Any,
        *,
        max_new_tokens: int | None = None,
        hyperparams: "Any | None" = None,
    ) -> str:
        """Run Temporal Contrastive Decoding (Li et al. 2026, arXiv:2604.15383).

        Cannot be built as a ``model.generate(logits_processor=[...])`` bolt-on
        the way VCD/IFCD/PAI are: a :class:`~transformers.LogitsProcessor` only
        ever sees ``(input_ids, scores)``, never the forward pass's attention
        weights, and TCD's gate (Eq. 8) needs the CURRENT step's decoder
        attention to audio tokens. So this runs its own greedy decode loop,
        holding two KV caches (original audio / Hann-blurred slow-path audio,
        Eq. 1) and calling the model directly each step -- exactly what VCD's
        processor already does for its single contrastive branch, just applied
        to both branches here, with ``output_attentions=True`` on the original
        branch to get the audio-attention ratio for free from the same forward
        (matches the paper's own reported ~1.00x decode-step overhead, Table 7:
        the attention weights are already computed internally, not an extra
        pass).

        The per-example blur window and update scale (Eq. 5-6) are derived
        from a stability score (Eq. 2-4) computed ONCE before decoding starts,
        from (a) the audio encoder's own per-layer hidden-state trajectory on
        the *unblurred* audio, and (b) the decoder's per-layer attention to
        audio tokens during the prefill -- see
        :func:`evalrx.models.paper_methods.tcd.aggregate_stability` for
        why those two must have equal layer counts to be faithful, and
        ``paper_method_fidelity("tcd")`` for which registered specs qualify.
        """
        from evalrx.core.case import Inputs
        from evalrx.models.paper_methods import tcd

        hp = hyperparams or tcd.TCDHyperparams()
        logger.debug(
            "%s: generate_tcd(l_attn=%s, tau=%s, gamma_gate=%s)",
            self.spec.key, hp.l_attn, hp.tau, hp.gamma_gate,
        )
        if self.spec.audio is None:
            raise ValueError(f"{self.spec.key}: TCD requires an audio-capable spec")
        audio = getattr(inputs, "audio", None) if isinstance(inputs, Inputs) else None
        if audio is None or isinstance(audio, (list, tuple)):
            raise ValueError("TCD requires exactly one audio clip")

        import torch

        model, processor = self._loaded
        tok = getattr(processor, "tokenizer", processor)
        waveform = _resolve_audio(audio)
        original_inputs = Inputs(
            prompt=self._as_prompt(inputs),
            image=getattr(inputs, "image", None),
            audio=waveform,
            video=getattr(inputs, "video", None),
        )
        enc, ids, _tokens, _ttm = self._encode_vlm(original_inputs, model, processor)
        enc.pop("token_type_ids", None)
        audio_token_id = _read_nested_attr(
            model.config, self.spec.audio.audio_token_id_attr, default=None
        )
        if audio_token_id is None:
            raise ValueError(f"{self.spec.key}: could not resolve the audio-token id from config")
        audio_mask = torch.tensor(ids, device=enc["input_ids"].device) == int(audio_token_id)
        if not bool(audio_mask.any()):
            raise ValueError(f"{self.spec.key}: no audio tokens found in the encoded prompt")

        audio_tower = _read_nested_attr(model, self.spec.audio.audio_tower, default=None)
        if audio_tower is None:
            raise ValueError(
                f"{self.spec.key}: could not resolve audio_tower={self.spec.audio.audio_tower!r}"
            )

        # -- Eq. 2-3: encoder-side per-layer stability, on the UNBLURRED audio --
        with torch.no_grad():
            audio_out = audio_tower(
                enc["input_features"], output_hidden_states=True, return_dict=True
            )
        # hidden_states[0] is pre-layer-0 embeddings, not a layer's own output.
        # Verified against Qwen2AudioEncoder.forward: entries 1..N-1 are the
        # RAW pre-pool output of layers 0..N-2 (seq_len == max_source_positions,
        # e.g. 1500), but the LAST entry (N) is layer N-1's output after
        # avg_pooler + layer_norm have already run -- half the seq_len and a
        # LayerNorm-pinned scale, not a like-for-like continuation of the rest.
        # layer_stability's M_l/F_l are computed independently per entry (no
        # cross-layer diffs), so this doesn't break Eq. 2-3, but it does mean
        # the LAST layer's S_l sits on a different footing before Eq. 4
        # softmax-weights it back in -- an approximation, not a bug, and one
        # this codebase's convention is to say plainly rather than round off.
        encoder_states = [h[0].float() for h in audio_out.hidden_states[1:]]
        layer_stability_scores = tcd.layer_stability(encoder_states, eps=hp.eps)

        # -- prefill: original branch (also seeds its KV cache) --
        with torch.no_grad():
            prefill = model(**enc, use_cache=True, output_attentions=True, return_dict=True)
        if not getattr(prefill, "attentions", None):
            raise ValueError(f"{self.spec.key}: TCD requires eager self-attention outputs")

        # -- Eq. 4: aggregate stability, weighted by the decoder's per-layer
        # audio-attention ratio from that same prefill. See
        # paper_method_fidelity("tcd") for the equal-layer-count requirement
        # this truncation is standing in for on a mismatched-depth spec.
        # Indexed straight off prefill.attentions (still bf16) one layer at a
        # time -- audio_attention_ratio does its own float() per call, so this
        # never holds more than one layer's fp32 copy at once. A real MMAU clip
        # is ~780 tokens; materializing all 32 layers' (heads, 780, 780) fp32
        # attentions up front, as an earlier version of this method did, would
        # be several GB of copies purely for a scalar-per-layer reduction. --
        n_layers = min(layer_stability_scores.shape[0], len(prefill.attentions))
        layer_ratio = torch.stack(
            [
                tcd.audio_attention_ratio(prefill.attentions[i][0], audio_mask)
                for i in range(n_layers)
            ]
        )
        stability = tcd.aggregate_stability(
            layer_stability_scores[:n_layers], layer_ratio, temperature=hp.tau
        )
        window_ms, lam = tcd.adaptive_blur_params(stability, hp)
        logger.debug(
            "%s: generate_tcd stability=%.4f window_ms=%.2f lam=%.4f",
            self.spec.key, stability, window_ms, lam,
        )

        # -- Eq. 1: blur + re-encode (once, up front -- not per decode step) --
        blurred_waveform = tcd.hann_blur_waveform(waveform, AUDIO_SAMPLE_RATE, window_ms)
        blurred_inputs = Inputs(
            prompt=original_inputs.prompt, image=original_inputs.image,
            audio=blurred_waveform, video=original_inputs.video,
        )
        blurred_enc, _, _, _ = self._encode_vlm(blurred_inputs, model, processor)
        blurred_enc.pop("token_type_ids", None)
        with torch.no_grad():
            blurred_prefill = model(**blurred_enc, use_cache=True, return_dict=True)

        eos_ids = set()
        if getattr(tok, "eos_token_id", None) is not None:
            eos_ids.add(int(tok.eos_token_id))
        gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        if isinstance(gen_eos, (list, tuple, set)):
            eos_ids.update(int(e) for e in gen_eos)
        elif gen_eos is not None:
            eos_ids.add(int(gen_eos))

        max_new = int(max_new_tokens or self.runtime.max_new_tokens)
        original_kv = prefill.past_key_values
        blurred_kv = blurred_prefill.past_key_values
        z = prefill.logits[0, -1].float()
        z_tilde = blurred_prefill.logits[0, -1].float()
        last_layers_attn = [a[0].float() for a in prefill.attentions[-hp.l_attn :]]
        mask = audio_mask.clone()
        generated: list[int] = []

        with torch.no_grad():
            for _ in range(max_new):
                r_t = float(
                    torch.stack(
                        [tcd.audio_attention_ratio(a, mask) for a in last_layers_attn]
                    ).mean()
                )
                entropy_hat = tcd.topk_renormalized_entropy(z, hp.k_ent)
                gate_value = tcd.reliance_gate(r_t, entropy_hat, hp)
                fused = tcd.fuse_logits(z, z_tilde, lam=lam, gate_value=gate_value, hp=hp)
                next_id = int(torch.argmax(fused))
                if next_id in eos_ids:
                    break
                generated.append(next_id)

                next_input = torch.tensor([[next_id]], device=z.device)
                out = model(
                    input_ids=next_input, past_key_values=original_kv,
                    use_cache=True, output_attentions=True, return_dict=True,
                )
                out_tilde = model(
                    input_ids=next_input, past_key_values=blurred_kv,
                    use_cache=True, return_dict=True,
                )
                original_kv = out.past_key_values
                blurred_kv = out_tilde.past_key_values
                z = out.logits[0, -1].float()
                z_tilde = out_tilde.logits[0, -1].float()
                last_layers_attn = [a[0].float() for a in out.attentions[-hp.l_attn :]]
                mask = torch.cat([mask, torch.zeros(1, dtype=torch.bool, device=mask.device)])

        return tok.decode(generated, skip_special_tokens=True)


    def generate_tcd_baseline(self, inputs: Any, *, max_new_tokens: int | None = None) -> str:
        """Greedy baseline paired with :meth:`generate_tcd`.

        ``generate_tcd`` is greedy by construction (Eq. 9's fused logits feed
        a plain argmax, no sampler). ``Qwen2-Audio-7B-Instruct``'s own
        ``generation_config`` defaults to ``do_sample=True`` (temperature 0.7,
        top_p 0.5, top_k 20) -- calling the bare :meth:`generate` for a
        baseline would inherit that and silently pair a sampled arm against a
        greedy one, exactly the mismatch ``generate_vcd_baseline`` exists to
        avoid for VCD (see its docstring). The paper's own baseline is
        greedy too (Table 7's "Baseline (Greedy)"; Section 4.1).
        """
        logger.debug("%s: generate_tcd_baseline()", self.spec.key)
        if self.spec.audio is None:
            raise ValueError(f"{self.spec.key}: TCD requires an audio-capable spec")
        return self.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)


    def generate_vicrop_consensus(
        self,
        inputs: Any,
        *,
        baseline_answer: str,
        layer: int | float = 14,
    ) -> str:
        """Use ViCrop only when independent crop and fused-view answers agree.

        This is a deployment safety guard, not part of the source ViCrop
        method. It prevents a single misleading attention crop from replacing
        an otherwise stable baseline answer.
        """
        import re

        logger.debug("%s: generate_vicrop_consensus(layer=%s)", self.spec.key, layer)
        if self.spec.model_type != "llava":
            raise ValueError(f"{self.spec.key}: ViCrop is only native on the LLaVA executor")
        from evalrx.models.paper_methods.vicrop import prepare_views

        image, crop, fused_prompt = prepare_views(self, inputs, layer=layer)
        fused = self.generate(Inputs(fused_prompt, [image, crop]))
        crop_only_prompt = (
            "The image is a task-relative crop selected from a larger scene. "
            "Answer only from visible evidence in this crop.\n\n"
            + self._as_prompt(inputs)
        )
        crop_only = self.generate(Inputs(crop_only_prompt, crop))

        def decision(text: str) -> str:
            lowered = str(text).lower()
            yes_no = re.search(r"\b(yes|no)\b", lowered)
            if yes_no:
                return yes_no.group(1)
            choices = re.findall(r"\b([a-d])\b", lowered)
            if choices:
                return choices[-1]
            return re.sub(r"\W+", "", lowered)

        return fused if decision(fused) and decision(fused) == decision(crop_only) else baseline_answer


    def generate_pai(
        self,
        inputs: Any,
        *,
        alpha: float = 0.2,
        guidance_scale: float = 2.0,
        start_layer: int = 2,
        end_layer: int = 32,
        max_new_tokens: int | None = None,
    ) -> str:
        """Run PAI's attention and classifier-free-guidance route on LLaVA.

        The attention boost and the image-free classifier-free-guidance cache
        follow the released PAI decoding path.  It remains an architecture
        specialization because the source uses its pinned LLaVA stack.
        """
        import torch

        logger.debug(
            "%s: generate_pai(alpha=%s, guidance_scale=%s, start_layer=%s, end_layer=%s)",
            self.spec.key, alpha, guidance_scale, start_layer, end_layer,
        )
        if self.spec.model_type != "llava":
            raise ValueError(f"{self.spec.key}: PAI is only native on the LLaVA executor")
        model, processor = self._loaded
        enc, _, _, _ = self._encode_vlm(inputs, model, processor)
        enc.pop("token_type_ids", None)
        image_id = getattr(model.config, "image_token_index", None)
        if image_id is None:
            raise ValueError(f"{self.spec.key}: PAI requires config.image_token_index")
        positions = (enc["input_ids"][0] == int(image_id)).nonzero().flatten()
        if positions.numel() == 0 or not bool((positions[1:] == positions[:-1] + 1).all()):
            raise ValueError(f"{self.spec.key}: PAI requires one contiguous image-token span")
        from transformers.generation.logits_process import LogitsProcessorList

        from evalrx.models.paper_methods.pai import (
            PAICFGLogitsProcessor,
            image_attention_boost,
        )

        # PAI's reference code patches the LLaMA language model, whereas HF
        # LLaVA wraps it in ``LlavaForConditionalGeneration``.
        language_model = getattr(model, "language_model", model)
        unconditional_ids = torch.cat(
            (enc["input_ids"][:, : positions[0]], enc["input_ids"][:, positions[-1] + 1 :]),
            dim=1,
        )
        cfg = PAICFGLogitsProcessor(
            model,
            unconditional_ids,
            guidance_scale=guidance_scale,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        with image_attention_boost(
            language_model,
            image_start=int(positions[0]),
            image_end=int(positions[-1]) + 1,
            alpha=alpha,
            start_layer=start_layer,
            end_layer=end_layer,
        ):
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens or self.runtime.max_new_tokens,
                    do_sample=False,
                    logits_processor=LogitsProcessorList([cfg]),
                )
        tok = getattr(processor, "tokenizer", processor)
        return tok.decode(_new_tokens(out[0], enc["input_ids"][0]), skip_special_tokens=True)
