"""L2 calibrated specialists for the ``hf_local`` backend: a second model (an
open-vocabulary detector, CLAP, or a hosted vision / audio API) grounds or
replaces the answer on the question types it is calibrated for.

:class:`SpecialistRepairsMixin` is mixed into
:class:`~evalrx.models.backends.hf.model.HFLocalModel`. The specialists only
call the model through ``generate``, so they are backend-independent in
substance; they live here until another backend needs them.
"""

from __future__ import annotations

import logging
from typing import Any

from evalrx.core.case import Inputs
from evalrx.models._media import resolve_audio as _resolve_audio
from evalrx.models._media import resolve_image as _resolve_image
from evalrx.models.backends.base import RuntimeConfig
from evalrx.models.backends.hf._util import (  # noqa: F401
    _check_audio_duration,
    _collect_message_images,
    _new_tokens,
    _populate_audio_extras,
    _populate_vision_extras,
    _read_nested_attr,
)

logger = logging.getLogger(__name__)


class SpecialistRepairsMixin:
    """L2 calibrated specialist repairs (see the module docstring)."""

    def generate_detector_grounded_presence(
        self,
        inputs: Any,
        *,
        baseline_answer: str,
        objects: list[str] | tuple[str, ...],
        detector_threshold: float = 0.25,
        detector_engine: Any | None = None,
    ) -> str:
        """Override a negative binary answer only with calibrated detector evidence.

        The object allowlist and score threshold are explicit experiment payload,
        not inferred from the case label or metadata. Grounding DINO is queried
        only when the baseline is negative and the prompt names an allowlisted
        object, keeping the intervention narrow and deterministic.
        """
        import re

        if re.search(r"\byes\b", str(baseline_answer), flags=re.IGNORECASE):
            return baseline_answer
        if not re.search(r"\bno\b", str(baseline_answer), flags=re.IGNORECASE):
            return baseline_answer

        prompt = self._as_prompt(inputs)
        match = re.search(
            r"\b(?:is|are) there (?:an?\s+|any\s+)?(.+?)\s+in (?:the|this) image\b",
            prompt,
            flags=re.IGNORECASE,
        )
        if match is None:
            return baseline_answer
        query = re.sub(r"\s+", " ", match.group(1).strip().lower())
        allowed = {str(name).strip().lower() for name in objects}
        if query not in allowed:
            return baseline_answer

        image = getattr(inputs, "image", None)
        if image is None or isinstance(image, (list, tuple)):
            return baseline_answer
        engine = detector_engine
        if engine is None:
            model, _ = self._loaded
            device = str(next(model.parameters()).device)
            engine = self._detect_engines.get(device)
            if engine is None:
                from evalrx.models.tools.perception import default_detect_engine

                engine = default_detect_engine(
                    device=device, threshold=0.01, text_threshold=0.01
                )
                self._detect_engines[device] = engine
        detections = engine(_resolve_image(image), query)
        score = max((float(item.get("score", 0.0)) for item in detections), default=0.0)
        return "Yes" if score >= float(detector_threshold) else baseline_answer


    def generate_clap_grounded_presence(
        self,
        inputs: Any,
        *,
        baseline_answer: str,
        negative_threshold: float = -0.05,
        positive_threshold: float = 0.275,
        model_id: str = "laion/clap-htsat-unfused",
        similarity_engine: Any | None = None,
    ) -> str:
        """Gate binary sound-presence answers with calibrated CLAP evidence."""
        import re

        baseline = str(baseline_answer)
        is_yes = re.search(r"\byes\b", baseline, flags=re.IGNORECASE) is not None
        is_no = re.search(r"\bno\b", baseline, flags=re.IGNORECASE) is not None
        if not (is_yes or is_no):
            return baseline_answer
        match = re.search(
            r"\bsound of (?:an?\s+)?(.+?)(?:\s+in (?:the|this) audio|\?)",
            self._as_prompt(inputs),
            flags=re.IGNORECASE,
        )
        if match is None:
            return baseline_answer
        query = re.sub(r"\s+", " ", match.group(1).strip().lower())
        audio = getattr(inputs, "audio", None)
        if audio is None or isinstance(audio, (list, tuple)):
            return baseline_answer

        engine = similarity_engine
        if engine is None:
            model, _ = self._loaded
            device = str(next(model.parameters()).device)
            key = (str(model_id), device)
            engine = self._audio_text_engines.get(key)
            if engine is None:
                import torch
                from scipy.signal import resample_poly
                from transformers import ClapModel, ClapProcessor

                processor = ClapProcessor.from_pretrained(model_id)
                clap = ClapModel.from_pretrained(model_id).to(device).eval()

                def engine(waveform: Any, text: str) -> float:
                    samples = resample_poly(waveform, 3, 1).astype("float32")
                    encoded = processor(
                        text=["the sound of " + text], audio=[samples],
                        sampling_rate=48000, return_tensors="pt", padding=True,
                    )
                    encoded = {name: value.to(device) for name, value in encoded.items()}
                    with torch.no_grad():
                        audio_features = clap.get_audio_features(
                            input_features=encoded["input_features"],
                            is_longer=encoded.get("is_longer"),
                        ).pooler_output
                        text_features = clap.get_text_features(
                            input_ids=encoded["input_ids"],
                            attention_mask=encoded["attention_mask"],
                        ).pooler_output
                    return float(torch.nn.functional.cosine_similarity(
                        audio_features, text_features
                    )[0])

                self._audio_text_engines[key] = engine
        score = float(engine(_resolve_audio(audio), query))
        if is_yes and score < float(negative_threshold):
            return "No"
        if is_no and score >= float(positive_threshold):
            return "Yes"
        return baseline_answer


    def generate_audio_api_specialist(
        self,
        inputs: Any,
        *,
        baseline_answer: str,
        model_id: str = "gemini-3.7-flash",
        allowed_disagreements: list[str] | tuple[str, ...] | None = None,
        allowed_disagreements_by_route: dict[str, list[str]] | None = None,
        specialist_engine: Any | None = None,
    ) -> str:
        """Answer an audio question with a frozen external multimodal specialist."""
        import re

        prompt = self._as_prompt(inputs)
        audio = getattr(inputs, "audio", None)
        if audio is None or isinstance(audio, (list, tuple)):
            return baseline_answer
        engine = specialist_engine or self._audio_api_specialist_engines.get(str(model_id))
        if engine is None:
            from evalrx.models.backends.api.gemini_model import GeminiModel

            specialist = GeminiModel(model_id=str(model_id))

            def engine(specialist_inputs: Inputs) -> str:
                turn = specialist.chat(
                    [{
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": specialist_inputs.audio},
                            {"type": "text", "text": specialist_inputs.prompt},
                        ],
                    }]
                )
                return turn.text

            self._audio_api_specialist_engines[str(model_id)] = engine
        answer = str(engine(Inputs(prompt=prompt, audio=audio)))
        route: str | None = None
        if allowed_disagreements_by_route is not None:
            question = prompt.split("\n", 1)[0]
            if re.search(
                r"\b(?:conversation|speakers?|utterance|words?|sentence|phoneme|"
                r"sarcas\w*|dialogue|people talking|emotional?\s+state|emotion of)\b",
                question,
                flags=re.IGNORECASE,
            ):
                route = "speech"
            elif re.search(
                r"\b(?:music|musical|melody|chord|guitar|drum|bass|synth|instrument|"
                r"rhythm|tempo|key signature|tonic|chant|singer|vocal|genre|harmonica|"
                r"percussive|song|vocals|time signature|audio quality|sound texture|"
                r"texture of the sound|e-guitar)\b",
                question,
                flags=re.IGNORECASE,
            ):
                route = "music"
            else:
                route = "sound"
            allowed_disagreements = allowed_disagreements_by_route.get(route, [])
        if allowed_disagreements is not None:
            baseline_letters = re.findall(r"\b([A-D])\b", str(baseline_answer).upper())
            specialist_letters = re.findall(r"\b([A-D])\b", answer.upper())
            if not baseline_letters or not specialist_letters:
                return baseline_answer
            pair = baseline_letters[0] + specialist_letters[0]
            if pair not in {str(item).upper() for item in allowed_disagreements}:
                return baseline_answer
        return answer


    def generate_noncolor_spatial_specialist(
        self,
        inputs: Any,
        *,
        baseline_answer: str,
        model_id: str = "qwen2.5-vl-7b-instruct",
        specialist_engine: Any | None = None,
    ) -> str:
        """Route non-color spatial questions to a frozen vision specialist."""
        prompt = self._as_prompt(inputs)
        lowered = prompt.lower()
        if "shape" not in lowered and "color" in lowered:
            return baseline_answer
        image = getattr(inputs, "image", None)
        if image is None or isinstance(image, (list, tuple)):
            return baseline_answer

        engine = specialist_engine or self._vision_specialist_engines.get(str(model_id))
        if engine is None:
            import gc

            import torch

            from evalrx.specs import get_spec

            subject, _ = self._loaded
            device = str(next(subject.parameters()).device)
            subject.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            from evalrx.models.backends.hf.model import HFLocalModel

            specialist = HFLocalModel(
                get_spec(model_id),
                RuntimeConfig(
                    device=device,
                    dtype=self.runtime.dtype,
                    attn_impl=self.runtime.attn_impl,
                    max_new_tokens=self.runtime.max_new_tokens,
                    apply_chat_template=True,
                ),
            )
            specialist.load()
            engine = specialist.generate
            self._vision_specialist_engines[str(model_id)] = engine
        return str(engine(Inputs(prompt=prompt, image=image)))


    def generate_chart_vision_specialist(
        self,
        inputs: Any,
        *,
        baseline_answer: str,
        model_id: str = "qwen2.5-vl-7b-instruct",
        specialist_engine: Any | None = None,
    ) -> str:
        """Answer a chart question with a frozen, independently loaded VLM."""
        prompt = self._as_prompt(inputs)
        image = getattr(inputs, "image", None)
        if image is None or isinstance(image, (list, tuple)):
            return baseline_answer
        engine = specialist_engine or self._vision_specialist_engines.get(str(model_id))
        if engine is None:
            import gc

            import torch

            from evalrx.specs import get_spec

            subject, _ = self._loaded
            device = str(next(subject.parameters()).device)
            subject.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            from evalrx.models.backends.hf.model import HFLocalModel

            specialist = HFLocalModel(
                get_spec(model_id),
                RuntimeConfig(
                    device=device,
                    dtype=self.runtime.dtype,
                    attn_impl=self.runtime.attn_impl,
                    max_new_tokens=self.runtime.max_new_tokens,
                    apply_chat_template=True,
                ),
            )
            specialist.load()
            engine = specialist.generate
            self._vision_specialist_engines[str(model_id)] = engine
        return str(engine(Inputs(prompt=prompt, image=image)))


    def generate_vision_api_specialist(
        self,
        inputs: Any,
        *,
        baseline_answer: str,
        model_id: str = "gemini-3.7-flash",
        specialist_engine: Any | None = None,
    ) -> str:
        """Answer an image question with a frozen external multimodal specialist."""
        import re

        prompt = self._as_prompt(inputs)
        image = getattr(inputs, "image", None)
        if image is None or isinstance(image, (list, tuple)):
            return baseline_answer
        engine = specialist_engine or self._vision_api_specialist_engines.get(str(model_id))
        if engine is None:
            from evalrx.models.backends.api.gemini_model import GeminiModel

            specialist = GeminiModel(model_id=str(model_id))
            engine = specialist.generate
            self._vision_api_specialist_engines[str(model_id)] = engine
        strict_prompt = prompt + (
            "\nReturn only the final short answer with no explanation, units, or formatting."
        )
        answer = str(engine(Inputs(prompt=strict_prompt, image=image))).strip()
        ratio = re.fullmatch(r"\s*([-+]?\d+(?:\.\d+)?)\s*:\s*([-+]?\d+(?:\.\d+)?)\s*", answer)
        if ratio is not None and float(ratio.group(2)) != 0:
            return str(float(ratio.group(1)) / float(ratio.group(2)))
        return answer
