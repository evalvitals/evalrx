"""Plain-language descriptions of repair candidates.

One home for the plain-language wording, shared by the fix agent, the
dynamic report and the contract emitter instead of each reaching into the
agent's private names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evalrx.eval_agent.stages.fix_agent import FixCandidate


#: What each of the host's own repairs does, in one sentence, for a reader who
#: has never heard of this model or this field.  It lives here rather than in
#: the frontend because only this module knows what these candidates actually
#: do; a consumer handed ``vcd_diffusion_noise`` can title-case it and nothing
#: more, which produces a label that looks explained and is not.
#:
#: Judge-invented candidates are not in this table -- they carry their own
#: sentence from ``what_it_does`` in the proposal.  A name in neither place
#: resolves to "" and is rendered blank, on purpose.
BUILTIN_DESCRIPTIONS: "dict[str, str]" = {
    "aad_silence_contrast":
        "Compares the model's answer with what it says when the sound is replaced by "
        "silence, and keeps only the part the sound itself explains.",
    "aad_silence_contrast_gated_false_yes":
        "Compares the answer with what the model says when the sound is replaced by "
        "silence, applied only where it said yes to something the audio does not support.",
    "annotate_horizontal_band_count":
        "Counts the bars in the chart and writes that number onto the picture before asking.",
    "answer_bbox_crop":
        "Crops the picture down to the region the answer is about before asking.",
    "assertive_grounding":
        "Tells the model to answer from what is actually in the picture rather than from "
        "what it expects to be there.",
    "attend_carefully":
        "Adds an instruction to examine the input closely before answering.",
    "chain_of_verification":
        "Has the model draft an answer, then check it with its own follow-up questions "
        "before committing.",
    "coded_pipeline":
        "Runs a short program the agent wrote that re-asks the model several different "
        "ways and keeps the answer those attempts agree on.",
    "detector_visual_search_consensus":
        "Uses an object detector to pick regions to look at, asks about each one, and "
        "keeps the answer the regions agree on.",
    "finetune_recipe":
        "Writes down a retraining plan for a person to run later. Nothing is changed "
        "automatically.",
    "guided_visual_search_consensus":
        "Uses the question to choose which parts of the picture to zoom into, then keeps "
        "the answer those views agree on.",
    "icd_instruction_disturbance":
        "Compares the answer with what the model says under a deliberately misleading "
        "instruction, and discounts whatever the misleading version produced too.",
    "icd_instruction_disturbance_gated_false_yes":
        "Discounts whatever a deliberately misleading instruction also produced, applied "
        "only where the model said yes to something the picture does not support.",
    "icd_instruction_disturbance_question":
        "Attaches a deliberately misleading instruction to the question itself, and "
        "discounts whatever the model says in both versions.",
    "ifcd_truthx_contrast":
        "Nudges the model's internal state toward the pattern it shows when it is being "
        "truthful, and compares that with the untouched run.",
    "increase_max_tokens":
        "Gives the model more room to write, so answers are not cut off part-way.",
    "least_to_most":
        "Breaks the question into smaller steps and has the model work through them in order.",
    "opera_overtrust_binary":
        "Stops the model leaning too hard on a few words it has already written when it "
        "makes a yes-or-no call.",
    "pai_image_attention":
        "Makes the model weigh the picture more heavily and its own prior expectations less.",
    "salient_crop":
        "Crops the picture to its most eye-catching region before asking.",
    "self_consistency_5":
        "Asks the same question five times and keeps the answer the model gives most often.",
    "self_refine":
        "Has the model criticise its own first answer and then rewrite it.",
    "separate_horizontal_bands":
        "Splits the chart into separate bars so each one can be read on its own.",
    "tcd_temporal_blur":
        "Compares the answer with what the model says when the timing in the clip is "
        "smeared out, and keeps the part real timing explains.",
    "upscale_sharpen":
        "Enlarges and sharpens the picture before asking.",
    "vcd_diffusion_noise":
        "Compares the answer with what the model says when the picture is replaced by "
        "noise, and discounts whatever it would have said without seeing anything.",
    "vcd_diffusion_noise_gated_false_yes":
        "Discounts whatever the model would say without seeing the picture, applied only "
        "where it said yes to something the picture does not support.",
    "vicrop_consensus_guard":
        "Zooms into the region the model was already looking at, and changes the answer "
        "only when the zoomed views agree.",
    "vicrop_relative_attention":
        "Finds the region the model was already looking at and zooms into it before asking again.",
    "visual_grounding":
        "Tells the model to read the answer off the picture first and to fall back on "
        "general knowledge only after that.",
    "zoom_equalize":
        "Zooms in and evens out the brightness so faint details become visible.",
}


def judge_description(proposal: "Any") -> str:
    """The proposal's own ``what_it_does``, cleaned, or "".

    Judges that ignore the field, or answer it with the snake_case name in
    title case, contribute nothing a reader could not already see -- both come
    back empty rather than as a sentence that only looks like one.
    """
    if not isinstance(proposal, dict):
        return ""
    text = " ".join(str(proposal.get("what_it_does", "") or "").split()).strip()
    if not text:
        return ""
    name = str(proposal.get("name", "") or "").strip()
    if name and text.lower().rstrip(".") == name.replace("_", " ").lower():
        return ""
    return text[:300]


def code_description(code: str) -> str:
    """The ``# WHAT_IT_DOES:`` header the codegen prompt asks for.

    The prompt asks for one line and shows a one-line example, but a model that
    wraps it anyway should not lose the second half of its own sentence, so
    immediately following comment lines are folded in until the code starts.
    """
    lines = code.splitlines()[:12]
    for index, line in enumerate(lines):
        if not line.strip().upper().startswith("# WHAT_IT_DOES:"):
            continue
        parts = [line.strip().split(":", 1)[1]]
        for follow in lines[index + 1:]:
            stripped = follow.strip()
            if not stripped.startswith("#"):
                break
            body = stripped.lstrip("#").strip()
            if not body or body.upper().startswith("WHAT_IT_DOES"):
                break
            parts.append(body)
        return " ".join(" ".join(parts).split()).strip()[:300]
    return ""


def plain_description(candidate: "FixCandidate") -> str:
    """One sentence for a reader, or "" when there is honestly nothing to say.

    Never falls back to the slug.  ``audio_evidence_then_answer`` rendered as
    "Audio Evidence Then Answer" reads like an explanation the run never
    produced, and a blank is the more honest signal that it did not.
    """
    described = (candidate.description or "").strip()
    return described or BUILTIN_DESCRIPTIONS.get(candidate.name, "")
