"""ask_judge: typed output, validation retry with feedback, loud failure."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from evalrx.agent_runtime.judge_call import (
    JudgeCallError,
    JudgeOutputError,
    ask_judge,
    first_json_value,
)


class Pick(BaseModel):
    tool: Literal["probe", "stats"]
    rationale: str = Field(min_length=1)


class ScriptedJudge:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0)


def test_valid_json_in_prose_and_fences_is_one_call():
    judge = ScriptedJudge('Sure:\n```json\n{"tool": "stats", "rationale": "next"}\n```\nDone.')
    reply = ask_judge(judge, "pick one", Pick, role="decide")
    assert reply.output == Pick(tool="stats", rationale="next")
    assert len(judge.prompts) == 1
    assert '"tool"' in judge.prompts[0] and "pick one" in judge.prompts[0]  # schema + request
    assert [e.attempt for e in reply.exchanges] == [1]
    assert reply.raw.startswith("Sure:")


def test_invalid_answer_is_sent_back_with_the_errors():
    judge = ScriptedJudge('{"tool": "launch", "rationale": "x"}', '{"tool": "probe", "rationale": "ok"}')
    reply = ask_judge(judge, "pick one", Pick, role="decide", retries=1)
    assert reply.output.tool == "probe"
    second = judge.prompts[1]
    assert "Your previous answer:" in second and "That answer could not be used:" in second
    assert '"launch"' in second  # the judge sees its own answer ...
    assert "probe" in second and "stats" in second  # ... and what was allowed
    assert [e.record()["metadata"]["attempt"] for e in reply.exchanges] == [1, 2]


def test_exhausted_retries_raise_with_every_exchange():
    judge = ScriptedJudge("no json here", "still none")
    with pytest.raises(JudgeOutputError) as info:
        ask_judge(judge, "pick one", Pick, role="decide", retries=1)
    assert [e.output for e in info.value.exchanges] == ["no json here", "still none"]
    assert info.value.raw == "still none"


def test_a_failing_judge_raises_judge_call_error():
    class Broken:
        def generate(self, prompt, **kwargs):
            raise RuntimeError("account limit reached")

    with pytest.raises(JudgeCallError, match="account limit") as info:
        ask_judge(Broken(), "pick one", Pick, role="decide")
    (exchange,) = info.value.exchanges
    assert exchange.output is None and "account limit" in exchange.error


def test_check_asks_for_a_revision_then_accepts_on_the_last_attempt():
    judge = ScriptedJudge(*['{"tool": "probe", "rationale": "jargon"}'] * 2)
    reply = ask_judge(
        judge, "pick", Pick, role="decide", retries=1,
        check=lambda p: "too technical" if p.rationale == "jargon" else None,
    )
    assert reply.output.rationale == "jargon"
    assert len(judge.prompts) == 2 and "too technical" in judge.prompts[1]


def test_images_reach_only_a_judge_that_takes_them(tmp_path: Path):
    figure = tmp_path / "fig.png"
    figure.write_bytes(b"png")
    seen = {}

    class Seeing:
        def generate(self, prompt, *, images=None):
            seen["images"] = images
            return '{"tool": "probe", "rationale": "r"}'

    reply = ask_judge(Seeing(), "p", Pick, role="r", images=[figure])
    assert seen["images"] == [figure]
    assert reply.exchanges[0].record()["inputs"]["images"] == [str(figure)]
    blind = ScriptedJudge('{"tool": "probe", "rationale": "r"}')
    assert ask_judge(blind, "p", Pick, role="r", images=[figure]).exchanges[0].images == ()


def test_works_from_threads_and_inside_a_running_event_loop():
    def one(_):
        return ask_judge(ScriptedJudge('{"tool": "probe", "rationale": "r"}'), "p", Pick, role="r").output.tool

    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(one, range(4))) == ["probe"] * 4

    async def inside():
        return one(0)

    assert asyncio.run(inside()) == "probe"


def test_first_json_value_skips_braces_that_do_not_start_json():
    assert first_json_value('see {this} then [1, 2]') == [1, 2]
    with pytest.raises(ValueError):
        first_json_value("nothing")
