"""Transcript rendering for CLI coding-agent event streams."""

from __future__ import annotations

import json

RAW_OUTPUT_CAP = 3000
STREAM_TEXT_CAP = 4000
STREAM_TOOL_INPUT_CAP = 2000
STREAM_TOOL_RESULT_CAP = 2000
STREAM_FALLBACK_CAP = 8000


def trunc(text: str, cap: int) -> str:
    text = text or ""
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n... [truncated {len(text) - cap} chars]"


def _blocks_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                out.append(str(block.get("text", "")))
            elif isinstance(block, str):
                out.append(block)
        return "\n".join(out)
    return ""


def _summarise_tool_use(name: str, inp: object) -> str:
    if not isinstance(inp, dict):
        return trunc(str(inp), STREAM_TOOL_INPUT_CAP)
    if name == "Bash":
        return trunc("$ " + str(inp.get("command", "")), STREAM_TOOL_INPUT_CAP)
    if name == "Write":
        content_len = len(str(inp.get("content", "")))
        return f"write {inp.get('file_path', '?')} ({content_len} chars)"
    if name == "Edit":
        return f"edit {inp.get('file_path', '?')}"
    if name == "Read":
        span = ""
        if inp.get("offset") is not None or inp.get("limit") is not None:
            span = f" [offset={inp.get('offset')}, limit={inp.get('limit')}]"
        return f"read {inp.get('file_path', '?')}{span}"
    try:
        return trunc(json.dumps(inp, default=str), STREAM_TOOL_INPUT_CAP)
    except (TypeError, ValueError):
        return trunc(str(inp), STREAM_TOOL_INPUT_CAP)


def render_claude_stream(stdout: str) -> tuple[str, dict | None]:
    """Render ``claude -p --output-format stream-json`` into a coding trajectory."""
    events: list[dict] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(event, dict):
            events.append(event)
    if not events:
        return trunc(stdout, STREAM_FALLBACK_CAP), None

    out: list[str] = []
    usage: dict | None = None
    tool_seq: dict[str, int] = {}
    step = 0

    for event in events:
        etype = event.get("type")
        if etype == "system" and event.get("subtype") == "init":
            tools = event.get("tools") or []
            out.append(
                f"=== session start | model={event.get('model', '?')} | "
                f"tools={','.join(map(str, tools)) if tools else '-'} ==="
            )
        elif etype == "assistant":
            for block in (event.get("message") or {}).get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    txt = str(block.get("text", "")).strip()
                    if txt:
                        out.append(f"[assistant] {trunc(txt, STREAM_TEXT_CAP)}")
                elif block.get("type") == "tool_use":
                    step += 1
                    if block.get("id"):
                        tool_seq[block["id"]] = step
                    name = str(block.get("name", "tool"))
                    summary = _summarise_tool_use(name, block.get("input"))
                    out.append(f"[#{step} {name}] {summary}")
        elif etype == "user":
            for block in (event.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    n = tool_seq.get(block.get("tool_use_id"), "?")
                    err = " ERROR" if block.get("is_error") else ""
                    out.append(
                        f"[#{n} result{err}] "
                        f"{trunc(_blocks_text(block.get('content')), STREAM_TOOL_RESULT_CAP)}"
                    )
        elif etype == "result":
            token_usage = event.get("usage") or {}
            usage = {
                "cost_usd": event.get("total_cost_usd"),
                "num_turns": event.get("num_turns"),
                "duration_ms": event.get("duration_ms"),
                "input_tokens": token_usage.get("input_tokens"),
                "output_tokens": token_usage.get("output_tokens"),
                "cache_read_input_tokens": token_usage.get("cache_read_input_tokens"),
                "cache_creation_input_tokens": token_usage.get("cache_creation_input_tokens"),
            }
            final = str(event.get("result", "")).strip()
            if final:
                out.append(f"[final] {trunc(final, STREAM_TEXT_CAP)}")
            cost = usage["cost_usd"]
            dur = event.get("duration_ms")
            out.append(
                "=== result: {sub} | turns={turns} | cost={cost} | "
                "tokens in={inp} out={outp} (cache_read={cache}) | wall={wall} ===".format(
                    sub=event.get("subtype", "?"),
                    turns=usage["num_turns"],
                    cost=(f"${cost:.4f}" if isinstance(cost, (int, float)) else "n/a"),
                    inp=usage["input_tokens"],
                    outp=usage["output_tokens"],
                    cache=usage["cache_read_input_tokens"],
                    wall=(f"{dur / 1000:.1f}s" if isinstance(dur, (int, float)) else "?"),
                )
            )

    return "\n".join(out), usage
