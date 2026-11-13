# Test fixtures

`chartqa-gemini.json` is the `{data, layout}` payload a portable export
(`evalrx report`) embeds, taken from a real M1–M5 run: gemini-3.6-flash on
ChartQA (`examples/benchmark/vlm/gemini`, run `chartqa.m1m5-repair-20261012-v2`).
Media is not embedded. It was produced with:

```python
import json
from pathlib import Path
from evalrx.reporting.dynamic import build_report_data, fallback_spec, validate_spec

data = build_report_data(Path(run_dir).resolve())
data["generated_at"] = "FIXED"
layout = {"spec": validate_spec(fallback_spec(data), data=data),
          "generated_by": {"mode": "deterministic", "model": None}}
Path("chartqa-gemini.json").write_text(json.dumps({"data": data, "layout": layout}, ensure_ascii=False))
```

The unit tests (`src/*.test.ts`) and the smoke test (`e2e/`) both read it.
Regenerate it when `ReportData` changes shape.
