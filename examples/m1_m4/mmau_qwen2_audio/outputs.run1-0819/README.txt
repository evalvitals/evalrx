outputs/  —  run output guide

run_log.jsonl
    one JSON line per M1/M2/M3/M4/M5 event

report/
    human-facing deliverables: run summary, hypotheses, M5 results
      report/hypotheses.json
      report/m5_results.json
      report/summary.json
      report/summary.md

artifacts/
    M1 heavy numeric data (.npy tensors, .json finding dumps)
      artifacts/c0_answer_extraction_audit.result.json
      artifacts/c0_calibration.result.json
      artifacts/c0_coverage_verification_gap.result.json
      artifacts/c0_format_sensitivity.result.json
      artifacts/c0_logprob_entropy.result.json
      artifacts/c0_logprob_entropy_token_logprobs.json
      artifacts/c0_m2_stats_results.json
      artifacts/c0_m2_stats_tool_results.json
      artifacts/c0_self_consistency.result.json
      artifacts/c0_self_consistency_samples.json
      artifacts/c0_selfcheck_consistency.result.json
      artifacts/c0_termination_audit.result.json

prompts/
    verbatim judge prompt + response for each M1/M2/M3 call
      prompts/c0_m2_analysis.prompt.txt
      prompts/c0_m2_analysis.response.txt
      prompts/c0_m3_diagnosis.prompt.txt
      prompts/c0_m3_diagnosis.response.txt

fixes/
    one self-contained folder per repair attempt (code + sandbox + record.md + result.json), see new_trial(); outcome.md summarises all
      fixes/outcome.md

other/
    misc files
      run_log.jsonl
