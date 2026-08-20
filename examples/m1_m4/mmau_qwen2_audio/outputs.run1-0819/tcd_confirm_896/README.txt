tcd_confirm_896/  —  run output guide

run_log.jsonl
    one JSON line per M1/M2/M3/M4/M5 event

artifacts/
    M1 heavy numeric data (.npy tensors, .json finding dumps)
      artifacts/baseline.json

fixes/
    one self-contained folder per repair attempt (code + sandbox + record.md + result.json), see new_trial(); outcome.md summarises all
      fixes/01_L3a_tcd_temporal_blur_confirm/outputs.jsonl
      fixes/01_L3a_tcd_temporal_blur_confirm/record.md
      fixes/01_L3a_tcd_temporal_blur_confirm/result.json
      fixes/outcome.md

other/
    misc files
      run_log.jsonl
