# Plan: speed up self-play (TFLite, no batching)

Full plan: ~/.claude/plans/find-ways-to-speed-noble-shannon.md

Baseline (measured 2026-07-09, MAX_ITER=160, spatial_list v2 model):
- decode_spatial_list: 2928 µs/eval (~470 ms/move)
- get_move_matrix (convolutional): 1159 µs/call
- non-NN MCTS overhead: ~2875 µs/iter (~930 ms/move total non-NN)

## Phases

- [ ] Phase 0: fix profile_inference (stale 26-row shape check at util.py:1097), record baseline numbers, capture fixed-seed game trace
- [ ] Phase 1: native spatial_list policy in MCTS — no per-eval decode; get_move_list gathers spatial indices directly. KEEP old policy logic in util.py (user wants to test later). Mirror in amcts.
- [ ] Phase 2: movegen micro-opts — shared bool policy matrix (drop per-piece zeros+logical_or), vectorized _get_highest_row
- [ ] Phase 3: copy path (Game.__new__/Player.__new__) + TFLite input buffer reuse in evaluate_from_tflite
- [ ] Phase 4: int8 board grid (codes 0=empty, 1-7=ZLOSIJT, 8=garbage) — user approved renumbering
- [ ] Phase 5: opt-in float16 tflite flag + 200-game battle gate (user runs the gate)

## Verification per phase

- pytest tests.py
- fixed-seed trace identical (Phases 1-4 must be bit-identical)
- profile_inference / profile_game before+after, numbers logged below

## Results

### Phase 0 baseline (profile_inference n=300, MAX_ITER=160)

| metric | value |
|---|---|
| game_to_X | 21.6 µs/call |
| tensor setup + set_tensor | 34.0 µs/call |
| interpreter.invoke() | 1507.6 µs/call (96.2% of raw inference) |
| get_tensor + reshape | 4.6 µs/call |
| full evaluate() incl. spatial decode | 3585.5 µs/call (decode ≈ 2018 µs) |
| MCTS avg wall time | 787.5 ms/move |
| non-inference overhead | 213.8 ms/move |

XNNPACK delegate active (warning about one dynamic-sized tensor). profile_inference's stale (1,26,10) shape check fixed; it now mirrors evaluate_from_tflite and times full evaluate() separately.

- [x] Phase 0 complete (trace: scratchpad/trace_baseline.json)
