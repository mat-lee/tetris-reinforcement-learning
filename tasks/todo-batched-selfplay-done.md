# Plan: async cross-game batched self-play (N=128, K=128)

## Goal

Each training set runs **128 games concurrently in one process** with all NN evaluations coalesced into a single `model.forward(batch=128)` per "wave". MCTS quality stays identical to today (no virtual loss, no algorithmic change). Game 0 renders with pygame if `config.visual=True`; games 1–127 are headless.

## Speedup model

Self-play wall-clock today ≈ `games × moves × MAX_ITER × t_eval(BS=1)`.
After refactor: each evaluator-tick processes up to 128 leaf positions in `~t_eval(BS=128)`. On MPS, BS=128 BaseResNet is roughly 1.5–3× the cost of BS=1, not 128×. Realistic expectation: **10–25× faster** per set; the new bottleneck is CPU-side tree work (game copies, move generation), not the GPU.

## Architectural change

### New `BatchedEvaluator` (ai.py, near `evaluate`)

```python
class BatchedEvaluator:
    """One-process NN batching server. Coalesces concurrent evaluate() calls
    from N async games into a single model.forward(batch=K) and dispatches
    results back via per-call futures."""
    def __init__(self, model, config, max_batch=128, timeout_ms=2):
        self.model = model
        self.config = config
        self.max_batch = max_batch
        self.timeout_s = timeout_ms / 1000
        self._queue: asyncio.Queue[tuple[list, asyncio.Future]] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._stop = False

    async def start(self):
        self._task = asyncio.create_task(self._worker())

    async def stop(self):
        self._stop = True
        await self._task

    async def submit(self, x_features: list) -> tuple[float, np.ndarray]:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        await self._queue.put((x_features, fut))
        return await fut

    async def _worker(self):
        while not self._stop or not self._queue.empty():
            batch = []
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout=0.05))
            except asyncio.TimeoutError:
                continue
            # opportunistically drain up to max_batch within a tiny timeout
            deadline = asyncio.get_running_loop().time() + self.timeout_s
            while len(batch) < self.max_batch:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
                except asyncio.TimeoutError:
                    break
            self._dispatch(batch)

    def _dispatch(self, batch):
        # Stack features along batch dim. Each x_features is a list[Tensor|scalar]
        # matching evaluate_pytorch's input order.
        xs = [torch.stack([b[0][i] for b in batch], dim=0).to(device) for i in range(len(batch[0][0]))]
        with torch.no_grad():
            out = self.model.forward(*xs)
        values, policies = out[0], out[1]
        policies = torch.softmax(policies.reshape(len(batch), -1), dim=1)
        values = values.cpu().numpy()
        policies = policies.cpu().numpy().reshape(len(batch), *POLICY_SHAPE)
        for i, (_, fut) in enumerate(batch):
            fut.set_result((float(values[i]), policies[i]))
```

Notes:
- `timeout_ms=2` is tiny; with 128 games all hitting `await evaluate()` simultaneously, the queue fills instantly and the timeout is rarely needed.
- The 50 ms get-timeout in the outer loop just keeps the worker responsive to `stop()`.
- For keras/tflite we'd build a parallel batching server; defer for now (PyTorch only path first).

### New `async def aevaluate(config, game, evaluator)` (ai.py)

Replaces `evaluate(config, game, network)` in the async codepath. Builds the same feature list as `evaluate_pytorch` (unsqueezed per-sample, *no* batch dim — the batcher stacks), then `await evaluator.submit(features)`.

### Convert `MCTS` → `async def amcts` (ai.py:291)

- Mechanical: every `evaluate(config, ..., network)` → `await aevaluate(config, ..., evaluator)`.
- The function takes `evaluator` instead of `interference_network`.
- All other logic identical.
- Keep the old sync `MCTS` for `battle_networks`, `main.py`, `optimize.py` callers.

### Convert `play_game` → `async def aplay_game` (ai.py:1218)

- Two call sites updated to `await amcts(...)` (line 1251 random-moves loop, line 1259 main loop).
- `screen` parameter: if `config.visual and game_number == 0`, render normally. For game_number != 0, force `config.visual=False` locally (use a `config.copy()` with `visual=False`) so the rest of the function doesn't try to grab a pygame surface.
- Note: `pygame.event.get()` and `pygame.display.update()` only run on the rendered game.

### Rewrite `make_training_set` (ai.py:1349)

```python
def make_training_set(config, interference_network, num_games, save_game=False, save_stats=True, screen=None):
    if config.model == 'pytorch':
        return asyncio.run(_make_training_set_async(
            config, interference_network, num_games, save_game, save_stats, screen))
    else:
        # keras/tflite path unchanged (serial)
        return _make_training_set_sync(config, interference_network, num_games, save_game, save_stats, screen)
```

The async path:

```python
async def _make_training_set_async(config, model, num_games, save_game, save_stats, screen):
    evaluator = BatchedEvaluator(model, config, max_batch=num_games)
    await evaluator.start()
    try:
        coros = [
            aplay_game(config, evaluator,
                       game_number=idx,
                       screen=screen if idx == 1 else None)  # render game 1 only
            for idx in range(1, num_games + 1)
        ]
        results = await asyncio.gather(*coros)
    finally:
        await evaluator.stop()
    # rest is identical: assemble series_data, save, stats
```

`max_batch=num_games` so a complete wave fits in one forward. With `num_games=128` that's BS=128.

### `pygame` & async

Pygame doesn't block on `pygame.display.update()`. The only call that could block is `pygame.event.get()` — it's non-blocking. We can call it directly from `aplay_game` when `game_number == 1 and config.visual`. No issue.

The pygame event loop won't pump on its own under asyncio, but we only need it active enough to keep the OS window responsive. A tiny `await asyncio.sleep(0)` after `display.update()` yields to other coroutines.

### Config / call-site changes

- `simulation.py`: no API change needed; `num_games` will be the training-set size (e.g. `training_games=128` in Config).
- `Config.training_games`: confirm this is the value we want set to 128. (User: "128 is best ... each training set.")
- Visual: rendering game 1 (1-indexed in the existing loop).

### Async battle_networks (200 games, two evaluators)

Convert `battle_networks` to async too. 200 games concurrent, with **two** `BatchedEvaluator` instances — one wrapping NN_1, one wrapping NN_2. Each `aplay_game_battle` picks the correct evaluator based on whose turn it is.

At any given move number, ~half the games are on NN_1's turn and half on NN_2's turn (sides alternate per game), so each batcher sees ~100 concurrent submissions per wave. Batch size 100 still gets near GPU-saturation.

```python
async def _battle_networks_async(model_1, config_1, model_2, config_2, games, ...):
    ev1 = BatchedEvaluator(model_1, config_1, max_batch=games)
    ev2 = BatchedEvaluator(model_2, config_2, max_batch=games)
    await ev1.start(); await ev2.start()
    try:
        coros = [aplay_battle_game(config_1, config_2, ev1, ev2, game_idx=i,
                                   side=i % 2)  # alternate who goes first
                 for i in range(games)]
        outcomes = await asyncio.gather(*coros)
    finally:
        await ev1.stop(); await ev2.stop()
    # tally wins[0], wins[1] as today
```

Update Config: `battle_games` default → 200 (was lower).

### Out of scope (this change)

- `optimize.py` and `main.py` — keep sync MCTS path.
- Keras / tflite async — defer; the codebase has now defaulted to pytorch.
- `torch.compile` / fp16 — separate follow-up. Stack on top once async is stable.

## Files changed

- `src/ai.py`:
  - add `import asyncio`
  - add `BatchedEvaluator` class
  - add `async def aevaluate(...)`
  - add `async def amcts(...)` (copy of `MCTS` with `await`)
  - add `async def aplay_game(...)` (copy of `play_game` with `await amcts`, game-1-only render)
  - add `async def _make_training_set_async(...)`
  - branch `make_training_set` on `config.model == 'pytorch'` to call async path
  - add `async def aplay_battle_game(...)` (two evaluators, side-aware)
  - add `async def _battle_networks_async(...)`
  - branch `battle_networks` on `config.model == 'pytorch'`
  - update `Config.battle_games` default to 200 (or whatever name it has — verify)

No changes to `architectures.py`, `move_generation.py`, `board.py`, `player.py`, `game.py`, `simulation.py`.

## Verification

1. **Smoke**: `python simulation.py` with `training_games=4`, `MAX_ITER=10`, `visual=False`. Must produce a valid `s2.2.X/{N}.txt` with the same sample shape as today.
2. **Diff**: Run serial path (sync `MCTS`, BS=1) vs async path on same seed; confirm identical game outcomes when `num_games=1` and `max_batch=1`. (Same NN, same seed → must match exactly.)
3. **Throughput**: Time 1 training set of 128 games at MAX_ITER=50 before vs after. Record in `tasks/todo.md` review section. Target: ≥10× wall-clock reduction.
4. **Visual**: `visual=True, training_games=128` — game 1 window opens, others run headless, no pygame errors.
5. **No regressions**: `pytest tests.py::test_reflections` still passes.

## Risks

- **CPU saturation**: 128 trees of tree-traversal + game copies run serially on one Python thread. If this dominates, asyncio gives smaller speedup than expected. Mitigation: measure first; if needed, add `concurrent.futures.ThreadPoolExecutor` for `Game.copy()` (releases GIL? — needs check) or shrink N to ~32.
- **Memory**: 128 active MCTS trees with full game-state copies at leaves. Estimate ~50 KB per game state × ~400 leaves × 128 games = ~2.5 GB peak. Should be fine on 32 GB Mac; flag if it isn't.
- **Determinism**: Async leaf ordering may differ from sync. For training data this is fine; for the verification step at num_games=1, batch=1 forces deterministic order.
- **Pygame in async**: Untested combination. Fallback: skip pygame entirely when `num_games > 1` if it misbehaves.

## Sequence

- [x] Step 1: Add `BatchedEvaluator`, `aevaluate`, `amcts`. Verified: `tasks/verify_amcts.py` PASS — same move, 51/51 children matched, 0 visit-count mismatches.
- [x] Step 2: Add `aplay_game`. Verified: `tasks/verify_aplay.py` PASS — 29 samples both sync and async, identical sample width and stats keys.
- [x] Step 3: Add `_make_training_set_async` and branch in `make_training_set`. Verified: `tasks/verify_training_set.py` 1.65× speedup at N=4, boards 40×10, sample width 13.
- [x] Step 4: Add `aplay_battle_game` + `_battle_networks_async` and branch in `battle_networks`. Verified: `tasks/verify_battle.py` PASS — 6 games, wins tally correctly.
- [ ] Step 5: Run N=128 self-play and 200-game battle; time them; record review notes. (Hand back to user — needs their real Config + time budget.)
- [ ] Step 6 (deferred): wire game-1 pygame rendering, validate visually.

## Caveats

- `_battle_networks_async` has **no early termination**. The sync path bails when one side hits the threshold; the async path plays all `games` games. For 200 games at MAX_ITER=50 this trades a bit of best-case time for higher batch saturation. Add back if needed (cancel the gather when threshold met).
- Async path is **pytorch only**. Keras / tflite stays serial; the `model == 'pytorch'` branch decides.
- Pygame rendering during async self-play is wired (`screen` is passed only to game 1) but not yet visually validated. Step 6.

## Review (filled in after implementation)

_TBD_
