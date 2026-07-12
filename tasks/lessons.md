# Lessons

## 2026-07-12 — coordinate system migration left legacy indexing behind
- Correction: after switching the policy to mino-anchored coordinates, I left move
  generation's internal structures (checked_list, movement graphs) on the legacy
  flat `x + 2` offset, treating them as "internal, doesn't matter". The owner's
  design: checked_list IS the action space in policy format — internal spatial
  structures must use the same representation as the external one.
- Rule: when migrating a representation/coordinate system, enumerate EVERY structure
  indexed in the old system and convert them all — or explicitly ask whether the
  internal ones should follow. "It still works" is not "it's right".

## 2026-07-12 — plans must explain non-obvious mechanisms concretely
- Correction: the owner rejected a plan because the mask-cropping mechanism read
  like "another coordinate system adding complexity". It was actually the device
  that ELIMINATES a coordinate offset — but the plan asserted that instead of
  showing it.
- Rule: when a plan hinges on a non-obvious trick, include a one-paragraph worked
  example (concrete piece/number) showing why it's equivalent/simpler, and state
  explicitly how many representations exist before vs after.
