"""Multi-env (vectorized) replay of recorded BEHAVIOR demos.

`harness.VectorReplayHarness` drives a `num_envs > 1` environment through
state-restore replay of recorded demonstrations: recorded state is injected per
scene each step, observations are omitted, recorded transitions are applied per
scene in that scene's frame, and the task success condition is re-evaluated
every step so the fresh verdict can be compared against the recorded one.

It is included here because it is what exercises the library fixes in this PR at
`num_envs > 1`, and because how a vector env is driven for replay is worth a
second pair of eyes -- the batch lifecycle, the per-scene injection, and the
per-scene transition placement are all places where a wrong assumption would
produce plausible-looking but contaminated results. Two such assumptions are
called out in the class docstring: scene slots are inhomogeneous by
construction, and cross-scene isolation is spatial tiling alone.

Deliberately not included: the campaign that runs it over the full 20,000-demo
dataset -- shard CLIs, fleet worker, result checking and reporting, container
images, and the calibration numbers. That machinery is specific to one
organization's hardware, nobody else can run it, and it is not useful to review.
It lives in Calder's fork. So there is no entry point in this package; the
harness is here to be read against the fixes, not invoked.
"""
