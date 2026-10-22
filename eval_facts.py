"""Scratch: evaluate scene facts on breakfast_box support + val."""
import sys, time, collections
sys.path.insert(0, "/home/hjh/Desktop/SymbolicAD")
from pathlib import Path
from symbolicAD.rules.facts import GroundingDinoDetector, SceneFactBuilder, SceneFactConfig

det = GroundingDinoDetector(device="cuda:0")
builder = SceneFactBuilder(det, SceneFactConfig())

paths = [Path(p) for p in sys.argv[1:]]
t0 = time.time()
for p in paths:
    ctx = builder.build(p)
    scene = ctx.scene
    counts = collections.Counter(
        scene.component_type(i) for i in range(len(scene.components))
    )
    md = ctx.metadata
    print("%-34s div=%.3f conf=%.3f complete=%s n=%d %s" % (
        p.name, md["divider_fraction"], scene.grounding_confidence,
        scene.observation_complete, len(scene.components),
        dict(counts)))
print("elapsed %.1fs for %d images" % (time.time() - t0, len(paths)))
