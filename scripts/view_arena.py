"""Open the arena in MuJoCo's interactive viewer.

Run:  python scripts/view_arena.py [--config configs/arena_lab.yaml] [--seed 0] [--shape K]

One random episode from sim.randomize (arena shape, walls, panes, light,
camera, robot start); change --seed for another one. --shape K forces arena
shape K (0 = straight; the list is printed at start).

Viewer controls
  left-drag            rotate
  right-drag           pan
  scroll               zoom
  [ and ]              cycle through cameras (robot_cam is one of them)
  Esc                  back to the free camera
  double-click a body  select it; Ctrl + right-drag moves a selected mocap body
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sim  # noqa: F401,E402  selects the GL backend; must come before mujoco

import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

from sim.arena import load_config  # noqa: E402
from sim.randomize import ArenaPool  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/arena_lab.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shape", type=int, default=None, help="force arena shape K")
    args = ap.parse_args()
    pool = ArenaPool(load_config(args.config))
    for k, lay in enumerate(pool.layouts):
        print(f"  shape {k}: {lay.name}")
    rng = np.random.default_rng(args.seed)
    if args.shape is None:
        k, model, data, rand = pool.sample(rng)
    else:
        k, (model, data, rand) = args.shape, pool.get(args.shape)
    info = rand.reset(data, rng)
    print(f"shape {k} ({info['layout']}), walls: {info['walls']}")
    mujoco.viewer.launch(model, data)
