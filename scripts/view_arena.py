"""Open the arena in MuJoCo's interactive viewer.

Run:  python scripts/view_arena.py

Needs sim/arena.py to be implemented (load_config, build_arena).

Viewer controls
  left-drag            rotate
  right-drag           pan
  scroll               zoom
  [ and ]              cycle through cameras (robot_cam is one of them)
  Esc                  back to the free camera
  double-click a body  select it; Ctrl + right-drag moves a selected mocap body
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sim  # noqa: F401,E402  selects the GL backend; must come before mujoco

import mujoco.viewer  # noqa: E402

from sim.arena import build_arena, load_config  # noqa: E402

if __name__ == "__main__":
    model, data, _ = build_arena(load_config())
    mujoco.viewer.launch(model, data)
