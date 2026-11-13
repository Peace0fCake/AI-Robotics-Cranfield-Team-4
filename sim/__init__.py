"""Simulation package.

Importing this package picks the MuJoCo offscreen GL backend for the current
OS, so it must be imported BEFORE `import mujoco`:

    import sim            # first
    import mujoco         # second

  - Linux / Colab: EGL (headless). Override with MUJOCO_GL=osmesa if EGL is missing.
  - Windows / macOS: MuJoCo's default (GLFW / WGL). MUJOCO_GL=egl is invalid
    there and makes `import mujoco` raise, so it is never set.
"""

import os
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
elif os.environ.get("MUJOCO_GL", "").lower() in ("egl", "osmesa"):
    del os.environ["MUJOCO_GL"]
