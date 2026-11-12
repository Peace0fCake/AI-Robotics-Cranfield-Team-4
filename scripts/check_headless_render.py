"""Phase 1 check: headless MuJoCo rendering (RGB + segmentation) and FPS benchmark.

Run:  python scripts/check_headless_render.py
Done when:
  - scripts/out/rgb.png shows the box on the floor from robot height
  - segmentation contains the box and floor geom IDs
  - FPS is printed for 640x480 and 160x48

See docs/PROJECT_GUIDE.md, section 5.1.
"""

# 1. Choose the GL backend BEFORE importing mujoco (default: egl).

# 2. Imports: mujoco, numpy, PIL, time, pathlib.

# 3. MJCF scene string: light, floor plane (half-extents!), one named box,
#    camera "robot_cam" at ~0.15 m looking along +x.

# 4. Load model + data, run mj_forward.

# 5. RGB render -> save scripts/out/rgb.png

# 6. Segmentation render -> check box/floor IDs present -> save colourised PNG.

# 7. Benchmark ~200 renders at 640x480 and at 160x48; print FPS, mujoco version, backend.
