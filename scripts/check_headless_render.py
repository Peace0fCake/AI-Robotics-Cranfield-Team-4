"""Phase 1 check: headless MuJoCo rendering (RGB + segmentation) and FPS benchmark.

Run:  python scripts/check_headless_render.py
Done when:
  - scripts/out/rgb.png shows the box on the floor from robot height
  - segmentation contains the box and floor geom IDs
  - FPS is printed for 256x192 (the training render size, D-014) and 640x480 (for comparison)

See docs/PROJECT_GUIDE.md, section 5.1.
"""

# 0. Put the repo root on sys.path (scripts/ is not a package), so `import sim` works.

# 1. `import sim` BEFORE importing mujoco: it picks the GL backend for this OS
#    (EGL on Linux, MuJoCo's default on Windows).

# 2. Imports: mujoco, numpy, PIL, time, pathlib.

# 3. MJCF scene string: light, floor plane (half-extents!), one named box,
#    camera "robot_cam" at 0.145 m looking along +x, fovy 41.
#    Set <visual><global offwidth=... offheight=.../> to at least 640x480.

# 4. Load model + data, run mj_forward.

# 5. RGB render -> save scripts/out/rgb.png

# 6. Segmentation render -> check box/floor IDs present -> save colourised PNG.

# 7. Benchmark ~200 renders at 256x192 and at 640x480; print FPS, mujoco version, backend.
