# AI for Robotics: Autonomous A-to-B Navigation

Cranfield MSc Robotics, N-RBT-AIR group project. The full plan is in
[docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md).

## Setup (Linux, Python 3.11)

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/check_headless_render.py   # headless rendering check
```

Headless rendering uses EGL (`MUJOCO_GL=egl`). If EGL is unavailable, install
OSMesa (`libosmesa6` on Debian/Ubuntu) and use `MUJOCO_GL=osmesa`.

## Google Colab

```python
!git clone <repo-url> robot && cd robot && pip install -r requirements.txt
%env MUJOCO_GL=egl
!python scripts/check_headless_render.py
```

## Repository layout
See section 13 of the project guide.
