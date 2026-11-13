# AI for Robotics: Autonomous A-to-B Navigation

Cranfield MSc Robotics, N-RBT-AIR group project. The full plan is in
[docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md).

## Setup (Python 3.11)

The virtual environment is per machine and is not shared: never copy `venv/`
between computers or operating systems. Create it locally from
`requirements.txt`.

**Linux**

```bash
git clone <repo-url> robot && cd robot
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
git clone <repo-url> robot; cd robot
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If activation is blocked, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

**Then, on either OS** (run from the repo root):

```
python scripts/make_placeholders.py       # only if assets/placeholders/ is empty
python scripts/check_headless_render.py   # offscreen rendering check
```

### Rendering backend

Scripts must `import sim` before `import mujoco`. That selects the offscreen GL
backend for the OS:

| OS | Backend | Notes |
|---|---|---|
| Linux, Colab | EGL (`MUJOCO_GL=egl`) | if EGL is unavailable: install `libosmesa6` and set `MUJOCO_GL=osmesa` |
| Windows | MuJoCo default (GLFW) | do **not** set `MUJOCO_GL=egl`; `import mujoco` fails with it |

### Rules that keep the code portable

- Paths: build them with `pathlib` relative to the repo root. No absolute
  paths, no hard-coded `/` or `\` separators.
- Run scripts from the repo root.
- Multiprocessing (parallel envs): put the entry point under
  `if __name__ == "__main__":`. Windows starts worker processes by re-importing
  the script, and fails without it.
- New dependency: add it to `requirements.txt` in the same commit.

## Google Colab

```python
!git clone <repo-url> robot
%cd robot
!pip install -r requirements.txt
!python scripts/check_headless_render.py
```

## Repository layout
See section 13 of the project guide.
