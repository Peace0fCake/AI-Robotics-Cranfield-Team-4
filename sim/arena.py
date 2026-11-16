"""MuJoCo arena builder: config -> MjSpec -> compiled MjModel.

See docs/PROJECT_GUIDE.md sections 3 and 5.1, and configs/arena.yaml.

Conventions
  - metres, z up. The arena shape comes from sim/layout.py (straight or L);
    origin at the centre of the second plaque, which for the straight arena is
    the arena centre with x along the 3 m length, as in configs/arena.yaml.
  - every geom gets a systematic name so segmentation IDs map back to meaning:
      floor_<i>            plaque i, in driving order (the last one has the line)
      carpet               lab floor around the arena
      wall_<k>             flat wall on outline segment k
      box_<k>_<i>          cardboard box i on outline segment k
      room_<side>          wall of the lab room on that side of the arena
      pane_<j>_paper       white sheet of pre-allocated pane j
      pane_<j>_image_<c>_<front|back>   printed picture of class c on pane j
  - MuJoCo sizes are HALF-extents.
"""

from pathlib import Path

import mujoco
import numpy as np
import yaml
from PIL import Image

from sim.layout import all_layouts

REPO_ROOT = Path(__file__).resolve().parent.parent
SIDES = ("north", "south", "east", "west")
PARK_Z = -10.0  # unused panes are parked below the floor


def load_config(path="configs/arena.yaml"):
    """Read the YAML config (path relative to the repo root) and return a dict."""
    with open(REPO_ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _texture_2d(spec, name, file):
    spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(REPO_ROOT / file))


def _material(spec, name, texture, texrepeat=(1, 1), texuniform=False, **kw):
    mat = spec.add_material(name=name, texrepeat=list(texrepeat), texuniform=texuniform, **kw)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture
    return mat


def _yaw_quat(deg):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.array([0.0, 0.0, 1.0]), np.radians(deg))
    return q


def _facing_quat(dx, dy):
    """Stand a geom up (local y -> world z) with its local +z along (dx, dy).

    Used for anything carrying a 2D texture on a vertical face: MuJoCo projects
    2D textures along the geom's local z.
    """
    q_up = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q_up, np.array([1.0, 0.0, 0.0]), np.pi / 2)  # local z -> world -y
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, _yaw_quat(np.degrees(np.arctan2(dy, dx)) + 90.0), q_up)
    return q


def add_floor(spec, cfg, layout):
    """One thin box per plaque, top surface at z = 0, plus the lab carpet around.

    Plaque i takes floor.textures[i] (the last texture carries the black line).
    The texture's +x edge is the plaque's front edge, so turning the plaque to
    its heading puts the line on the finish edge whatever the arena shape.
    """
    floor = cfg["floor"]
    t, textures = floor["thickness"], floor["textures"]
    for i, file in enumerate(textures):
        _texture_2d(spec, f"floor_tex_{i}", file)
    n = len(layout.plaques)
    for i, p in enumerate(layout.plaques):
        tex = len(textures) - 1 if i == n - 1 else min(i, len(textures) - 2)
        _material(spec, f"floor_{i}", f"floor_tex_{tex}", reflectance=0.05)
        cx, cy = p.centre
        spec.worldbody.add_geom(
            name=f"floor_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5, 0.75, t / 2],
            pos=[cx, cy, -t / 2], quat=_yaw_quat(p.yaw_deg), material=f"floor_{i}")

    sur = floor.get("surround")
    if sur:
        _texture_2d(spec, "carpet", sur["texture"])
        # texuniform: the texture repeats every `tile_size` metres, whatever the plane size.
        _material(spec, "carpet", "carpet", texrepeat=(1 / sur["tile_size"],) * 2, texuniform=True)
        h = sur["half_extent"]
        x0, y0, x1, y1 = layout.bbox
        spec.worldbody.add_geom(
            name="carpet", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[h, h, 0.1],
            pos=[(x0 + x1) / 2, (y0 + y1) / 2, -t], material="carpet")


def _square_png(rel_path, size=512):
    """Square copy of a texture, cached in assets/textures/_cache (git-ignored).

    MuJoCo cube textures need square faces. Each face is stretched back to the
    box face's shape when rendered, so squashing it to a square loses nothing
    but resolution.
    """
    src = REPO_ROOT / rel_path
    dst = REPO_ROOT / "assets" / "textures" / "_cache" / f"square_{src.stem}.png"
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.open(src).convert("RGB").resize((size, size), Image.Resampling.LANCZOS).save(dst)
    return dst


def _add_box_assets(spec, cfg):
    tex = cfg["walls"]["box"]["textures"]
    end, top = _square_png(tex["end"]), _square_png(tex["top"])
    for name, big in (("box_sticker", tex["big_sticker"]), ("box_plain", tex["big_plain"])):
        big = _square_png(big)
        # Cube faces: right/left = +-x (ends), up/down = +-y (big faces), front/back = +-z (top, bottom).
        spec.add_texture(
            name=name, type=mujoco.mjtTexture.mjTEXTURE_CUBE,
            cubefiles=[str(end), str(end), str(big), str(big), str(top), str(top)])
        _material(spec, name, name)


def _segment_frame(seg, offset):
    """Centre, unit direction along the segment and outward normal, `offset` m outside."""
    n = np.array(seg.normal, dtype=float)
    d = (seg.p1 - seg.p0) / seg.length
    return (seg.p0 + seg.p1) / 2 + offset * n, d, n


def box_slots(cfg, seg):
    """Nominal (x, y, yaw_deg) of every box along one outline segment, before jitter."""
    box = cfg["walls"]["box"]
    c, d, _ = _segment_frame(seg, box["thickness"] / 2)
    count = int(np.ceil(seg.length / box["length"] - 1e-6))
    yaw = np.degrees(np.arctan2(d[1], d[0]))
    return [(*(c + (-count * box["length"] / 2 + box["length"] * (i + 0.5)) * d), yaw)
            for i in range(count)]


def add_walls(spec, cfg, layout):
    """Pre-allocate BOTH wall types on every outline segment except the finish:
    one flat wall geom and a row of box geoms. set_segment() shows one of them
    per episode without recompiling. The finish edge stays open (carpet beyond).

    Walls sit just OUTSIDE the plaques.
    """
    walls = cfg["walls"]
    box, flat = walls["box"], walls["flat"]
    _add_box_assets(spec, cfg)
    _texture_2d(spec, "wall_flat", flat["texture"])
    for k, seg in enumerate(layout.segments):
        if seg.finish:
            continue
        # A 2D texture is projected along the geom's local z, so the flat wall is
        # stood up with local z pointing into the arena (local y = up).
        c, _, n = _segment_frame(seg, flat["thickness"] / 2)
        length = seg.length + 2 * flat["thickness"]
        # One material per segment: texrepeat counts repeats over the whole face
        # (texuniform off keeps the skirting at the bottom), and lengths differ.
        _material(spec, f"wall_flat_{k}", "wall_flat",
                  texrepeat=(length / flat["tile_width"], 1), emission=flat.get("emission", 0.0))
        spec.worldbody.add_geom(
            name=f"wall_{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[length / 2, flat["height"] / 2, flat["thickness"] / 2],
            pos=[c[0], c[1], flat["height"] / 2], quat=_facing_quat(-n[0], -n[1]),
            material=f"wall_flat_{k}")
        for i, (bx, by, yaw) in enumerate(box_slots(cfg, seg)):
            spec.worldbody.add_geom(
                name=f"box_{k}_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[box["length"] / 2, box["thickness"] / 2, box["height"] / 2],
                pos=[bx, by, box["height"] / 2], quat=_yaw_quat(yaw), material="box_plain")


def _show_geom(model, g, visible):
    """Hide a geom from rendering, segmentation and collisions, or show it again."""
    model.geom_rgba[g, 3] = 1.0 if visible else 0.0
    model.geom_contype[g] = model.geom_conaffinity[g] = 1 if visible else 0


def set_segment(model, cfg, layout, k, kind):
    """Show the walls of outline segment k as "boxes" or "flat" ("open" hides both)."""
    if kind not in ("boxes", "flat", "open"):
        raise ValueError(f"segment {k}: unknown wall type {kind!r}")
    _show_geom(model, _gid(model, f"wall_{k}"), kind == "flat")
    for i in range(len(box_slots(cfg, layout.segments[k]))):
        _show_geom(model, _gid(model, f"box_{k}_{i}"), kind == "boxes")


def add_room(spec, cfg, layout):
    """The lab room: four tall textured walls around the arena (cfg['room']).

    Each wall is built long enough to still meet its neighbours when the
    randomiser moves it out to room.distance[1], so corners never open.
    """
    room = cfg.get("room")
    if not room:
        return
    _texture_2d(spec, "room_wall", room["texture"])
    x0, y0, x1, y1 = layout.bbox
    d0, d_max = room["distance"][0], room["distance"][1]
    t, h = room["thickness"], room["height"]
    sides = {"north": ((x0 + x1) / 2, y1 + d0, x1 - x0, (0, -1)),
             "south": ((x0 + x1) / 2, y0 - d0, x1 - x0, (0, 1)),
             "east": (x1 + d0, (y0 + y1) / 2, y1 - y0, (-1, 0)),
             "west": (x0 - d0, (y0 + y1) / 2, y1 - y0, (1, 0))}
    for side, (x, y, run, inward) in sides.items():
        length = run + 2 * d_max + 2 * t
        _material(spec, f"room_{side}", "room_wall", texrepeat=(length / room["tile_width"], 1),
                  emission=room.get("emission", 0.0))
        # Shift by t/2 so the inner face, not the centre, is at distance d0.
        spec.worldbody.add_geom(
            name=f"room_{side}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[length / 2, h / 2, t / 2],
            pos=[x - inward[0] * t / 2, y - inward[1] * t / 2, h / 2 - cfg["floor"]["thickness"]],
            quat=_facing_quat(*inward), material=f"room_{side}")


def pane_faces(cfg):
    """Printed faces of a sheet: ("front",) for one-sided prints (the lab sheets:
    blank white back), ("front", "back") when panes.two_sided is true."""
    return ("front", "back") if cfg["panes"].get("two_sided", True) else ("front",)


def pane_dims(cfg, cls, height_scale=1.0, margins=None):
    """Paper and picture size (metres) for one class. margins = (side, top, bottom)."""
    p = cfg["panes"]
    c = p["classes"][cls]
    if margins is None:
        margins = tuple(np.mean(p["paper"][k]) for k in ("margin_side", "margin_top", "margin_bottom"))
    side, top, bottom = margins
    paper_h = c["paper_height"] * height_scale
    pic_h = paper_h - top - bottom
    pic_w = pic_h * c["aspect"]
    return {"paper_w": pic_w + 2 * side, "paper_h": paper_h,
            "pic_w": pic_w, "pic_h": pic_h, "pic_y": bottom + pic_h / 2}


def add_panes(spec, cfg):
    """Add cfg['panes']['max_count'] mocap bodies, each a two-layer pane.

    Body frame: origin at the middle of the paper's BOTTOM edge (so pitch
    leans the sheet about the edge that stands on the floor), local x along
    the paper width, local y up the paper, local z out of the printed face.

    Geoms per body j:
      pane_<j>_paper                   white box, the sheet
      pane_<j>_image_<class>_front     textured box on the +z face
      pane_<j>_image_<class>_back      same picture on the -z face, turned 180 deg
                                       about local y so it reads un-mirrored from behind
    One front/back pair per class; only the chosen class is shown per episode
    (see set_pane).
    """
    p = cfg["panes"]
    t, it, off = p["thickness"], p["image_thickness"], p["image_offset"]
    for cls, c in p["classes"].items():
        _texture_2d(spec, f"pane_{cls}", c["texture"])
        _material(spec, f"pane_{cls}", f"pane_{cls}")
    spec.add_material(name="paper", rgba=[0.95, 0.95, 0.95, 1])
    back = np.array([0.0, 0.0, 1.0, 0.0])  # 180 deg about local y
    for j in range(p["max_count"]):
        body = spec.worldbody.add_body(name=f"pane_{j}", mocap=True, pos=[0, 0, PARK_Z])
        d0 = pane_dims(cfg, "building")
        body.add_geom(name=f"pane_{j}_paper", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[d0["paper_w"] / 2, d0["paper_h"] / 2, t / 2],
                      pos=[0, d0["paper_h"] / 2, 0], material="paper", contype=0, conaffinity=0)
        for cls in p["classes"]:
            d = pane_dims(cfg, cls)
            sides = (("front", off, [1, 0, 0, 0]), ("back", -off, back))
            for face, z, quat in sides[:len(pane_faces(cfg))]:
                body.add_geom(
                    name=f"pane_{j}_image_{cls}_{face}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[d["pic_w"] / 2, d["pic_h"] / 2, it / 2], pos=[0, d["pic_y"], z],
                    quat=quat, material=f"pane_{cls}", rgba=[1, 1, 1, 0],
                    contype=0, conaffinity=0)


def add_robot_camera(spec, cfg):
    """Add a mocap body 'robot' carrying the camera cfg['camera']['name'].

    Robot frame: origin at the front bumper on the floor, x forward.
    """
    r, cam = cfg["robot"], cfg["camera"]
    body = spec.worldbody.add_body(name="robot", mocap=True, pos=[0, 0, 0])
    body.add_geom(name="robot_body", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[r["length"] / 2, r["width"] / 2, 0.04], pos=[-r["length"] / 2, 0, 0.04],
                  rgba=[0.2, 0.4, 0.9, 1], contype=0, conaffinity=0, group=3)
    # A camera looks along its local -z with local y up; xyaxes puts -z on body +x.
    tilt = np.radians(cam["tilt_deg"])
    c = body.add_camera(name=cam["name"], pos=[r["camera_x"], 0, cam["height"]], fovy=cam["fovy_deg"])
    c.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    c.alt.xyaxes = [0, -1, 0, np.sin(tilt), 0, np.cos(tilt)]


def add_lights(spec, cfg, layout):
    """Ceiling panels: lights pointing straight down, one above each plaque.

    Without a `lighting` section (the first room's config) defaults are used.
    """
    light = cfg.get("lighting", {"height": 2.5, "diffuse": 0.6, "ambient": 0.4})
    for i, p in enumerate(layout.plaques):
        spec.worldbody.add_light(name=f"ceiling_{i}", pos=[*p.centre, light["height"]],
                                 dir=[0, 0, -1], diffuse=[light["diffuse"]] * 3,
                                 castshadow=True, cutoff=80)
    spec.visual.headlight.ambient = [light["ambient"]] * 3
    spec.visual.headlight.diffuse = [0.0] * 3


def build_arena(cfg, layout=None):
    """Build the full scene for one arena shape. Returns (model, data, spec).

    layout: a sim.layout.Layout; default the first of all_layouts(cfg), the
    straight arena. Everything random is applied after compiling
    (sim/randomize.py), so one compiled model serves every episode of a shape.
    """
    layout = layout or all_layouts(cfg)[0]
    spec = mujoco.MjSpec()
    spec.modelname = f"arena_{layout.name}"
    w, h = max(cfg["camera"].get("yolo_width", 0), 1280), max(cfg["camera"].get("yolo_height", 0), 960)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = w, h
    add_floor(spec, cfg, layout)
    add_walls(spec, cfg, layout)
    add_room(spec, cfg, layout)
    add_panes(spec, cfg)
    add_robot_camera(spec, cfg)
    add_lights(spec, cfg, layout)
    model = spec.compile()
    sides = cfg["walls"].get("sides", {})
    for k, seg in enumerate(layout.segments):
        if not seg.finish:
            kind = sides.get(seg.side, "boxes")
            set_segment(model, cfg, layout, k, "boxes" if kind == "open" else kind)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, spec


def geom_classes(model):
    """Return {geom_id: label} from geom names.

    Labels: 'floor', 'carpet', 'wall', 'box', 'room', 'paper', 'robot',
    'human', 'car', 'building'.
    """
    labels = {}
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        parts = name.split("_")
        if parts[0] == "pane":
            labels[g] = parts[3] if parts[2] == "image" else "paper"
        else:
            labels[g] = parts[0]
    return labels


# ---- per-episode placement, no recompile ---------------------------------

def _gid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def set_pane(model, data, cfg, j, cls=None, x=0.0, y=0.0, yaw_deg=0.0, pitch_deg=0.0,
             roll_deg=0.0, height_scale=1.0, margins=None):
    """Show pane j as class `cls` at (x, y), or park it when cls is None.

    yaw 0 = the sheet stands across the arena, printed face towards -x (the start).
    pitch leans it about its bottom edge (positive = top towards the printed
    face's side, i.e. towards -x at yaw 0). roll turns it in its own plane.
    height_scale and margins (side, top, bottom) resize the sheet and picture.
    """
    mid = model.body(f"pane_{j}").mocapid[0]
    for c in cfg["panes"]["classes"]:
        for face in pane_faces(cfg):
            model.geom_rgba[_gid(model, f"pane_{j}_image_{c}_{face}"), 3] = float(c == cls)
    if cls is None:
        data.mocap_pos[mid] = [0, 0, PARK_Z]
        return
    p = cfg["panes"]
    d = pane_dims(cfg, cls, height_scale, margins)
    paper = _gid(model, f"pane_{j}_paper")
    model.geom_size[paper] = [d["paper_w"] / 2, d["paper_h"] / 2, p["thickness"] / 2]
    model.geom_pos[paper] = [0, d["paper_h"] / 2, 0]
    model.geom_rbound[paper] = np.linalg.norm(model.geom_size[paper])
    for face in pane_faces(cfg):
        g = _gid(model, f"pane_{j}_image_{cls}_{face}")
        model.geom_size[g] = [d["pic_w"] / 2, d["pic_h"] / 2, p["image_thickness"] / 2]
        model.geom_pos[g][1] = d["pic_y"]
        model.geom_rbound[g] = np.linalg.norm(model.geom_size[g])
    # Base orientation: local x -> world -y, local y -> world z, local z -> world -x.
    base = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, base.flatten())
    for axis, deg in (((1.0, 0.0, 0.0), pitch_deg), ((0.0, 0.0, 1.0), roll_deg)):
        r = np.zeros(4)
        mujoco.mju_axisAngle2Quat(r, np.array(axis), np.radians(deg))
        mujoco.mju_mulQuat(q, q.copy(), r)
    mujoco.mju_mulQuat(q, _yaw_quat(yaw_deg), q.copy())
    data.mocap_pos[mid] = [x, y, p.get("base_height", 0.0)]
    data.mocap_quat[mid] = q


def set_robot(model, data, x, y, yaw_deg=0.0):
    """Put the robot's front bumper at (x, y), heading yaw (0 = +x)."""
    mid = model.body("robot").mocapid[0]
    data.mocap_pos[mid] = [x, y, 0.0]
    data.mocap_quat[mid] = _yaw_quat(yaw_deg)

