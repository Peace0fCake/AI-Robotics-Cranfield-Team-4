"""Per-episode domain randomisation of the arena (PROJECT_GUIDE section 5.3).

    pool = ArenaPool(cfg)                     # one compiled model per arena shape
    k, model, data, rand = pool.sample(rng)   # or: model, data, rand = pool.get(k)
    info = rand.reset(data, rng)              # new walls, panes, light, camera, robot pose

Inside one shape, everything is changed in the compiled model and data arrays
(positions, sizes, colours, materials, visibility), never by recompiling, so
a reset is cheap. The nominal values are copied once at construction and
every reset starts from them, so random changes never accumulate.

Ranges come from configs/<arena>.yaml: the `randomization` section, plus the
`panes` and `layout.start` ranges written by Matthew.
"""

import mujoco
import numpy as np

from sim.arena import _gid, box_slots, build_arena, set_pane, set_robot, set_segment
from sim.layout import CELL, all_layouts, sample_layout_index


def _uniform(rng, rng_range):
    return rng.uniform(*rng_range)


def _half(plaque):
    """Half extents (x, y) of a plaque in metres."""
    return np.array([plaque.x1 - plaque.x0, plaque.y1 - plaque.y0]) * CELL / 2


class EpisodeRandomizer:
    def __init__(self, model, cfg, layout):
        self.model, self.cfg, self.layout = model, cfg, layout
        self.r = cfg.get("randomization", {})
        m = model
        # Nominal copies, restored at the start of every reset.
        self.nom_geom_pos = m.geom_pos.copy()
        self.nom_geom_quat = m.geom_quat.copy()
        self.nom_mat_rgba = m.mat_rgba.copy()
        self.nom_light_pos = m.light_pos.copy()
        self.nom_light_diffuse = m.light_diffuse.copy()
        self.cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cfg["camera"]["name"])
        self.nom_cam_pos = m.cam_pos[self.cam].copy()
        self.nom_cam_quat = m.cam_quat[self.cam].copy()
        self.walls = [k for k, s in enumerate(layout.segments) if not s.finish]
        self.boxes = [_gid(m, f"box_{k}_{i}") for k in self.walls
                      for i in range(len(box_slots(cfg, layout.segments[k])))]
        self.mat = {name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, name)
                    for name in ("box_sticker", "box_plain")}
        self.floor_mats = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, f"floor_{i}")
                           for i in range(len(layout.plaques))]
        self.room = [(g, s) for g, s in ((_gid(m, f"room_{s}"), s) for s in ("north", "south", "east", "west"))
                     if g >= 0]

    # -- pieces ----------------------------------------------------------------

    def _walls(self, rng):
        """Wall type per outline segment. The finish edge is always open."""
        wr = self.r.get("walls")
        kinds = {}
        for k in self.walls:
            if wr:
                allowed = wr["allowed"]
                w = np.array([wr["weights"][a] for a in allowed], dtype=float)
                kinds[k] = str(rng.choice(allowed, p=w / w.sum()))
            else:
                kind = self.cfg["walls"]["sides"].get(self.layout.segments[k].side, "boxes")
                kinds[k] = "boxes" if kind == "open" else kind
            set_segment(self.model, self.cfg, self.layout, k, kinds[k])
        return kinds

    def _boxes(self, rng):
        """Hand-placement error on each box, and which face carries the sticker."""
        m, box = self.model, self.cfg["walls"]["box"]
        jit = box["jitter"]
        for g in self.boxes:
            yaw = np.zeros(4)
            mujoco.mju_axisAngle2Quat(yaw, np.array([0.0, 0.0, 1.0]),
                                      np.radians(_uniform(rng, jit["yaw_deg"])))
            mujoco.mju_mulQuat(m.geom_quat[g], yaw, self.nom_geom_quat[g])
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, self.nom_geom_quat[g])
            ex, ey = rot.reshape(3, 3)[:, 0], rot.reshape(3, 3)[:, 1]
            # Local y is across the wall; "out" must never push a box into the arena,
            # so take the sign of ey that points away from the nearest plaque.
            centre = self.nom_geom_pos[g]
            nearest = min(self.layout.plaques, key=lambda p: np.hypot(*(p.centre - centre[:2])))
            outward = ey if np.dot(ey[:2], centre[:2] - nearest.centre) > 0 else -ey
            m.geom_pos[g] = centre + _uniform(rng, jit["along"]) * ex + _uniform(rng, jit["out"]) * outward
            sticker = rng.random() < box["sticker_prob"]
            m.geom_matid[g] = self.mat["box_sticker" if sticker else "box_plain"]

    def _room(self, rng):
        """Move each room wall out to a random distance from the arena (room size unknown)."""
        room = self.cfg.get("room")
        if not room:
            return
        out = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
        for g, side in self.room:
            extra = _uniform(rng, room["distance"]) - room["distance"][0]
            self.model.geom_pos[g, :2] = self.nom_geom_pos[g, :2] + extra * np.array(out[side])

    def _floor(self, rng):
        fr = self.r.get("floor")
        if not fr:
            return
        for mid in self.floor_mats:
            gain = _uniform(rng, fr["brightness"]) * (1 + rng.uniform(-1, 1, 3) * fr["tint"])
            self.model.mat_rgba[mid, :3] = np.clip(self.nom_mat_rgba[mid, :3] * gain, 0, 1)
            self.model.mat_reflectance[mid] = _uniform(rng, fr["reflectance"])

    def _lights(self, rng):
        lr = self.r.get("lighting")
        if not lr:
            return
        m = self.model
        for i in range(m.nlight):
            m.light_pos[i, :2] = self.nom_light_pos[i, :2] + rng.uniform(-1, 1, 2) * lr["pos_jitter"]
            m.light_diffuse[i] = self.nom_light_diffuse[i] * _uniform(rng, lr["diffuse_scale"])
        m.vis.headlight.ambient[:] = _uniform(rng, lr["ambient"])

    def _camera(self, rng):
        cr = self.r.get("camera")
        if not cr:
            return
        m, c = self.model, self.cam
        m.cam_pos[c] = self.nom_cam_pos + [0, 0, rng.uniform(-1, 1) * cr["height_jitter"]]
        # Tilt about the camera's local x (pitch down = positive), yaw about its local y.
        q = self.nom_cam_quat.copy()
        for axis, deg in (((1.0, 0.0, 0.0), -_uniform(rng, cr["tilt_deg"])),
                          ((0.0, 1.0, 0.0), _uniform(rng, cr["yaw_deg"]))):
            r = np.zeros(4)
            mujoco.mju_axisAngle2Quat(r, np.array(axis), np.radians(deg))
            mujoco.mju_mulQuat(q, q.copy(), r)
        m.cam_quat[c] = q

    def _panes(self, data, rng):
        """1-2 panes of each class on every plaque but the first (the clear start
        plaque), each facing the robot as it arrives on that plaque."""
        cfg, p = self.cfg, self.cfg["panes"]
        margin, spacing = p["placement"]["wall_margin"], p["placement"]["min_spacing"]
        pose, paper = p["pose"], p["paper"]
        regions = self.layout.plaques[1:]
        placed, j = [], 0
        for cls in p["classes"]:
            lo, hi = p["count_per_class"]
            for _ in range(rng.integers(lo, hi + 1)):
                if j >= p["max_count"]:
                    break
                for _try in range(200):
                    pl = regions[rng.integers(len(regions))]
                    (x0, y0), (x1, y1) = pl.centre - _half(pl), pl.centre + _half(pl)
                    x, y = rng.uniform(x0 + margin, x1 - margin), rng.uniform(y0 + margin, y1 - margin)
                    if all(np.hypot(x - a, y - b) >= spacing for a, b, *_ in placed):
                        break
                else:
                    continue  # no free spot: skip this pane rather than overlap
                wide = rng.random() < pose["yaw_wide_prob"]
                # yaw 0 in set_pane = printed face towards -x, i.e. facing a robot
                # driving along +x; add the plaque's heading for turned plaques.
                yaw = pl.yaw_deg + _uniform(rng, pose["yaw_wide_deg"] if wide else pose["yaw_deg"])
                margins = tuple(_uniform(rng, paper[k]) for k in ("margin_side", "margin_top", "margin_bottom"))
                set_pane(self.model, data, cfg, j, cls, x, y, yaw_deg=yaw,
                         pitch_deg=_uniform(rng, pose["pitch_deg"]),
                         roll_deg=_uniform(rng, pose["roll_deg"]),
                         height_scale=_uniform(rng, paper["height_scale"]), margins=margins)
                placed.append((x, y, cls, yaw))
                j += 1
        for k in range(j, p["max_count"]):
            set_pane(self.model, data, cfg, k, None)
        return [{"pane": i, "class": c, "x": x, "y": y, "yaw_deg": yw}
                for i, (x, y, c, yw) in enumerate(placed)]

    def _robot(self, data, rng):
        """Start on the first plaque, facing its direction of travel, with the
        layout.start jitter applied in that plaque's frame."""
        st = self.cfg["layout"]["start"]
        (cx, cy), heading = self.layout.start
        if self.r.get("robot_start", True):
            a, b = _uniform(rng, st["x_jitter"]), _uniform(rng, st["y_jitter"])
            yaw = heading + _uniform(rng, st["yaw_deg"])
        else:
            a, b, yaw = 0.0, 0.0, heading
        h = np.radians(heading)
        x = cx + a * np.cos(h) - b * np.sin(h)
        y = cy + a * np.sin(h) + b * np.cos(h)
        set_robot(self.model, data, x, y, yaw)
        return {"x": float(x), "y": float(y), "yaw_deg": float(yaw)}

    # -- public ----------------------------------------------------------------

    def reset(self, data, rng):
        """Randomise everything for a new episode. Returns a dict describing it."""
        m = self.model
        m.geom_pos[:] = self.nom_geom_pos
        m.geom_quat[:] = self.nom_geom_quat
        info = {"layout": self.layout.name, "walls": self._walls(rng)}
        self._boxes(rng)
        self._room(rng)
        self._floor(rng)
        self._lights(rng)
        self._camera(rng)
        info["panes"] = self._panes(data, rng)
        info["robot"] = self._robot(data, rng)
        mujoco.mj_forward(m, data)
        return info


class ArenaPool:
    """One compiled model (and randomiser) per arena shape, built on first use."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.layouts = all_layouts(cfg)
        self._built = {}

    def get(self, k):
        if k not in self._built:
            model, data, _ = build_arena(self.cfg, self.layouts[k])
            self._built[k] = (model, data, EpisodeRandomizer(model, self.cfg, self.layouts[k]))
        return self._built[k]

    def sample(self, rng):
        k = sample_layout_index(self.cfg, self.layouts, rng)
        return (k, *self.get(k))
