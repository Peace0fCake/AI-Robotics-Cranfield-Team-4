"""Arena shapes: a chain of floor plaques, straight or with turns (an L, ...).

Every plaque is 1.0 m along the direction of travel and 1.5 m across, as the
real white plaques. The robot starts on the first plaque and the run ends at
the black line on the far edge of the last one. Each next plaque either
continues straight on, or turns left or right: it is then laid against the
side of the previous plaque, flush with that plaque's front edge or with its
back edge.

Geometry is computed on a 0.25 m grid (1.0 m = 4 cells, 1.5 m = 6 cells), so
outlines are exact and walls can be found by checking which cell edges have
floor on one side only.

    layouts = all_layouts(cfg)         # every distinct shape, straight first
    lay = layouts[k]
    lay.plaques, lay.segments, lay.start, lay.finish
"""

from dataclasses import dataclass, field
from itertools import product

import numpy as np

CELL = 0.25                          # metres per grid cell
ALONG, ACROSS = 4, 6                 # plaque size in cells: 1.0 m along travel, 1.5 m across
DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))   # heading 0..3 = +x, +y, -x, -y
STEPS = ("straight", "left_front", "left_back", "right_front", "right_back")


@dataclass
class Plaque:
    x0: int
    y0: int
    x1: int
    y1: int
    heading: int                     # direction of travel on this plaque, 0..3

    @property
    def centre(self):
        return np.array([(self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2]) * CELL

    @property
    def yaw_deg(self):
        return 90.0 * self.heading

    @property
    def area_cells(self):
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def overlaps(self, o):
        return self.x0 < o.x1 and o.x0 < self.x1 and self.y0 < o.y1 and o.y0 < self.y1


@dataclass
class Segment:
    """A straight piece of the arena outline. normal points OUT of the arena."""
    p0: np.ndarray
    p1: np.ndarray
    normal: tuple
    finish: bool

    @property
    def length(self):
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def side(self):
        """Compass name of the outward normal, used for per-side config values."""
        return {(0, 1): "north", (0, -1): "south", (1, 0): "east", (-1, 0): "west"}[self.normal]


@dataclass
class Layout:
    name: str
    plaques: list
    segments: list = field(default_factory=list)

    @property
    def start(self):
        """Centre of the first plaque, and the heading (deg) towards the second
        plaque: when the arena turns right after the start plaque, the robot
        is put down facing the way on, not the wall."""
        nxt = self.plaques[1] if len(self.plaques) > 1 else self.plaques[0]
        return self.plaques[0].centre, nxt.yaw_deg

    @property
    def finish(self):
        """The finish segment (front edge of the last plaque)."""
        return next(s for s in self.segments if s.finish)

    @property
    def bbox(self):
        xs = [v for p in self.plaques for v in (p.x0, p.x1)]
        ys = [v for p in self.plaques for v in (p.y0, p.y1)]
        return np.array([min(xs), min(ys), max(xs), max(ys)]) * CELL


def _first_plaque():
    return Plaque(-ALONG // 2, -ACROSS // 2, ALONG // 2, ACROSS // 2, heading=0)


def _next_plaque(prev, step):
    """Place the plaque after `prev` for one of STEPS."""
    if step == "straight":
        h = prev.heading
    else:
        h = (prev.heading + (1 if step.startswith("left") else -1)) % 4
    dx, dy = DIRS[h]
    # Range along the new direction of travel: just beyond prev's edge facing it.
    if dx:
        edge = prev.x1 if dx > 0 else prev.x0
        rx = (edge, edge + ALONG) if dx > 0 else (edge - ALONG, edge)
        lo, hi = prev.y0, prev.y1
    else:
        edge = prev.y1 if dy > 0 else prev.y0
        rx = (edge, edge + ALONG) if dy > 0 else (edge - ALONG, edge)
        lo, hi = prev.x0, prev.x1
    # Range across: the same as prev when going straight; after a turn prev is
    # only 4 cells long that way, so the 6-cell plaque is flush with prev's
    # front edge or with its back edge.
    if step == "straight":
        across = (lo, hi)
    else:
        pdx, pdy = DIRS[prev.heading]
        s = pdx if dx == 0 else pdy           # sign of prev's travel along this axis
        front = (hi - ACROSS, hi) if s > 0 else (lo, lo + ACROSS)
        back = (lo, lo + ACROSS) if s > 0 else (hi - ACROSS, hi)
        across = front if step.endswith("front") else back
    if dx:
        return Plaque(rx[0], across[0], rx[1], across[1], h)
    return Plaque(across[0], rx[0], across[1], rx[1], h)


def _outline(plaques):
    """Outline segments of the union of plaques; the last plaque's front edge is the finish."""
    occ = {}
    for i, p in enumerate(plaques):
        for cx in range(p.x0, p.x1):
            for cy in range(p.y0, p.y1):
                occ[(cx, cy)] = i
    last = plaques[-1]
    fdx, fdy = DIRS[last.heading]
    # Unit edges keyed by (normal, line coordinate) -> list of (position along, finish)
    edges = {}
    for (cx, cy), i in occ.items():
        for n in DIRS:
            if (cx + n[0], cy + n[1]) in occ:
                continue
            finish = i == len(plaques) - 1 and n == (fdx, fdy)
            if n[0]:
                line, pos = (cx + (n[0] > 0)), cy
            else:
                line, pos = (cy + (n[1] > 0)), cx
            edges.setdefault((n, line), []).append((pos, finish))
    segments = []
    for (n, line), items in edges.items():
        items.sort()
        run = [items[0]]
        for pos, fin in items[1:] + [(None, None)]:
            if pos is not None and pos == run[-1][0] + 1 and fin == run[-1][1]:
                run.append((pos, fin))
                continue
            a, b = run[0][0], run[-1][0] + 1
            if n[0]:
                p0, p1 = np.array([line, a]) * CELL, np.array([line, b]) * CELL
            else:
                p0, p1 = np.array([a, line]) * CELL, np.array([b, line]) * CELL
            segments.append(Segment(p0, p1, n, run[0][1]))
            if pos is not None:
                run = [(pos, fin)]
    return segments


def make_layout(steps):
    """Build the layout for a sequence of STEPS (one per plaque after the first).
    Returns None if two plaques would overlap."""
    plaques = [_first_plaque()]
    for st in steps:
        nxt = _next_plaque(plaques[-1], st)
        if any(nxt.overlaps(p) for p in plaques):
            return None
        plaques.append(nxt)
    # Origin at the centre of the second plaque: for the straight arena this is
    # the arena centre, as in configs/arena.yaml (start at x = -1, line near x = +1.45).
    ref = plaques[1] if len(plaques) > 1 else plaques[0]
    ox, oy = (ref.x0 + ref.x1) // 2, (ref.y0 + ref.y1) // 2
    plaques = [Plaque(p.x0 - ox, p.y0 - oy, p.x1 - ox, p.y1 - oy, p.heading) for p in plaques]
    lay = Layout(name="-".join(steps) or "single", plaques=plaques)
    lay.segments = _outline(plaques)
    return lay


def all_layouts(cfg):
    """Every distinct layout allowed by cfg['layout']['shapes'] (straight first)."""
    shapes = cfg.get("layout", {}).get("shapes", {})
    n = shapes.get("plaques", cfg["floor"]["plaques"])
    max_turns = shapes.get("max_turns", 0)
    out = []
    for steps in product(STEPS, repeat=n - 1):
        if sum(s != "straight" for s in steps) > max_turns:
            continue
        lay = make_layout(steps)
        if lay is not None:
            out.append(lay)
    out.sort(key=lambda lay: sum(s != "straight" for s in lay.name.split("-")))
    return out


def sample_layout_index(cfg, layouts, rng):
    """Straight with probability shapes.straight_prob, otherwise any turning layout."""
    p = cfg.get("layout", {}).get("shapes", {}).get("straight_prob", 1.0)
    if len(layouts) == 1 or rng.random() < p:
        return 0
    return int(rng.integers(1, len(layouts)))
