"""MuJoCo arena builder: config -> MjSpec -> compiled MjModel.

See docs/PROJECT_GUIDE.md sections 3 and 5.1, and configs/arena.yaml.

Conventions
  - metres, z up, origin at arena centre, x along the 3 m length.
  - every geom gets a systematic name so segmentation IDs map back to meaning:
      floor_<i>            i = 0..2, increasing x
      wall_<side>          flat wall on side north/south/east/west
      box_<side>_<i>       cardboard box i on that side
      pane_<j>_paper       white sheet of pre-allocated pane j
      pane_<j>_image_<c>_<front|back>   printed picture of class c on pane j
  - MuJoCo sizes are HALF-extents.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path="configs/arena.yaml"):
    """Read the YAML config (path relative to the repo root) and return a dict."""
    raise NotImplementedError


def add_floor(spec, cfg):
    """Add one texture + material + thin box geom per plaque, top surface at z = 0."""
    raise NotImplementedError


def add_walls(spec, cfg):
    """For each side add either one flat wall geom or a row of box geoms.

    Walls sit just OUTSIDE the inner 3.0 x 1.5 m area.
    """
    raise NotImplementedError


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
    One front/back pair per class; only the chosen class is shown per episode.

    Unused panes are parked out of sight. Per episode, with no recompile:
      pose          data.mocap_pos / data.mocap_quat
      paper size    model.geom_size (also update model.geom_rbound)
      picture size and place on the paper   model.geom_size / model.geom_pos
      which class   model.geom_rgba alpha (0 hides a geom)
    """
    raise NotImplementedError


def add_robot_camera(spec, cfg):
    """Add a mocap body 'robot' carrying the camera cfg['camera']['name']."""
    raise NotImplementedError


def build_arena(cfg):
    """Build the full scene. Returns (model, data, spec)."""
    raise NotImplementedError


def geom_classes(model):
    """Return {geom_id: label} from geom names.

    Labels: 'floor', 'wall', 'box', 'paper', 'human', 'car', 'building'.
    """
    raise NotImplementedError
