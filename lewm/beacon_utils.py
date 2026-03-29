"""Beacon panel generation, distractor patches, and shade variation.

A beacon is a thin coloured panel placed on a wall.  In the simulator it is
just another axis-aligned box, but it carries semantic metadata (identity,
colour family, facing direction) that the training pipeline records as labels.

Colour families
---------------
Each canonical beacon identity maps to a *family* of similar shades so the
model learns to recognise the beacon by its colour cluster, not an exact RGB.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional

import json
import numpy as np

from .obstacle_utils import ObstacleSpec


# --------------------------------------------------------------------------- #
# Canonical colour families  (hue centre ± jitter)
# --------------------------------------------------------------------------- #

BEACON_FAMILIES: Dict[str, Tuple[float, float, float]] = {
    "red":    (0.85, 0.15, 0.15),
    "green":  (0.15, 0.80, 0.20),
    "blue":   (0.20, 0.25, 0.85),
    "yellow": (0.90, 0.85, 0.15),
    "purple": (0.65, 0.15, 0.75),
    "orange": (0.95, 0.55, 0.10),
    "cyan":   (0.10, 0.80, 0.80),
    "white":  (0.90, 0.90, 0.90),
}


@dataclass
class BeaconSpec:
    """A coloured beacon panel with identity metadata."""
    pos: Tuple[float, float, float]
    size: Tuple[float, float, float]
    color: Tuple[float, float, float]
    identity: str                                # canonical name (e.g. "red")
    normal: Tuple[float, float] = (1.0, 0.0)    # 2-D direction the panel faces

    def to_obstacle(self) -> ObstacleSpec:
        """Convert to a plain obstacle for scene building."""
        return ObstacleSpec(pos=self.pos, size=self.size, color=self.color)


@dataclass
class BeaconLayout:
    """All beacon panels and distractors for one scene."""
    beacons: List[BeaconSpec] = field(default_factory=list)
    distractors: List[ObstacleSpec] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({
            "beacons": [asdict(b) for b in self.beacons],
            "distractors": [asdict(d) for d in self.distractors],
        })

    @staticmethod
    def from_json(s: str) -> "BeaconLayout":
        data = json.loads(s)
        beacons = [BeaconSpec(**b) for b in data.get("beacons", [])]
        distractors = [ObstacleSpec(**d) for d in data.get("distractors", [])]
        return BeaconLayout(beacons=beacons, distractors=distractors)

    def all_obstacles(self) -> List[ObstacleSpec]:
        """All panels (beacon + distractor) as plain obstacles for rendering."""
        return [b.to_obstacle() for b in self.beacons] + self.distractors


# --------------------------------------------------------------------------- #
# Colour sampling
# --------------------------------------------------------------------------- #

def sample_beacon_color(
    identity: str,
    rng: np.random.RandomState,
    shade_jitter: float = 0.10,
) -> Tuple[float, float, float]:
    """Sample a colour from a beacon family with shade variation.

    The jitter ensures the model learns identity from colour *cluster*,
    not an exact RGB triple.
    """
    base = np.array(BEACON_FAMILIES[identity], dtype=np.float64)
    jitter = rng.uniform(-shade_jitter, shade_jitter, size=3)
    c = np.clip(base + jitter, 0.0, 1.0)
    return (float(c[0]), float(c[1]), float(c[2]))


def sample_distractor_color(
    rng: np.random.RandomState,
    near_identity: Optional[str] = None,
    closeness: float = 0.15,
) -> Tuple[float, float, float]:
    """Sample a colour that looks beacon-like but isn't a true beacon.

    If *near_identity* is given, the distractor colour is close to that
    family, making it a harder negative.  Otherwise a random bright patch.
    """
    if near_identity and near_identity in BEACON_FAMILIES:
        base = np.array(BEACON_FAMILIES[near_identity], dtype=np.float64)
        offset = rng.uniform(-closeness, closeness, size=3)
        # Push at least one channel far enough that it's visually distinct
        channel = rng.randint(3)
        offset[channel] += rng.choice([-0.25, 0.25])
        c = np.clip(base + offset, 0.0, 1.0)
    else:
        c = rng.uniform(0.3, 0.95, size=3)
    return (float(c[0]), float(c[1]), float(c[2]))


# --------------------------------------------------------------------------- #
# Beacon panel placement
# --------------------------------------------------------------------------- #

def make_beacon_panel(
    wall_pos: Tuple[float, float, float],
    wall_normal: Tuple[float, float],
    identity: str,
    rng: np.random.RandomState,
    panel_width: Optional[float] = None,
    panel_height: Optional[float] = None,
    offset_fraction: float = 0.0,
    shade_jitter: float = 0.10,
) -> BeaconSpec:
    """Place a beacon panel on the surface of a wall.

    Args:
        wall_pos: centre of the host wall (x, y, z).
        wall_normal: 2-D unit-normal of the wall face the panel sits on.
        identity: canonical beacon name.
        rng: random state.
        panel_width: width override (default random 0.08–0.18 m).
        panel_height: height override (default random 0.08–0.18 m).
        offset_fraction: lateral offset as fraction of wall width (0 = centred).
        shade_jitter: per-channel colour jitter magnitude.
    """
    pw = panel_width or rng.uniform(0.08, 0.18)
    ph = panel_height or rng.uniform(0.08, 0.18)
    thickness = 0.015  # very thin panel

    nx, ny = wall_normal
    # Panel is placed slightly in front of the wall surface
    bx = wall_pos[0] + nx * 0.035
    by = wall_pos[1] + ny * 0.035
    bz = wall_pos[2]  # same height centre as wall

    # Lateral offset (perpendicular to normal)
    if abs(offset_fraction) > 1e-6:
        perp_x, perp_y = -ny, nx
        bx += perp_x * offset_fraction
        by += perp_y * offset_fraction

    # Size depends on orientation of the normal
    if abs(nx) > abs(ny):
        size = (thickness, pw, ph)
    else:
        size = (pw, thickness, ph)

    color = sample_beacon_color(identity, rng, shade_jitter)

    return BeaconSpec(
        pos=(float(bx), float(by), float(bz)),
        size=(float(size[0]), float(size[1]), float(size[2])),
        color=color,
        identity=identity,
        normal=(float(nx), float(ny)),
    )


def make_distractor_patch(
    pos: Tuple[float, float, float],
    rng: np.random.RandomState,
    near_identity: Optional[str] = None,
    size_range: Tuple[float, float] = (0.06, 0.20),
) -> ObstacleSpec:
    """A coloured patch that looks beacon-like but is not a true beacon."""
    sw = rng.uniform(size_range[0], size_range[1])
    sh = rng.uniform(size_range[0], size_range[1])
    thickness = 0.015
    orient = rng.choice(["x", "y"])
    if orient == "x":
        size = (thickness, sw, sh)
    else:
        size = (sw, thickness, sh)

    color = sample_distractor_color(rng, near_identity=near_identity)
    return ObstacleSpec(pos=pos, size=size, color=color)


# --------------------------------------------------------------------------- #
# Beacon-like wall colouring
# --------------------------------------------------------------------------- #

def beacon_like_wall_color(
    rng: np.random.RandomState,
    beacon_identities: Optional[List[str]] = None,
) -> Tuple[float, float, float]:
    """Sample a wall colour from a beacon colour family.

    Makes walls look like beacons so colour alone is not sufficient to
    identify a beacon — the model must also attend to shape / context.
    """
    if not beacon_identities:
        beacon_identities = list(BEACON_FAMILIES.keys())
    identity = rng.choice(beacon_identities)
    base = np.array(BEACON_FAMILIES[identity], dtype=np.float64)
    # Desaturate significantly so it's not an exact match
    c = base * rng.uniform(0.4, 0.7) + rng.uniform(0.0, 0.2, size=3)
    c = np.clip(c, 0.05, 0.95)
    return (float(c[0]), float(c[1]), float(c[2]))


# --------------------------------------------------------------------------- #
# Composite beacon layout generation
# --------------------------------------------------------------------------- #

def generate_beacon_layout(
    beacon_positions: List[Tuple[Tuple[float, float, float], Tuple[float, float], str]],
    rng: np.random.RandomState,
    n_distractors: int = 0,
    distractor_positions: Optional[List[Tuple[float, float, float]]] = None,
    shade_jitter: float = 0.10,
    size_variation: bool = True,
) -> BeaconLayout:
    """Build a complete beacon layout from a list of placements.

    Args:
        beacon_positions: list of (wall_centre, wall_normal, identity).
        n_distractors: how many distractor patches to add.
        distractor_positions: explicit positions; random if None.
        shade_jitter: colour jitter per beacon.
        size_variation: randomise panel width/height.
    """
    beacons: List[BeaconSpec] = []
    for wall_pos, wall_normal, identity in beacon_positions:
        pw = None if size_variation else 0.12
        ph = None if size_variation else 0.12
        offset = rng.uniform(-0.05, 0.05) if size_variation else 0.0
        b = make_beacon_panel(
            wall_pos, wall_normal, identity, rng,
            panel_width=pw, panel_height=ph,
            offset_fraction=offset, shade_jitter=shade_jitter,
        )
        beacons.append(b)

    distractors: List[ObstacleSpec] = []
    if n_distractors > 0:
        identities = [b.identity for b in beacons] if beacons else list(BEACON_FAMILIES.keys())
        if distractor_positions is None:
            distractor_positions = [
                (float(rng.uniform(-2.5, 2.5)),
                 float(rng.uniform(-2.5, 2.5)),
                 float(rng.uniform(0.10, 0.30)))
                for _ in range(n_distractors)
            ]
        for pos in distractor_positions[:n_distractors]:
            near = rng.choice(identities) if identities else None
            distractors.append(make_distractor_patch(pos, rng, near_identity=near))

    return BeaconLayout(beacons=beacons, distractors=distractors)
