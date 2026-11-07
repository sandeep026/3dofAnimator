# import matplotlib.pyplot as plt
# import motion_viz as mv
#
# states, t = mv.circular_turn(speed=3.0, turn_rate_deg_s=12.0, duration=10.0)
#
## Save only. No window.
# mv.save_motion("car_turn.gif", "car", states, length=4.5, fps=20, step=2)
# mv.save_motion("quad_turn.gif", "quadcopter", states, length=2.6, fps=20)
#
## Optional live view. Only when you want it.
##anim = mv.animate_motion("quadcopter", states, length=2.6)
##plt.show()

"""Quadcopter avoiding circle and square obstacles while following a moving quadcopter.

Only geometry from shapes.toml is drawn:
  "quadcopter" for the chaser and the target, "circle" and "square" for obstacles.

The chaser's path comes from a small potential-field planner. It stands in for
whatever produces your trajectory: any (N, 3) array of [x, y, theta_deg] works.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import motion_viz as mv
import numpy as np
from matplotlib import animation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

DT = 0.05
QUAD_LENGTH = 2.6


# --------------------------------------------------------------------------- #
# Obstacles: "circle" and "square" from shapes.toml
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Obstacle:
    shape: str  # "circle" or "square", as named in shapes.toml
    x: float
    y: float
    length: float  # diameter for a circle, side for a square
    theta_deg: float = 0.0

    def __post_init__(self):
        if self.shape not in ("circle", "square"):
            raise ValueError(
                f"obstacle shape must be 'circle' or 'square', got {self.shape!r}"
            )


def _half_extent(shape, length):
    """Circle radius, or square half-side, at the requested length."""
    kind, geom, _ = shape["parts"][0]
    scale = length / shape["length"]
    if kind == "circle":
        return float(geom[1]) * scale
    return float(np.abs(geom).max()) * scale


def signed_distance(obs, shapes, x, y):
    """Distance from (x, y) to the obstacle edge. Negative inside."""
    half = _half_extent(shapes[obs.shape], obs.length)
    dx, dy = x - obs.x, y - obs.y
    if obs.shape == "circle":
        return math.hypot(dx, dy) - half
    # Express the offset in the square's own frame by undoing its rotation.
    t = math.radians(-obs.theta_deg)
    c, s = math.cos(t), math.sin(t)
    qx, qy = c * dx - s * dy, s * dx + c * dy
    wx, wy = abs(qx) - half, abs(qy) - half
    return math.hypot(max(wx, 0.0), max(wy, 0.0)) + min(max(wx, wy), 0.0)


def _outward_normal(obs, shapes, x, y, eps=1e-4):
    """Unit vector pointing away from the obstacle, from a finite-difference gradient."""
    gx = (
        signed_distance(obs, shapes, x + eps, y)
        - signed_distance(obs, shapes, x - eps, y)
    ) / (2 * eps)
    gy = (
        signed_distance(obs, shapes, x, y + eps)
        - signed_distance(obs, shapes, x, y - eps)
    ) / (2 * eps)
    norm = math.hypot(gx, gy)
    return np.array([gx, gy]) / norm if norm > 0 else np.array([1.0, 0.0])


# --------------------------------------------------------------------------- #
# Target and planner
# --------------------------------------------------------------------------- #


def orbit(center, radius, speed, duration, start_deg=180.0, ccw=True):
    """Target flying a circle. Returns (states, times) with states of shape (N, 3)."""
    t = np.arange(0.0, duration + DT / 2, DT)
    w = (speed / radius) * (1.0 if ccw else -1.0)
    ang = math.radians(start_deg) + w * t
    xy = np.column_stack(
        [center[0] + radius * np.cos(ang), center[1] + radius * np.sin(ang)]
    )
    vel = np.gradient(xy, DT, axis=0)
    theta = (
        np.degrees(np.arctan2(vel[:, 1], vel[:, 0])) - 90.0
    )  # nose-along-+y convention
    return np.column_stack([xy, theta]), t


@dataclass(frozen=True)
class Gains:
    k_goal: float = 1.0  # pull toward the point behind the target (1/s)
    k_rep: float = 4.0  # strength of obstacle push
    influence: float = 3.0  # obstacle push starts this far from the drone edge (m)
    target_influence: float = 2.0  # same, for the other drone
    k_swirl: float = 0.5  # sideways bias that helps slip around obstacles
    standoff: float = 6.0  # how far behind the target to fly (m)
    min_gap: float = 1.0  # gap kept between the two drone bodies (m)
    v_max: float = 3.0  # m/s
    a_max: float = 2.0  # m/s^2


def _push(d, normal, gains, influence):
    """Repulsive force from a surface at clearance d (Khatib-style potential)."""
    if d >= influence:
        return np.zeros(2)
    d = max(d, 0.05)
    mag = gains.k_rep * (1.0 / d - 1.0 / influence) / (d * d)
    tangent = np.array([-normal[1], normal[0]])
    return mag * (normal + gains.k_swirl * tangent)


def _reach(shapes, length):
    """Largest distance from a quadcopter's centre to its edge, from the TOML geometry."""
    ax = Figure().add_subplot()
    return mv.Drawable(ax, shapes["quadcopter"], length).reach


def plan_chaser(start, target, obstacles, shapes, gains=Gains(), length=QUAD_LENGTH):
    """Return (N, 3) chaser states [x, y, theta_deg], one per target sample."""
    reach = _reach(shapes, length)
    tp = target[:, :2]
    tv = np.gradient(tp, DT, axis=0)

    p = np.asarray(start, dtype=float)
    v = np.zeros(2)
    heading = math.degrees(math.atan2(tp[0, 1] - p[1], tp[0, 0] - p[0])) - 90.0
    out = np.empty((len(target), 3))

    for k in range(len(target)):
        tgt = tp[k]
        speed_t = math.hypot(tv[k, 0], tv[k, 1])
        fwd = tv[k] / speed_t if speed_t > 1e-6 else np.array([0.0, 1.0])
        goal = tgt - gains.standoff * fwd

        force = gains.k_goal * (goal - p)

        for obs in obstacles:
            d = signed_distance(obs, shapes, p[0], p[1]) - reach
            if d < gains.influence:
                force += _push(
                    d, _outward_normal(obs, shapes, p[0], p[1]), gains, gains.influence
                )

        offset = p - tgt
        dist = math.hypot(offset[0], offset[1])
        gap = dist - (2 * reach + gains.min_gap)
        if gap < gains.target_influence:
            normal = offset / dist if dist > 1e-9 else np.array([1.0, 0.0])
            force += _push(gap, normal, gains, gains.target_influence)

        # Limit the speed command, then limit how fast the velocity can change.
        speed_cmd = math.hypot(force[0], force[1])
        if speed_cmd > gains.v_max:
            force *= gains.v_max / speed_cmd
        dv = force - v
        dv_norm = math.hypot(dv[0], dv[1])
        limit = gains.a_max * DT
        if dv_norm > limit:
            dv *= limit / dv_norm
        v = v + dv

        if math.hypot(v[0], v[1]) > 0.05:
            heading = math.degrees(math.atan2(v[1], v[0])) - 90.0
        out[k] = (p[0], p[1], heading)
        p = p + v * DT

    return out


def clearance(chaser, target, obstacles, shapes, length=QUAD_LENGTH):
    """Smallest gap to any obstacle edge, and to the target body, over the whole run."""
    reach = _reach(shapes, length)
    obstacle_gap = min(
        signed_distance(obs, shapes, x, y) - reach
        for obs in obstacles
        for x, y in chaser[:, :2]
    )
    target_gap = float(np.hypot(*(chaser[:, :2] - target[:, :2]).T).min()) - 2 * reach
    return obstacle_gap, target_gap


# --------------------------------------------------------------------------- #
# Drawing and output
# --------------------------------------------------------------------------- #


def _build_scene(fig, chaser, target, obstacles, shapes, length=QUAD_LENGTH):
    """Draw obstacles once, create both quadcopters and trails, return the update function."""
    ax = fig.add_subplot()
    ax.set_aspect("equal")
    ax.set_axis_off()
    title = ax.set_title("")

    for obs in obstacles:
        mv.Drawable(ax, shapes[obs.shape], obs.length).move(obs.x, obs.y, obs.theta_deg)

    chaser_body = mv.Drawable(ax, shapes["quadcopter"], length)
    target_body = mv.Drawable(ax, shapes["quadcopter"], length)
    for patch in target_body.patches:
        patch.set_alpha(0.45)  # the target is drawn faded so the chaser stands out

    (chaser_trail,) = ax.plot([], [], lw=1.2, color="#1F618D")
    (target_trail,) = ax.plot([], [], lw=1.0, ls="--", color="#7F8C8D")

    pts = [chaser[:, :2], target[:, :2]]
    pts += [
        np.array([[o.x - o.length, o.y - o.length], [o.x + o.length, o.y + o.length]])
        for o in obstacles
    ]
    pts = np.vstack(pts)
    lo, hi = pts.min(axis=0) - 2.0, pts.max(axis=0) + 2.0
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])

    def update(i):
        chaser_body.move(*chaser[i])
        target_body.move(*target[i])
        chaser_trail.set_data(chaser[: i + 1, 0], chaser[: i + 1, 1])
        target_trail.set_data(target[: i + 1, 0], target[: i + 1, 1])
        title.set_text(f"t = {i * DT:.1f} s   (solid: chaser, faded: target)")

    return update


def save_scene(path, chaser, target, obstacles, shapes, fps=10, step=2, dpi=80):
    """Save the chase to .gif or .mp4. No window is opened.

    With DT = 0.05, step = 2 and fps = 10 the file plays back in real time.
    """
    path = Path(path).expanduser().resolve()
    writer = mv.WRITERS.get(path.suffix.lower())
    if writer is None:
        raise ValueError(
            f"unsupported file type {path.suffix!r}; use one of {sorted(mv.WRITERS)}"
        )
    if not animation.writers.is_available(writer):
        raise RuntimeError(
            f"the '{writer}' writer is not installed; it is needed for {path.suffix}"
        )

    fig = Figure(figsize=(7, 7), dpi=dpi)
    FigureCanvasAgg(fig)
    update = _build_scene(fig, chaser, target, obstacles, shapes)
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=range(0, len(chaser), step),
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=writer, fps=fps, dpi=dpi)
    print(f"Animation saved to: {path}")
    return path


def show_scene(chaser, target, obstacles, shapes, interval=30):
    """Live window. Keep the returned object, then call plt.show()."""
    fig = plt.figure(figsize=(7, 7))
    update = _build_scene(fig, chaser, target, obstacles, shapes)
    return animation.FuncAnimation(
        fig, update, frames=len(chaser), interval=interval, blit=False, repeat=True
    )


# --------------------------------------------------------------------------- #
# Scene
# --------------------------------------------------------------------------- #

START = (-14.0, -8.0)
OBSTACLES = [
    Obstacle("circle", -9.0, -2.0, 2.6),
    Obstacle("square", -2.0, 2.0, 3.0, 30.0),
    Obstacle("circle", 1.0, -4.5, 2.2),
    Obstacle("square", 5.0, 5.5, 2.4, -20.0),
    Obstacle("circle", 9.0, -1.0, 2.8),
    Obstacle("square", -6.0, -9.0, 2.0, 0.0),
]


def main():
    shapes = mv.load_shapes()
    target, _ = orbit(center=(4.0, 0.0), radius=10.0, speed=2.0, duration=60.0)
    chaser = plan_chaser(START, target, OBSTACLES, shapes)

    obs_gap, tgt_gap = clearance(chaser, target, OBSTACLES, shapes)
    print(f"closest approach to an obstacle edge: {obs_gap:.2f} m")
    print(f"closest approach to the target:       {tgt_gap:.2f} m")

    save_scene("obstacle_chase.gif", chaser, target, OBSTACLES, shapes)

    # Optional live view:
    # anim = show_scene(chaser, target, OBSTACLES, shapes)
    # plt.show()


if __name__ == "__main__":
    main()
