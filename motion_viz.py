"""Fast 3-DOF shape plotting: load a shape, place it with translation and rotation, plot."""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon


from matplotlib import animation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

WRITERS = {".gif": "pillow", ".mp4": "ffmpeg"}

SHAPES_FILE = Path(__file__).with_name("shapes.toml")
RESERVED = ("defaults", "gallery")
GEOMETRY_KEYS = ("kind", "verts", "center", "radius")


def load_shapes(path=SHAPES_FILE):
    """Return {name: {"length": L, "parts": [(kind, geometry, style), ...]}}."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    defaults = data.get("defaults", {})
    shapes = {}
    for name, table in data.items():
        if name in RESERVED or not isinstance(table, dict) or "parts" not in table:
            continue
        parts = []
        for raw in table["parts"]:
            kind = raw["kind"]
            style = {**defaults.get(kind, {}),
                     **{k: v for k, v in raw.items() if k not in GEOMETRY_KEYS}}
            if kind == "polygon":
                geom = np.asarray(raw["verts"], dtype=float)
            else:
                geom = (np.asarray(raw["center"], dtype=float), float(raw["radius"]))
            parts.append((kind, geom, style))
        shapes[name] = {"length": float(table["length"]), "parts": parts}
    return shapes


def _rotation(theta_deg):
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]])


class Drawable:
    """Patches for one shape. Scaling happens once here; move() only updates positions."""

    def __init__(self, ax, shape, length=None):
        scale = (shape["length"] if length is None else length) / shape["length"]
        self._parts = []   # (patch, local geometry, is_circle)
        reach = 0.0
        for kind, geom, style in shape["parts"]:
            if kind == "polygon":
                pts = geom * scale
                patch = Polygon(pts, closed=True, **style)
                reach = max(reach, float(np.hypot(pts[:, 0], pts[:, 1]).max()))
                self._parts.append((patch, pts, False))
            else:
                center, radius = geom
                center = center * scale
                radius = radius * scale
                patch = Circle(center, radius, **style)
                reach = max(reach, float(np.hypot(*center)) + radius)
                self._parts.append((patch, center, True))
            ax.add_patch(patch)
        self.patches = [p for p, _, _ in self._parts]
        self.reach = reach

    def move(self, x, y, theta_deg):
        """Rotate about the local origin, then translate to (x, y)."""
        R = _rotation(theta_deg)
        t = np.array((x, y), dtype=float)
        for patch, local, is_circle in self._parts:
            world = local @ R.T + t
            if is_circle:
                patch.set_center(world)
            else:
                patch.set_xy(world)


def _clean(ax):
    ax.set_aspect("equal")
    ax.set_axis_off()


def plot_pose(name, x, y, theta_deg, length=None, shapes=None):
    """Plot one shape at one pose."""
    shapes = shapes or load_shapes()
    fig, ax = plt.subplots()
    d = Drawable(ax, shapes[name], length)
    d.move(x, y, theta_deg)
    r = d.reach * 1.3
    ax.set_xlim(x - r, x + r)
    ax.set_ylim(y - r, y + r)
    _clean(ax)
    return fig


def _build_motion(fig, name, states, length, trail, shapes):
    """Draw the shape and optional trail on fig. Returns (update_fn, frame_count)."""
    shapes = shapes or load_shapes()
    states = np.asarray(states, dtype=float)
    xs, ys = states[:, 0], states[:, 1]

    ax = fig.add_subplot()
    _clean(ax)
    d = Drawable(ax, shapes[name], length)
    artists = list(d.patches)

    pad = d.reach * 1.5
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)

    line = None
    if trail:
        line, = ax.plot([], [], lw=1.0, color="#7F8C8D")
        artists.append(line)

    def update(i):
        x, y, th = states[i]
        d.move(x, y, th)
        if line is not None:
            line.set_data(xs[:i + 1], ys[:i + 1])
        return artists

    return update, len(states)


def animate_motion(name, states, length=None, interval=30, trail=True,
                   blit=False, shapes=None):
    """Live animation in a matplotlib window. Call plt.show() to display it.

    Keep the returned object alive, or it will be garbage collected.
    """
    fig = plt.figure()
    update, n = _build_motion(fig, name, states, length, trail, shapes)
    return FuncAnimation(fig, update, frames=n, interval=interval,
                         blit=blit, repeat=True)


def save_motion(path, name, states, length=None, fps=20, step=1,
                trail=True, dpi=100, shapes=None):
    """Save the motion to a .gif or .mp4 file. No window is opened.

    fps   : frames per second in the output file
    step  : use every step-th sample (reduces file size and render time)
    Real-time playback needs fps = 1 / (dt * step). For dt = 0.05 and step = 1, fps = 20.
    """
    path = Path(path).expanduser().resolve()
    writer = WRITERS.get(path.suffix.lower())
    if writer is None:
        raise ValueError(f"unsupported file type {path.suffix!r}; use one of {sorted(WRITERS)}")
    if not animation.writers.is_available(writer):
        raise RuntimeError(f"the '{writer}' writer is not installed; it is needed for {path.suffix}")

    states = np.asarray(states, dtype=float)[::step]

    fig = Figure(figsize=(6, 6), dpi=dpi)
    FigureCanvasAgg(fig)
    update, n = _build_motion(fig, name, states, length, trail, shapes)
    anim = FuncAnimation(fig, update, frames=n, interval=1000 / fps,
                         blit=False, repeat=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=writer, fps=fps, dpi=dpi)
    print(f"Animation saved to: {path}")
    return path


def plot_snapshots(name, states, every, length=None, shapes=None):
    """Draw the shape every `every` samples on one axes, fading older ones.

    `every` is an integer count of samples: every = round(seconds / dt).
    """
    shapes = shapes or load_shapes()
    states = np.asarray(states, dtype=float)
    idx = list(range(0, len(states), int(every)))

    fig, ax = plt.subplots()
    _clean(ax)
    ax.plot(states[:, 0], states[:, 1], color="#BDC3C7", lw=1.0)

    reach = 0.0
    for k, i in enumerate(idx):
        d = Drawable(ax, shapes[name], length)
        d.move(*states[i])
        alpha = 0.3 + 0.7 * (k + 1) / len(idx)
        for p in d.patches:
            p.set_alpha(alpha)
        reach = d.reach

    pad = reach * 1.5
    ax.set_xlim(states[:, 0].min() - pad, states[:, 0].max() + pad)
    ax.set_ylim(states[:, 1].min() - pad, states[:, 1].max() + pad)
    return fig


def plot_all_shapes(length=None, shapes=None, columns=4):
    """One figure with every shape at the origin, on the same kind of scale."""
    shapes = shapes or load_shapes()
    names = list(shapes)
    rows = math.ceil(len(names) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        d = Drawable(ax, shapes[name], length)
        r = d.reach * 1.2
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        _clean(ax)
        ax.set_title(name.replace("_", " ").title())
    for ax in axes.flat[len(names):]:
        ax.remove()
    return fig


def check_motion(shapes=None):
    """Fail loudly if moving a shape does not move its geometry.

    Car body vertex (0.45, 2.25), rotated 90 degrees CCW and shifted by (5, 0),
    must land at (2.75, 0.45).
    """
    shapes = shapes or load_shapes()
    fig, ax = plt.subplots()
    d = Drawable(ax, shapes["car"])
    d.move(5.0, 0.0, 90.0)
    xy = d.patches[0].get_xy()
    target = np.array([2.75, 0.45])
    assert np.min(np.hypot(*(xy - target).T)) < 1e-9, "shape did not move as expected"
    plt.close(fig)


def circular_turn(speed, turn_rate_deg_s, duration, dt=0.05):
    """Example trajectory: constant-speed turn. Returns (states, times)."""
    t = np.arange(0.0, duration + dt / 2, dt)
    w = math.radians(turn_rate_deg_s)
    th = w * t
    states = np.column_stack([speed / w * (np.cos(th) - 1),
                              speed / w * np.sin(th),
                              np.degrees(th)])
    return states, t


if __name__ == "__main__":
    check_motion()
    shapes = load_shapes()
    states, t = circular_turn(speed=3.0, turn_rate_deg_s=12.0, duration=30.0)

    plot_pose("car", 0, 0, 45, shapes=shapes)
    plot_snapshots("car", states, every=100, length=4.5, shapes=shapes)  # every 5 s at dt = 0.05
    anim = animate_motion("quadcopter", states, length=2.6, shapes=shapes)
    plt.show()


    import time
    t0 = time.perf_counter()
    anim = animate_motion("quadcopter", states, length=2.6)
    print(f"setup: {time.perf_counter() - t0:.3f} s")


    import time
    import matplotlib
    matplotlib.use("TkAgg")  # or your usual interactive backend; Agg does not show live timing
    import matplotlib.pyplot as plt
    import motion_viz as mv

    shapes = mv.load_shapes()
    states, _ = mv.circular_turn(speed=3.0, turn_rate_deg_s=12.0, duration=30.0)

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    d = mv.Drawable(ax, shapes["quadcopter"], length=2.6)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    fig.canvas.draw()

    # 1. Pose update only
    t0 = time.perf_counter()
    for x, y, th in states:
        d.move(x, y, th)
    print(f"move only:   {(time.perf_counter() - t0) / len(states) * 1e6:.1f} us per frame")

    # 2. Full redraw of the figure
    t0 = time.perf_counter()
    for x, y, th in states[:100]:
        d.move(x, y, th)
        fig.canvas.draw()
    print(f"full redraw: {(time.perf_counter() - t0) / 100 * 1e3:.2f} ms per frame")