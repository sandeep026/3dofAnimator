"""Tests for motion_viz.py.

Setup and run:
    pip install pytest pillow
    pytest -v test_motion_viz.py

Uses the non-interactive Agg backend, so no windows open.
"""

from __future__ import annotations

import math
import textwrap

import matplotlib

matplotlib.use("Agg")  # must run before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

import motion_viz as mv  # noqa: E402

ALL_SHAPES = [
    "car", "airplane", "rocket", "ship", "satellite",
    "quadcopter", "rectangle", "square", "triangle", "circle",
]


# --------------------------------------------------------------------------- #
# Helpers. These restate the maths independently of motion_viz, so a bug in
# the module cannot hide itself by matching the same bug in the helper.
# --------------------------------------------------------------------------- #

def rot(theta_deg):
    t = math.radians(theta_deg)
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


def world(local, x, y, theta_deg):
    """Rotate local points CCW about the origin, then translate by (x, y)."""
    return np.asarray(local, dtype=float) @ rot(theta_deg).T + np.array([x, y])


def scale_for(shape, length=None):
    return (shape["length"] if length is None else length) / shape["length"]


def expected_reach(shape, length=None):
    s = scale_for(shape, length)
    best = 0.0
    for kind, geom, _ in shape["parts"]:
        if kind == "polygon":
            best = max(best, float(np.max(np.hypot(geom[:, 0] * s, geom[:, 1] * s))))
        else:
            center, radius = geom
            best = max(best, float(np.hypot(*(center * s))) + radius * s)
    return best


def polygon_xy(patch):
    """Polygon vertices without matplotlib's duplicated closing point."""
    xy = np.asarray(patch.get_xy(), dtype=float)
    if len(xy) > 1 and np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    return xy


def assert_patches_match(patches, shape, x, y, theta_deg, length=None):
    """Every drawn patch must sit where the reference transform puts it."""
    s = scale_for(shape, length)
    assert len(patches) == len(shape["parts"])
    for patch, (kind, geom, _) in zip(patches, shape["parts"]):
        if kind == "polygon":
            np.testing.assert_allclose(polygon_xy(patch), world(geom * s, x, y, theta_deg), atol=1e-9)
        else:
            center, _radius = geom
            np.testing.assert_allclose(patch.get_center(), world(center * s, x, y, theta_deg), atol=1e-9)


def write_toml(tmp_path, text):
    path = tmp_path / "shapes.toml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def shapes():
    return mv.load_shapes()


@pytest.fixture
def trajectory():
    return mv.circular_turn(speed=3.0, turn_rate_deg_s=12.0, duration=30.0)


@pytest.fixture
def ax():
    _, axes = plt.subplots()
    return axes


@pytest.fixture
def captured_anim(monkeypatch):
    """Replace FuncAnimation with a recorder, so the update function can be called directly."""
    record = {}

    class RecordingAnimation:
        def __init__(self, fig, func, frames, interval, blit, repeat):
            record.update(fig=fig, func=func, frames=frames,
                          interval=interval, blit=blit, repeat=repeat)

    monkeypatch.setattr(mv, "FuncAnimation", RecordingAnimation)
    return record


# --------------------------------------------------------------------------- #
# _rotation
# --------------------------------------------------------------------------- #

class TestRotation:
    def test_zero_is_identity(self):
        np.testing.assert_allclose(mv._rotation(0.0), np.eye(2), atol=1e-12)

    def test_quarter_turn_is_counter_clockwise(self):
        np.testing.assert_allclose(mv._rotation(90.0) @ [1.0, 0.0], [0.0, 1.0], atol=1e-12)

    @pytest.mark.parametrize("angle", [0.0, 37.0, 90.0, 180.0, 270.0, -45.0, 1234.5])
    def test_is_orthonormal_with_unit_determinant(self, angle):
        R = mv._rotation(angle)
        np.testing.assert_allclose(R @ R.T, np.eye(2), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)

    def test_full_turn_equals_no_turn(self):
        np.testing.assert_allclose(mv._rotation(360.0), mv._rotation(0.0), atol=1e-12)


# --------------------------------------------------------------------------- #
# load_shapes
# --------------------------------------------------------------------------- #

class TestLoadShapes:
    def test_default_file_is_beside_module(self):
        assert mv.SHAPES_FILE.name == "shapes.toml"
        assert mv.SHAPES_FILE.is_file()

    def test_finds_exactly_the_expected_shapes(self, shapes):
        assert set(shapes) == set(ALL_SHAPES)

    def test_reserved_tables_are_not_shapes(self, shapes):
        assert "gallery" not in shapes
        assert "defaults" not in shapes

    @pytest.mark.parametrize("name", ALL_SHAPES)
    def test_length_is_positive_float(self, shapes, name):
        assert isinstance(shapes[name]["length"], float)
        assert shapes[name]["length"] > 0

    @pytest.mark.parametrize("name", ALL_SHAPES)
    def test_parts_are_well_formed(self, shapes, name):
        assert len(shapes[name]["parts"]) > 0
        for kind, geom, style in shapes[name]["parts"]:
            assert kind in ("polygon", "circle")
            assert isinstance(style, dict)
            if kind == "polygon":
                assert isinstance(geom, np.ndarray)
                assert geom.dtype == float
                assert geom.ndim == 2 and geom.shape[1] == 2
                assert len(geom) >= 3
                assert np.all(np.isfinite(geom))
            else:
                center, radius = geom
                assert center.shape == (2,)
                assert radius > 0

    def test_geometry_keys_are_removed_from_style(self, shapes):
        for _, _, style in shapes["car"]["parts"]:
            assert not {"kind", "verts", "center", "radius"} & set(style)

    def test_defaults_are_merged_into_style(self, shapes):
        # car body sets facecolor itself but inherits edgecolor and linewidth from [defaults.polygon]
        _, _, body_style = shapes["car"]["parts"][0]
        assert body_style["facecolor"] == "#C0392B"
        assert body_style["edgecolor"] == "#1B2631"
        assert body_style["linewidth"] == 1.0

    def test_circle_defaults_apply(self, shapes):
        # the 'circle' shape's part sets no zorder, so it inherits [defaults.circle]
        _, _, style = shapes["circle"]["parts"][0]
        assert style["zorder"] == 3

    def test_part_values_override_defaults(self, shapes):
        # quadcopter body overrides the circle default zorder of 3 with 4
        _, _, style = shapes["quadcopter"]["parts"][-1]
        assert style["zorder"] == 4

    def test_part_order_is_preserved(self, shapes):
        first_airplane_part = shapes["airplane"]["parts"][0][2]
        assert first_airplane_part["facecolor"] == "#ECF0F1"  # fuselage is first in the file

    def test_alpha_from_toml_is_kept(self, shapes):
        rotor_style = shapes["quadcopter"]["parts"][2][2]
        assert rotor_style["alpha"] == pytest.approx(0.85)

    def test_custom_file_loads_and_skips_non_shapes(self, tmp_path):
        path = write_toml(tmp_path, """
            title = "not a shape"

            [notes]
            text = "no parts array here"

            [tri]
            length = 2.0

            [[tri.parts]]
            kind = "polygon"
            verts = [[0.0, 1.0], [0.6, -0.6], [-0.6, -0.6]]
        """)
        shapes = mv.load_shapes(path)
        assert list(shapes) == ["tri"]
        assert shapes["tri"]["length"] == 2.0

    def test_accepts_string_path(self, tmp_path):
        path = write_toml(tmp_path, """
            [tri]
            length = 1.0

            [[tri.parts]]
            kind = "polygon"
            verts = [[0.0, 1.0], [0.6, -0.6], [-0.6, -0.6]]
        """)
        assert "tri" in mv.load_shapes(str(path))

    def test_missing_length_raises(self, tmp_path):
        path = write_toml(tmp_path, """
            [bad]
            [[bad.parts]]
            kind = "polygon"
            verts = [[0.0, 1.0], [0.6, -0.6], [-0.6, -0.6]]
        """)
        with pytest.raises(KeyError):
            mv.load_shapes(path)

    def test_missing_kind_raises(self, tmp_path):
        path = write_toml(tmp_path, """
            [bad]
            length = 1.0
            [[bad.parts]]
            verts = [[0.0, 1.0], [0.6, -0.6], [-0.6, -0.6]]
        """)
        with pytest.raises(KeyError):
            mv.load_shapes(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            mv.load_shapes(tmp_path / "does_not_exist.toml")


# --------------------------------------------------------------------------- #
# Drawable
# --------------------------------------------------------------------------- #

class TestDrawable:
    @pytest.mark.parametrize("name", ALL_SHAPES)
    def test_one_patch_per_part(self, ax, shapes, name):
        d = mv.Drawable(ax, shapes[name])
        assert len(d.patches) == len(shapes[name]["parts"])
        assert len(ax.patches) == len(d.patches)

    @pytest.mark.parametrize("name", ALL_SHAPES)
    def test_reach_matches_reference(self, ax, shapes, name):
        d = mv.Drawable(ax, shapes[name], length=3.0)
        assert d.reach == pytest.approx(expected_reach(shapes[name], 3.0))

    def test_native_length_is_default(self, ax, shapes):
        d = mv.Drawable(ax, shapes["car"])
        assert d.reach == pytest.approx(expected_reach(shapes["car"]))

    def test_circle_reach_includes_radius(self, ax, shapes):
        # quadcopter: farthest point is a rotor centre at (0.9, 0.9) plus radius 0.4
        d = mv.Drawable(ax, shapes["quadcopter"])
        assert d.reach == pytest.approx(math.hypot(0.9, 0.9) + 0.4)

    def test_length_scales_geometry_proportionally(self, ax, shapes):
        shape = shapes["car"]
        d = mv.Drawable(ax, shape, length=2 * shape["length"])
        d.move(0.0, 0.0, 0.0)
        assert_patches_match(d.patches, shape, 0.0, 0.0, 0.0, length=2 * shape["length"])
        assert d.reach == pytest.approx(2 * expected_reach(shape))

    @pytest.mark.parametrize("name", ALL_SHAPES)
    @pytest.mark.parametrize("pose", [(0, 0, 0), (5, -3, 0), (2, 1, 90),
                                      (-4, 6, 137.5), (1, 1, -270)])
    def test_move_applies_rotation_then_translation(self, ax, shapes, name, pose):
        d = mv.Drawable(ax, shapes[name])
        d.move(*pose)
        assert_patches_match(d.patches, shapes[name], *pose)

    def test_heading_90_points_nose_along_minus_x(self, ax, shapes):
        # airplane nose is at local (0, 2). A 90 degree CCW turn sends it to (-2, 0).
        d = mv.Drawable(ax, shapes["airplane"])
        d.move(0.0, 0.0, 90.0)
        np.testing.assert_allclose(polygon_xy(d.patches[0])[0], [-2.0, 0.0], atol=1e-9)

    def test_move_is_absolute_not_cumulative(self, ax, shapes):
        d = mv.Drawable(ax, shapes["car"])
        for pose in [(1, 2, 30), (9, -7, 200), (0.5, 0.5, 75), (3, 3, 3)]:
            d.move(*pose)
        assert_patches_match(d.patches, shapes["car"], 3, 3, 3)

    def test_repeated_identical_moves_are_stable(self, ax, shapes):
        d = mv.Drawable(ax, shapes["ship"])
        d.move(1, 2, 30)
        first = [polygon_xy(p).copy() for p in d.patches]
        d.move(1, 2, 30)
        for before, patch in zip(first, d.patches):
            np.testing.assert_allclose(polygon_xy(patch), before, atol=1e-12)

    def test_style_from_toml_reaches_patches(self, ax, shapes):
        d = mv.Drawable(ax, shapes["quadcopter"])
        front_left_rotor = d.patches[2]
        assert front_left_rotor.get_alpha() == pytest.approx(0.85)
        np.testing.assert_allclose(front_left_rotor.get_facecolor()[:3], to_rgb("#E74C3C"), atol=1e-6)
        assert d.patches[-1].get_zorder() == 4  # central body

        car = mv.Drawable(ax, shapes["car"])
        assert car.patches[1].get_zorder() == 3  # windscreen


# --------------------------------------------------------------------------- #
# plot_pose
# --------------------------------------------------------------------------- #

class TestPlotPose:
    def test_returns_figure_with_one_axes(self, shapes):
        fig = mv.plot_pose("car", 0, 0, 0, shapes=shapes)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1

    def test_shape_is_drawn_at_pose(self, shapes):
        fig = mv.plot_pose("airplane", 10, -4, 45, shapes=shapes)
        assert_patches_match(fig.axes[0].patches, shapes["airplane"], 10, -4, 45)

    def test_view_is_centred_on_pose(self, shapes):
        fig = mv.plot_pose("airplane", 10, -4, 45, shapes=shapes)
        ax = fig.axes[0]
        assert np.mean(ax.get_xlim()) == pytest.approx(10)
        assert np.mean(ax.get_ylim()) == pytest.approx(-4)

    def test_view_half_width_is_1_3_times_reach(self, shapes):
        fig = mv.plot_pose("airplane", 0, 0, 0, shapes=shapes)
        ax = fig.axes[0]
        half_width = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 2
        assert half_width == pytest.approx(1.3 * expected_reach(shapes["airplane"]))

    def test_aspect_is_equal(self, shapes):
        fig = mv.plot_pose("car", 0, 0, 0, shapes=shapes)
        assert fig.axes[0].get_aspect() == 1.0

    def test_length_argument_scales_view(self, shapes):
        fig = mv.plot_pose("car", 0, 0, 0, length=9.0, shapes=shapes)
        ax = fig.axes[0]
        half_width = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 2
        assert half_width == pytest.approx(1.3 * expected_reach(shapes["car"], 9.0))


# --------------------------------------------------------------------------- #
# animate_motion
# --------------------------------------------------------------------------- #

class TestAnimateMotion:
    def test_passes_frame_count_and_blit_options(self, captured_anim, shapes, trajectory):
        states, _ = trajectory
        mv.animate_motion("quadcopter", states, length=2.6, shapes=shapes)
        assert captured_anim["frames"] == len(states)
        assert captured_anim["blit"] is True
        assert captured_anim["repeat"] is True
        assert captured_anim["interval"] == 30

    def test_interval_is_forwarded(self, captured_anim, shapes, trajectory):
        states, _ = trajectory
        mv.animate_motion("car", states, interval=77, shapes=shapes)
        assert captured_anim["interval"] == 77

    def test_each_frame_places_shape_at_its_state(self, captured_anim, shapes, trajectory):
        states, _ = trajectory
        mv.animate_motion("quadcopter", states, length=2.6, shapes=shapes)
        update = captured_anim["func"]
        patches = captured_anim["fig"].axes[0].patches
        for i in (0, 1, len(states) // 2, len(states) - 1):
            update(i)
            assert_patches_match(patches, shapes["quadcopter"], *states[i], length=2.6)

    def test_trail_grows_by_one_sample_per_frame(self, captured_anim, shapes, trajectory):
        states, _ = trajectory
        mv.animate_motion("car", states, length=4.5, shapes=shapes)
        update = captured_anim["func"]
        for i in (0, 5, 100, len(states) - 1):
            line = update(i)[-1]
            np.testing.assert_array_equal(line.get_xdata(), states[: i + 1, 0])
            np.testing.assert_array_equal(line.get_ydata(), states[: i + 1, 1])

    def test_returned_artists_include_patches_and_trail(self, captured_anim, shapes, trajectory):
        states, _ = trajectory
        mv.animate_motion("car", states, shapes=shapes)
        patches = captured_anim["fig"].axes[0].patches
        artists = captured_anim["func"](10)
        for patch in patches:
            assert patch in artists
        assert len(artists) == len(patches) + 1

    def test_trail_false_adds_no_line(self, captured_anim, shapes, trajectory):
        states, _ = trajectory
        mv.animate_motion("car", states, trail=False, shapes=shapes)
        ax = captured_anim["fig"].axes[0]
        assert len(ax.lines) == 0
        artists = captured_anim["func"](5)
        assert len(artists) == len(ax.patches)

    def test_axis_limits_contain_whole_path(self, captured_anim, shapes, trajectory):
        states, _ = trajectory
        mv.animate_motion("car", states, length=4.5, shapes=shapes)
        ax = captured_anim["fig"].axes[0]
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        assert x0 < states[:, 0].min() and x1 > states[:, 0].max()
        assert y0 < states[:, 1].min() and y1 > states[:, 1].max()

    def test_unknown_shape_raises(self, shapes, trajectory):
        states, _ = trajectory
        with pytest.raises(KeyError):
            mv.animate_motion("spaceship", states, shapes=shapes)

    def test_real_animation_saves_to_gif(self, tmp_path, shapes, trajectory):
        pytest.importorskip("PIL")
        states, _ = trajectory
        anim = mv.animate_motion("car", states[:10], length=4.5, shapes=shapes)
        out = tmp_path / "car.gif"
        anim.save(out, writer="pillow", fps=10)
        assert out.stat().st_size > 0


# --------------------------------------------------------------------------- #
# plot_snapshots
# --------------------------------------------------------------------------- #

class TestPlotSnapshots:
    def test_snapshot_count_matches_stride(self, shapes, trajectory):
        states, _ = trajectory          # 601 samples
        fig = mv.plot_snapshots("car", states, every=100, length=4.5, shapes=shapes)
        ax = fig.axes[0]
        n_parts = len(shapes["car"]["parts"])
        assert len(ax.patches) == 7 * n_parts   # samples 0, 100, ..., 600

    def test_each_snapshot_sits_at_its_sample(self, shapes, trajectory):
        states, _ = trajectory
        every = 100
        fig = mv.plot_snapshots("car", states, every=every, length=4.5, shapes=shapes)
        patches = fig.axes[0].patches
        n = len(shapes["car"]["parts"])
        for k, i in enumerate(range(0, len(states), every)):
            assert_patches_match(patches[k * n:(k + 1) * n], shapes["car"], *states[i], length=4.5)

    def test_every_one_draws_every_sample(self, shapes):
        states = np.zeros((5, 3))
        fig = mv.plot_snapshots("circle", states, every=1, shapes=shapes)
        assert len(fig.axes[0].patches) == 5

    def test_large_stride_draws_only_first_sample(self, shapes, trajectory):
        states, _ = trajectory
        fig = mv.plot_snapshots("circle", states, every=1000, shapes=shapes)
        assert len(fig.axes[0].patches) == 1

    def test_alpha_increases_to_fully_opaque(self, shapes, trajectory):
        states, _ = trajectory
        fig = mv.plot_snapshots("car", states, every=100, length=4.5, shapes=shapes)
        n = len(shapes["car"]["parts"])
        alphas = [fig.axes[0].patches[k * n].get_alpha() for k in range(7)]
        assert np.all(np.diff(alphas) > 0)
        assert alphas[-1] == pytest.approx(1.0)
        assert 0.0 < alphas[0] < 1.0

    def test_trajectory_line_is_drawn(self, shapes, trajectory):
        states, _ = trajectory
        fig = mv.plot_snapshots("car", states, every=100, shapes=shapes)
        line = fig.axes[0].lines[0]
        np.testing.assert_array_equal(line.get_xdata(), states[:, 0])
        np.testing.assert_array_equal(line.get_ydata(), states[:, 1])

    def test_axis_limits_contain_whole_path(self, shapes, trajectory):
        states, _ = trajectory
        fig = mv.plot_snapshots("car", states, every=100, shapes=shapes)
        ax = fig.axes[0]
        assert ax.get_xlim()[0] < states[:, 0].min()
        assert ax.get_xlim()[1] > states[:, 0].max()
        assert ax.get_ylim()[0] < states[:, 1].min()
        assert ax.get_ylim()[1] > states[:, 1].max()

    def test_zero_stride_raises(self, shapes, trajectory):
        states, _ = trajectory
        with pytest.raises(ValueError):
            mv.plot_snapshots("car", states, every=0, shapes=shapes)


# --------------------------------------------------------------------------- #
# plot_all_shapes
# --------------------------------------------------------------------------- #

class TestPlotAllShapes:
    def test_one_panel_per_shape(self, shapes):
        fig = mv.plot_all_shapes(shapes=shapes)
        assert len(fig.axes) == len(ALL_SHAPES)

    def test_panel_titles_match_shape_names(self, shapes):
        fig = mv.plot_all_shapes(shapes=shapes)
        titles = [ax.get_title() for ax in fig.axes]
        assert titles == [name.replace("_", " ").title() for name in shapes]

    def test_each_panel_contains_its_shape(self, shapes):
        fig = mv.plot_all_shapes(shapes=shapes)
        for ax, name in zip(fig.axes, shapes):
            assert len(ax.patches) == len(shapes[name]["parts"])

    def test_each_panel_is_centred_on_origin(self, shapes):
        fig = mv.plot_all_shapes(shapes=shapes)
        for ax in fig.axes:
            assert ax.get_xlim()[0] == pytest.approx(-ax.get_xlim()[1])
            assert ax.get_ylim()[0] == pytest.approx(-ax.get_ylim()[1])

    def test_panel_size_follows_reach(self, shapes):
        fig = mv.plot_all_shapes(shapes=shapes)
        for ax, name in zip(fig.axes, shapes):
            assert ax.get_xlim()[1] == pytest.approx(1.2 * expected_reach(shapes[name]))

    def test_length_argument_applies_to_every_panel(self, shapes):
        fig = mv.plot_all_shapes(length=1.0, shapes=shapes)
        for ax, name in zip(fig.axes, shapes):
            assert ax.get_xlim()[1] == pytest.approx(1.2 * expected_reach(shapes[name], 1.0))

    @pytest.mark.parametrize("columns", [1, 3, 4, 5, 10])
    def test_unused_grid_cells_are_removed(self, shapes, columns):
        fig = mv.plot_all_shapes(shapes=shapes, columns=columns)
        assert len(fig.axes) == len(ALL_SHAPES)


# --------------------------------------------------------------------------- #
# check_motion
# --------------------------------------------------------------------------- #

class TestCheckMotion:
    def test_passes_on_shipped_shapes(self):
        mv.check_motion()

    def test_detects_a_broken_move(self, monkeypatch):
        monkeypatch.setattr(mv.Drawable, "move", lambda self, x, y, theta: None)
        with pytest.raises(AssertionError):
            mv.check_motion()

    def test_closes_its_figure(self):
        before = set(plt.get_fignums())
        mv.check_motion()
        assert set(plt.get_fignums()) == before


# --------------------------------------------------------------------------- #
# circular_turn
# --------------------------------------------------------------------------- #

class TestCircularTurn:
    def test_shapes_of_outputs(self):
        states, times = mv.circular_turn(speed=3.0, turn_rate_deg_s=12.0, duration=30.0)
        assert states.shape == (601, 3)
        assert times.shape == (601,)

    def test_time_stamps(self):
        _, times = mv.circular_turn(speed=3.0, turn_rate_deg_s=12.0, duration=30.0)
        assert times[0] == 0.0
        assert times[-1] == pytest.approx(30.0)
        np.testing.assert_allclose(np.diff(times), 0.05, atol=1e-12)

    def test_starts_at_origin_with_zero_heading(self):
        states, _ = mv.circular_turn(speed=3.0, turn_rate_deg_s=12.0, duration=30.0)
        np.testing.assert_allclose(states[0], [0.0, 0.0, 0.0], atol=1e-12)

    def test_heading_is_linear_in_time(self):
        states, times = mv.circular_turn(speed=3.0, turn_rate_deg_s=12.0, duration=30.0)
        np.testing.assert_allclose(states[:, 2], 12.0 * times, atol=1e-9)

    def test_step_length_equals_speed_times_step(self):
        # Chord of an arc of radius v/w subtending angle w*dt
        speed, rate, dt = 3.0, 12.0, 0.05
        states, _ = mv.circular_turn(speed=speed, turn_rate_deg_s=rate, duration=30.0, dt=dt)
        w = math.radians(rate)
        expected = 2 * (speed / w) * math.sin(w * dt / 2)
        steps = np.hypot(np.diff(states[:, 0]), np.diff(states[:, 1]))
        np.testing.assert_allclose(steps, expected, rtol=1e-9)

    def test_velocity_is_perpendicular_to_heading_on_nose_convention(self):
        # Nose points along (-sin h, cos h), so velocity direction is heading + 90 degrees.
        states, _ = mv.circular_turn(speed=3.0, turn_rate_deg_s=12.0, duration=30.0)
        dx = np.diff(states[:, 0])
        dy = np.diff(states[:, 1])
        chord_deg = np.degrees(np.arctan2(dy, dx))
        mid_heading = (states[:-1, 2] + states[1:, 2]) / 2
        diff = (chord_deg - (mid_heading + 90.0) + 180.0) % 360.0 - 180.0
        assert np.max(np.abs(diff)) < 1e-6

    def test_path_lies_on_turning_circle(self):
        speed, rate = 3.0, 12.0
        states, _ = mv.circular_turn(speed=speed, turn_rate_deg_s=rate, duration=30.0)
        R = speed / math.radians(rate)
        # circle centre is (-R, 0), radius |R|
        np.testing.assert_allclose(np.hypot(states[:, 0] + R, states[:, 1]), abs(R), atol=1e-9)

    def test_negative_rate_turns_clockwise(self):
        states, _ = mv.circular_turn(speed=3.0, turn_rate_deg_s=-12.0, duration=30.0)
        assert states[-1, 2] < 0.0
        assert np.all(np.diff(states[:, 2]) < 0.0)

    @pytest.mark.xfail(strict=True, reason="turn_rate_deg_s=0 divides by zero; no straight-line branch")
    def test_zero_turn_rate_is_straight_line(self):
        with np.errstate(all="ignore"):
            states, times = mv.circular_turn(speed=3.0, turn_rate_deg_s=0.0, duration=10.0)
        assert np.all(np.isfinite(states))
        np.testing.assert_allclose(states[:, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(states[:, 1], 3.0 * times, atol=1e-9)


class TestSaveMotion:
    def test_writes_gif_file(self, tmp_path, shapes, trajectory):
        pytest.importorskip("PIL")
        states, _ = trajectory
        out = mv.save_motion(tmp_path / "car.gif", "car", states[:20], length=4.5, shapes=shapes)
        assert out == (tmp_path / "car.gif").resolve()
        assert out.stat().st_size > 0

    def test_does_not_open_a_figure(self, tmp_path, shapes, trajectory):
        pytest.importorskip("PIL")
        states, _ = trajectory
        before = set(plt.get_fignums())
        mv.save_motion(tmp_path / "car.gif", "car", states[:20], shapes=shapes)
        assert set(plt.get_fignums()) == before

    def test_step_reduces_frame_count(self, tmp_path, monkeypatch, shapes, trajectory):
        pytest.importorskip("PIL")
        seen = {}
        real = mv.FuncAnimation

        def spy(*args, **kwargs):
            seen["frames"] = kwargs["frames"]
            return real(*args, **kwargs)

        monkeypatch.setattr(mv, "FuncAnimation", spy)
        states, _ = trajectory          # 601 samples
        mv.save_motion(tmp_path / "car.gif", "car", states, step=10, shapes=shapes)
        assert seen["frames"] == 61     # samples 0, 10, ..., 600

    def test_unknown_extension_raises(self, tmp_path, shapes, trajectory):
        states, _ = trajectory
        with pytest.raises(ValueError):
            mv.save_motion(tmp_path / "car.avi", "car", states, shapes=shapes)

    def test_mp4_without_ffmpeg_raises(self, tmp_path, monkeypatch, shapes, trajectory):
        monkeypatch.setattr(mv.animation.writers, "is_available", lambda name: False)
        states, _ = trajectory
        with pytest.raises(RuntimeError):
            mv.save_motion(tmp_path / "car.mp4", "car", states, shapes=shapes)

    def test_writes_mp4_when_ffmpeg_present(self, tmp_path, shapes, trajectory):
        if not mv.animation.writers.is_available("ffmpeg"):
            pytest.skip("ffmpeg not installed")
        states, _ = trajectory
        out = mv.save_motion(tmp_path / "car.mp4", "car", states[:20], shapes=shapes)
        assert out.stat().st_size > 0