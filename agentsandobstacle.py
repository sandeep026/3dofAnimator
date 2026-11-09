from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

import tomllib

@dataclass
class Time:
    t0: float = 0
    tf: float = 1
    points: int = 100

    def get_sampling_time(self, milli: bool = True) -> float:
        c = 1e3 if milli else 1
        return (self.tf - self.t0) / self.points * c


@dataclass
class Agent:
    name: str
    vertices: list[list[float]]
    path: Callable[["Time"], tuple[np.ndarray, np.ndarray, np.ndarray]]
    color: str = "skyblue"
    edge_color: str = "navy"
    _vertices_new: np.ndarray | None = None
    _x: np.ndarray | None = None
    _y: np.ndarray | None = None
    _theta: np.ndarray | None = None
    _polygon:Polygon|None=None

    def get_vertices_new(self, time: Time):
        # how the shape changes over time
        x, y, theta = self.path(time)
        # Vectorized batch transformation
        R = np.empty((time.points, 2, 2))
        c, s = np.cos(theta), np.sin(theta)
        R[:, 0, 0] = c
        R[:, 0, 1] = -s
        R[:, 1, 0] = s
        R[:, 1, 1] = c
        v = np.tile(np.array(self.vertices), (time.points, 1, 1))
        v_new = (
            v @ np.transpose(R, (0, 2, 1)) + np.column_stack((x, y))[:, np.newaxis, :]
        )
        self._vertices_new = v_new
        self._x = x
        self._y = y
        self._theta = theta
        return x, y, v_new

    def get_polygon_new(self):
        poly = Polygon(
            self._vertices_new[0,:,:],
            closed=True,
            facecolor=self.color,
            edgecolor=self.edge_color,
            linewidth=1.5,
            alpha=0.7,
            hatch="//",
            label=self.name,
        )
        self._polygon=poly

@dataclass
class Obstacle:
    name: str
    vertices: list[list[float]]
    center: tuple[float, float] = (0.0, 0.0)
    theta: float = 0.0
    color: str = "dimgray"
    edge_color: str = "black"
    _vertices_new:np.ndarray|None=None
    _polygon:Polygon|None=None

    def get_vertices_new(self):
        v = np.array(self.vertices)
        c, s = np.cos(self.theta), np.sin(self.theta)
        R = np.array([[c, -s], [s, c]])
        v_new = v @ R.T + np.array(self.center)
        self._vertices_new=v_new

    def get_polygon_new(self):
        if self._vertices_new is None:
            raise ValueError('get_vertices_new must be called before get_polygon_new')
        poly = Polygon(
            self._vertices_new,
            closed=True,
            facecolor=self.color,
            edgecolor=self.edge_color,
            linewidth=1.5,
            alpha=0.7,
            hatch="//",
            label=self.name,
        )
        self._polygon=poly


def visualize(agents: list[Agent], obstacles: list[Obstacle], time: Time):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")

    all_x, all_y = [], []

    # Plot static obstacles first
    for obstacle in obstacles:
        obstacle.get_vertices_new()
        obstacle.get_polygon_new()
        ax.add_patch(obstacle._polygon)
        all_x.extend(obstacle._vertices_new[:, 0])
        all_y.extend(obstacle._vertices_new[:, 1])

    agent_data = []

    # Process agents and pre-allocate artists for blitting
    for agent in agents:
        agent.get_vertices_new(time)
        agent.get_polygon_new()
        all_x.extend(agent._x)
        all_y.extend(agent._y)

        # Plot static path guideline
        ax.plot(agent._x, agent._y, "--", color=agent.color, alpha=0.4)

        # Main moving vehicle patch
        ax.add_patch(agent._polygon)

        agent_data.append({"V_new": agent._vertices_new, "main_poly": agent._polygon})

    # Auto-scale limits to fit everything safely
    if all_x and all_y:
        max_range = max(np.ptp(all_x), np.ptp(all_y)) / 2 * 1.15
        mid_x, mid_y = np.mean(all_x), np.mean(all_y)
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)

    ax.set_title("Multi-Agent Simulation")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))

    def update(i):
        artist=[]
        for agent in agents:
            agent._polygon.set_xy(agent._vertices_new[i,:,:])
            artist.append(agent._polygon)
        return tuple(artist)

    # Run animation with true blitting
    ani = animation.FuncAnimation(
        fig, update, frames=time.points, interval=time.get_sampling_time(), blit=True
    )
    plt.show()


def agent_1_path(time: Time):
    t = np.linspace(time.t0, time.tf, time.points)
    w = 8
    h = 5
    v = 1
    x = w * np.sin(v * t)
    y = h * np.sin(2 * v * t)
    xd = w * np.cos(v * t) * v
    yd = h * np.cos(2 * v * t) * 2 * v
    theta = np.arctan2(yd, xd)-np.pi/2
    return x, y, theta


def agent_2_path(time: Time):
    t = np.linspace(time.t0, time.tf, time.points)
    w = 9
    h = 7
    v = 1.2
    x = w * np.sin(v * t)
    y = h * np.sin(2 * v * t)
    xd = w * np.cos(v * t) * v
    yd = h * np.cos(2 * v * t) * 2 * v
    theta = np.arctan2(yd, xd)-np.pi/2
    return x, y, theta

'''
if __name__ == "__main__":
    # Define simple polygon footprints
    car_vertices = [[-1.0, -0.5], [1.0, -0.5], [1.5, 0.0], [1.0, 0.5], [-1.0, 0.5]]
    box_vertices = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    # Setup simulation time configuration
    with open('vehicle_shapes.toml','rb') as f:
        shapes=tomllib.load(f)

    time_cfg = Time(t0=0.0, tf=10.0, points=200)

    # Instantiate agents
    agent1 = Agent(
        name="Agent Alpha",
        vertices=shapes['aircraft'],
        path=agent_1_path,
        color="skyblue",
        edge_color="navy",
    )

    agent2 = Agent(
        name="Agent Beta",
        vertices=shapes['rocket'],
        path=agent_2_path,
        color="salmon",
        edge_color="darkred",
    )

    # Instantiate a static obstacle
    obstacle1 = Obstacle(
        name="Barrier Box",
        vertices=shapes['satellite'],
        center=(0.0, 5.0),
        theta=0.785,  # Rotated by 45 degrees
        color="dimgray",
        edge_color="black",
    )

    # Instantiate a static obstacle
    obstacle1 = Obstacle(
        name="satellite",
        vertices=shapes['satellite'],
        center=(5.0, 0.0),
        theta=0,  # Rotated by 45 degrees
        color="dimgray",
        edge_color="black",
    )

    # Instantiate a static obstacle
    obstacle2 = Obstacle(
        name="catamaran",
        vertices=shapes['catamaran'],
        center=(-5.0, 0.0),
        theta=0,  # Rotated by 45 degrees
        color="dimgray",
        edge_color="black",
    )

    # Run the visualization
    visualize(agents=[agent1, agent2], obstacles=[obstacle1,obstacle2], time=time_cfg)
'''


# --- Trajectory Path Generators ---
def make_lissajous(w, h, v_freq, phase_x=0, phase_y=0):
    def path(time: Time):
        t = np.linspace(time.t0, time.tf, time.points)
        x = w * np.sin(v_freq * t + phase_x)
        y = h * np.sin(2 * v_freq * t + phase_y)
        xd = w * np.cos(v_freq * t + phase_x) * v_freq
        yd = h * np.cos(2 * v_freq * t + phase_y) * 2 * v_freq
        theta = np.arctan2(yd, xd) - np.pi / 2
        return x, y, theta
    return path


def make_orbit(radius, speed, cx=0, cy=0, phase=0):
    def path(time: Time):
        t = np.linspace(time.t0, time.tf, time.points)
        x = cx + radius * np.cos(speed * t + phase)
        y = cy + radius * np.sin(speed * t + phase)
        xd = -radius * np.sin(speed * t + phase) * speed
        yd = radius * np.cos(speed * t + phase) * speed
        theta = np.arctan2(yd, xd) - np.pi / 2
        return x, y, theta
    return path


def make_figure_eight(scale_w, scale_h, speed):
    def path(time: Time):
        t = np.linspace(time.t0, time.tf, time.points)
        x = scale_w * np.sin(speed * t)
        y = scale_h * np.sin(speed * t) * np.cos(speed * t)
        xd = scale_w * speed * np.cos(speed * t)
        yd = scale_h * speed * (np.cos(speed * t)**2 - np.sin(speed * t)**2)
        theta = np.arctan2(yd, xd) - np.pi / 2
        return x, y, theta
    return path


if __name__ == "__main__":
    # --- Shape Library ---
    shapes = {
        "car": [[-0.2, 1.0], [0.2, 1.0], [0.4, 0.7], [0.42, 0.3], [0.42, -0.85], [0.38, -1.0], [-0.38, -1.0], [-0.42, -0.85], [-0.42, 0.3], [-0.4, 0.7]],
        "bus": [[-0.25, 1.0], [0.25, 1.0], [0.38, 0.7], [0.35, -0.6], [0.35, -1.0], [-0.35, -1.0], [-0.35, -0.6], [-0.38, 0.7]],
        "roomba": [[-0.6, 0.95], [0.6, 0.95], [0.85, 0.65], [0.98, 0.3], [1.0, -0.1], [0.9, -0.5], [0.65, -0.85], [0.35, -0.98], [0.0, -1.0], [-0.35, -0.98], [-0.65, -0.85], [-0.9, -0.5], [-1.0, -0.1], [-0.98, 0.3], [-0.85, 0.65]],
        "fighter_jet": [[0.0, 1.0], [0.1, 0.4], [0.95, -0.85], [0.0, -0.6], [-0.95, -0.85], [-0.1, 0.4]],
        "rocket": [[0.0, 1.5], [0.15, 0.8], [0.15, -0.6], [0.45, -1.3], [0.0, -1.1], [-0.45, -1.3], [-0.15, -0.6], [-0.15, 0.8]],
        "aircraft": [[0.00, 0.90], [0.06, 0.40], [0.10, 0.24], [1.00, 0.16], [1.00, -0.04], [0.10, -0.10], [0.04, -0.60], [0.40, -0.76], [0.04, -0.80], [0.00, -0.84], [-0.04, -0.80], [-0.40, -0.76], [-0.04, -0.60], [-0.10, -0.10], [-1.00, -0.04], [-1.00, 0.16], [-0.10, 0.24], [-0.06, 0.40]],
        "satellite": [[-1.0, 0.2], [-0.2, 0.2], [0.0, 0.5], [0.2, 0.2], [1.0, 0.2], [1.0, -0.2], [0.2, -0.2], [0.0, -0.5], [-0.2, -0.2], [-1.0, -0.2]],
        "airship": [[0.0, 1.0], [0.35, 0.85], [0.5, 0.45], [0.5, -0.45], [0.35, -0.85], [0.0, -1.0], [-0.35, -0.85], [-0.5, -0.45], [-0.5, 0.45], [-0.35, 0.85]],
        "catamaran": [[0.25, 0.85], [0.5, 0.4], [0.5, -0.85], [0.2, -1.0], [0.2, -0.4], [-0.2, -0.4], [-0.2, -1.0], [-0.5, -0.85], [-0.5, 0.4], [-0.25, 0.85]],
        "cargoship": [[0.0, 1.0], [0.3, 0.7], [0.35, 0.2], [0.35, -0.85], [0.25, -1.0], [-0.25, -1.0], [-0.35, -0.85], [-0.35, 0.2], [-0.3, 0.7]],
        "speedboat": [[0.0, 1.0], [0.2, 0.6], [0.38, 0.0], [0.4, -0.7], [0.3, -0.95], [-0.3, -0.95], [-0.4, -0.7], [-0.38, 0.0], [-0.2, 0.6]]
    }

    time_cfg = Time(t0=0.0, tf=20.0, points=400)

    # --- Instantiate 20 Agents ---
    agents = [
        Agent("Agent 1 (Jet)", shapes["fighter_jet"], make_lissajous(10, 8, 0.8, 0, 0), "skyblue", "navy"),
        Agent("Agent 2 (Rocket)", shapes["rocket"], make_lissajous(12, 6, 1.0, np.pi/4, 0), "salmon", "darkred"),
        Agent("Agent 3 (Aircraft)", shapes["aircraft"], make_orbit(7.0, 1.1, 0, 0, 0), "limegreen", "darkgreen"),
        Agent("Agent 4 (Car)", shapes["car"], make_figure_eight(9, 5, 0.9), "gold", "orange"),
        Agent("Agent 5 (Bus)", shapes["bus"], make_lissajous(8, 10, 0.7, np.pi/2, np.pi/3), "orchid", "purple"),
        Agent("Agent 6 (Speedboat)", shapes["speedboat"], make_orbit(10.0, -0.8, 0, 0, 1.0), "cyan", "teal"),
        Agent("Agent 7 (Airship)", shapes["airship"], make_figure_eight(11, 7, 0.6), "hotpink", "deeppink"),
        Agent("Agent 8 (Catamaran)", shapes["catamaran"], make_lissajous(6, 12, 1.3, 1.0, 0), "tan", "sienna"),
        Agent("Agent 9 (Cargoship)", shapes["cargoship"], make_orbit(8.5, 0.9, 0, 0, 2.5), "khaki", "olive"),
        Agent("Agent 10 (Roomba)", shapes["roomba"], make_lissajous(9, 9, 1.1, 0.5, 1.5), "tomato", "firebrick"),
        Agent("Agent 11 (Jet 2)", shapes["fighter_jet"], make_orbit(11.0, -1.0, 0, 0, 0.5), "dodgerblue", "midnightblue"),
        Agent("Agent 12 (Rocket 2)", shapes["rocket"], make_figure_eight(7, 10, 0.8), "coral", "saddlebrown"),
        Agent("Agent 13 (Aircraft 2)", shapes["aircraft"], make_lissajous(13, 5, 0.7, 2.0, 1.0), "mediumseagreen", "darkslategray"),
        Agent("Agent 14 (Car 2)", shapes["car"], make_orbit(6.0, 1.4, 0, 0, 3.0), "violet", "purple"),
        Agent("Agent 15 (Bus 2)", shapes["bus"], make_lissajous(7, 7, 1.2, 0, np.pi), "gold", "darkgoldenrod"),
        Agent("Agent 16 (Speedboat 2)", shapes["speedboat"], make_figure_eight(10, 8, 0.7), "turquoise", "darkcyan"),
        Agent("Agent 17 (Airship 2)", shapes["airship"], make_orbit(9.0, -0.7, 0, 0, 4.0), "lightcoral", "maroon"),
        Agent("Agent 18 (Catamaran 2)", shapes["catamaran"], make_lissajous(11, 9, 0.9, 1.5, 2.0), "yellowgreen", "olivedrab"),
        Agent("Agent 19 (Cargoship 2)", shapes["cargoship"], make_figure_eight(8, 11, 1.0), "skyblue", "darkblue"),
        Agent("Agent 20 (Roomba 2)", shapes["roomba"], make_orbit(12.0, 0.8, 0, 0, 1.5), "plum", "indigo")
    ]

    # --- Instantiate 10 Static Obstacles ---
    obstacles = [
        Obstacle("Obs 1 (Satellite)", shapes["satellite"], (6.0, 6.0), 0.5, "dimgray", "black"),
        Obstacle("Obs 2 (Catamaran)", shapes["catamaran"], (-7.0, 5.0), 1.2, "dimgray", "black"),
        Obstacle("Obs 3 (Cargoship)", shapes["cargoship"], (8.0, -5.0), -0.8, "dimgray", "black"),
        Obstacle("Obs 4 (Satellite)", shapes["satellite"], (-8.0, -6.0), 0.2, "dimgray", "black"),
        Obstacle("Obs 5 (Airship)", shapes["airship"], (0.0, 9.0), 1.57, "dimgray", "black"),
        Obstacle("Obs 6 (Speedboat)", shapes["speedboat"], (0.0, -9.0), 0.0, "dimgray", "black"),
        Obstacle("Obs 7 (Roomba)", shapes["roomba"], (10.0, 0.0), 0.78, "dimgray", "black"),
        Obstacle("Obs 8 (Rocket)", shapes["rocket"], (-10.0, 0.0), -1.1, "dimgray", "black"),
        Obstacle("Obs 9 (Satellite)", shapes["satellite"], (4.0, -8.0), 2.3, "dimgray", "black"),
        Obstacle("Obs 10 (Catamaran)", shapes["catamaran"], (-4.0, 8.0), -0.4, "dimgray", "black")
    ]

    # --- Run Stress-Test Simulation ---
    visualize(agents=agents, obstacles=obstacles, time=time_cfg)