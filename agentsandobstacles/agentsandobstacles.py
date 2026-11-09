import tomllib
from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


@dataclass
class Time:
    """

    t0 initial time

    tf final time

    intervals number of intervals
    """

    t0: float = 0
    tf: float = 1
    intervals: int = 100
    _t: np.ndarray | None = None

    def get_sampling_time(self, milli: bool = True) -> float:
        """
        sampling time in milli secs or secs

        Parameters
        ----------
        milli : bool, optional
            option to switch to ms or s, by default True

        Returns
        -------
        float
            sampling time
        """
        c = 1e3 if milli else 1
        return (self.tf - self.t0) / self.intervals * c


@dataclass
class Agent:
    """
    Agents are moving objects in the 2D plane

    The agent is represented as a polygon and a polygon
    is defined using its vertices. The center of the polygon
    is the origin and it moves with time and also rotates.
    The rotation is capture using a rotation matrix
    and the center is shifted.


    name - name of agent

    vertices - point which define the polygon

    path - the function which provides the motion of the center and its orientation

    color - polygon's color

    edgecolor - polygons'e edge color

    _vertices_new - vertices of polygon after rotation and translation

    _x - x-axis motion of the center

    _y - y-axis motion of the center

    _theta - rotation of the polygon

    _polygon - polygon object created (create and use once)
    """

    name: str
    vertices: list[list[float]]
    path: Callable[["Time"], tuple[np.ndarray, np.ndarray, np.ndarray]]
    color: str = "skyblue"
    edge_color: str = "navy"
    _vertices_new: np.ndarray | None = None
    _x: np.ndarray | None = None
    _y: np.ndarray | None = None
    _theta: np.ndarray | None = None
    _polygon: Polygon | None = None

    def get_vertices_new(self, time: Time) -> None:
        """
        compute the transformed polygon vertices for all times

        Parameters
        ----------
        time : Time
            time config file
        """
        # how the shape changes over time
        x, y, theta = self.path(time)
        # Vectorized batch transformation
        R = np.empty((time.intervals + 1, 2, 2))
        c, s = np.cos(theta), np.sin(theta)
        R[:, 0, 0] = c
        R[:, 0, 1] = -s
        R[:, 1, 0] = s
        R[:, 1, 1] = c
        v = np.tile(np.array(self.vertices), (time.intervals + 1, 1, 1))
        v_new = (
            v @ np.transpose(R, (0, 2, 1)) + np.column_stack((x, y))[:, np.newaxis, :]
        )
        self._vertices_new = v_new
        self._x = x
        self._y = y
        self._theta = theta

    def get_polygon_new(self) -> None:
        """
        initializes the polygon object

        this will be initialized once but
        reused to update the coordinates
        to save time
        """
        poly = Polygon(
            self._vertices_new[0, :, :],
            closed=True,
            facecolor=self.color,
            edgecolor=self.edge_color,
            linewidth=1.5,
            alpha=0.7,
            hatch="//",
            label=self.name,
        )
        self._polygon = poly


@dataclass
class Obstacle:
    """
    Obstacle are stationary objects in the 2D plane

    The obstacle is represented as a polygon and a polygon
    is defined using its vertices. The center of the polygon
    is the origin. The obstacle can be moved to any position
    with a fixed orientation.

    name - name of obstacle

    vertices - point which define the polygon

    center - position of the obstacle

    color - polygon's color

    edgecolor - polygons'e edge color

    _vertices_new - vertices of polygon after rotation and translation

    _polygon - polygon object created (create and use once)
    """

    name: str
    vertices: list[list[float]]
    center: tuple[float, float] = (0.0, 0.0)
    theta: float = 0.0
    color: str = "dimgray"
    edge_color: str = "black"
    _vertices_new: np.ndarray | None = None
    _polygon: Polygon | None = None

    def get_vertices_new(self) -> None:
        """
        compute the transformed coordinates for
        vertices
        """
        v = np.array(self.vertices)
        c, s = np.cos(self.theta), np.sin(self.theta)
        R = np.array([[c, -s], [s, c]])
        v_new = v @ R.T + np.array(self.center)
        self._vertices_new = v_new

    def get_polygon_new(self) -> None:
        """
        initialized the polygon

        for the obstacle this polygon will
        be used once

        Raises
        ------
        ValueError
            if transformed vertices are not calculated
        """
        if self._vertices_new is None:
            raise ValueError("get_vertices_new must be called before get_polygon_new")
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
        self._polygon = poly


@dataclass
class AgentsAndObstacles:
    time: Time
    agents: list[Agent]
    obstacles: list[Obstacle]

    def visualize(self, save:bool=True,filename: str = "motion"):
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.set_aspect("equal")

        all_x, all_y = [], []

        # Plot static obstacles first
        for obstacle in self.obstacles:
            obstacle.get_vertices_new()
            obstacle.get_polygon_new()
            ax.add_patch(obstacle._polygon)
            all_x.extend(obstacle._vertices_new[:, 0])
            all_y.extend(obstacle._vertices_new[:, 1])

        for agent in self.agents:
            # for each agent compute the transformed vertices
            agent.get_vertices_new(self.time)
            agent.get_polygon_new()
            # collect data to automate scale
            all_x.extend(agent._x)
            all_y.extend(agent._y)
            # line plot of the path of the agent
            ax.plot(agent._x, agent._y, "--", color=agent.color, alpha=0.4)
            # tranfer the polygon object to figure
            ax.add_patch(agent._polygon)

        self.autoscale(ax, all_x=all_x, all_y=all_y)

        ax.set_title("Multi-Agent Simulation")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))

        def update(i):
            artist = []
            for agent in self.agents:
                agent._polygon.set_xy(agent._vertices_new[i, :, :])
                artist.append(agent._polygon)
            return tuple(artist)

        # Run animation with true blitting
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=self.time.intervals + 1,
            interval=self.time.get_sampling_time(),
            blit=True,
        )

        if save:
            ani.save(
                f"{filename}.mp4",
                writer="ffmpeg",
                fps=(self.time.intervals) / (self.time.tf - self.time.t0),
                dpi=200,
                bitrate=2000,
                extra_args=['-preset', 'ultrafast']
            )
            print('File save successfull')
        else:
            plt.show()

    @classmethod
    def autoscale(cls, ax, all_x, all_y):
        if all_x and all_y:
            max_range = max(np.ptp(all_x), np.ptp(all_y)) / 2 * 1.15
            mid_x, mid_y = np.mean(all_x), np.mean(all_y)
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
