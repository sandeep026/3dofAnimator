import tomllib

import numpy as np

from agentsandobstacle import Agent, AgentsAndObstacle, Time, Obstacle


def make_lissajous(w, h, v_freq, phase_x=0, phase_y=0):
    def path(time: Time):
        t = np.linspace(time.t0, time.tf, time.intervals + 1)
        x = w * np.sin(v_freq * t + phase_x)
        y = h * np.sin(2 * v_freq * t + phase_y)
        xd = w * np.cos(v_freq * t + phase_x) * v_freq
        yd = h * np.cos(2 * v_freq * t + phase_y) * 2 * v_freq
        theta = np.arctan2(yd, xd) - np.pi / 2
        return x, y, theta

    return path


def make_orbit(radius, speed, cx=0, cy=0, phase=0):
    def path(time: Time):
        t = np.linspace(time.t0, time.tf, time.intervals + 1)
        x = cx + radius * np.cos(speed * t + phase)
        y = cy + radius * np.sin(speed * t + phase)
        xd = -radius * np.sin(speed * t + phase) * speed
        yd = radius * np.cos(speed * t + phase) * speed
        theta = np.arctan2(yd, xd) - np.pi / 2
        return x, y, theta

    return path


def make_figure_eight(scale_w, scale_h, speed):
    def path(time: Time):
        t = np.linspace(time.t0, time.tf, time.intervals + 1)
        x = scale_w * np.sin(speed * t)
        y = scale_h * np.sin(speed * t) * np.cos(speed * t)
        xd = scale_w * speed * np.cos(speed * t)
        yd = scale_h * speed * (np.cos(speed * t) ** 2 - np.sin(speed * t) ** 2)
        theta = np.arctan2(yd, xd) - np.pi / 2
        return x, y, theta

    return path


def setup():

    with open("vehicle_shapes.toml", "rb") as f:
        shapes = tomllib.load(f)

    time_cfg = Time(t0=0.0, tf=20.0, intervals=400)

    # --- Instantiate 20 Agents ---
    agents = [
        Agent(
            "Agent 1 (Jet)",
            shapes["fighter_jet"],
            make_lissajous(10, 8, 0.8, 0, 0),
            "skyblue",
            "navy",
        ),
        Agent(
            "Agent 2 (Rocket)",
            shapes["rocket"],
            make_lissajous(12, 6, 1.0, np.pi / 4, 0),
            "salmon",
            "darkred",
        ),
        Agent(
            "Agent 3 (Aircraft)",
            shapes["aircraft"],
            make_orbit(7.0, 1.1, 0, 0, 0),
            "limegreen",
            "darkgreen",
        ),
        Agent(
            "Agent 4 (Car)",
            shapes["car"],
            make_figure_eight(9, 5, 0.9),
            "gold",
            "orange",
        ),
        Agent(
            "Agent 5 (Bus)",
            shapes["bus"],
            make_lissajous(8, 10, 0.7, np.pi / 2, np.pi / 3),
            "orchid",
            "purple",
        ),
        Agent(
            "Agent 6 (Speedboat)",
            shapes["speedboat"],
            make_orbit(10.0, -0.8, 0, 0, 1.0),
            "cyan",
            "teal",
        ),
        Agent(
            "Agent 7 (Airship)",
            shapes["airship"],
            make_figure_eight(11, 7, 0.6),
            "hotpink",
            "deeppink",
        ),
        Agent(
            "Agent 8 (Catamaran)",
            shapes["catamaran"],
            make_lissajous(6, 12, 1.3, 1.0, 0),
            "tan",
            "sienna",
        ),
        Agent(
            "Agent 9 (Cargoship)",
            shapes["cargoship"],
            make_orbit(8.5, 0.9, 0, 0, 2.5),
            "khaki",
            "olive",
        ),
        Agent(
            "Agent 10 (Roomba)",
            shapes["roomba"],
            make_lissajous(9, 9, 1.1, 0.5, 1.5),
            "tomato",
            "firebrick",
        ),
        Agent(
            "Agent 11 (Jet 2)",
            shapes["fighter_jet"],
            make_orbit(11.0, -1.0, 0, 0, 0.5),
            "dodgerblue",
            "midnightblue",
        ),
        Agent(
            "Agent 12 (Rocket 2)",
            shapes["rocket"],
            make_figure_eight(7, 10, 0.8),
            "coral",
            "saddlebrown",
        ),
        Agent(
            "Agent 13 (Aircraft 2)",
            shapes["aircraft"],
            make_lissajous(13, 5, 0.7, 2.0, 1.0),
            "mediumseagreen",
            "darkslategray",
        ),
        Agent(
            "Agent 14 (Car 2)",
            shapes["car"],
            make_orbit(6.0, 1.4, 0, 0, 3.0),
            "violet",
            "purple",
        ),
        Agent(
            "Agent 15 (Bus 2)",
            shapes["bus"],
            make_lissajous(7, 7, 1.2, 0, np.pi),
            "gold",
            "darkgoldenrod",
        ),
        Agent(
            "Agent 16 (Speedboat 2)",
            shapes["speedboat"],
            make_figure_eight(10, 8, 0.7),
            "turquoise",
            "darkcyan",
        ),
        Agent(
            "Agent 17 (Airship 2)",
            shapes["airship"],
            make_orbit(9.0, -0.7, 0, 0, 4.0),
            "lightcoral",
            "maroon",
        ),
        Agent(
            "Agent 18 (Catamaran 2)",
            shapes["catamaran"],
            make_lissajous(11, 9, 0.9, 1.5, 2.0),
            "yellowgreen",
            "olivedrab",
        ),
        Agent(
            "Agent 19 (Cargoship 2)",
            shapes["cargoship"],
            make_figure_eight(8, 11, 1.0),
            "skyblue",
            "darkblue",
        ),
        Agent(
            "Agent 20 (Roomba 2)",
            shapes["roomba"],
            make_orbit(12.0, 0.8, 0, 0, 1.5),
            "plum",
            "indigo",
        ),
    ]

    # --- Instantiate 10 Static Obstacles ---
    obstacles = [
        Obstacle(
            "Obs 1 (Satellite)",
            shapes["satellite"],
            (6.0, 6.0),
            0.5,
            "dimgray",
            "black",
        ),
        Obstacle(
            "Obs 2 (Catamaran)",
            shapes["catamaran"],
            (-7.0, 5.0),
            1.2,
            "dimgray",
            "black",
        ),
        Obstacle(
            "Obs 3 (Cargoship)",
            shapes["cargoship"],
            (8.0, -5.0),
            -0.8,
            "dimgray",
            "black",
        ),
        Obstacle(
            "Obs 4 (Satellite)",
            shapes["satellite"],
            (-8.0, -6.0),
            0.2,
            "dimgray",
            "black",
        ),
        Obstacle(
            "Obs 5 (Airship)", shapes["airship"], (0.0, 9.0), 1.57, "dimgray", "black"
        ),
        Obstacle(
            "Obs 6 (Speedboat)",
            shapes["speedboat"],
            (0.0, -9.0),
            0.0,
            "dimgray",
            "black",
        ),
        Obstacle(
            "Obs 7 (Roomba)", shapes["roomba"], (10.0, 0.0), 0.78, "dimgray", "black"
        ),
        Obstacle(
            "Obs 8 (Rocket)", shapes["rocket"], (-10.0, 0.0), -1.1, "dimgray", "black"
        ),
        Obstacle(
            "Obs 9 (Satellite)",
            shapes["satellite"],
            (4.0, -8.0),
            2.3,
            "dimgray",
            "black",
        ),
        Obstacle(
            "Obs 10 (Catamaran)",
            shapes["catamaran"],
            (-4.0, 8.0),
            -0.4,
            "dimgray",
            "black",
        ),
    ]

    # --- Run Stress-Test Simulation ---
    ao = AgentsAndObstacle(agents=agents, obstacles=obstacles, time=time_cfg)
    ao.visualize()


if __name__ == "__main__":
    setup()
