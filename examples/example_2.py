import tomllib

import numpy as np

from agentsandobstacles import Agent, AgentsAndObstacles, Obstacle, Time


def agent_1_path(time: Time):
    t = np.linspace(time.t0, time.tf, time.intervals + 1)
    w = 8
    h = 5
    v = 1
    x = w * np.sin(v * t)
    y = h * np.sin(2 * v * t)
    xd = w * np.cos(v * t) * v
    yd = h * np.cos(2 * v * t) * 2 * v
    theta = np.arctan2(yd, xd) - np.pi / 2
    return x, y, theta


def agent_2_path(time: Time):
    t = np.linspace(time.t0, time.tf, time.intervals + 1)
    w = 9
    h = 7
    v = 1.2
    x = w * np.sin(v * t)
    y = h * np.sin(2 * v * t)
    xd = w * np.cos(v * t) * v
    yd = h * np.cos(2 * v * t) * 2 * v
    theta = np.arctan2(yd, xd) - np.pi / 2
    return x, y, theta



def setup():
 
    with open(r'..\agentsandobstacles\vehicle_shapes.toml','rb') as f:
        shapes=tomllib.load(f)

    time_cfg = Time(t0=0.0, tf=10.0, intervals=200)

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
        theta=45*np.pi/180, 
        color="dimgray",
        edge_color="black",
    )

    # Instantiate a static obstacle
    obstacle2 = Obstacle(
        name="catamaran",
        vertices=shapes['catamaran'],
        center=(-5.0, 0.0),
        theta=-45*np.pi/180, 
        color="dimgray",
        edge_color="black",
    )

    # Run the visualization
    ao=AgentsAndObstacles(agents=[agent1, agent2], obstacles=[obstacle1,obstacle2], time=time_cfg)
    ao.visualize(filename='light')

if __name__ == "__main__":
    setup()