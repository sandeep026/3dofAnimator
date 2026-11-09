import tomllib
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.animation as animation

with open("vehicle_shapes.toml", "rb") as f:
    registry = tomllib.load(f)

def plot_all_shapes():
    for vehicle, coordinates in registry.items():
        # polygon object requires the coordinates
        fig, ax = plt.subplots(figsize=(6, 6))
        poly = Polygon(
            coordinates,
            closed=True,
            facecolor="skyblue",
            edgecolor="navy",
            linewidth=2,
            alpha=0.8,
        )

        ax.add_patch(poly)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.set_title(f"{vehicle}", fontsize=14)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.show()


def get_trajectory(t0=0,tf=10,points=500):
    t = np.linspace(t0, tf, points)
    w = 8
    h = 5
    v = 1
    x = w * np.sin(v * t)
    y = h * np.sin(2 * v * t)
    xd = w * np.cos(v * t) * v
    yd = h * np.cos(2 * v * t) * 2 * v
    theta = np.arctan2(yd, xd)
    return x, y, theta


def get_trajectory2(t0=0,tf=10,points=500):
    t = np.linspace(t0, tf, points)
    w = 9
    h = 6
    v = 1
    x = w * np.sin(v * t)
    y = h * np.sin(2 * v * t)
    xd = w * np.cos(v * t) * v
    yd = h * np.cos(2 * v * t) * 2 * v
    theta = np.arctan2(yd, xd)
    return x, y, theta


def visualize_v1():
    ver = registry["satellite"]
    t0=0
    tf=10
    points=100
    x, y, theta = get_trajectory(t0,tf,points)
    view_radius=10

    #generate transformed coordinates of vertices of polygon
    ver_n=[]
    for i in range(points):
        ver_n.append(np.array(ver)@rotate(theta[i]-np.pi/2).T+np.array([x[i],y[i]]))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.set_xlim(-view_radius,view_radius)
    ax.set_ylim(-view_radius,view_radius)
    ax.plot(x,y)


    poly = Polygon(
        ver_n[0],
        closed=True,
        facecolor="skyblue",
        edgecolor="navy",
        linewidth=2,
        alpha=0.8,
    )
    ax.add_patch(poly)


    plt.ion()
    for i in range(len(x)):
        poly.set_xy(ver_n[i])
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause((tf-t0)/points)
    plt.show()

def rotate(theta):
    # theta in radians
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

def visualize_v2():
    ver = registry["aircraft"]
    #seconds
    t0=0
    tf=10
    points=1000
    ts=(tf-t0)/points
    #matplotlib needs time in milli secs
    ts_ms=ts*1e3
    x, y, theta = get_trajectory(t0,tf,points)

    #generate transformed coordinates of vertices of polygon
    ver_n=[]

    R=np.empty((points,2,2))
    c=np.cos(theta-np.pi/2)
    s=np.sin(theta-np.pi/2)
    R[:,0,0]=c
    R[:,0,1]=-s
    R[:,1,0]=s
    R[:,1,1]=c

    V=np.tile(np.array(ver),(points,1,1))
    V_new=V@np.transpose(R,(0,2,1)) + np.column_stack((x, y))[:, np.newaxis, :]


    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    view_radius=10
    ax.set_xlim(-view_radius,view_radius)
    ax.set_ylim(-view_radius,view_radius)
    ax.plot(x,y)

    poly = Polygon(
        V_new[0,:,:],
        closed=True,
        facecolor="skyblue",
        edgecolor="navy",
        linewidth=2,
        alpha=0.8,
    )
    ax.add_patch(poly)


    def update(i):
        poly.set_xy(V_new[i,:,:])
        return (poly,)


# Create animation object
    ani = animation.FuncAnimation(
        fig, update, frames=points, interval=ts_ms, blit=True
    )
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon

def timelapse(timelapse_step=50, max_ghosts=10):
    ver = registry["aircraft"]
    #seconds
    t0 = 0
    tf = 10
    points = 1000
    ts = (tf - t0) / points
    #matplotlib needs time in milli secs
    ts_ms = ts * 1e3
    x, y, theta = get_trajectory(t0, tf, points)

    #generate transformed coordinates of vertices of polygon
    R = np.empty((points, 2, 2))
    c = np.cos(theta - np.pi / 2)
    s = np.sin(theta - np.pi / 2)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c

    V = np.tile(np.array(ver), (points, 1, 1))
    V_new = V @ np.transpose(R, (0, 2, 1)) + np.column_stack((x, y))[:, np.newaxis, :]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    view_radius = 10
    ax.set_xlim(-view_radius, view_radius)
    ax.set_ylim(-view_radius, view_radius)
    ax.plot(x, y, 'r--', alpha=0.5, label='Trajectory')

    # Main active vehicle polygon
    poly = Polygon(
        V_new[0, :, :],
        closed=True,
        facecolor="skyblue",
        edgecolor="navy",
        linewidth=2,
        alpha=0.9,
    )
    ax.add_patch(poly)

    # Container to hold active timelapse/ghost polygons on the plot
    ghost_patches = []

    def update(i):
        # 1. Update main moving vehicle
        poly.set_xy(V_new[i, :, :])
        
        # 2. Add a timelapse ghost polygon at specified intervals
        if i % timelapse_step == 0:
            ghost_poly = Polygon(
                V_new[i, :, :],
                closed=True,
                facecolor="gray",
                edgecolor="black",
                linewidth=1,
                alpha=0.3,
            )
            ax.add_patch(ghost_poly)
            ghost_patches.append(ghost_poly)
            
            # Limit the number of ghosts on screen to avoid clutter
            if len(ghost_patches) > max_ghosts:
                old_ghost = ghost_patches.pop(0)
                old_ghost.remove()
                
        return (poly, *ghost_patches)

    # Create animation object
    ani = animation.FuncAnimation(
        fig, update, frames=points, interval=ts_ms, blit=False  # blit=False required when dynamically adding patches
    )
    plt.legend()
    plt.show()

def visualize_timelapse_snapshots(timelapse_step=100):
    ver = registry["aircraft"]
    t0 = 0
    tf = 10
    points = 1000
    x, y, theta = get_trajectory(t0, tf, points)

    # Generate transformed coordinates for all points
    R = np.empty((points, 2, 2))
    c = np.cos(theta - np.pi / 2)
    s = np.sin(theta - np.pi / 2)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c

    V = np.tile(np.array(ver), (points, 1, 1))
    V_new = V @ np.transpose(R, (0, 2, 1)) + np.column_stack((x, y))[:, np.newaxis, :]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    view_radius = 10
    ax.set_xlim(-view_radius, view_radius)
    ax.set_ylim(-view_radius, view_radius)
    
    # Plot the center trajectory path
    ax.plot(x, y, 'r--', alpha=0.5, label='Trajectory Path')

    # 1. Highlight Initial Position (Index 0)
    start_poly = Polygon(
        V_new[0, :, :],
        closed=True,
        facecolor="lightgreen",
        edgecolor="darkgreen",
        linewidth=2,
        alpha=0.9,
        label='Start Position'
    )
    ax.add_patch(start_poly)

    # 2. Plot intermediate timelapse snapshots with uniform alpha
    snapshots = V_new[::timelapse_step]
    for frame_verts in snapshots:
        poly = Polygon(
            frame_verts,
            closed=True,
            facecolor="skyblue",
            edgecolor="navy",
            linewidth=1,
            alpha=0.3,  # Uniform transparency for intermediate ghosts
        )
        ax.add_patch(poly)

    # 3. Highlight Final Position (Last Index)
    final_poly = Polygon(
        V_new[-1, :, :],
        closed=True,
        facecolor="orange",
        edgecolor="darkred",
        linewidth=2,
        alpha=0.9,
        label='Final Position'
    )
    ax.add_patch(final_poly)

    ax.legend()
    ax.set_title(f"Vehicle Trajectory Snapshots (Step: {timelapse_step})")
    plt.show()




def visualize(agents: list[Agent], obstacles: list[Obstacle], t0=0, tf=10, points=1000):
    """
    Runs a high-performance animated scene with blitting enabled, 
    driven cleanly by Agent and Obstacle dataclasses.
    """
    ts = (tf - t0) / points
    ts_ms = ts * 1e3

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect('equal')

    all_x, all_y = [], []

    # 1. Process and Plot Static Obstacles
    for obs in obstacles:
        shape = np.array(obs.shape)
        c, s = np.cos(obs.theta - np.pi / 2), np.sin(obs.theta - np.pi / 2)
        R_single = np.array([[c, s], [-s, c]])
        transformed_shape = shape @ R_single + np.array(obs.pos)
        
        poly = Polygon(
            transformed_shape, closed=True,
            facecolor=obs.color, edgecolor=obs.edge_color,
            linewidth=1.5, alpha=0.7, hatch='//', label=obs.label
        )
        ax.add_patch(poly)
        all_x.extend(transformed_shape[:, 0])
        all_y.extend(transformed_shape[:, 1])

    # 2. Precompute Trajectories & Pre-allocate Artists for Blitting
    agent_data = []

    for agent in agents:
        x, y, theta = agent.trajectory(t0, tf, points)
        all_x.extend(x)
        all_y.extend(y)

        # Vectorized batch transformation
        R = np.empty((points, 2, 2))
        c, s = np.cos(theta ), np.sin(theta )
        R[:, 0, 0] = c
        R[:, 0, 1] = -s
        R[:, 1, 0] = s
        R[:, 1, 1] = c

        V = np.tile(np.array(agent.shape), (points, 1, 1))
        V_new = V @ np.transpose(R, (0, 2, 1)) + np.column_stack((x, y))[:, np.newaxis, :]

        # Plot static path guideline
        ax.plot(x, y, '--', color=agent.color, alpha=0.4)

        # Main moving vehicle patch
        main_poly = Polygon(
            V_new[0, :, :], closed=True,
            facecolor=agent.color, edgecolor=agent.edge_color,
            linewidth=1.5, alpha=0.9, label=agent.name
        )
        ax.add_patch(main_poly)

        # Pre-allocate timelapse ghost patches (hidden initially with alpha=0.0)
        ghost_indices = np.arange(0, points, agent.timelapse_step)
        ghost_polys = []
        for g_idx in ghost_indices:
            g_poly = Polygon(
                V_new[g_idx, :, :], closed=True,
                facecolor=agent.color, edgecolor=agent.edge_color,
                linewidth=1.0, alpha=0.0
            )
            ax.add_patch(g_poly)
            ghost_polys.append(g_poly)

        agent_data.append({
            'V_new': V_new,
            'main_poly': main_poly,
            'ghost_indices': ghost_indices,
            'ghost_polys': ghost_polys,
            'alpha_ghost': agent.alpha_ghost
        })

    # Auto-scale canvas bounds securely
    if all_x and all_y:
        max_range = max(np.ptp(all_x), np.ptp(all_y)) / 2 * 1.15
        mid_x, mid_y = np.mean(all_x), np.mean(all_y)
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)

    ax.set_title("Multi-Agent Scene Animation (Blit Enabled)")
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    # 3. Optimized Update Function for Blitting
    def update(i):
        updated_artists = []
        for data in agent_data:
            V_new = data['V_new']
            main_poly = data['main_poly']
            
            # Update main vehicle position
            main_poly.set_xy(V_new[i, :, :])
            updated_artists.append(main_poly)

            # Fade-in ghost snapshots as time step 'i' passes them
            for g_idx, g_poly in zip(data['ghost_indices'], data['ghost_polys']):
                if g_idx <= i:
                    g_poly.set_alpha(data['alpha_ghost'])
                else:
                    g_poly.set_alpha(0.0)
                updated_artists.append(g_poly)

        return tuple(updated_artists)

    # Run animation with true blitting
    ani = animation.FuncAnimation(
        fig, update, frames=points, interval=ts_ms, blit=True
    )
    plt.show()

visualize_v1()