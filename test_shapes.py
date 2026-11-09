from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

# ==========================================
# 1. DATA LAYER (Data Classes & Storage)
# ==========================================

@dataclass
class VehiclePolygon:
    name: str
    coords: List[Tuple[float, float]] = field(default_factory=list)

    def generate_polygon(self) -> List[Tuple[float, float]]:
        """
        Generates a closed loop path of the vehicle polygon for rendering.
        """
        if not self.coords:
            return []
        # Append only the first tuple element to close the loop cleanly
        return self.coords + [self.coords[0]]


# Master registry containing the raw coordinate definitions.
# Max amplitude is 1.0, centered around (0,0), and oriented facing the +Y axis.
# Default limit is 10 points per vehicle. Shapes that cannot be represented
# within 10 points use up to 20 points (bicycle, airliner, helicopter, drone).
# Each entry represents a silhouette family; the comment lists the shapes it covers.
VEHICLE_REGISTRY: Dict[str, List[Tuple[float, float]]] = {
    # ---------- Ground Vehicles ----------
    "car": [  # Narrow tapered front, squared rear. Covers: sedan, suv, pickup, race_car
        (-0.2, 1.0), (0.2, 1.0), (0.4, 0.7), (0.42, 0.3), (0.42, -0.85),
        (0.38, -1.0), (-0.38, -1.0), (-0.42, -0.85), (-0.42, 0.3), (-0.4, 0.7)
    ],
    "bus": [  # Wide flat front, tapered rear. Covers: box_truck, van
        (-0.35, 1.0), (0.35, 1.0), (0.35, 0.6), (0.38, -0.7),
        (0.25, -1.0), (-0.25, -1.0), (-0.38, -0.7), (-0.35, 0.6)
    ],

    # ---------- Two-Wheelers ----------
    "bicycle": [  # Wide handlebar with grips, seat, equal front and rear wheels.
                  # Covers: motorcycle, bike, scooter
        (0.0, 1.0), (0.07, 0.8), (0.03, 0.72), (0.5, 0.58), (0.5, 0.42),
        (0.03, 0.5), (0.03, -0.3), (0.2, -0.45), (0.03, -0.7), (0.07, -0.8),
        (0.0, -1.0), (-0.07, -0.8), (-0.03, -0.7), (-0.2, -0.45), (-0.03, -0.3),
        (-0.03, 0.5), (-0.5, 0.42), (-0.5, 0.58), (-0.03, 0.72), (-0.07, 0.8)
    ],

    # ---------- Mobile Robots ----------
    "roomba": [  # Round body with flat front bumper. Covers: agv, sidewalk_robot, rover, lawn_mower_robot
        (-0.7, 0.75), (0.7, 0.75), (0.95, 0.45), (1.0, 0.0), (0.95, -0.45),
        (0.7, -0.8), (-0.7, -0.8), (-0.95, -0.45), (-1.0, 0.0), (-0.95, 0.45)
    ],

    # ---------- Aerospace ----------
    "airliner": [  # Light propeller aircraft: long straight wings, short fuselage, tailplane.
                   # Reference: top-view single-engine prop plane
        (0.0, 0.88), (0.13, 0.7), (0.15, 0.49), (1.0, 0.45), (1.0, 0.22),
        (0.15, 0.2), (0.15, -0.28), (0.36, -0.28), (0.36, -0.48), (0.05, -0.5),
        (0.0, -0.63), (-0.05, -0.5), (-0.36, -0.48), (-0.36, -0.28), (-0.15, -0.28),
        (-0.15, 0.2), (-1.0, 0.22), (-1.0, 0.45), (-0.15, 0.49), (-0.13, 0.7)
    ],
    "fighter_jet": [  # Delta wing with a center notch. Covers: bomber, space_shuttle
        (0.0, 1.0), (0.1, 0.4), (0.95, -0.85),
        (0.0, -0.6), (-0.95, -0.85), (-0.1, 0.4)
    ],
    rocket = [
    [0.0, 1.5], [0.15, 0.8], [0.15, -0.6], [0.45, -1.3],
    [0.0, -1.1], [-0.45, -1.3], [-0.15, -0.6], [-0.15, 0.8]
]
    "satellite": [
        (-1.0, 0.2), (-0.2, 0.2), (0.0, 0.5), (0.2, 0.2), (1.0, 0.2),
        (1.0, -0.2), (0.2, -0.2), (0.0, -0.5), (-0.2, -0.2), (-1.0, -0.2)
    ],
    "airship": [
        (0.0, 1.0), (0.35, 0.85), (0.5, 0.45), (0.5, -0.45), (0.35, -0.85),
        (0.0, -1.0), (-0.35, -0.85), (-0.5, -0.45), (-0.5, 0.45), (-0.35, 0.85)
    ],

    # ---------- Water Vehicles ----------
    "catamaran": [
        (0.25, 0.85), (0.5, 0.4), (0.5, -0.85), (0.2, -1.0), (0.2, -0.4),
        (-0.2, -0.4), (-0.2, -1.0), (-0.5, -0.85), (-0.5, 0.4), (-0.25, 0.85)
    ],
    "cargo_ship": [  # Covers: sailboat, tugboat, aircraft_carrier
        (0.0, 1.0), (0.3, 0.7), (0.35, 0.2), (0.35, -0.85), (0.25, -1.0),
        (-0.25, -1.0), (-0.35, -0.85), (-0.35, 0.2), (-0.3, 0.7)
    ],
    "speedboat": [  # Covers: yacht, submarine
        (0.0, 1.0), (0.2, 0.6), (0.38, 0.0), (0.4, -0.7), (0.3, -0.95),
        (-0.3, -0.95), (-0.4, -0.7), (-0.38, 0.0), (-0.2, 0.6)
    ],
}

# ==========================================
# 2. PROCESSING LAYER (Operations & UI)
# ==========================================

def plot_all_vehicles_together():
    """
    Renders all vehicles from the registry onto a single subplot grid figure canvas.
    Layout cleanly spans a 3 rows by 5 columns configuration matrix.
    """
    keys = list(VEHICLE_REGISTRY.keys())
    
    # Generate a unified 3 rows by 5 columns subplot frame matrix
    fig, axes = plt.subplots(3, 5, figsize=(15, 11))
    axes = axes.flatten()  # Flatten the 2D grid matrix into a sequential array loop
    
    for i, key in enumerate(keys):
        ax = axes[i]
        coords = VEHICLE_REGISTRY[key]
        
        # Instantiate clean data class properties from data registry
        vehicle = VehiclePolygon(name=key.capitalize(), coords=coords)
        polygon_points = vehicle.generate_polygon()
        xs, ys = zip(*polygon_points)
        
        # Draw geometry contours and filled regions
        ax.fill(xs, ys, color='#2c3e50', alpha=0.15)
        ax.plot(xs, ys, marker='o', color='#2c3e50', linewidth=1.5, markersize=3)
        
        # Position a distinct center indicator (0,0) and heading reference vector arrow (+Y direction)
        ax.plot(0, 0, marker='X', color='#e74c3c', markersize=5)
        ax.arrow(0, 0, 0, 0.4, head_width=0.08, head_length=0.1, fc='#e74c3c', ec='#e74c3c', zorder=5)
        
        # Boundary constraints enforcing absolute size index limit to 1
        ax.set_title(vehicle.name, fontsize=10, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
        ax.axvline(0, color='gray', linewidth=0.5, alpha=0.3)
        
    # Hide any unused trailing subplots in the grid panel
    for j in range(len(keys), len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_all_vehicles_together()