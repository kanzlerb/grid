"""
Unified pipeline: DXF → Contours → Grid → Delaunay Triangulation → Thermal Simulation
"""
from class_dxf import DXF
from class_grid import GRID
from delauney import Grid as DelaunayGrid
import matplotlib.pyplot as plt
import numpy as np


def main():
    # 1. Load DXF and extract contours
    dxf_file = "screwhead.DXF"  # Change as needed
    mesh_density = 1.0
    contour_density_factor = 1.0
    dxf = DXF(dxf_file, mesh_density, contour_density_factor)
    contours = dxf.connected_contours
    dxf.plot_connected_contours()

    # 2. Generate grid points inside contours
    grid = GRID(dxf.bbox, dxf.nx, dxf.ny, contours, filter=True)
    grid_points = grid.grid

    # 3. Triangulation (Delaunay)
    delaunay_grid = DelaunayGrid(grid_points)
    delaunay_grid.compute_delaunay_triangulation(default_T=25.0, default_C=1.0)

    # 4. Set boundary/initial conditions (example: heat source at first element)
    if len(delaunay_grid.elements) > 0:
        delaunay_grid.elements[0].T = 100.0

    # 5. Run thermal simulation
    temperature_history = delaunay_grid.simulate_heat_flow(steps=100, k=50.0, dt=0.01)

    # 6. Visualization
    for step in [0, 10, 50, 99]:
        delaunay_grid.plot(temperatures=temperature_history[step], step=step)

if __name__ == "__main__":
    main()
