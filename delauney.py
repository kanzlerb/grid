import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap

class Element:
    def __init__(self, nodes, T=0.0, C=1.0):
        self.nodes = nodes
        self.T = T
        self.C = C
        self.neighbors = set()

    def add_neighbor(self, element):
        self.neighbors.add(element)

    def update_temperature(self, k=50.0, dt=0.01, points=None):
        if not self.neighbors:
            return

        sum_T = self.T * self.C
        sum_C = self.C

        for neighbor in self.neighbors:
            shared_edges = set()
            for edge in [(self.nodes[0], self.nodes[1]), (self.nodes[1], self.nodes[2]), (self.nodes[0], self.nodes[2])]:
                if edge in [(neighbor.nodes[0], neighbor.nodes[1]), (neighbor.nodes[1], neighbor.nodes[2]), (neighbor.nodes[0], neighbor.nodes[2])]:
                    shared_edges.add(edge)
                if edge[::-1] in [(neighbor.nodes[0], neighbor.nodes[1]), (neighbor.nodes[1], neighbor.nodes[2]), (neighbor.nodes[0], neighbor.nodes[2])]:
                    shared_edges.add(edge)

            if shared_edges:
                edge = shared_edges.pop()
                a, b = edge
                L = np.linalg.norm(points[a] - points[b])
                sum_T += neighbor.T * L * k * dt
                sum_C += L * k * dt

        if sum_C > 0:
            self.T = sum_T / sum_C

class Grid:
    def __init__(self, points=None):
        self.points = np.array(points) if points is not None else np.empty((0, 2))
        self.elements = []
        self.edge_map = defaultdict(set)

    def add_points(self, new_points):
        self.points = np.vstack((self.points, new_points)) if self.points.size else np.array(new_points)

    def compute_delaunay_triangulation(self, default_T=25.0, default_C=1.0):
        if len(self.points) < 3:
            raise ValueError("Mindestens 3 Punkte sind für eine Triangulation erforderlich.")

        tri = Delaunay(self.points)
        self.elements = [Element(tuple(sorted(simplex)), default_T, default_C) for simplex in tri.simplices]

        for elem in self.elements:
            for edge in [(elem.nodes[0], elem.nodes[1]), (elem.nodes[1], elem.nodes[2]), (elem.nodes[0], elem.nodes[2])]:
                self.edge_map[edge].add(elem)
                self.edge_map[(edge[1], edge[0])].add(elem)

        for elem in self.elements:
            for edge in [(elem.nodes[0], elem.nodes[1]), (elem.nodes[1], elem.nodes[2]), (elem.nodes[0], elem.nodes[2])]:
                for neighbor in self.edge_map[edge]:
                    if neighbor != elem:
                        elem.add_neighbor(neighbor)

    def simulate_heat_flow(self, steps=100, k=50.0, dt=0.01):
        temperatures = []
        initial_temps = [elem.T for elem in self.elements]
        temperatures.append(initial_temps)

        for _ in range(steps):
            new_temps = [elem.T for elem in self.elements]

            for i, elem in enumerate(self.elements):
                elem.update_temperature(k, dt, self.points)
                new_temps[i] = elem.T

            current_temps = [elem.T for elem in self.elements]
            temperatures.append(current_temps)

        return temperatures

    def plot(self, temperatures=None, step=None):
        if len(self.points) == 0:
            print("Keine Punkte zum Plotten vorhanden.")
            return

        triangles = [elem.nodes for elem in self.elements]
        fig, ax = plt.subplots(figsize=(8, 6))

        cmap = LinearSegmentedColormap.from_list('temp_cmap', ['darkblue', 'blue', 'cyan', 'green', 'yellow', 'orange', 'red', 'darkred'])
        norm = plt.Normalize(vmin=25, vmax=100)

        plt.triplot(self.points[:, 0], self.points[:, 1], triangles, lw=0.5, color='gray')

        if temperatures is not None:
            tpc = ax.tripcolor(self.points[:, 0], self.points[:, 1], triangles, facecolors=temperatures, norm=norm, cmap=cmap, shading='flat')
            plt.colorbar(tpc, label='Temperatur (°C)')

        plt.plot(self.points[:, 0], self.points[:, 1], 'o', markersize=2, color='black')
        plt.title(f"Temperaturverteilung (Schritt {step})" if step is not None else "Temperaturverteilung")
        plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    points = np.random.rand(100, 2)

    grid = Grid(points)
    grid.compute_delaunay_triangulation(default_T=25.0, default_C=1.0)

    grid.elements[0].T = 1000.0

    temperature_history = grid.simulate_heat_flow(steps=1000, k=50.0, dt=0.01)

    for step in [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,200,300,400,500,600,700,800,900,999]:
        grid.plot(temperatures=temperature_history[step], step=step)