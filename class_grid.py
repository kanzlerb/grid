import numpy as np
from class_dxf import *

class GRID:
    """
    A class to generate a grid with optional contour filtering and density adjustment.

    Attributes:
        grid (numpy.ndarray): Generated grid points.
        contours (list): List of contours.
    """

    def __init__(self, bbox, nx, ny, contours=[], filter=True):
        """
        Initialize the GRIDClass with the given parameters.

        Args:
            bbox (list): Bounding box coordinates [min_x, min_y, max_x, max_y].
            nx (int): Number of grid divisions along the x-axis.
            ny (int): Number of grid divisions along the y-axis.
            contours (list): List of contours.
            filter (bool): Whether to filter grid points using ray tracing.
            diffusion (bool): Whether to adjust grid density.
        """
        x_step = (bbox[2] - bbox[0]) / (nx - 1)
        y_step = (bbox[3] - bbox[1]) / (ny - 1)
        points = []
        for i in range(nx):
            for j in range(ny):
                x = bbox[0] + i * x_step
                y = bbox[1] + j * y_step
                points.append([x, y])
        self.grid = np.array(points)
        self.contours = contours
        if filter:
            self.grid = np.array([x for x in self.grid if self.raytracing(x, self.contours)])
        if contours:
            self.contours.append(self.grid)
            self.grid = np.concatenate(self.contours, axis=0)
        
    def raytracing(self, point, contours):
        """
        Perform ray tracing to check if a point is inside the contours.

        Args:
            point (list): Coordinates of the point [x, y].
            contours (list): List of contours.

        Returns:
            bool: True if the point is inside a contour, False otherwise.
        """
        odd = 0
        for c in contours:
            x, y = point
            prev_x, prev_y = c[-1]
            odd_in_c = False
            for curr_x, curr_y in c:
                if (prev_y < y and curr_y >= y) or (curr_y < y and prev_y >= y):
                    if prev_x + (y - prev_y) / (curr_y - prev_y) * (curr_x - prev_x) < x:
                        odd_in_c = not odd_in_c
                prev_x, prev_y = curr_x, curr_y
            if odd_in_c:
                odd += 1
                if odd > 1:
                    return False
        return odd == 1
    
# Beispielaufruf
# test_dxf_file = "dxf_files/test_pcb_1.dxf"
# dxf = DXF(test_dxf_file, mesh_density=1.0, contour_density_factor=1.0)
# grid = GRID(dxf.bbox, dxf.nx, dxf.ny, dxf.connected_contours)
# plt.figure()
# plt.plot(grid.grid[:, 0], grid.grid[:, 1])
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Connected Contours (2D)')
# plt.grid(True)
# plt.axis('equal')
# plt.show()