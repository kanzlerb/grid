import numpy as np
import ezdxf
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

class DXF:
    """
    A class to handle points, arcs, lines, and circles as entities in DXF files.

    Attributes:
        doc (ezdxf.drawing.Drawing): The DXF document.
        msp (ezdxf.layouts.BaseLayout): The modelspace of the DXF document.
        contours (list): A list to store contours.
        mesh_density (float): Density of the mesh.
        contour_density_factor (float): Contour density factor.
        bbox (Bounding Box, np.array()), contains:
            min_x (float): Minimum x-coordinate of bounding box.
            max_x (float): Maximum x-coordinate of bounding box.
            min_y (float): Minimum y-coordinate of bounding box.
            max_y (float): Maximum y-coordinate of bounding box.
        nx (int): Number of divisions along the x-axis.
        ny (int): Number of divisions along the y-axis.
        connected_contours (list): A list to store connected contours.
    """
    def __init__(self, dxf_file, mesh_density, contour_density_factor):
        """
        Initialize the DXFClass with the given parameters.

        Args:
            dxf_file (str): Path to the DXF file.
            mesh_density (float): Density of the mesh.
            contour_density_factor (float): Contour density factor.
        """
        self.doc                    = ezdxf.readfile(dxf_file)
        self.msp                    = self.doc.modelspace()
        self.contours               = []
        self.mesh_density           = mesh_density
        self.contour_density_factor = contour_density_factor
        self.main()

    def calculate_bounding_box(self, entity):
        """
        Calculate the bounding box of the given entity.

        Args:
            entity (ezdxf.entities.Entity): DXF entity.

        Returns:
            list or None: Bounding box coordinates [(min_x, min_y), (max_x, max_y)] or None if entity type is not handled.
        """
        if entity.dxftype() == 'CIRCLE':
            center = entity.dxf.center
            radius = entity.dxf.radius
            return [(center[0] - radius, center[1] - radius), (center[0] + radius, center[1] + radius)]
        elif entity.dxftype() == 'LINE':
            start = entity.dxf.start
            end = entity.dxf.end
            return [(min(start[0], end[0]), min(start[1], end[1])), (max(start[0], end[0]), max(start[1], end[1]))]
        elif entity.dxftype() == 'ARC':
            center = entity.dxf.center
            radius = entity.dxf.radius
            start_angle = entity.dxf.start_angle
            end_angle = entity.dxf.end_angle
            start_point = (center[0] + radius * np.cos(np.radians(start_angle)), center[1] + radius * np.sin(np.radians(start_angle)))
            end_point = (center[0] + radius * np.cos(np.radians(end_angle)), center[1] + radius * np.sin(np.radians(end_angle)))
            return [(min(start_point[0], end_point[0]), min(start_point[1], end_point[1])), 
                    (max(start_point[0], end_point[0]), max(start_point[1], end_point[1]))]
        elif entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
            points = [(vertex[0], vertex[1]) for vertex in entity.points()]
            min_x = min(point[0] for point in points)
            max_x = max(point[0] for point in points)
            min_y = min(point[1] for point in points)
            max_y = max(point[1] for point in points)
            return [(min_x, min_y), (max_x, max_y)]
        else:
            return None

    def discretize_line(self, srt, end, dc):
        """
        Discretize a line into points.

        Args:
            srt (tuple): Start point of the line (x, y).
            end (tuple): End point of the line (x, y).
            dc (int): Discretization count.

        Returns:
            list: List of discretized points.
        """
        return [
            (srt[0] + i * (end[0] - srt[0]) / dc,
             srt[1] + i * (end[1] - srt[1]) / dc)
            for i in range(dc + 1)
        ]

    def discretize_arc(self, ctr, rad, ang_s, ang_e, dc):
        """
        Discretize an arc into points.

        Args:
            ctr (tuple): Center of the arc (x, y).
            rad (float): Radius of the arc.
            ang_s (float): Start angle of the arc (in degrees).
            ang_e (float): End angle of the arc (in degrees).
            dc (int): Discretization count.

        Returns:
            tuple: Start point, end point, and list of discretized points.
        """
        arc = [
            (ctr[0] + rad * np.cos(np.radians(angle)),
             ctr[1] + rad * np.sin(np.radians(angle)))
            for angle in np.linspace(ang_s, ang_e, dc)
        ]
        srt = (ctr[0] + rad * np.cos(np.radians(ang_s)),
               ctr[1] + rad * np.sin(np.radians(ang_s)))
        end = (ctr[0] + rad * np.cos(np.radians(ang_e)),
               ctr[1] + rad * np.sin(np.radians(ang_e)))
        return srt, end, arc

    def main(self):
        """
        Main method to process the DXF entities and calculate contours.
        """
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')
        for entity in self.msp:
            bbox = self.calculate_bounding_box(entity)
            if bbox:
                min_x = min(min_x, bbox[0][0])
                max_x = max(max_x, bbox[1][0])
                min_y = min(min_y, bbox[0][1])
                max_y = max(max_y, bbox[1][1])
        self.bbox = np.array([min_x - 1, min_y - 1, max_x + 1, max_y + 1])
        self.nx = int((max_x - min_x) * self.mesh_density)
        self.ny = int((max_y - min_y) * self.mesh_density)
        self.connected_contours = []
        
        dc = int(max(self.nx, self.ny) * self.contour_density_factor)
        for entity in self.msp:
            if entity.dxftype() == "POINT":
                self.contours.append([entity.dxf.location[0], entity.dxf.location[1]])
            if entity.dxftype() == "LINE":
                srt = (entity.dxf.start[0], entity.dxf.start[1])
                end = (entity.dxf.end[0], entity.dxf.end[1])
                contour = self.discretize_line(srt, end, dc)
                self.contours.append(contour)
            elif entity.dxftype() == "ARC":
                ctr = (entity.dxf.center[0], entity.dxf.center[1])
                rad = entity.dxf.radius
                ang_s = entity.dxf.start_angle
                ang_e = entity.dxf.end_angle
                srt, end, arc = self.discretize_arc(ctr, rad, ang_s, ang_e, dc)
                self.contours.append(arc)
            elif entity.dxftype() == "CIRCLE":
                ctr = (entity.dxf.center[0], entity.dxf.center[1])
                rad = entity.dxf.radius
                srt, end, arc = self.discretize_arc(ctr, rad, 0, 360, dc)
                self.contours.append(arc)

        while self.contours:
            curr_contour = self.contours.pop(0)
            for other_contour in self.contours[:]:
                if np.allclose(curr_contour[-1], other_contour[0], atol=0.001):
                    curr_contour.extend(other_contour[1:])
                    self.contours.remove(other_contour)
                elif np.allclose(curr_contour[0], other_contour[-1], atol=0.001):
                    curr_contour[:0] = other_contour[:-1][::-1]
                    self.contours.remove(other_contour)
                elif np.allclose(curr_contour[0], other_contour[0], atol=0.001):
                    curr_contour[:0] = other_contour[1:][::-1]
                    self.contours.remove(other_contour)
                elif np.allclose(curr_contour[-1], other_contour[-1], atol=0.001):
                    curr_contour.extend(other_contour[:-1][::-1])
                    self.contours.remove(other_contour)
            self.connected_contours.append(np.array(curr_contour))
            
def plot_connected_contours(contours):
    plt.figure()
    for contour in contours:
        plt.plot(contour[:, 0], contour[:, 1])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Connected Contours (2D)')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# Beispielaufruf mit Toleranz
test_dxf_file = "dxf_files/test_pcb_1.dxf"
dxf = DXF(test_dxf_file, mesh_density=1.0, contour_density_factor=1.0)
plot_connected_contours(dxf.connected_contours)

