import matplotlib.pyplot as plt
from class_dxf import *
from class_grid import *

def main(file):
    dxf_file                = file
    mesh_density            = 0.5
    contour_density_factor  = 3
    dxf                     = DXF(dxf_file, mesh_density, contour_density_factor)
    grid                    = GRID(dxf.bbox, dxf.nx, dxf.ny, dxf.connected_contours, filter=True)
    plt.figure(figsize=(10, 5))
    plt.scatter(grid.grid[:, 0], grid.grid[:, 1], label='Filtered Grid Points', s=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Plot of Grid Points, Circles, and Filtered Grid Points')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()
    
# Problems remain:
#   - Ellipsen müssen noch behandelbar sein  
#   - max min boundbing box search bei arcs versagt leider noch
#   - Multithreading muss für das raytracing unbedingt eingeführt werden
#   - diffusionsfunktion muss gebaut werden 

# Function rests:

    # def logarithmic(dist, factor):
    #     return np.log(dist + 1) * factor
    # def exponential(dist, factor):
    #     return np.exp(dist) * factor
    # def linear(dist, factor):
    #     return dist * factor
    # def quadratic(dist, factor):
    #     return dist**2 * factor
    # def inverse(dist, factor):
    #     return 1/dist * factor
    # def adjust_grid_density(self, grid, contours, adjustment_function=linear, factor=0.000000000000000001, dist_metric='euclidean'):
    #     weighted_grid = []
    #     for p_g in grid:
    #         if dist_metric == 'euclidean':
    #             dists = [euclidean(p_g, p_c) for contour in contours for p_c in contour]
    #         elif dist_metric == 'mahalanobis':
    #             cov = np.cov(np.vstack(contours).T)
    #             dists = [mahalanobis(p_g, p_c, cov) for contour in contours for p_c in contour]
    #         elif dist_metric == 'manhattan':
    #             dists = [cityblock(p_g, p_c) for contour in contours for p_c in contour]  
    #         elif dist_metric == 'chebyshev':
    #             dists = [chebyshev(p_g, p_c) for contour in contours for p_c in contour]
    #         min_dist = min(dists)
    #         closest_contour = contours[dists.index(min_dist) // len(contours[0])]
    #         closest_p_c = closest_contour[dists.index(min_dist) % len(contours[0])]
    #         v_n = np.array([p_g[0] - closest_p_c[0], p_g[1] - closest_p_c[1]])
    #         v_n /= np.linalg.norm(v_n)
    #         factor = adjustment_function(min_dist, factor)
    #         new_point = p_g - v_n * factor
    #         weighted_grid.append(new_point)
    #     return np.array(weighted_grid)