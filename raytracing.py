import matplotlib.pyplot as plt
import numpy as np

def raytracing(point, contours):
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

# Definition der Konturen für ein generisches PCB
contours = []

# Äußere Kontur für den Rand des PCB hinzufügen
outer_contour = [(0, 0), (6, 0), (6, 8), (0, 8), (0, 0)]
contours.append(outer_contour)

# Leiterbahnen hinzufügen (Beispiel)
contours.append([(1, 1), (4, 1), (4, 2), (3.5, 2.5), (2, 5), (4, 5), (4, 7), (1, 7), (1, 1)])

# Pads hinzufügen (Beispiel)
pad_center = (3, 2)
pad_radius = 0.5
num_pad_points = 20
pad_contour = [(pad_center[0] + pad_radius * np.cos(2 * np.pi * i / num_pad_points),
                pad_center[1] + pad_radius * np.sin(2 * np.pi * i / num_pad_points))
               for i in range(num_pad_points)]
pad_contour.append(pad_contour[0])  # Anfangspunkt hinzufügen
contours.append(pad_contour)

# Bohrungen hinzufügen (Beispiel)
hole_center = (2, 6)
hole_radius = 0.2
num_hole_points = 20
hole_contour = [(hole_center[0] + hole_radius * np.cos(2 * np.pi * i / num_hole_points),
                 hole_center[1] + hole_radius * np.sin(2 * np.pi * i / num_hole_points))
                for i in range(num_hole_points)]
hole_contour.append(hole_contour[0])  # Anfangspunkt hinzufügen
contours.append(hole_contour)

# Generieren von 1000 gleichmäßig verteilten Testpunkten innerhalb des äußeren Konturs
x_values = np.linspace(0.1, 5.9, 50)
y_values = np.linspace(0.1, 7.9, 50)
test_points = [[x, y] for x in x_values for y in y_values]

# Überprüfen, ob die Testpunkte innerhalb der Konturen liegen
results = [raytracing(point, contours) for point in test_points]

# Plotten der Konturen und der Testpunkte
for c in contours:
    x, y = zip(*c)
    plt.plot(x, y, '-', color='black')

inside_points = [test_points[i] for i in range(len(test_points)) if results[i]]
outside_points = [test_points[i] for i in range(len(test_points)) if not results[i]]

x_in, y_in = zip(*inside_points)
x_out, y_out = zip(*outside_points)

plt.plot(x_in, y_in, 'o', color='green', markersize=1)  # Punkte innerhalb der Konturen (Grün)
plt.plot(x_out, y_out, 'o', color='red', markersize=1)  # Punkte außerhalb der Konturen (Rot)

plt.title("Ray Tracing Anwendungsfall mit generischem PCB")
plt.xlabel("X-Achse")
plt.ylabel("Y-Achse")
plt.grid(True)
plt.axis('equal')
plt.show()