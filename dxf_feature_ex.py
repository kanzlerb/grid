import matplotlib.pyplot as plt
from ezdxf import readfile

def extract_geometry(dxf_path):
    geometry_info = []

    # DXF-Datei lesen
    doc = readfile(dxf_path)

    # Modelspace abrufen
    msp = doc.modelspace()

    # Durchlaufen der Entitäten
    for entity in msp:
        # Überprüfen, ob die Entität eine Linie ist
        if entity.dxftype() == 'LINE':
            start_point = entity.dxf.start
            end_point = entity.dxf.end
            geometry_info.append(('Line', start_point, end_point))
        # Überprüfen, ob die Entität ein Kreis ist
        elif entity.dxftype() == 'CIRCLE':
            center = entity.dxf.center
            radius = entity.dxf.radius
            geometry_info.append(('Circle', center, radius))
        # Weitere Entitätentypen können hier hinzugefügt werden

    return geometry_info

def plot_dxf(dxf_path):
    geometries = extract_geometry(dxf_path)

    fig, ax = plt.subplots()
    for geometry in geometries:
        if geometry[0] == 'Line':
            start_point, end_point = geometry[1], geometry[2]
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='blue')
        elif geometry[0] == 'Circle':
            center, radius = geometry[1], geometry[2]
            circle = plt.Circle((center[0], center[1]), radius, color='red', fill=False)
            ax.add_artist(circle)

    ax.set_aspect('equal', 'datalim')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('DXF Plot')
    plt.grid(True)
    plt.show()

# Beispielaufruf
dxf_file = "dxf_files/test_pcb_1.dxf"
plot_dxf(dxf_file)