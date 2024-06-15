import numpy as np
import pandas as pd
import trimesh


Data_point_of_edge = np.array(pd.read_excel('./Right-Stack0052.xlsx'))
x_coordintate = Data_point_of_edge[:, 0]
y_coordintate = Data_point_of_edge[:, 1]

angles = np.linspace(0, 2 * np.pi, 100)

x_surface = np.outer(np.sin(angles), y_coordintate)
y_surface = np.outer(np.cos(angles), y_coordintate)
z_surface = np.outer(np.ones(np.size(angles)), x_coordintate)

with open('./Right-Stack0052.obj', 'w') as file:
    for i in range(len(x_surface)):
        for j in range(len(x_surface[0])):
            file.write(f"v {x_surface[i][j]} {y_surface[i][j]} {z_surface[i][j]}\n")

    for i in range(len(x_surface) - 1):
        for j in range(len(x_surface[0]) - 1):
            vert1 = i * len(x_surface[0]) + j + 1
            vert2 = i * len(x_surface[0]) + j + 2
            vert3 = (i + 1) * len(x_surface[0]) + j + 2
            vert4 = (i + 1) * len(x_surface[0]) + j + 1
            file.write(f"f {vert1} {vert2} {vert3}\n")
            file.write(f"f {vert1} {vert3} {vert4}\n")
            
def Volume_estimation(mesh, sample_count=35000):
    min, max = mesh.bounds
    inside_count = 0
    inside = []
    outside = []

    for _ in range(sample_count):
        sample = np.random.uniform(min, max)
        if mesh.ray.contains_points([sample])[0]:
            inside_count += 1
            inside.append(sample)
        else:
            outside.append(sample)

    np.savetxt('./inside_points-Stack0052-Right.txt', np.array(inside))
    np.savetxt('./outside_points-Stack0052-Right.txt', np.array(outside))

    bounding_box_vol = np.prod(max - min)
    Estimated_volume = (inside_count / sample_count) * bounding_box_vol

    return Estimated_volume

mesh_data = trimesh.load_mesh('./Right-Stack0052.obj')
volume = Volume_estimation(mesh_data)

print("Estimated volume:", volume * (3.3567e-11))
print("Estimated volume:", volume)

