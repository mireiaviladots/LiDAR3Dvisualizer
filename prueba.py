import open3d as o3d
import numpy as np

def create_mesh_from_point_cloud(points, colors, alpha=0.5):
    """
    Crea una malla a partir de una nube de puntos y eleva la malla en la dirección z.

    :param points: Numpy array de puntos (N, 3).
    :param colors: Numpy array de colores (N, 3).
    :param alpha: Valor de transparencia (0.0 a 1.0).
    :return: Malla elevada en la dirección z.
    """
    # Crear una nube de puntos
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Estimar las normales de la nube de puntos
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Crear la malla a partir de la nube de puntos
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)

    # Calcular el promedio de z para cada celda de la malla
    vertices = np.asarray(mesh.vertices)
    z_mean = np.mean(vertices[:, 2])

    # Elevar la malla en la dirección z
    vertices[:, 2] += z_mean
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    return mesh

# Ejemplo de uso
points = np.random.rand(1000, 3)  # Puntos de ejemplo
colors = np.random.rand(1000, 3)  # Colores de ejemplo

# Crear la malla a partir de la nube de puntos y elevarla en z
mesh = create_mesh_from_point_cloud(points, colors, alpha=0.5)

# Visualizar la malla
o3d.visualization.draw_geometries([mesh])