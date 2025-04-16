import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull

# Parameters for the square
num_points = 1000  # Number of points
x_min, x_max = 0, 10  # Range for X-axis
y_min, y_max = 0, 10  # Range for Y-axis

# Generate random points
x_coords = np.random.uniform(x_min, x_max, num_points)
y_coords = np.random.uniform(y_min, y_max, num_points)
z_coords = np.zeros(num_points)  # Z-axis is 0 for a 2D square

# Create the point cloud
points = np.column_stack((x_coords, y_coords, z_coords))
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# Use 2D points for ConvexHull
points_2d = np.column_stack((x_coords, y_coords))
hull = ConvexHull(points_2d)

# Create colors: default gray
colors = np.tile([0.5, 0.5, 0.5], (num_points, 1))

# Mark convex hull points in pink (rosa)
colors[hull.vertices] = [1.0, 0.0, 1.0]  # RGB for pink
point_cloud.colors = o3d.utility.Vector3dVector(colors)

# Create LineSet for convex hull edges
hull_indices = hull.vertices
lines = []

# Create edges by connecting the hull points in order
for i in range(len(hull_indices)):
    start = hull_indices[i]
    end = hull_indices[(i + 1) % len(hull_indices)]  # Wrap around to close the loop
    lines.append([start, end])

# Define line colors (all pink like the points)
line_colors = [[1.0, 0.0, 1.0] for _ in lines]

# Create Open3D LineSet
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(line_colors)

# Visualize point cloud with convex hull lines
o3d.visualization.draw_geometries(
    [point_cloud, line_set],
    window_name="2D Point Cloud with Convex Hull (Pink Points + Lines)"
)
