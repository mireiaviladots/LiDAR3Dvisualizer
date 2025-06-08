# Importing all necessary libraries
import numpy as np
import open3d as o3d
import csv
from scipy.spatial import cKDTree
from customtkinter import *
from tkinter import filedialog
import xml.etree.ElementTree as ET
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
import fnmatch
import matplotlib.pyplot as plt
import os
import sys
import math
import utm
import random
import ctypes
import threading
from tkinter import ttk
from tkinter import messagebox
import laspy
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import scipy.ndimage as ndi
from collections import Counter
import json
from pyproj import CRS, Transformer
from shapely.geometry import LineString, box
import gc

source_location = None
pc_filepath = None
csv_filepath = None
xml_filepath = None
point_size = None
vox_size = None
altura_extra = None
dose_min_csv = None
dose_max_csv = None
low_max = None
medium_min = None
medium_max = None
high_min = None
high_max = None
previous_point_value = ""
previous_voxel_value = ""
previous_downsample_value = ""
show_dose_layer = False
downsample = None
show_source = False
vis = None
heatmap = None
xcenter = None
ycenter = None
FAltcenter = None
Hcenter = None
lonmin = None
lonmax = None
latmin = None
latmax = None
run_prueba = False
right_frame_width = False
right_frame_height = False
screen_width = False
left_frame_width = False
title_bar_height = False
progress_bar = None
original_left_frame_widgets = []
legend_frame = None
legend_canvas = None
las_object = None
panel_canvas = None
button_seg = None
seg_frame = None
button_from = None
from_frame = None
button_to = None
to_frame = None
button_visualize = None
button_return = None
build_seg = False
from_set = False
to_set = False
show_latlon = False
show_utm = False
latlon_set = 0
utm_set = 0
posiciones = None
mi_set = False
selected_positions = None
combined_mesh = None
pixels_x = None
pixels_y = None
delta_x = None
delta_y = None
cell_stats = None
combined_mesh_pixels_x = None
combined_mesh_pixels_y = None
las = None
min_x_las = max_x_las = min_y_las = max_y_las = None
rects_top = None
building_prism_cells = None


# Displays a point cloud without voxelization, optionally applying dose layer visualization
def point_cloud_no_vox(show_dose_layer, pc_filepath, downsample, xml_filepath, csv_filepath, high_dose_rgb, medium_dose_rgb,
                        low_dose_rgb, dose_min_csv, low_max, medium_min, medium_max, high_min, altura_extra, show_source, source_location, point_size, progress_bar):

    def run():  # Inner function to be run in a separate thread
        global vis
        try:
            # Load the point cloud file
            pcd = o3d.io.read_point_cloud(pc_filepath)

            # Apply downsampling if specified (between 1% and 100%)
            if downsample is not None:
                if not (1 <= downsample <= 100):
                    messagebox.showerror("Error", "The downsample value must be between 1 and 100.")
                    return
                downsample_value = float(downsample) / 100.0
                if 0 < downsample_value <= 1:
                    if downsample_value == 1:
                        downsample_value = 0.99  # Avoid using full value
                    voxel_size = 1 * downsample_value
                    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
                    pcd = downsampled_pcd

            nube_puntos = np.asarray(pcd.points)  # Extract point coordinates

            update_progress_bar(progress_bar, 20)

            # Get color data if available, else assign white
            if pcd.has_colors():
                rgb = np.asarray(pcd.colors)
            else:
                rgb = np.ones_like(nube_puntos)

            if show_dose_layer:
                # Extract origin from XML for coordinate translation
                origin = get_origin_from_xml(xml_filepath)
                geo_points = nube_puntos + origin  # Convert local points to georeferenced

                # Read UTM coordinates and dose values from CSV
                utm_coords = np.genfromtxt(csv_filepath, delimiter=',', skip_header=1)
                utm_points = utm_coords[:, :2]
                dosis = utm_coords[:, 2]

                update_progress_bar(progress_bar, 30)

                # Build a spatial index to speed up nearest-neighbor search
                tree = cKDTree(utm_points)

                update_progress_bar(progress_bar, 40)

                # Filter points that lie within the bounds of the dose grid
                x_min, y_min = np.min(utm_points, axis=0)
                x_max, y_max = np.max(utm_points, axis=0)
                dentro_area = (
                    (geo_points[:, 0] >= x_min) & (geo_points[:, 0] <= x_max) &
                    (geo_points[:, 1] >= y_min) & (geo_points[:, 1] <= y_max)
                )

                update_progress_bar(progress_bar, 60)

                # Compute closest dose values for each 3D point within bounds
                puntos_dentro = geo_points[dentro_area]
                dosis_nube = np.full(len(puntos_dentro), np.nan)
                distancias, indices_mas_cercanos = tree.query(puntos_dentro[:, :2])
                dosis_nube[:] = dosis[indices_mas_cercanos]

                # Filter valid dose points
                valid_points = ~np.isnan(dosis_nube)
                puntos_dosis_elevados = puntos_dentro[valid_points]
                dosis_filtrada = dosis_nube[valid_points]

                update_progress_bar(progress_bar, 80)

                # Assign color based on dose levels
                colores_dosis = get_dose_color(
                    dosis_filtrada, high_dose_rgb, medium_dose_rgb, low_dose_rgb,
                    dose_min_csv, low_max, medium_min, medium_max, high_min
                )

                # Raise dose points vertically for visibility
                puntos_dosis_elevados[:, 2] += altura_extra

                # Restore original point cloud
                pcd.points = o3d.utility.Vector3dVector(geo_points)
                pcd.colors = o3d.utility.Vector3dVector(rgb)

                update_progress_bar(progress_bar, 90)

                # Create a new point cloud for dose layer
                pcd_dosis = o3d.geometry.PointCloud()
                pcd_dosis.points = o3d.utility.Vector3dVector(puntos_dosis_elevados)
                pcd_dosis.colors = o3d.utility.Vector3dVector(colores_dosis)

            else:
                # If dose layer not shown, still show progress
                for v in [30, 40, 50, 60, 70, 80]:
                    update_progress_bar(progress_bar, v)

            update_progress_bar(progress_bar, 100)
            progress_bar.grid_forget()  # Hide progress bar

            # Initialize Open3D visualizer
            vis = o3d.visualization.Visualizer()

            # Get dimensions of GUI layout
            right_frame.update_idletasks()
            right_frame_width = right_frame.winfo_width()
            right_frame_height = right_frame.winfo_height()
            left_frame.update_idletasks()
            left_frame_width = left_frame.winfo_width()
            title_bar_height = ctypes.windll.user32.GetSystemMetrics(4)

            # Create Open3D window inside the GUI
            vis.create_window(window_name='Open3D',
                              width=right_frame_width,
                              height=right_frame_height,
                              left=left_frame_width,
                              top=title_bar_height)
            vis.clear_geometries()
            vis.add_geometry(pcd)  # Add main point cloud

            # Add dose layer visualization if available
            if show_dose_layer:
                vis.add_geometry(pcd_dosis)

            # If radiation source is enabled, draw a black sphere at its location
            if show_dose_layer and show_source and source_location is not None:
                source_point = [source_location[0], source_location[1], np.max(puntos_dosis_elevados[:, 2])]
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)
                sphere.translate(source_point)
                sphere.paint_uniform_color([0, 0, 0])
                vis.add_geometry(sphere)

            # Set render options such as point size
            render_option = vis.get_render_option()
            render_option.point_size = point_size

            # Continuously update the 3D visualization
            while True:
                vis.poll_events()
                vis.update_renderer()

                if not vis.poll_events():  # If visualizer is closed
                    enable_left_frame()  # Re-enable GUI frame
                    break

        except Exception as e:
            # Show error message if anything goes wrong
            messagebox.showerror("Error", f"An error occurred: {e}")
            enable_left_frame()  # Restore GUI state even on error

    # Run the visualization logic in a separate thread
    threading.Thread(target=run, daemon=True).start()


# Displays a point cloud with voxelization, optionally applying dose layer visualization
def point_cloud_vox(show_dose_layer, pc_filepath, xml_filepath, csv_filepath, high_dose_rgb, medium_dose_rgb, low_dose_rgb, dose_min_csv, low_max,
            medium_min, medium_max, high_min, altura_extra, progress_bar):

    def run():  # Inner function to be run in a separate thread
        global vis
        try:
            # Load the point cloud from file
            pcd = o3d.io.read_point_cloud(pc_filepath)
            xyz = np.asarray(pcd.points)  # Extract XYZ coordinates

            update_progress_bar(progress_bar, 20)

            # Get colors if available, otherwise set all white
            if pcd.has_colors():
                rgb = np.asarray(pcd.colors)
            else:
                rgb = np.ones_like(xyz)

            # If dose layer not shown, just assign original points and colors
            if not show_dose_layer:
                pcd.points = o3d.utility.Vector3dVector(xyz)
                pcd.colors = o3d.utility.Vector3dVector(rgb)

            # If dose layer is shown, transform point coordinates using origin offset
            if show_dose_layer:
                origin = get_origin_from_xml (xml_filepath)
                geo_points = xyz + origin
                pcd.points = o3d.utility.Vector3dVector(geo_points)
                pcd.colors = o3d.utility.Vector3dVector(rgb)

            # Create voxel grid from the point cloud with user-defined voxel size
            vsize = vox_size
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=vsize)

            # Prepare a cube shape for voxel visualization
            cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
            cube.paint_uniform_color([1, 0, 0])  # Red color
            cube.compute_vertex_normals()

            # Build voxel mesh geometry
            voxels = voxel_grid.get_voxels()
            vox_mesh = o3d.geometry.TriangleMesh()
            for v in voxels:
                cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
                cube.paint_uniform_color(v.color)
                cube.translate(v.grid_index, relative=False)
                vox_mesh += cube

            # Scale and position voxel mesh
            vox_mesh.translate([0.5, 0.5, 0.5], relative=True)
            vox_mesh.scale(vsize, [0, 0, 0])
            vox_mesh.translate(voxel_grid.origin, relative=True)

            update_progress_bar(progress_bar, 30)

            # If dose layer is enabled, match point cloud with dose values from CSV
            if show_dose_layer:
                utm_coords = np.genfromtxt(csv_filepath, delimiter=',', skip_header=1)
                utm_points = utm_coords[:, :2]  # Extract XY UTM coordinates
                dosis = utm_coords[:, 2]        # Extract dose values

                tree = cKDTree(utm_points)      # Build KDTree for spatial matching

                update_progress_bar(progress_bar, 40)

                # Define bounding box of the CSV data
                x_min, y_min = np.min(utm_points, axis=0)
                x_max, y_max = np.max(utm_points, axis=0)

                # Filter cloud points that fall within CSV bounds
                dentro_area = (
                        (geo_points[:, 0] >= x_min) & (geo_points[:, 0] <= x_max) &
                        (geo_points[:, 1] >= y_min) & (geo_points[:, 1] <= y_max)
                )
                puntos_dentro = geo_points[dentro_area]
                dosis_nube = np.full(len(puntos_dentro), np.nan)

                # Match closest CSV dose point to each point in cloud
                distancias, indices_mas_cercanos = tree.query(puntos_dentro[:,:2])

                update_progress_bar(progress_bar, 60)

                # Assign dose values
                dosis_nube[:] = dosis[indices_mas_cercanos]
                valid_points = ~np.isnan(dosis_nube)

                # Keep only valid points and corresponding dose values
                puntos_dosis_elevados = puntos_dentro[valid_points]
                dosis_filtrada = dosis_nube[valid_points]

                # Get RGB colors based on dose levels
                colores_dosis = get_dose_color(dosis_filtrada, high_dose_rgb, medium_dose_rgb,
                                                    low_dose_rgb, dose_min_csv, low_max,
                                                    medium_min, medium_max, high_min)

                # Raise Z-axis for visualization if specified
                puntos_dosis_elevados[:, 2] += altura_extra

                # Build a new point cloud for dose visualization
                pcd_dosis = o3d.geometry.PointCloud()
                pcd_dosis.points = o3d.utility.Vector3dVector(puntos_dosis_elevados)
                pcd_dosis.colors = o3d.utility.Vector3dVector(colores_dosis)

                # Create voxel grid from dose point cloud
                voxel_grid_dosis = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_dosis, voxel_size=vsize)
                voxels_dosis = voxel_grid_dosis.get_voxels()
                vox_mesh_dosis = o3d.geometry.TriangleMesh()

                update_progress_bar(progress_bar, 80)

                # Reuse cube to visualize dose voxels
                cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
                cube.paint_uniform_color([1, 0, 0])
                cube.compute_vertex_normals()
                for v in voxels_dosis:
                    cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
                    cube.paint_uniform_color(v.color)
                    cube.translate(v.grid_index, relative=False)
                    vox_mesh_dosis += cube

                update_progress_bar(progress_bar, 90)

                # Adjust dose voxel mesh size and position
                vox_mesh_dosis.translate([0.5, 0.5, 0.5], relative=True)
                vox_mesh_dosis.scale(vsize, [0, 0, 0])
                vox_mesh_dosis.translate(voxel_grid_dosis.origin, relative=True)

            else:
                # If dose layer not shown, still show progress
                for v in [30, 40, 50, 60, 70, 80]:
                    update_progress_bar(progress_bar, v)

            update_progress_bar(progress_bar, 100)
            progress_bar.grid_forget()  # Hide progress bar

            # Initialize Open3D visualizer
            vis = o3d.visualization.Visualizer()

            # Get dimensions of GUI layout
            right_frame.update_idletasks()
            right_frame_width = right_frame.winfo_width()
            right_frame_height = right_frame.winfo_height()
            left_frame.update_idletasks()
            left_frame_width = left_frame.winfo_width()
            title_bar_height = ctypes.windll.user32.GetSystemMetrics(4)

            # Create Open3D window inside the GUI
            vis.create_window(window_name='Open3D',
                              width=right_frame_width,
                              height=right_frame_height,
                              left=left_frame_width,
                              top=title_bar_height)
            vis.clear_geometries()
            vis.add_geometry(vox_mesh)  # Add voxel mesh to visualizer

            # If dose mesh is present, add it too
            if show_dose_layer:
                vis.add_geometry(vox_mesh_dosis)

            # If radiation source is enabled, draw a black sphere at its location
            if show_dose_layer and show_source and source_location is not None:
                source_point = [[source_location[0], source_location[1], np.max(puntos_dosis_elevados[:, 2])]]
                source_pcd = o3d.geometry.PointCloud()
                source_pcd.points = o3d.utility.Vector3dVector(source_point)
                source_pcd.paint_uniform_color([0, 0, 0])
                vis.add_geometry(source_pcd)

            # Continuously update the 3D visualization
            while True:
                vis.poll_events()
                vis.update_renderer()

                if not vis.poll_events():  # If visualizer is closed
                    enable_left_frame()  # Re-enable GUI frame
                    break

        except Exception as e:
            # Show error message if anything goes wrong
            messagebox.showerror("Error", f"An error occurred: {e}")
            enable_left_frame()  # Restore GUI state even on error

    # Run the visualization logic in a separate thread
    threading.Thread(target=run, daemon=True).start()


# Generates a 3D grid from a LAS object, creating prisms for each cell and extracting statistics.
# Also identifies rooftops of buildings and stores their coordinates.
def grid(las_object, pixels_x, pixels_y):
    global combined_mesh, delta_x, delta_y, cell_stats, rects_top, building_prism_cells

    progress_bar = create_progress_bar()
    update_progress_bar(progress_bar, 10)

    # Clear any previous mesh and statistics
    if combined_mesh is not None:
        del combined_mesh
    if cell_stats is not None:
        del cell_stats
    rects_top = []
    building_prism_cells = []

    gc.collect()  # Free unused memory

    # Load LAS points and prepare XYZ coordinates
    las = las_object
    las_points = np.vstack((las.x, las.y, las.z)).T

    # Extract and normalize RGB colors if available
    if hasattr(las, "red"):
        colors = np.vstack((las.red, las.green, las.blue)).T
        if colors.max() > 1.0:
            colors = colors / 65535.0
    else:
        colors = np.zeros_like(las_points)

    # Extract classification and tree-based classification data
    classifications = las.classification
    classificationtree = las['classificationtree']

    update_progress_bar(progress_bar, 20)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(las_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Get point and color arrays
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(points)

    # Calculate bounds for X and Y
    min_x, min_y = np.min(points[:, :2], axis=0)
    max_x, max_y = np.max(points[:, :2], axis=0)

    update_progress_bar(progress_bar, 30)

    # Calculate cell size for the grid
    delta_x = (max_x - min_x) / pixels_x
    delta_y = (max_y - min_y) / pixels_y

    # Initialize data structures for storing cell information
    z_values = np.full((pixels_y, pixels_x), np.nan)
    cell_stats = np.empty((pixels_y, pixels_x), dtype=object)
    for i in range(pixels_y):
        for j in range(pixels_x):
            cell_stats[i, j] = {'z_values': [], 'colors': [], 'classes': [], 'tree_classes': [], 'building': None}

    update_progress_bar(progress_bar, 40)

    # Compute cell indices for each point
    x_idx = ((points[:, 0] - min_x) / delta_x).astype(int)
    y_idx = ((points[:, 1] - min_y) / delta_y).astype(int)
    valid_mask = (
            (x_idx >= 0) & (x_idx < pixels_x) &
            (y_idx >= 0) & (y_idx < pixels_y)
    )

    # Fill in height and color information for each grid cell
    for xi, yi, z, color in zip(x_idx[valid_mask], y_idx[valid_mask], points[valid_mask][:, 2], colors[valid_mask]):
        cell_stats[yi, xi]['z_values'].append(z)
        cell_stats[yi, xi]['colors'].append(color)

    # Also fill classification info per cell
    x_idx_las = ((las_points[:, 0] - min_x) / delta_x).astype(int)
    y_idx_las = ((las_points[:, 1] - min_y) / delta_y).astype(int)
    valid_mask_las = (
            (x_idx_las >= 0) & (x_idx_las < pixels_x) &
            (y_idx_las >= 0) & (y_idx_las < pixels_y))

    update_progress_bar(progress_bar, 50)

    for xi, yi, cls, cls_tree in zip(x_idx_las[valid_mask_las], y_idx_las[valid_mask_las], classifications[valid_mask_las], classificationtree[valid_mask_las]):
        cell_stats[yi, xi]['classes'].append(cls)
        cell_stats[yi, xi]['tree_classes'].append(cls_tree)

    # Calculate filtered average Z height for each grid cell
    for i in range(pixels_y):
        for j in range(pixels_x):
            z_vals = cell_stats[i][j]['z_values']
            if z_vals:
                z_vals = np.array(z_vals)
                z_mean = np.mean(z_vals)
                z_std = np.std(z_vals)
                mask = z_vals <= z_mean + 2 * z_std
                filtered_z_vals = z_vals[mask]
                z_values[i, j] = np.mean(filtered_z_vals) + 2 * np.std(filtered_z_vals)
                cell_stats[i][j]['color'] = np.mean(cell_stats[i][j]['colors'], axis=0)

    update_progress_bar(progress_bar, 60)

    # Create 3D box (prism) geometry per cell
    prisms = []
    prisms_building = []
    building_prism_cells = []
    count_6 = 0  # Track number of class 6 (building) cells
    for i in range(pixels_y):
        for j in range(pixels_x):
            if not np.isnan(z_values[i, j]):
                z_final = z_values[i, j]
                z_min = np.min(cell_stats[i][j]['z_values'])
                height = z_final - z_min
                if height > 0:
                    # Create and place prism
                    prism = o3d.geometry.TriangleMesh.create_box(width=delta_x, height=delta_y, depth=height)
                    prism.translate((min_x + j * delta_x, min_y + i * delta_y, z_min))
                    prism.paint_uniform_color(cell_stats[i][j]['color'])

                    # Compute majority class per cell
                    classes = cell_stats[i][j]['classes']
                    tree_classes = cell_stats[i][j]['tree_classes']

                    if classes:
                        majority_class = Counter(classes).most_common(1)[0][0]
                    else:
                        majority_class = None

                    if tree_classes:
                        majority_tree_class = Counter(tree_classes).most_common(1)[0][0]
                    else:
                        majority_tree_class = None

                    cell_stats[i][j]['majority_class'] = majority_class
                    cell_stats[i][j]['majority_tree_class'] = majority_tree_class

                    # If class is building (6), track separately
                    if majority_class == 6:
                        count_6 += 1
                        prisms_building.append(prism)
                        building_prism_cells.append((i, j))

                    prisms.append(prism)

    update_progress_bar(progress_bar, 80)

    # Combine all prisms into one mesh
    combined_mesh = o3d.geometry.TriangleMesh()
    for prism in prisms:
        combined_mesh += prism

    rects_top = []

    # Identify and store rooftop (top face) rectangles of buildings
    for prism in prisms_building:
        vertices = np.asarray(prism.vertices)
        top_z = np.max(vertices[:, 2])
        top_face_vertices = vertices[np.isclose(vertices[:, 2], top_z)]

        if top_face_vertices.shape[0] >= 4:
            pts = top_face_vertices[:4, :2]
            center = np.mean(pts, axis=0)
            angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
            idx_sorted = np.argsort(angles)
            pts_ordered = pts[idx_sorted]
            rects_top.append(pts_ordered)

    update_progress_bar(progress_bar, 100)
    progress_bar.grid_forget()

    # Call UI update panel with processed data
    panel_left_frame(xcenter, ycenter, FAltcenter, las_object, pixels_x, pixels_y, building_prism_cells)


# Detects obstacles (trees and buildings) between a selected point and a point defined by latitude and longitude or UTM coordinates.
def tree_obstacles(las_object, entry_latitude, entry_longitude, entry_zone, entry_easting, entry_northing, latlon_set, utm_set, combined_mesh, pixels_x, pixels_y, delta_x, delta_y, cell_stats, selected_rect_indices, building_prism_cells):

    def run():
        global vis
        try:
            # Ensure a start point (selected_positions) is available
            if not selected_positions:
                messagebox.showwarning("Warning", "Please select one point before continuing.")
                return

            if latlon_set == 1:
                # Validate UTM zone input
                try:
                    zone = int(entry_zone.get())
                except (ValueError, AttributeError):
                    messagebox.showerror("Error", "UTM zone must be an integer between 1 and 60.")
                    return

                if not 1 <= zone <= 60:
                    messagebox.showerror("Error", "UTM zone must be between 1 and 60.")
                    return

                # Validate latitude input
                try:
                    lat = float(entry_latitude.get())
                except (ValueError, AttributeError):
                    messagebox.showerror("Error", "Latitude must be a number between -90 and 90.")
                    return

                if not -90 <= lat <= 90:
                    messagebox.showerror("Error", "Latitude must be between -90 and 90.")
                    return

                # Validate longitude input
                try:
                    lon = float(entry_longitude.get())
                except (ValueError, AttributeError):
                    messagebox.showerror("Error", "Longitude must be a number between -180 and 180.")
                    return

                if not -180 <= lon <= 180:
                    messagebox.showerror("Error", "Longitude must be between -180 and 180.")
                    return

                # Convert lat/lon to UTM coordinates
                utm_x, utm_y = latlon_to_utm(lat, lon, zone)

            elif utm_set == 1:
                # Validate UTM easting and northing input
                try:
                    easting = float(entry_easting.get())
                except (ValueError, AttributeError):
                    messagebox.showerror("Error", "Easting must be a number.")
                    return

                if not 100000 <= easting <= 900000:
                    messagebox.showerror("Error", "Easting must be between 100,000 and 900,000 meters.")
                    return

                try:
                    northing = float(entry_northing.get())
                except (ValueError, AttributeError):
                    messagebox.showerror("Error", "Northing must be a number.")
                    return

                if not 0 <= northing <= 10000000:
                    messagebox.showerror("Error", "Northing must be between 0 and 10,000,000 meters.")
                    return

                utm_x = easting
                utm_y = northing

            else:
                messagebox.showerror("Error", "Please select either Lat/Lon or UTM input mode.")
                return

            las = las_object

            # Check if the target point is within the point cloud bounds
            min_x, max_x = np.min(las.x), np.max(las.x)
            min_y, max_y = np.min(las.y), np.max(las.y)

            if not (min_x <= utm_x <= max_x and min_y <= utm_y <= max_y):
                messagebox.showerror("Error", "The entered position is outside the bounds of the point cloud.")
                return

            progress_bar = create_progress_bar()

            # Disable UI inputs during processing
            disable_left_frame()
            entry_latitude.configure(state='disabled')
            entry_longitude.configure(state='disabled')

            update_progress_bar(progress_bar, 10)

            # Mark building cells from selected rectangle indices
            for building_number, index_group in enumerate(selected_rect_indices, start=1):
                for idx in index_group:
                    if idx < len(building_prism_cells):
                        i, j = building_prism_cells[idx]
                        cell_stats[i][j]['building'] = building_number

            # Get drone start positions with estimated height above sea level
            posiciones_con_altura = []
            for (x, y, alt) in selected_positions:
                col = int((x - min_x) / delta_x)
                row = int((y - min_y) / delta_y)

                if 0 <= col < pixels_x and 0 <= row < pixels_y:
                    cell_z_vals = cell_stats[row][col]['z_values']
                    if cell_z_vals:
                        altitud_nivel_mar = np.mean(cell_z_vals)
                        z_dron = altitud_nivel_mar + alt
                        posiciones_con_altura.append((x, y, z_dron))

            update_progress_bar(progress_bar, 40)

            # Get altitude of the target point (from UTM) using mean cell elevation
            posiciones_latlonh = []
            col = int((utm_x - min_x) / delta_x)
            row = int((utm_y - min_y) / delta_y)

            update_progress_bar(progress_bar, 50)

            if 0 <= col < pixels_x and 0 <= row < pixels_y:
                cell_z_vals = cell_stats[row][col]['z_values']
                if cell_z_vals:
                    altitud_nivel_mar = np.mean(cell_z_vals)
                    posiciones_latlonh.append((utm_x, utm_y, altitud_nivel_mar))

            update_progress_bar(progress_bar, 70)

            # Function to generate a random color avoiding pink
            def generar_color_aleatorio_no_rosa():
                while True:
                    color = [random.random(), random.random(), random.random()]
                    if not (abs(color[0] - 1.0) < 0.1 and abs(color[1] - 0.0) < 0.1 and abs(
                            color[2] - 1.0) < 0.1):
                        return color

            update_progress_bar(progress_bar, 80)

            colores_puntos = []
            for _ in posiciones_con_altura:
                color = generar_color_aleatorio_no_rosa()
                colores_puntos.append(color)

            update_progress_bar(progress_bar, 95)

            # Initialize lists to hold crossed obstacles
            total_arboles_cruzados = []
            total_edificios_cruzados = []

            # If start and end points are valid, compute intersecting cells
            if posiciones_con_altura and posiciones_latlonh:
                puntos_linea = posiciones_con_altura + posiciones_latlonh
                line_indices = [[i, len(puntos_linea) - 1] for i in range(len(posiciones_con_altura))]

                # For each start point, trace a 2D line to the target point
                for idx, (start_point, _) in enumerate(zip(posiciones_con_altura, line_indices)):
                    end_point = posiciones_latlonh[0]
                    linea = LineString([start_point[:2], end_point[:2]])

                    clases_cruzadas = []        # Tree classes intersected
                    edificios_cruzados = []     # Buildings intersected

                    for i in range(pixels_y):
                        for j in range(pixels_x):
                            x0 = min_x + j * delta_x
                            y0 = min_y + i * delta_y
                            x1 = x0 + delta_x
                            y1 = y0 + delta_y
                            celda = box(x0, y0, x1, y1)
                            if linea.intersects(celda):
                                # Interpolate drone height at current cell (z_rayo)
                                x_celda = (x0 + x1) / 2
                                y_celda = (y0 + y1) / 2

                                x_start, y_start, z_start = start_point
                                x_end, y_end, z_end = end_point
                                dx = x_end - x_start
                                dy = y_end - y_start
                                dz = z_end - z_start

                                denom = dx ** 2 + dy ** 2
                                if denom > 0:
                                    t = ((x_celda - x_start) * dx + (y_celda - y_start) * dy) / denom
                                    t = max(0, min(1, t))
                                    z_rayo = z_start + t * dz
                                else:
                                    z_rayo = z_start

                                cell_z_vals = cell_stats[i][j].get('z_values', [])

                                if cell_z_vals:
                                    z_max = max(cell_z_vals)
                                else:
                                    z_max = -np.inf

                                # Compare cell elevation with drone height
                                if z_max >= z_rayo:
                                    tree_class = cell_stats[i][j].get('majority_tree_class')
                                    building_num = cell_stats[i][j].get('building')

                                    if tree_class and tree_class != 0:
                                        clases_cruzadas.append(tree_class)

                                    if building_num:
                                        edificios_cruzados.append(building_num)

                    total_arboles_cruzados.append(len(set(clases_cruzadas)))
                    total_edificios_cruzados.append(len(set(edificios_cruzados)))

            update_progress_bar(progress_bar, 100)
            progress_bar.grid_forget()

            # Show legend
            summary_lines(colores_puntos, total_arboles_cruzados, total_edificios_cruzados)

            # Initialize Open3D visualizer
            vis = o3d.visualization.Visualizer()

            # Get dimensions of GUI layout
            right_frame.update_idletasks()
            right_frame_width = right_frame.winfo_width()
            right_frame_height = right_frame.winfo_height()
            left_frame.update_idletasks()
            left_frame_width = left_frame.winfo_width()
            title_bar_height = ctypes.windll.user32.GetSystemMetrics(4)

            # Create Open3D window inside the GUI
            vis.create_window(window_name='Open3D',
                              width=right_frame_width,
                              height=right_frame_height,
                              left=left_frame_width,
                              top=title_bar_height)
            vis.clear_geometries()
            vis.add_geometry(combined_mesh)

            # Draw lines from each start point to destination
            if posiciones_con_altura and posiciones_latlonh:
                puntos_linea = posiciones_con_altura + posiciones_latlonh
                line_indices = [[i, len(puntos_linea) - 1] for i in range(len(posiciones_con_altura))]
                lines = o3d.geometry.LineSet()
                lines.points = o3d.utility.Vector3dVector(puntos_linea)
                lines.lines = o3d.utility.Vector2iVector(line_indices)
                lines.colors = o3d.utility.Vector3dVector([[0, 0, 0]] * len(line_indices))
                vis.add_geometry(lines)

            # Draw colored point clouds for each drone start position
            for punto, color in zip(posiciones_con_altura, colores_puntos):
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector([punto])
                pc.paint_uniform_color(color)
                vis.add_geometry(pc)

            # Draw target point in pink
            if posiciones_latlonh:
                puntos_rosas = o3d.geometry.PointCloud()
                puntos_rosas.points = o3d.utility.Vector3dVector(posiciones_latlonh)
                puntos_rosas.paint_uniform_color([1.0, 0, 1.0])
                vis.add_geometry(puntos_rosas)

            # Continuously update the 3D visualization
            while True:
                vis.poll_events()
                vis.update_renderer()

                if not vis.poll_events():  # If visualizer is closed
                    entry_latitude.configure(state='normal')
                    entry_longitude.configure(state='normal')
                    enable_left_frame()  #Re-enable GUI frame

                    # Clear building markers after visualization
                    for i in range(pixels_y):
                        for j in range(pixels_x):
                            cell_stats[i][j]['building'] = None
                    break

        except Exception as e:
            # Show error message if anything goes wrong
            messagebox.showerror("Error", f"An error occurred: {e}")
            enable_left_frame()  # Restore GUI state even on error

    # Run in a separate thread
    threading.Thread(target=run, daemon=True).start()


# Disables the left frame to prevent user interaction
def disable_left_frame():
    root.attributes('-disabled', True)


# Enables the left frame to allow user interaction
def enable_left_frame():
    root.attributes('-disabled', False)


# Creates a legend frame on the left side of the application for the Segmentation function
def legend_left_frame(counts=None, color_map=None):
    global legend_frame, legend_canvas

    # Get dimensions of the left frame
    left_frame.update_idletasks()
    width = left_frame.winfo_width()
    height = left_frame.winfo_height()

    # Create and place a canvas as the background for the legend with a dark gray color
    legend_canvas = CTkCanvas(left_frame, bg="#2E2E2E", highlightthickness=0, width=width, height=height)
    legend_canvas.place(x=0, y=0)
    legend_canvas.create_rectangle(0, 0, width, height, fill="#2E2E2E", outline="")

    # Create the main container for legend items
    legend_frame = CTkFrame(
        left_frame,
        fg_color="#2E2E2E",
        corner_radius=0,
        width=width,
        height=height,
        border_width=10,
        border_color="#2E2E2E"
    )
    legend_frame.place(x=0, y=0)

    # Remove any previously existing legend items
    for widget in legend_frame.winfo_children():
        widget.destroy()

    # Load default classification colors from JSON
    if color_map is None:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(script_dir, "classification_colors_s.json")

            with open(json_path, "r") as f:
                color_map_raw = json.load(f)["classifications"]
                color_map = {int(k): v for k, v in color_map_raw.items()}

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load color map: {e}")
            color_map = {}

    # Mapping from classification IDs
    class_labels = {
        0: "Created, never classified",
        1: "Unclassified",
        2: "Ground",
        3: "Low Vegetation",
        4: "Medium Vegetation",
        5: "High Vegetation",
        6: "Building",
        7: "Low Point (noise)",
        9: "Water",
        10: "Rail",
        11: "Road Surface",
        13: "Wire – Guard (Shield)",
        14: "Wire – Conductor (Phase)",
        15: "Transmission Tower",
        16: "Wire-structure Connector",
        17: "Bridge Deck",
        18: "High Noise"
    }

    # Create and display each legend item
    for class_id, rgb in color_map.items():
        # Convert RGB float [0-1] values to hex string
        hex_color = "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

        # Get the label name and the count for the class (if provided)
        label = class_labels.get(class_id, f"Customized Class ({class_id})")
        count = counts.get(class_id, 0) if counts else 0

        # Create a container for the legend item
        item_frame = CTkFrame(legend_frame, fg_color="#2E2E2E")
        item_frame.pack(anchor="w", padx=10, pady=3)

        # Draw a colored circle to represent the class
        circle = CTkCanvas(item_frame, width=20, height=20, bg="#2E2E2E", highlightthickness=0)
        circle.create_oval(2, 2, 18, 18, fill=hex_color, outline=hex_color)
        circle.pack(side="left")

        # Add the class label next to the circle
        text_label = CTkLabel(item_frame, text=label, text_color="#F0F0F0", font=("Arial", 12))
        text_label.pack(side="left", padx=(8, 2))

        # Add the point count in parentheses (if available)
        count_label = CTkLabel(item_frame, text=f"({count})", text_color="#A0A0A0", font=("Arial", 12))
        count_label.pack(side="left")


# Creates a left frame for the Obstacle Detection function
def panel_left_frame (xcenter, ycenter, FAltcenter, las_object, pixels_x, pixels_y, building_prism_cells):
        global panel_canvas, button_seg, seg_frame, button_from, from_frame, button_to, to_frame
        global button_visualize, button_return, progress_bar, posiciones, selected_positions
        global build_seg, from_set, to_set, latlon_set, utm_set, rects_top

        enable_left_frame()  # Enable GUI frame

        # Initialize lists for user interaction and selection
        botones = []
        posiciones = []
        selected_positions = []

        # Get dimensions of the left frame
        left_frame.update_idletasks()
        width = left_frame.winfo_width()
        height = left_frame.winfo_height()

        # Create and configure a canvas to draw
        panel_canvas = CTkCanvas(left_frame, bg="#2E2E2E", highlightthickness=0, width=width, height=height)
        panel_canvas.place(x=0, y=0)
        panel_canvas.create_rectangle(0, 0, width, height, fill="#2E2E2E", outline="")
        title = "OBSTACLE DETECTION"
        panel_canvas.create_text(width//2, 40, text=title, font=("Arial", 18, "bold"), fill="white")
        panel_canvas.pack_propagate(False)

        build_seg = False  # Flag to track if building segmentation section is open

        # Function to toggle the Building Segmentation section
        def toggle_parameters():
            global build_seg, from_set, to_set, button_from, from_frame
            build_seg = not build_seg

            if build_seg:
                button_seg.configure(text=" ▲ Building Segmentation")
                seg_frame.pack(pady=(10, 0), fill="x")
                panel_building_frame.pack(pady=(0, 0), fill="x")
                button_from.pack_forget()
                button_from.pack(fill="x", padx=(0, 0), pady=(10, 0))
                button_to.pack_forget()
                button_to.pack(fill="x", padx=(0, 0), pady=(10, 0))

                # Reset FROM section if active
                if from_set:
                    button_from.pack_forget()
                    button_from.pack(fill="x", padx=(0, 0), pady=(10, 0))
                    from_frame.pack_forget()
                    from_frame.pack(pady=(5, 0), fill="x")
                    button_to.pack_forget()
                    button_to.pack(fill="x", padx=(0, 0), pady=(10, 0))
                    from_set = False
                    button_from.configure(text=" ▼ FROM")
                    from_frame.pack_forget()

            else:
                button_seg.configure(text=" ▼ Building Segmentation")
                seg_frame.pack_forget()
                panel_building_frame.pack_forget()

            if to_set:
                to_frame.pack_forget()
                to_frame.pack(pady=(10, 0), fill="x")
                to_set = False
                button_to.configure(text=" ▼ TO")
                to_frame.pack_forget()

        # Create the button to toggle Building Segmentation section
        button_seg = CTkButton(panel_canvas, text=" ▼ Building Segmentation", text_color="#F0F0F0", fg_color="#3E3E3E",
                                           anchor="w", corner_radius=0, command=toggle_parameters)
        button_seg.pack(fill="x", padx=(0, 0), pady=(50, 0))

        # Frame that contains GUI elements for building segmentation
        seg_frame = CTkFrame(panel_canvas, fg_color="#2E2E2E", corner_radius=0)
        label_seg = CTkLabel(seg_frame, text="Selecciona X que formaran el edificio. Cada edificio clica OK.", text_color="white", font=("Arial", 12))
        label_seg.pack(fill="x", padx=(5, 5), pady=(0, 0))

        # Container to display the buildings
        panel_building_frame = CTkFrame(seg_frame, fg_color="#2E2E2E", height=150, corner_radius=0)
        panel_building = CTkFrame(panel_building_frame, height=150, width=250, fg_color="white", corner_radius=10)
        panel_building.grid(row=0, column=0, rowspan=2, padx=(10, 10), pady=(10, 0), sticky="nsew")

        # Canvas for displaying building shapes
        canvas_2d = CTkCanvas(panel_building, bg="white", width=250, height=150, highlightthickness=0)
        canvas_2d.pack(fill="both", expand=True)

        # Compute scale and center for geometry normalization
        start_x = start_y = None
        selection_rect = None
        drag_threshold = 5
        all_points = np.vstack(rects_top)
        min_x, min_y = np.min(all_points[:, :2], axis=0)
        max_x, max_y = np.max(all_points[:, :2], axis=0)
        scale_x = canvas_2d.winfo_reqwidth() / (max_x - min_x)
        scale_y = canvas_2d.winfo_reqheight() / (max_y - min_y)
        scale = min(scale_x, scale_y)
        x_center_geom = (min_x + max_x) / 2
        y_center_geom = (min_y + max_y) / 2

        buildings_canvas = []

        # Draw each building shape on the canvas
        for i, rect in enumerate(rects_top):
            rotated_reflected_points = []
            for x, y in rect[:, :2]:
                x_rot = 2 * x_center_geom - x
                y_rot = 2 * y_center_geom - y
                x_final = 2 * x_center_geom - x_rot
                x_scaled = scale * (x_final - min_x)
                y_scaled = scale * (y_rot - min_y)
                rotated_reflected_points.append((x_scaled, y_scaled))

            flat_coords = [coord for point in rotated_reflected_points for coord in point]
            fill_color = "#3A86FF"
            rect_id = canvas_2d.create_polygon(flat_coords, fill=fill_color, outline="black")

            # Bounding box used for hit detection
            bbox = (
                min(p[0] for p in rotated_reflected_points),
                min(p[1] for p in rotated_reflected_points),
                max(p[0] for p in rotated_reflected_points),
                max(p[1] for p in rotated_reflected_points)
            )

            buildings_canvas.append({
                "id": rect_id,
                "bbox": bbox,
                "color": fill_color,
                "selected": False,
                "building": None,
                "real_coords": rect,
                "index": i,
            })

        # Mouse interaction functions for selecting buildings
        def on_mouse_down(event):
            nonlocal start_x, start_y, selection_rect
            start_x, start_y = event.x, event.y
            selection_rect = canvas_2d.create_rectangle(start_x, start_y, start_x, start_y, outline="red", dash=(2, 2))

        def on_mouse_drag(event):
            canvas_2d.coords(selection_rect, start_x, start_y, event.x, event.y)

        def on_mouse_up(event):
            nonlocal selection_rect
            end_x, end_y = event.x, event.y
            dx, dy = abs(end_x - start_x), abs(end_y - start_y)

            if dx < drag_threshold and dy < drag_threshold:
                # Click selection
                for b in buildings_canvas:
                    x0, y0, x1, y1 = b["bbox"]
                    if x0 <= end_x <= x1 and y0 <= end_y <= y1:
                        if b["building"] is None:
                            b["selected"] = not b["selected"]
                            new_color = "pink" if b["selected"] else b.get("color", "#3A86FF")
                            canvas_2d.itemconfig(b["id"], fill=new_color)
                        break
            else:
                # Box selection
                x0_sel, y0_sel = min(start_x, end_x), min(start_y, end_y)
                x1_sel, y1_sel = max(start_x, end_x), max(start_y, end_y)

                for b in buildings_canvas:
                    bx0, by0, bx1, by1 = b["bbox"]
                    if bx1 >= x0_sel and bx0 <= x1_sel and by1 >= y0_sel and by0 <= y1_sel:
                        if b["building"] is None:
                            b["selected"] = True
                            canvas_2d.itemconfig(b["id"], fill="pink")

            if selection_rect is not None:
                canvas_2d.delete(selection_rect)
                selection_rect = None

        canvas_2d.bind("<ButtonPress-1>", on_mouse_down)
        canvas_2d.bind("<B1-Motion>", on_mouse_drag)
        canvas_2d.bind("<ButtonRelease-1>", on_mouse_up)

        # Button to confirm selected buildings
        ok_button = CTkButton(panel_building_frame, text="OK", text_color="white", width=70, fg_color="#1E3A5F")
        ok_button.grid(row=0, column=1, padx=(0, 10), pady=(10, 5), sticky="s")

        selected_rects_real_coords = []
        selected_rect_indices = []

        # Generate a random color not in exclude list
        def get_random_color(exclude=["pink"]):
            while True:
                r = lambda: random.randint(0, 255)
                color = f'#{r():02x}{r():02x}{r():02x}'
                if color.lower() not in [c.lower() for c in exclude]:
                    return color

        # Confirm building selection and assign a new color
        def confirm_selection():
            new_color = get_random_color(exclude=["pink"])
            current_indices = []
            for b in buildings_canvas:
                if b["selected"]:
                    selected_rects_real_coords.append(b["real_coords"].copy())
                    current_indices.append(b["index"])
                    canvas_2d.itemconfig(b["id"], fill=new_color)
                    b["building"] = "locked"
                    b["selected"] = False
                    b["color"] = new_color
            if current_indices:
                selected_rect_indices.append(tuple(current_indices))

        ok_button.configure(command=confirm_selection)

        # Button to undo last confirmed building selection
        reset_button = CTkButton(panel_building_frame, text="Undo", text_color="white", width=70, fg_color="#1E3A5F")
        reset_button.grid(row=1, column=1, padx=(0, 10), pady=(5, 10), sticky="n")

        def reset_selection():
            if not selected_rect_indices:
                return
            last_group = selected_rect_indices.pop()
            for idx in last_group:
                for b in buildings_canvas:
                    if b["index"] == idx and b["building"] == "locked":
                        b["building"] = None
                        b["selected"] = False
                        b["color"] = "#3A86FF"
                        canvas_2d.itemconfig(b["id"], fill=b["color"])
                        break

        reset_button.configure(command=reset_selection)

        # Configure layout weights for proper resizing
        panel_building_frame.grid_rowconfigure(0, weight=1)
        panel_building_frame.grid_rowconfigure(1, weight=1)
        panel_building_frame.grid_columnconfigure(0, weight=1)
        panel_building_frame.grid_columnconfigure(1, weight=0)

        from_set = False  # Flag to track whether FROM section is expanded

        # Function to toggle the FROM section
        def toggle_from():
            global from_set, build_seg, to_set, button_from, from_frame
            from_set = not from_set
            if from_set:
                button_from.configure(text=" ▲ FROM")
                from_frame.pack(pady=(5, 0), fill="x")
                button_to.pack_forget()
                button_to.pack(fill="x", padx=(0, 0), pady=(10, 0))
                if to_set:
                    button_to.pack_forget()
                    button_to.pack(fill="x", padx=(0, 0), pady=(10, 0))
                    to_frame.pack_forget()
                    to_frame.pack(pady=(10, 0), fill="x")
                if build_seg:
                    build_seg = False
                    button_seg.configure(text=" ▼ Building Segmentation")
                    seg_frame.pack_forget()
                    button_from.pack_forget()
                    button_from.pack(fill="x", padx=(0, 0), pady=(10, 0))
                    from_frame.pack_forget()
                    from_frame.pack(pady=(5, 0), fill="x")
                    button_to.pack_forget()
                    button_to.pack(fill="x", padx=(0, 0), pady=(10, 0))
            else:
                button_from.configure(text=" ▼ FROM")
                from_frame.pack_forget()

            if to_set:
                button_to.pack_forget()
                button_to.pack(fill="x", padx=(0, 0), pady=(10, 0))
                to_frame.pack_forget()
                to_frame.pack(pady=(10, 0), fill="x")

        # FROM Button
        button_from = CTkButton(panel_canvas, text=" ▼ FROM", text_color="#F0F0F0", fg_color="#666666",
                                           anchor="w", corner_radius=0, command=toggle_from)
        button_from.pack(fill="x", padx=(0, 0), pady=(10, 0))

        # Frame for FROM input fields
        from_frame = CTkFrame(panel_canvas, fg_color="#2E2E2E", corner_radius=0)

        dronPos_label = CTkLabel(from_frame, text="Dron position:", text_color="white", font=("Arial", 12))
        dronPos_label.pack(padx=(5, 5), pady=(0, 0))

        panel_dronPos = CTkFrame(master=from_frame, width=300, height=200, fg_color="white", corner_radius=10)
        panel_dronPos.pack(padx=(10, 10), pady=(10, 0))

        to_set = False  # Flag for TO section

        # Function to toggle the TO section
        def toggle_to():
            global to_set, build_seg
            to_set = not to_set
            if to_set:
                button_to.configure(text=" ▲ TO")
                to_frame.pack(pady=(10, 0), fill="x")
                if build_seg:
                    build_seg = False
                    button_seg.configure(text=" ▼ Building Segmentation")
                    seg_frame.pack_forget()
                    button_from.pack_forget()
                    button_from.pack(fill="x", padx=(0, 0), pady=(10, 0))
                    button_to.pack_forget()
                    button_to.pack(fill="x", padx=(0, 0), pady=(10, 0))
                    to_frame.pack_forget()
                    to_frame.pack(pady=(10, 0), fill="x")
            else:
                button_to.configure(text=" ▼ TO")
                to_frame.pack_forget()

        # TO Button
        button_to = CTkButton(panel_canvas, text=" ▼ TO", text_color="#F0F0F0", fg_color="#666666", anchor="w", corner_radius=0, command=toggle_to)
        button_to.pack(fill="x", padx=(0, 0), pady=(10, 0))

        # Frame for TO input fields
        to_frame = CTkFrame(panel_canvas, fg_color="#2E2E2E", height=140, corner_radius=0)
        to_frame.grid_propagate(False)

        show_latlon = False
        show_utm = False

        # Flags to indicate current coordinate mode
        latlon_set = 0  # 1 if Lat/Lon is selected
        utm_set = 0     # 1 if UTM is selected

        # Function to toggle display of geographic coordinates input
        def toggle_latlon():
            global latlon_set, utm_set, show_latlon, show_utm
            show_latlon = not show_latlon

            if show_latlon:
                # Activate Lat/Lon mode
                latlon_set = 1
                button_coordLatLng.configure(text="▲ Geographic coordinates")
                label_hint_left.pack_forget()
                label_hint_right.pack_forget()
                zone_frame.pack(pady=(5, 0), anchor='center')
                lat_frame.pack(pady=(5, 0), anchor='center')
                lon_frame.pack(pady=(5, 0), anchor='center')

                # If UTM is active, deactivate it
                if show_utm:
                    utm_set = 0
                    show_utm = False
                    button_coordUTM.configure(text="▼ UTM coordinates")
                    easting_frame.pack_forget()
                    northing_frame.pack_forget()

            else:
                # Collapse Lat/Lon mode
                latlon_set = 0
                button_coordLatLng.configure(text="▼ Geographic coordinates")
                zone_frame.pack_forget()
                lat_frame.pack_forget()
                lon_frame.pack_forget()
                label_hint_left.pack(pady=(10, 10))
                label_hint_right.pack(pady=(10, 10))

        # Function to toggle display of UTM coordinates input
        def toggle_utm():
            global latlon_set, utm_set, show_latlon, show_utm
            show_utm = not show_utm

            if show_utm:
                # Activate UTM mode
                utm_set = 1
                button_coordUTM.configure(text="▲ UTM coordinates")
                label_hint_right.pack_forget()
                label_hint_left.pack_forget()
                easting_frame.pack(pady=(5, 0), anchor='center')
                northing_frame.pack(pady=(5, 0), anchor='center')

                # If Lat/Lon is active, deactivate it
                if show_latlon:
                    latlon_set = 0
                    show_latlon = False
                    button_coordLatLng.configure(text="▼ Geographic coordinates")
                    zone_frame.pack_forget()
                    lat_frame.pack_forget()
                    lon_frame.pack_forget()

            else:
                # Collapse UTM mode
                utm_set = 0
                button_coordUTM.configure(text="▼ UTM coordinates")
                easting_frame.pack_forget()
                northing_frame.pack_forget()
                label_hint_right.pack(pady=(10, 10))
                label_hint_left.pack(pady=(10, 10))

        # Layout for coordinate input section
        left_to_frame = CTkFrame(to_frame, fg_color="#999999", corner_radius=10)
        left_to_frame.grid(row=0, column=0, padx=(10, 5), pady=(10, 0))
        left_to_frame.grid_propagate(False)

        right_to_frame = CTkFrame(to_frame, fg_color="#999999", corner_radius=10)
        right_to_frame.grid(row=0, column=1, padx=(5, 10), pady=(10, 0))
        right_to_frame.grid_propagate(False)

        to_frame.grid_rowconfigure(0, weight=1)
        to_frame.grid_columnconfigure(0, weight=1)
        to_frame.grid_columnconfigure(1, weight=1)

        # Container for geographic coordinate inputs
        left_to_frame_container = CTkFrame(left_to_frame, fg_color="#999999", corner_radius=10, height=140)
        left_to_frame_container.pack(padx=(0, 0), pady=(0, 0))
        left_to_frame_container.pack_propagate(False)

        # Button to expand/collapse geographic coordinates input
        button_coordLatLng = CTkButton(left_to_frame_container, text="▼ Geographic coordinates", text_color="#F0F0F0", fg_color="#3E3E3E", corner_radius=0, height=10, command=toggle_latlon)
        button_coordLatLng.pack(fill='x', padx=0, pady=0)

        # Hint label before selecting coordinate type
        label_hint_left = CTkLabel(
            left_to_frame_container,
            text="Select geographic or UTM coordinates",
            text_color="#F0F0F0",
            font=("Arial", 12),
            bg_color="#999999",
            wraplength=180,
            justify="center"
        )
        label_hint_left.pack(padx=(5, 5), pady=(10, 10), fill="x")

        # Create a frame container for zone input
        zone_frame = CTkFrame(left_to_frame_container, fg_color="#999999", corner_radius=0)

        # Label for the UTM Zone input
        label_zone = CTkLabel(zone_frame, text="Zone: ", text_color="#F0F0F0", bg_color="#999999")
        label_zone.pack(side='left', padx=(0, 5))

        # Entry field to input the UTM Zone
        entry_zone = CTkEntry(zone_frame, width=50, text_color="black", state="normal", font=("Arial", 12))
        entry_zone.pack(side='left')

        # Frame container for Latitude input
        lat_frame = CTkFrame(left_to_frame_container, fg_color="#999999", corner_radius=0)

        # Label for Latitude
        label_lat = CTkLabel(lat_frame, text="Lat: ", text_color="#F0F0F0", bg_color="#999999")
        label_lat.pack(side='left', padx=(0, 5))

        # Entry field for Latitude input
        entry_lat = CTkEntry(lat_frame, width=50, text_color="black", state="normal", font=("Arial", 12))
        entry_lat.pack(side='left')

        # Frame container for Longitude input
        lon_frame = CTkFrame(left_to_frame_container, fg_color="#999999", corner_radius=0)

        # Label for Longitude
        label_lon = CTkLabel(lon_frame, text="Lon: ", text_color="#F0F0F0", bg_color="#999999")
        label_lon.pack(side='left', padx=(0, 5))

        # Entry field for Longitude input
        entry_lon = CTkEntry(lon_frame, width=50, text_color="black", state="normal", font=("Arial", 12))
        entry_lon.pack(side='left')

        # Create a frame container for UTM coordinate inputs
        right_to_frame_container = CTkFrame(right_to_frame, fg_color="#999999", corner_radius=10, height=140)
        right_to_frame_container.pack(padx=(0, 0), pady=(0, 0))
        right_to_frame_container.pack_propagate(False)

        # Button to toggle the display of UTM coordinates input fields
        button_coordUTM = CTkButton(right_to_frame_container, text="▼ UTM coordinates", text_color="#F0F0F0",
                                    fg_color="#3E3E3E",
                                    corner_radius=0, height=10, command=toggle_utm)
        button_coordUTM.pack(fill='x', padx=0, pady=0)

        # Hint label before selecting coordinate type
        label_hint_right = CTkLabel(
            right_to_frame_container,
            text="Select geographic or UTM coordinates",
            text_color="#F0F0F0",
            font=("Arial", 12),
            bg_color="#999999",
            wraplength=180,
            justify="center"
        )
        label_hint_right.pack(padx=(5, 5), pady=(10, 10), fill="x")

        # Frame container for Easting coordinate input
        easting_frame = CTkFrame(right_to_frame_container, fg_color="#999999", corner_radius=0)

        # Label for Easting
        label_easting = CTkLabel(easting_frame, text="Easting: ", text_color="#F0F0F0", bg_color="#999999")
        label_easting.pack(side='left', padx=(0, 5))

        # Entry field for Easting input
        entry_easting = CTkEntry(easting_frame, width=50, text_color="black", state="normal", font=("Arial", 12))
        entry_easting.pack(side='left')

        # Frame container for Northing coordinate input
        northing_frame = CTkFrame(right_to_frame_container, fg_color="#999999", corner_radius=0)

        # Label for Northing
        label_northing = CTkLabel(northing_frame, text="Northing: ", text_color="#F0F0F0", bg_color="#999999")
        label_northing.pack(side='left', padx=(0, 5))

        # Entry field for Northing input
        entry_northing = CTkEntry(northing_frame, width=50, text_color="black", state="normal", font=("Arial", 12))
        entry_northing.pack(side='left')

        # Button to return
        button_return = CTkButton(left_frame, text="Return", text_color="#F0F0F0", fg_color="#B71C1C", hover_color="#C62828", corner_radius=0, border_color="#D3D3D3", border_width=2, command=btn_return)
        button_return.pack(side="bottom", padx=(0, 0), pady=(5, 0))

        # Button to trigger visualization
        button_visualize = CTkButton(left_frame, text="Visualize", text_color="#F0F0F0", fg_color="#1E3A5F",
                                     hover_color="#2E4A7F", corner_radius=0, border_color="#D3D3D3", border_width=2,
                                     command=lambda: tree_obstacles(las_object, entry_lat, entry_lon, entry_zone, entry_easting,
                                                                    entry_northing, latlon_set, utm_set,
                                                                    combined_mesh, pixels_x, pixels_y, delta_x,
                                                                    delta_y, cell_stats, selected_rect_indices, building_prism_cells))
        button_visualize.pack(side="bottom", padx=(0, 0), pady=(5, 0))

        # Force update of layout
        left_frame.update_idletasks()

        # Prepare coordinate arrays and calculate min/max and geometric centers for scaling points
        x_array = np.array(xcenter)
        y_array = np.array(ycenter)
        x_min, x_max = x_array.min(), x_array.max()
        y_min, y_max = y_array.min(), y_array.max()
        x_center_geom = (x_min + x_max) / 2
        y_center_geom = (y_min + y_max) / 2

        # Function to scale a value from one range to another (linear scaling)
        def escalar(val, min_val, max_val, new_min, new_max):
            return new_min + (val - min_val) / (max_val - min_val) * (new_max - new_min)

        # Loop through each drone position to create buttons representing each drone location on the panel
        for i in range(len(xcenter)):
            # Perform some rotation/reflection transformations on coordinates
            x_rotado = 2 * x_center_geom - xcenter[i]
            y_rotado = 2 * y_center_geom - ycenter[i]
            x_rotado = 2 * x_center_geom - x_rotado

            # Scale rotated coordinates to fit within the panel's coordinate system
            x = escalar(x_rotado, x_min, x_max, 10, 290)
            y = escalar(y_rotado, y_min, y_max, 10, 190)

            # Save the original positions with altitude
            posiciones.append((xcenter[i], ycenter[i], FAltcenter[i]))

            # Button to visually represent the drone position on the panel
            btn = CTkButton(panel_dronPos, text="", width=6, height=6, fg_color="blue", hover_color="darkblue", corner_radius=3, command=lambda b=i: toggle_color(botones[b], b))
            btn.place(x=x, y=y, anchor="center")
            botones.append(btn)


# Function to return to the main menu
def btn_return():
    global root
    if 'panel_canvas' in globals() and panel_canvas.winfo_exists():
        panel_canvas.place_forget()

    if 'button_seg' in globals() and button_seg.winfo_exists():
        button_seg.pack_forget()

    if 'seg_frame' in globals() and seg_frame.winfo_exists():
        seg_frame.pack_forget()

    if 'button_from' in globals() and button_from.winfo_exists():
        button_from.pack_forget()

    if 'from_frame' in globals() and from_frame.winfo_exists():
        from_frame.pack_forget()

    if 'button_to' in globals() and button_to.winfo_exists():
        button_to.pack_forget()

    if 'to_frame' in globals() and to_frame.winfo_exists():
        to_frame.pack_forget()

    if 'button_return' in globals() and button_return.winfo_exists():
        button_return.pack_forget()

    if 'button_visualize' in globals() and button_visualize.winfo_exists():
        button_visualize.pack_forget()

    root.num_pixeles_x_entry.configure(state="normal")
    root.num_pixeles_y_entry.configure(state="normal")


# Function to toggle the color of the button and manage selection
def toggle_color(boton, index):
    global selected_positions, posiciones, botones

    x, y, alt = posiciones[index]  # Retrieve the coordinates and altitude of the button's corresponding drone position

    # Check if the button is currently unselected (blue)
    if boton.cget("fg_color") == "blue":
        # If no positions are currently selected, select this one
        if not selected_positions:
            # Change button color to indicate selection
            boton.configure(fg_color="pink", hover_color="#ff69b4")
            # Add this position to the list of selected positions
            selected_positions.append((x, y, alt))

    else:
        # If button is already selected (pink), deselect it
        boton.configure(fg_color="blue", hover_color="darkblue")
        # Remove this position from the list of selected positions
        selected_positions = [pos for pos in selected_positions if pos != (x, y, alt)]


# Function to convert latitude and longitude to UTM coordinates
def latlon_to_utm(lat, lon, zone):
    # Define the source coordinate reference system as WGS84 (EPSG:4326), which uses lat/lon
    wgs84_crs = CRS.from_epsg(4326)
    # Define the target coordinate reference system as UTM for the specified zone
    utm_crs = CRS.from_proj4(f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs")
    # Convert coordinates from WGS84 to UTM
    transformer = Transformer.from_crs(wgs84_crs, utm_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y


# Function to display a summary of the lines detected
def summary_lines(colores_puntos, total_arboles_cruzados, total_edificios_cruzados):

    # Inner function to create and display the GUI window
    def ventana():
        # Create a new top-level window for the summary
        resumen = CTkToplevel()
        resumen.title("Obstacle Detection")
        resumen.geometry("300x400")
        resumen.configure(fg_color="#1E1E1E")  # Set dark background

        # Scrollable frame to contain the summary items
        scroll = CTkScrollableFrame(resumen, fg_color="#1E1E1E")
        scroll.pack(expand=True, fill="both", padx=10, pady=10)

        # Iterate through each line's color and obstacle counts
        for idx, (color, num_arboles, num_edificios) in enumerate(zip(colores_puntos, total_arboles_cruzados, total_edificios_cruzados)):
            # Frame for each summary item
            item_frame = CTkFrame(scroll, fg_color="#1E1E1E")
            item_frame.pack(anchor="w", padx=10, pady=5)

            # Convert RGB tuple (0–1 range) to hex color string
            hex_color = "#{:02x}{:02x}{:02x}".format(int(color[0]*255), int(color[1]*255), int(color[2]*255))

            # Create a small colored circle representing the line color
            circle = CTkCanvas(item_frame, width=20, height=20, bg="#1E1E1E", highlightthickness=0)
            circle.create_oval(2, 2, 18, 18, fill=hex_color, outline=hex_color)
            circle.pack(side="left")

            # Label showing the number of trees and buildings for the line
            label = CTkLabel(
                item_frame,
                text=f"{num_arboles} trees, {num_edificios} buildings",
                text_color="#F0F0F0",
                font=("Arial", 12)
            )
            label.pack(side="left", padx=8)

    # Run in a separate thread
    threading.Thread(target=ventana, daemon=True).start()


# Function to create a progress bar
def create_progress_bar():
    progress_bar = ttk.Progressbar(right_frame, orient="horizontal", length=300, mode="determinate", style="TProgressbar")
    right_frame.grid_rowconfigure(0, weight=1)
    right_frame.grid_rowconfigure(1, weight=1)
    right_frame.grid_rowconfigure(2, weight=1)
    right_frame.grid_columnconfigure(0, weight=1)
    progress_bar.grid(row=1, column=0, pady=10)
    return progress_bar


# Function to update the progress bar
def update_progress_bar(progress_bar, value):
    progress_bar['maximum'] = 100
    progress_bar['value'] = value
    root.update_idletasks()


# Function to load a point cloud file
def load_point_cloud():
    global pc_filepath
    root.point_size_entry.delete(0, "end")
    filepath = filedialog.askopenfilename(filetypes=[("PCD Files", "*.pcd")])
    if filepath:
        pc_filepath = filepath
        root.point_size_entry.configure(state="normal")
        root.point_size_entry.insert(0, 2)
        root.voxelizer_switch.configure(state="normal")


# Function to load an XML metadata file
def load_xml_metadata():
    global xml_filepath
    xml_filepath = filedialog.askopenfilename(filetypes=[("XML Files", "*.xml")])


# Function to process .n42 files
def process_n42_files():
    global dose_min_csv, dose_max_csv, csv_filepath, heatmap, xcenter, ycenter, FAltcenter, Hcenter, lonmin, lonmax, latmin, latmax
    fp = filedialog.askdirectory(title="Select Folder with .n42 Files")

    if not fp:
        return

    pathN42 = fp
    pathN42mod = os.path.join(fp)

    # Internal_background. This should be determined in a ultra low bagcgroundo facility like the UDOII facility at PTB
    Dose_int = 0.0
    H10_int = 0.0

    # Cosmic dose measured by the instrument. This should be determined in a lake platform or in the sea after the internal background has already beenn determined
    # current values are obtained in Banyoles lake
    Dose_cosmic = 0.0
    H10_cosmic = 0.0

    # Influence of radon progeny. This values are really complicated. In outdoors, an estimation of 2 nSv/h per 10 Bq/m3 could be used (Vargas, Cornejo and Camp. 2017)
    Dose_Radon = 0.0
    H10_Radon = 0.0

    # low_ROI_counts/high_ROI_counts when no artifical source is present
    R = 13.5  # SiPM 50 mm

    # Define lists with none values with a maximum of 4096 bin in each spectrum
    En_ch = [None] * 4096
    Conv_coeff = [None] * 4096
    F = [None] * 4096

    xcenter = [None] * 100000
    ycenter = [None] * 100000
    Hcenter = [None] * 100000
    FAltcenter = [None] * 100000

    sys.path.insert(0, pathN42)
    os.chdir(pathN42)
    listOfFiles = os.listdir(pathN42)
    f_name = fnmatch.filter(listOfFiles, '*.n42')

    # Loop for each *.n42 spectrum
    cont = 0
    for idx, file in enumerate(f_name):
        cont = cont + 1
        os.chdir(pathN42)
        f = open(file, "r")
        tree = ET.parse(file)
        roots = tree.getroot()

        # Read Start Date Time, LiveTime, DeadTime, ChannelData
        for each in roots.findall('.//RadMeasurement'):
            # Read LiveTime
            rating = each.find('.//LiveTimeDuration')
            LiveTime = rating.text

            # Find substring and convert to float
            LTime = LiveTime[LiveTime.find("T") + 1:LiveTime.find("S")]
            FLTime = float(LTime)
            LTime = FLTime

            # Read counts in each energy bin
            rating = each.find('.//ChannelData')
            ChannelData = rating.text

            # Convert string of counts in a list of integers
            Split_channels = ChannelData.split()

            icounts = list(map(float, Split_channels))

            # The channel index starts with 0 up to n_channels-1
            n_channels = len(icounts)

        # Read Energy calibration
        for each in roots.findall('.//EnergyCalibration'):
            rating = each.find('.//CoefficientValues')
            Ecal = rating.text

            # Convert string of counts in a list of integers
            Split_coeff = Ecal.split()
            float_coeff = list(map(float, Split_coeff))

        # Read altitude a.g.l.
        for each in roots.findall('.//GeographicPoint'):
            rating = each.find('.//ElevationValue')
            Altitude = rating.text
            FAltitude = float(Altitude)

            rating = each.find('.//LongitudeValue')
            Longitude = rating.text
            FLongitude = float(Longitude)

            rating = each.find('.//LatitudeValue')
            Latitude = rating.text
            FLatitude = float(Latitude)

        # Calculation of absorbed Dose and H10 using band method function
        Dose_conv_meas = 0
        H10_conv_meas = 0
        low_ROI = 0
        high_ROI = 0
        for i in range(0, n_channels):
            En_ch[i] = float_coeff[0] + float_coeff[1] * (i + 1)

            # Calculate Man Made Gross Count MMGC
            if ((En_ch[i] > 200) and (En_ch[i] <= 1340)):
                low_ROI = low_ROI + int(icounts[i])

            if ((En_ch[i] > 1340) and (En_ch[i] <= 2980)):
                high_ROI = high_ROI + int(icounts[i])

            # Calculate de conversion coefficent for the energy nGy/h per cps WITH capsule and total surface 30 keV   50 mmm
            Conv_coeff[i] = 0
            if ((En_ch[i] >= 30) and (En_ch[i] <= 55)):
                Conv_coeff[i] = 5206.355632 * En_ch[i] ** (-2.853969336)

            if ((En_ch[i] > 55) and (En_ch[i] <= 350)):
                Conv_coeff[i] = -4.43804048E-13 * En_ch[i] ** 5 + 4.852144251E-10 * En_ch[
                    i] ** 4 - 1.997841663E-07 * En_ch[i] ** 3 + 4.123346655E-05 * En_ch[i] ** 2 - 0.0035372218034 * \
                                En_ch[i] + 0.1532763827
            if ((En_ch[i] > 350) and (En_ch[i] <= 3000)):
                Conv_coeff[i] = -4.434768244E-07 * En_ch[i] ** 2 + 0.003033563589 * En_ch[i] - 0.6528052087

            # Calculate the conversion coefcient Gy --> Sv
            F[i] = 1
            if (En_ch[i] >= 3):
                a1 = math.log(En_ch[i] / 9.85)
                F[i] = a1 / (1.465 * a1 ** 2 - 4.414 * a1 + 4.789) + 0.7003 * math.atan(0.6519 * a1)

            Dose_conv_meas = Dose_conv_meas + icounts[i] / FLTime * Conv_coeff[i]
            H10_conv_meas = H10_conv_meas + (icounts[i] / FLTime * Conv_coeff[i]) * F[i]

        # H*(10) conversion factor at reference altitude 1 m
        # using the attenuation coefficient from H*(10) at pla del vent: 0.007
        Dose_conv_meas = Dose_conv_meas - Dose_int - Dose_cosmic - Dose_Radon
        H10_conv_meas = H10_conv_meas - H10_int - H10_cosmic - H10_Radon
        H10_conv_1m = H10_conv_meas * math.exp(0.008 * (FAltitude - 1))
        MMGC = (float(low_ROI) - R * float(high_ROI)) / float(LTime)
        u_MMGC = ((float(low_ROI)) / (float(LTime)) ** 2 + ((float(R) / float(LTime)) ** 2) * (float(high_ROI)) + (
                (float(high_ROI) / float(LTime)) ** 2) * (0.05 * float(R)) ** 2) ** 0.5
        Dose_conv_meas = "%.2f" % Dose_conv_meas
        H10_conv_meas = "%.2f" % H10_conv_meas
        H10_conv_1m = "%.2f" % H10_conv_1m
        MMGC = "%.2f" % MMGC
        u_MMGC = "%.2f" % u_MMGC

        for dose in roots.iter('DoseRateValue'):
            dose.text = str(Dose_conv_meas)

        for Ader in roots.iter('AmbientDoseEquivalentRateValue'):
            Ader.text = str(H10_conv_meas)

        for Ader1m in roots.iter('AmbientDoseEquivalentRateValue_1m'):
            Ader1m.text = str(H10_conv_1m)

        for Man_Made in roots.iter('MMGC'):
            Man_Made.text = str(MMGC)

        for uMan_Made in roots.iter('uncertainty_MMGC'):
            uMan_Made.text = str(u_MMGC)

        # Tranform x,y
        x0, y0, zone_number, zone_letter = utm.from_latlon(FLatitude, FLongitude, )

        xcenter[cont] = x0
        ycenter[cont] = y0
        Hcenter[cont] = H10_conv_meas
        FAltcenter[cont] = FAltitude

        if cont == 1:
            latmin = y0
            latmax = y0
            lonmin = x0
            lonmax = x0

        if x0 < lonmin:
            lonmin = x0
        if x0 > lonmax:
            lonmax = x0
        if y0 < latmin:
            latmin = y0
        if y0 > latmax:
            latmax = y0

        os.chdir(pathN42mod)
        tree.write(file)

    lonmin = lonmin - 50
    lonmax = lonmax + 50
    latmin = latmin - 50
    latmax = latmax + 50

    xcenter = np.array(xcenter, dtype=float)
    ycenter = np.array(ycenter, dtype=float)
    Hcenter = np.array(Hcenter, dtype=float)

    # conversion to string and numbers to floats
    xcenter = np.array([float(i) for i in xcenter if str(i).replace('.', '', 1).isdigit()])
    ycenter = np.array([float(i) for i in ycenter if str(i).replace('.', '', 1).isdigit()])
    Hcenter = np.array([float(i) for i in Hcenter if str(i).replace('.', '', 1).isdigit()])
    FAltcenter = np.array([float(i) for i in FAltcenter if str(i).replace('.', '', 1).isdigit()])

    # Define a grid for the area of interest
    Resolution = 50
    ygrid = np.linspace(latmin, latmax, Resolution)
    xgrid = np.linspace(lonmin, lonmax, Resolution)
    xmesh, ymesh = np.meshgrid(xgrid, ygrid)

    # Initialize the map with very low values
    heatmap = np.full(xmesh.shape, -np.inf)

    # Iterate over each circle
    for xc, yc, radius, hval in zip(xcenter, ycenter, FAltcenter, Hcenter):
        # Distance from each grid point to the center of the circle
        distance = np.sqrt((xmesh - xc) ** 2 + (ymesh - yc) ** 2)
        # Mask to identify points inside the circle
        mask = distance <= radius
        # Update the maximum value on the map
        heatmap[mask] = np.maximum(heatmap[mask], hval)

    # Set minimum values to be visible (if necessary)
    heatmap[heatmap == -np.inf] = np.nan

    # Write to CSV
    output_filename = "dose_data_pla_20m_2ms.csv"
    csv_filepath = output_filename
    with open(output_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Latitude", "Longitude", "Dose"])

        # Flatten the arrays for iteration
        for i in range(xmesh.shape[0]):
            for j in range(xmesh.shape[1]):
                writer.writerow([xmesh[i, j], ymesh[i, j], heatmap[i, j]])

    # Process the existing CSV
    dosis_values = np.genfromtxt(csv_filepath, delimiter=',', skip_header=1, usecols=2)
    dosis_values = dosis_values[~np.isnan(dosis_values)]
    dose_min_csv, dose_max_csv = np.min(dosis_values), np.max(dosis_values)

    # Assign values to the Min and Max fields and disable them
    root.low_dose_min.configure(state="normal")
    root.low_dose_min.delete(0, "end")
    root.low_dose_min.insert(0, str(dose_min_csv))
    root.low_dose_min.configure(state="disabled")

    root.high_dose_max.configure(state="normal")
    root.high_dose_max.delete(0, "end")
    root.high_dose_max.insert(0, str(dose_max_csv))
    root.high_dose_max.configure(state="disabled")

    resta = dose_max_csv - dose_min_csv
    ranges = resta/3

    root.low_dose_max.configure(state="normal")
    root.low_dose_max.delete(0, "end")
    root.low_dose_max.insert(0, f"{ranges + dose_min_csv:.2f}")
    root.low_dose_max.configure(state="disabled")

    root.medium_dose_max.configure(state="normal")
    root.medium_dose_max.delete(0, "end")
    root.medium_dose_max.insert(0, f"{(2 * ranges) + dose_min_csv:.2f}")
    root.medium_dose_max.configure(state="disabled")

    root.dose_layer_switch.configure(state="normal")


# Function to assign colors based on dose values
def get_dose_color(dosis_nube, high_dose_rgb, medium_dose_rgb, low_dose_rgb, dose_min_csv, low_max, medium_min, medium_max, high_min):
    # Initialize an array of zeros with shape (number of doses, 3) for RGB colors
    colores_dosis = np.zeros((len(dosis_nube), 3))
    # Assign low dose color: values between dose_min_csv and low_max
    colores_dosis[(dosis_nube >= dose_min_csv) & (dosis_nube < low_max)] = low_dose_rgb
    # Assign medium dose color: values between medium_min and medium_max
    colores_dosis[(dosis_nube >= medium_min) & (dosis_nube < medium_max)] = medium_dose_rgb
    # Assign high dose color: values greater than or equal to high_min
    colores_dosis[dosis_nube >= high_min] = high_dose_rgb
    return colores_dosis


# Function to extract the origin from an XML file
def get_origin_from_xml(xml_filepath):
    try:
        # Parse the XML file
        tree = ET.parse(xml_filepath)
        roots = tree.getroot()

        # Look for the <SRSOrigin> tag in the XML structure
        srs_origin = roots.find("SRSOrigin")

        # If the tag doesn't exist or has no content, return None
        if srs_origin is None or not srs_origin.text:
            return None

        # Convert the comma-separated string to a NumPy array of floats
        return np.array([float(coord) for coord in srs_origin.text.split(",")])

    except Exception as e:
        messagebox.showerror("Error", f"Failed to read XML file: {e}")
        return None


# Function to toggle the voxel size input
def toggle_voxel_size():
    global previous_point_value, previous_voxel_value, previous_downsample_value

    # Check whether the voxelizer checkbox (BooleanVar) is enabled
    if root.voxelizer_var.get():
        # Save the current point size and downsample values before disabling them
        previous_point_value = root.point_size_entry.get()
        previous_downsample_value = root.downsample_entry.get()

        # Clear and disable point size and downsample fields
        root.vox_size_entry.delete(0, "end")
        root.point_size_entry.delete(0, "end")
        root.downsample_entry.delete(0, "end")
        root.point_size_entry.configure(state="disabled")

        # Enable and prepare voxel size entry
        root.downsample_entry.configure(state="disabled")
        root.vox_size_entry.configure(state="normal")

        # Restore previous voxel value if it exists, else insert default value of 2
        if previous_voxel_value == "":
            root.vox_size_entry.insert(0, 2)
        else:
            root.vox_size_entry.insert(0, previous_voxel_value)

    else:
        # Save current voxel value before disabling the field
        previous_voxel_value = root.vox_size_entry.get()


        # Clear and disable voxel size field
        root.vox_size_entry.delete(0, "end")
        root.point_size_entry.delete(0, "end")
        root.vox_size_entry.configure(state="disabled")

        # Enable and restore point size and downsample inputs
        root.point_size_entry.configure(state="normal")
        root.downsample_entry.configure(state="normal")
        if previous_point_value == "":
            root.point_size_entry.insert(0, 2)  # Default value
        else:
            root.point_size_entry.insert(0, previous_point_value)
        if previous_downsample_value != "":
            root.downsample_entry.insert(0, previous_downsample_value)


# Function to toggle the dose layer visibility
def toggle_dose_layer(source_location):
    global show_dose_layer

    # Check if the "dose layer" switch is turned on
    if root.dose_layer_switch.get() == 1:
        show_dose_layer = True
        root.low_dose_max.configure(state="normal")
        root.medium_dose_min.configure(state="normal")
        root.medium_dose_max.configure(state="normal")
        root.high_dose_min.configure(state="normal")
        root.low_dose_cb.configure(state="normal")
        root.medium_dose_cb.configure(state="normal")
        root.high_dose_cb.configure(state="normal")
        root.dosis_slider.configure(state="normal")

        # If no colors are set yet, initialize default dose colors
        if not root.low_dose_cb.get():
            root.low_dose_cb.set("green")
            root.high_dose_cb.set("red")
            root.medium_dose_cb.set("yellow")

        # If a source location is available, enable the toggle to show/hide it
        if source_location is not None:
            root.show_source_switch.configure(state="normal")

    else:
        # If dose layer is turned off, disable all related controls
        show_dose_layer = False
        root.low_dose_max.configure(state="disabled")
        root.medium_dose_min.configure(state="disabled")
        root.medium_dose_max.configure(state="disabled")
        root.high_dose_min.configure(state="disabled")
        root.low_dose_cb.configure(state="disabled")
        root.medium_dose_cb.configure(state="disabled")
        root.high_dose_cb.configure(state="disabled")


# Function to find the radioactive source using a genetic algorithm
def find_radioactive_source(csv_filepath):
    global source_location

    # Check if a file path was provided
    if not csv_filepath:
        messagebox.showwarning("Warning", "Please select a N42 file.")
        return

    # Load UTM coordinates from the CSV file, skipping the header
    utm_coords = np.genfromtxt(csv_filepath, delimiter=',', skip_header=1)

    # Remove any rows where the third column (likely the measurement) is NaN
    utm_coords = utm_coords[~np.isnan(utm_coords[:, 2])]

    # Create an instance of the GeneticAlgorithm class with the cleaned coordinates
    ga = GeneticAlgorithm(utm_coords)

    # Run the genetic algorithm
    source_location = ga.run()

    # Store the result in the global variable
    source_location = source_location

    # Show the estimated source location to the user
    messagebox.showinfo(
        "Source Location",
        f"Estimated source location: Easting = {source_location[0]:.2f} m, Northing = {source_location[1]:.2f} m"
    )

    root.show_source_switch.configure(state="normal")


# Function to toggle the source visibility
def toggle_source():
    global show_source

    # Check if the toggle switch is ON (value is 1)
    if root.show_source_switch.get() == 1:
        show_source = True  # Enable showing the radioactive source
    else:
        show_source = False  # Disable showing the radioactive source


# Function to validate dose ranges
def validate_dose_ranges(show_dose_layer, dose_min_csv, dose_max_csv):
    global low_max, medium_min, medium_max, high_min

    if not show_dose_layer:
        return

    try:
        # Convert input values from GUI fields to floats
        low_max = float(root.low_dose_max.get())
        medium_min = float(root.medium_dose_min.get())
        medium_max = float(root.medium_dose_max.get())
        high_min = float(root.high_dose_min.get())

    except ValueError:
        messagebox.showerror("Error", "Dose range values must be numeric.")
        raise ValueError("Dose range values must be numeric.")

    # Ensures the dose thresholds are ordered correctly and fall within CSV min/max bounds
    if not (dose_min_csv <= low_max <= medium_min <= medium_max <= high_min <= dose_max_csv):
        messagebox.showerror("Error","Dose ranges are not logical. Ensure: min < low_max < medium_min < medium_max < high_min < max.")
        raise ValueError("Dose ranges are not logical.")


# Function to plot the heatmap
def plot_heatmap(heatmap, xcenter, ycenter, Hcenter, lonmin, lonmax, latmin, latmax):

    if heatmap is None or xcenter is None or ycenter is None or Hcenter is None:
        messagebox.showerror("Error", "Please process the N42 files first.")
        return

    disable_left_frame()

    # Create a matplotlib figure and axis to draw the heatmap
    fig, ax = plt.subplots()

    # Display the heatmap using imshow with geographical extent and colormap
    cax = ax.imshow(
        heatmap,
        extent=(lonmin, lonmax, latmin, latmax),    # Geographic bounds for x (longitude) and y (latitude)
        origin='lower',                             # Ensure the origin is at the bottom-left
        cmap='viridis',                             # Color map for visualization
        alpha=0.8                                   # Transparency for better overlay on map (if applicable)
    )

    # Add a colorbar to the figure to indicate radiation intensity scale
    fig.colorbar(cax, label='H*(10) rate nSv/h', ax=ax)

    # Set title and axis labels
    ax.set_title('Heatmap H*(10) rate')
    ax.set_xlabel('LONGITUDE')
    ax.set_ylabel('LATITUDE')

    # Overlay scatter points of measurement locations colored by their dose values
    ax.scatter(
        xcenter, ycenter,
        c=Hcenter,
        edgecolor='black', s=50, label='Measurement'
    )

    # Add a grid
    ax.grid(visible=True, color='black', linestyle='--', linewidth=0.5)

    # Add a legend
    ax.legend()

    # Define a callback to re-enable the left frame once the plot window is closed
    def on_close(event):
        enable_left_frame()

    # Connect the close event of the figure window to the callback
    fig.canvas.mpl_connect('close_event', on_close)

    # Show the plot
    plt.show()


# Function to plot a three-color heatmap
def plot_three_color_heatmap(heatmap, xcenter, ycenter, Hcenter, lonmin, lonmax, latmin, latmax):

    if heatmap is None or xcenter is None or ycenter is None or Hcenter is None:
        messagebox.showerror("Error", "Please process the N42 files first.")
        return

    disable_left_frame()

    # Determine colors and dose thresholds based on user-selected GUI values if dose layer is enabled
    if root.dose_layer_switch.get() == 1:
        # Get dose colors from the GUI or use default if empty
        low_dose_color = root.low_dose_cb.get() if root.low_dose_cb.get() else 'green'
        medium_dose_color = root.medium_dose_cb.get() if root.medium_dose_cb.get() else 'yellow'
        high_dose_color = root.high_dose_cb.get() if root.high_dose_cb.get() else 'red'
        colors = [low_dose_color, medium_dose_color, high_dose_color]

        # Set dose thresholds with error handling, defaults to 80 and 120
        R0 = 0
        try:
            R1 = float(root.low_dose_max.get()) if root.low_dose_max.get() else 80
        except ValueError:
            R1 = 80
        try:
            R2 = float(root.medium_dose_max.get()) if root.medium_dose_max.get() else 120
        except ValueError:
            R2 = 120

    else:
        # Default colors and thresholds when dose layer is off
        colors = ['green', 'yellow', 'red']
        R0, R1, R2 = 0, 80, 120

    # Calculate maximum boundary (R3) based on squared max dose for color normalization
    R3 = max(Hcenter) * max(Hcenter)
    bounds = [R0, R1, R2, R3]

    # Create a colormap and normalization based on the bounds for discrete color ranges
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    # Create matplotlib figure and axis for plotting
    fig, ax = plt.subplots()

    # Plot heatmap using imshow with the custom colormap and normalization
    im = ax.imshow(
        heatmap,
        extent=(lonmin, lonmax, latmin, latmax),
        origin='lower',
        cmap=cmap,
        norm=norm,
        alpha=0.8
    )

    # Add a colorbar with ticks at meaningful dose values
    fig.colorbar(
        im,
        label='H*(10) rate (nSv/h)',
        boundaries=bounds,
        ticks=[
            R0, R0 + (R1 - R0) / 2, R1,
            R1 + (R2 - R1) / 2, R2,
            R2 + (R3 - R2) / 2, R3
        ]
    )

    # Set plot title and axis labels
    ax.set_title('Heatmap with Three Color Range')
    ax.set_xlabel('LONGITUDE')
    ax.set_ylabel('LATITUDE')

    # Overlay scatter points of measurement locations colored by their dose values
    ax.scatter(
        xcenter, ycenter,
        c=Hcenter, cmap=cmap, norm=norm,
        edgecolor='black', s=50, label='Measurement'
    )

    # Add a grid
    ax.grid(visible=True, color='black', linestyle='--', linewidth=0.5)

    # Add a legend
    ax.legend()

    # Define a callback to re-enable the left frame once the plot window is closed
    def on_close(event):
        enable_left_frame()

    # Connect the close event of the figure window to the callback
    fig.canvas.mpl_connect('close_event', on_close)

    # Show the plot
    plt.show()


# Function to set the flag
def set_flag(xcenter, ycenter, FAltcenter, pixels_x_str, pixels_y_str):
    global las_object, mi_set, pixels_x, pixels_y, combined_mesh_pixels_x, combined_mesh_pixels_y, las, building_prism_cells, combined_mesh

    if xcenter is None or len(xcenter) == 0 or ycenter is None or len(ycenter) == 0:
        messagebox.showerror("Error", "Please process the N42 files first.")
        return

    if not pixels_x_str.isdigit() or not pixels_y_str.isdigit():
        messagebox.showerror("Invalid Input", "Num Pixels X and Y must be integers.")
        return

    root.num_pixeles_x_entry.configure(state="disabled")
    root.num_pixeles_y_entry.configure(state="disabled")

    # Convert pixel counts to integers
    pixels_x = int(pixels_x_str)
    pixels_y = int(pixels_y_str)

    if las_object is None:
        mi_set = False
        tree_segmentation(mi_set, pixels_x, pixels_y)

        combined_mesh_pixels_x = pixels_x
        combined_mesh_pixels_y = pixels_y

    else:
        # If the combined mesh does not exist or the pixel grid size has changed
        if (combined_mesh is None or combined_mesh_pixels_x != pixels_x or combined_mesh_pixels_y != pixels_y):
            # Regenerate the grid mesh for the new pixel resolution
            grid(las_object, pixels_x, pixels_y)

            # Update combined mesh pixel tracking variables
            combined_mesh_pixels_x = pixels_x
            combined_mesh_pixels_y = pixels_y

        else:
            panel_left_frame(xcenter, ycenter, FAltcenter, las_object, pixels_x, pixels_y, building_prism_cells)


# Function to set the function Tree Segemntation
def set_trees():
    global mi_set, las
    mi_set = True
    tree_segmentation(mi_set, pixels_x, pixels_y)


# Function to visualize the point cloud and dose layer
def visualize(pc_filepath, csv_filepath, xml_filepath, show_dose_layer, dose_min_csv, dose_max_csv):
    global altura_extra, point_size, vox_size, high_dose_rgb, medium_dose_rgb, low_dose_rgb, downsample, progress_bar

    if not pc_filepath:
        messagebox.showwarning("Warning", "Please select a Point Cloud.")
        return

    if root.dose_layer_switch.get() == 1 and not csv_filepath:
        messagebox.showerror("Error", "Please select a N42 file.")
        return

    if root.dose_layer_switch.get() == 1 and not xml_filepath:
        messagebox.showerror("Error", "Please select an XML.")
        return

    validate_dose_ranges(show_dose_layer, dose_min_csv, dose_max_csv)

    disable_left_frame()
    progress_bar = create_progress_bar()
    update_progress_bar(progress_bar, 1)

    use_voxelization = root.voxelizer_switch.get() == 1

    # Retrieve sizes from GUI
    point_size_str = root.point_size_entry.get().strip()
    vox_size_str = root.vox_size_entry.get().strip()
    altura_extra = root.dosis_slider.get()

    # Insert default values if fields are empty
    if use_voxelization:
        if vox_size_str == "":
            root.vox_size_entry.insert(0, 2)
    else:
        if point_size_str == "":
            root.point_size_entry.insert(0, 2)

    if point_size_str == "":
        point_size = 2
    else:
        point_size = float(point_size_str)
        if point_size <= 0:
            raise ValueError("Point size must be positive.")

    if vox_size_str == "":
        vox_size = 2
    else:
        vox_size = float(vox_size_str)
        if vox_size <= 0:
            raise ValueError("Voxel size must be positive.")

    if show_dose_layer:
        # Get RGB colors for dose layers
        high_dose_color = root.high_dose_cb.get()
        medium_dose_color = root.medium_dose_cb.get()
        low_dose_color = root.low_dose_cb.get()
        high_dose_rgb = np.array(mcolors.to_rgb(high_dose_color))
        medium_dose_rgb = np.array(mcolors.to_rgb(medium_dose_color))
        low_dose_rgb = np.array(mcolors.to_rgb(low_dose_color))
    else:
        high_dose_rgb = None
        medium_dose_rgb = None
        low_dose_rgb = None

    # Get downsample value if provided
    if root.downsample_entry.get().strip():
        downsample = float(root.downsample_entry.get().strip())
    else:
        downsample = None

    update_progress_bar(progress_bar, 10)

    # Call corresponding function depending on voxelization flag
    if use_voxelization:
        point_cloud_vox(show_dose_layer, pc_filepath, xml_filepath, csv_filepath, high_dose_rgb, medium_dose_rgb, low_dose_rgb, dose_min_csv, low_max,
            medium_min, medium_max, high_min, altura_extra, progress_bar)
    else:
        point_cloud_no_vox(show_dose_layer, pc_filepath, downsample, xml_filepath, csv_filepath, high_dose_rgb, medium_dose_rgb,
            low_dose_rgb, dose_min_csv, low_max, medium_min, medium_max, high_min, altura_extra, show_source, source_location, point_size, progress_bar)


# Segmentation function
def segmentation():

    def run():
        # Open a file dialog to select a LAS file
        fp = filedialog.askopenfilename(filetypes=[("LAS Files", "*.las")])
        if fp:
            disable_left_frame()

            progress_bar = create_progress_bar()
            update_progress_bar(progress_bar, 1)

            las = laspy.read(fp)

            # Extract point coordinates and classifications into numpy arrays
            points = np.vstack((las.x, las.y, las.z)).transpose()
            classifications = np.array(las.classification)

            update_progress_bar(progress_bar, 10)

            # Count the occurrences of each classification
            counts = dict(Counter(classifications))

            update_progress_bar(progress_bar, 20)

            # Load classification colors from a JSON file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(script_dir, "classification_colors_s.json")

            with open(json_path, "r") as f:
                color_map = json.load(f)["classifications"]

            # Convert color values from JSON to numpy arrays
            color_map = {int(k): np.array(v) for k, v in color_map.items()}

            # For any classifications missing from color_map, assign a random color
            unique_classes = np.unique(classifications)
            for cls in unique_classes:
                if cls not in color_map:
                    color_map[cls] = np.random.rand(3)

            update_progress_bar(progress_bar, 40)

            # Create an Open3D PointCloud object and set points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            update_progress_bar(progress_bar, 50)

            colors = np.zeros((points.shape[0], 3))

            update_progress_bar(progress_bar, 60)

            # Assign colors to each point based on classification
            for classification, color in color_map.items():
                colors[classifications == classification] = color

            update_progress_bar(progress_bar, 80)

            pcd.colors = o3d.utility.Vector3dVector(colors)

            update_progress_bar(progress_bar, 100)
            progress_bar.grid_forget()

            # Display a legend in the left frame showing counts and colors of classifications
            legend_left_frame(counts, color_map)

            # Initialize Open3D visualizer
            vis = o3d.visualization.Visualizer()

            # Get dimensions of GUI layout
            right_frame.update_idletasks()
            right_frame_width = right_frame.winfo_width()
            right_frame_height = right_frame.winfo_height()
            left_frame.update_idletasks()
            left_frame_width = left_frame.winfo_width()
            title_bar_height = ctypes.windll.user32.GetSystemMetrics(4)

            # Create Open3D window inside the GUI
            vis.create_window(window_name='Open3D', width=right_frame_width, height=right_frame_height,
                              left=left_frame_width, top=title_bar_height)
            vis.clear_geometries()
            vis.add_geometry(pcd)

            while True:
                vis.poll_events()
                vis.update_renderer()

                if not vis.poll_events():  # If visualizer is closed
                    if 'legend_frame' in globals() and legend_frame.winfo_exists():
                        legend_frame.place_forget()

                    if 'legend_canvas' in globals() and legend_canvas.winfo_exists():
                        legend_canvas.place_forget()

                    enable_left_frame()
                    break

            if not pcd.has_points():
                return

    # Run in a separate thread
    threading.Thread(target=run, daemon=True).start()


# Tree segmentation function
def tree_segmentation(mi_set, pixels_x, pixels_y):
    def run():
        global las_object, progress_bar, classificationtree, labels_clean, chm, rows, cols, pcd, xmin, ymax, resolution, las

        if mi_set==True:
            if las is None:
                fp = filedialog.askopenfilename(filetypes=[("LAS Files", "*.las")])
                if not fp:
                    return
                else:
                    las = fp
            else:
                fp = las

        if mi_set==False:
            fp = las

        if fp is None:
            return

        # If no LAS object loaded yet
        if las_object is None:
            disable_left_frame()

            progress_bar = create_progress_bar()
            update_progress_bar(progress_bar, 1)

            las_r = laspy.read(fp)

            # Extract points and classifications
            points = np.vstack((las_r.x, las_r.y, las_r.z)).transpose()
            classifications = np.array(las_r.classification)

            update_progress_bar(progress_bar, 5)

            # Load classification colors from JSON
            script_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(script_dir, "classification_colors_sp.json")
            with open(json_path, "r") as f:
                color_map = json.load(f)["classifications"]
            color_map = {int(k): np.array(v) for k, v in color_map.items()}

            # Assign random gray color for missing classes
            unique_classes = np.unique(classifications)
            for cls in unique_classes:
                if cls not in color_map:
                    gray = np.random.uniform(0.3, 0.8)
                    color_map[cls] = [gray, gray, gray]

            update_progress_bar(progress_bar, 10)

            # Create Open3D point cloud and assign colors based on classification
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            colors = np.zeros((points.shape[0], 3))
            for classification, color in color_map.items():
                colors[classifications == classification] = color

            update_progress_bar(progress_bar, 15)

            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Filter points classified as medium vegetation (class 4)
            medium_veg_points = points[classifications == 4]
            if medium_veg_points.shape[0] == 0:
                return  # Exit if no medium vegetation points

            update_progress_bar(progress_bar, 20)

            # Filter medium vegetation points above a certain height threshold (2.3 m above min)
            min_medium_veg_height = np.min(medium_veg_points[:, 2])
            medium_veg_points = medium_veg_points[medium_veg_points[:, 2] >= min_medium_veg_height + 2.3]

            update_progress_bar(progress_bar, 25)

            # Define grid resolution and bounding box for canopy height model (CHM)
            resolution = 0.55
            xmin, ymin = medium_veg_points[:, 0].min(), medium_veg_points[:, 1].min()
            xmax, ymax = medium_veg_points[:, 0].max(), medium_veg_points[:, 1].max()

            update_progress_bar(progress_bar, 30)

            # Compute grid dimensions
            cols = int(np.ceil((xmax - xmin) / resolution))
            rows = int(np.ceil((ymax - ymin) / resolution))

            # Initialize CHM
            chm = np.full((rows, cols), -999.0)

            # Populate CHM by assigning max height per grid cell
            for x, y, z in medium_veg_points:
                col = int((x - xmin) / resolution)
                row = int((ymax - y) / resolution)
                if 0 <= row < rows and 0 <= col < cols:
                    if z > chm[row, col]:
                        chm[row, col] = z

            update_progress_bar(progress_bar, 40)

            # Smooth CHM using Gaussian filter
            chm[chm == -999.0] = np.nan
            chm_smooth = np.nan_to_num(chm)
            chm_smooth = gaussian_filter(chm_smooth, sigma=2)

            # Detect local maxima in CHM to use as tree top markers
            coordinates = peak_local_max(chm_smooth, min_distance=2, exclude_border=False)

            update_progress_bar(progress_bar, 50)

            # Create marker image for watershed segmentation
            markers = np.zeros_like(chm_smooth, dtype=int)
            for i, (r, c) in enumerate(coordinates, 1):
                markers[r, c] = i

            update_progress_bar(progress_bar, 60)

            # Perform watershed segmentation on negative CHM to separate trees
            elevation = -chm_smooth
            labels = watershed(elevation, markers, mask=~np.isnan(chm))

            # Calculate sizes of segmented labels
            label_sizes = ndi.sum(~np.isnan(chm), labels, index=np.arange(1, labels.max() + 1))

            min_size = 20  # Minimum segment size threshold
            mask = np.zeros_like(labels, dtype=bool)

            update_progress_bar(progress_bar, 70)

            # Filter small segments (likely noise)
            for i, size in enumerate(label_sizes, 1):
              if size >= min_size:
                    mask |= labels == i

            # Clean labels by masking out small segments
            labels_clean = labels * mask

            # Initialize classificationtree array to assign tree segment IDs to points
            classificationtree = np.zeros(len(points), dtype=int)
            for i, (x, y, z) in enumerate(points):
                col = int((x - xmin) / resolution)
                row = int((ymax - y) / resolution)
                if 0 <= row < labels_clean.shape[0] and 0 <= col < labels_clean.shape[1]:
                    tree_id = labels_clean[row, col]
                    if tree_id > 0:
                        classificationtree[i] = tree_id

            # Add new dimension "classificationtree" to LAS file
            las_r.add_extra_dim(
                laspy.ExtraBytesParams(name="classificationtree", type=np.int32)
            )
            las_r["classificationtree"] = classificationtree

            las_object = las_r

            if mi_set == False:
                update_progress_bar(progress_bar, 100)
                progress_bar.grid_forget()
                grid(las_object, pixels_x, pixels_y)

            # If mi_set is True, create colored point cloud visualization of segmented trees
            if mi_set == True:
                classifications = np.array(las_r.classificationtree)
                num_arboles = len(np.unique(labels_clean)) - 1  # Number of detected trees
                filtered_chm = np.where((labels_clean > 0), chm, np.nan)

                update_progress_bar(progress_bar, 80)

                max_label = np.max(labels_clean)
                colors = np.random.rand(max_label + 1, 3)  # Random colors per tree label

                # Create colored image for each tree
                tree_colors = np.zeros((labels_clean.shape[0], labels_clean.shape[1], 3))
                for label in range(max_label + 1):
                    tree_colors[labels_clean == label] = colors[label]

                update_progress_bar(progress_bar, 90)

                pcd_points = []
                pcd_colors = []

                # Construct point cloud of tree tops colored by tree ID
                for row in range(rows):
                    for col in range(cols):
                        if not np.isnan(chm[row, col]):
                            pcd_points.append(
                                [xmin + col * resolution, ymax - row * resolution, filtered_chm[row, col]])
                            pcd_colors.append(tree_colors[row, col])

                pcd_points = np.array(pcd_points)
                pcd_points[:, 2] = pcd_points[:, 2] + 3  # Slightly elevate points for visualization clarity

                pcd_tree = o3d.geometry.PointCloud()
                pcd_tree.points = o3d.utility.Vector3dVector(np.array(pcd_points))
                pcd_tree.colors = o3d.utility.Vector3dVector(np.array(pcd_colors))

                update_progress_bar(progress_bar, 100)
                progress_bar.grid_forget()

                messagebox.showinfo("Segmentation Complete", f"Number of detected trees: {num_arboles}")

                vis = o3d.visualization.Visualizer()

                right_frame.update_idletasks()
                right_frame_width = right_frame.winfo_width()
                right_frame_height = right_frame.winfo_height()
                left_frame.update_idletasks()
                left_frame_width = left_frame.winfo_width()
                title_bar_height = ctypes.windll.user32.GetSystemMetrics(4)

                vis.create_window(window_name='Open3D', width=right_frame_width + left_frame_width,
                                  height=right_frame_height,
                                  left=0, top=title_bar_height)
                vis.clear_geometries()
                vis.add_geometry(pcd)
                vis.add_geometry(pcd_tree)

                while True:
                    vis.poll_events()
                    vis.update_renderer()

                    if not vis.poll_events():
                        enable_left_frame()
                        break

                if not pcd.has_points():
                    return
        else:
            # If LAS object already exists
            progress_bar = create_progress_bar()
            update_progress_bar(progress_bar, 10)

            if mi_set == False:
                update_progress_bar(progress_bar, 100)
                progress_bar.grid_forget()
                grid(las_object, pixels_x, pixels_y)

            if mi_set == True:
                classifications = np.array(las_object.classificationtree)

                update_progress_bar(progress_bar, 20)

                num_arboles = len(np.unique(labels_clean)) - 1

                update_progress_bar(progress_bar, 40)

                filtered_chm = np.where((labels_clean > 0), chm, np.nan)

                update_progress_bar(progress_bar, 80)

                max_label = np.max(labels_clean)

                colors = np.random.rand(max_label + 1, 3)

                tree_colors = np.zeros((labels_clean.shape[0], labels_clean.shape[1], 3))
                for label in range(max_label + 1):
                    tree_colors[labels_clean == label] = colors[label]

                update_progress_bar(progress_bar, 90)

                pcd_points = []
                pcd_colors = []

                for row in range(rows):
                    for col in range(cols):
                        if not np.isnan(chm[row, col]):
                            pcd_points.append(
                                [xmin + col * resolution, ymax - row * resolution, filtered_chm[row, col]])
                            pcd_colors.append(tree_colors[row, col])

                pcd_points = np.array(pcd_points)
                pcd_points[:, 2] = pcd_points[:, 2] + 3

                pcd_tree = o3d.geometry.PointCloud()
                pcd_tree.points = o3d.utility.Vector3dVector(np.array(pcd_points))
                pcd_tree.colors = o3d.utility.Vector3dVector(np.array(pcd_colors))

                update_progress_bar(progress_bar, 100)
                progress_bar.grid_forget()

                messagebox.showinfo("Segmentation Complete", f"Number of detected trees: {num_arboles}")

                vis = o3d.visualization.Visualizer()

                right_frame.update_idletasks()
                right_frame_width = right_frame.winfo_width()
                right_frame_height = right_frame.winfo_height()
                left_frame.update_idletasks()
                left_frame_width = left_frame.winfo_width()
                title_bar_height = ctypes.windll.user32.GetSystemMetrics(4)

                vis.create_window(window_name='Open3D', width=right_frame_width + left_frame_width,
                                  height=right_frame_height,
                                  left=0, top=title_bar_height)
                vis.clear_geometries()
                vis.add_geometry(pcd)
                vis.add_geometry(pcd_tree)

                while True:
                    vis.poll_events()
                    vis.update_renderer()

                    if not vis.poll_events():
                        enable_left_frame()
                        break

    # Run in a separate thread
    threading.Thread(target=run, daemon=True).start()


# ----------- Main application window -----------
root = CTk()
root.title("Point Cloud Viewer")
root.configure(bg="#1E1E1E")

# Set the window size and position
def maximize_window():
    root.state("zoomed")

root.after(0, maximize_window)

# Disable the frame
def disable_frame():
    root.attributes('-disabled', True)

# Enable the frame
def enable_frame():
    root.attributes('-disabled', False)

disable_frame()

# Show a message to the user
def show_message():
    messagebox.showinfo("Information", "In this program, each time you want to edit the parameters, the Open3D window must be closed. Click the Accept button to start.")
    enable_frame()

root.after(1000, show_message)

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=4)

left_frame = CTkFrame(root, fg_color="#2E2E2E", corner_radius=0)
left_frame.grid(row=0, column=0, sticky="nsew")
left_frame.pack_propagate(False)

right_frame = CTkFrame(root, fg_color="white", corner_radius=0)
right_frame.grid(row=0, column=1, sticky="nsew")

menu_frame = CTkFrame(left_frame, fg_color="#2E2E2E", corner_radius=0)
menu_frame.pack(pady=(15, 0))

root.menu_visible = False

# Function to toggle the visibility of the menu buttons
def toggle_menu():
    root.menu_visible = not root.menu_visible

    if root.menu_visible:
        root.btn_open_pc.pack(pady=0)
        root.btn_open_csv.pack(pady=0)
        root.btn_open_xml.pack(pady=0)
    else:
        root.btn_open_pc.pack_forget()
        root.btn_open_csv.pack_forget()
        root.btn_open_xml.pack_forget()

root.btn_menu = CTkButton(menu_frame, text="Open ...", command=toggle_menu, fg_color="#3E3E3E")
root.btn_menu.pack(pady=(5, 0))

# Load Point Cloud, N42 File, and XML Metadata
def load_point_cloud_and_toggle():
    load_point_cloud()
    toggle_menu()

# Process N42 files and toggle the menu
def process_n42_files_and_toggle():
    process_n42_files()
    toggle_menu()

# Load XML metadata and toggle the menu
def load_xml_metadata_and_toggle():
    load_xml_metadata()
    toggle_menu()

root.btn_open_pc = CTkButton(menu_frame, text="Point Cloud", text_color="#2E2E2E", fg_color="#F0F0F0", border_color="#6E6E6E", border_width=1, font=("Arial", 12), command=load_point_cloud_and_toggle)
root.btn_open_csv = CTkButton(menu_frame, text="N42 File", text_color="#2E2E2E", fg_color="#F0F0F0", border_color="#6E6E6E", border_width=2, font=("Arial", 12), command=process_n42_files_and_toggle)
root.btn_open_xml = CTkButton(menu_frame, text="XML", text_color="#2E2E2E", fg_color="#F0F0F0", border_color="#6E6E6E", border_width=1, font=("Arial", 12), command=load_xml_metadata_and_toggle)

downsample_frame = CTkFrame(left_frame, fg_color="#2E2E2E", corner_radius=0)
downsample_frame.pack(pady=(10, 0))
label_downsample = CTkLabel(downsample_frame, text="Downsample:", text_color="#F0F0F0", font=("Arial", 12))
root.downsample_entry = CTkEntry(downsample_frame, width=50, font=("Arial", 12))
label_percent = CTkLabel(downsample_frame, text="%", text_color="#F0F0F0", font=("Arial", 12))
label_downsample.pack(side="left", padx=(0, 5))
root.downsample_entry.pack(side="left", padx=(0, 5))
label_percent.pack(side="left")

root.parameters_visible = False

# Function to toggle the visibility of the parameters frame
def toggle_parameters():
    root.parameters_visible = not root.parameters_visible

    if root.menu_visible:
        toggle_menu()

    if root.parameters_visible:
        root.button_parameters.configure(text=" ▲ Parameters")
        parameters_frame.pack(pady=(10, 0), fill="x")
        root.button_dose_layer.pack_forget()
        root.button_dose_layer.pack(fill="x", padx=(0, 0), pady=(10, 0))
        root.button_extra_computations.pack_forget()
        root.button_extra_computations.pack(fill="x", padx=(0, 0), pady=(10, 0))
        root.btn_visualize.pack_forget()
        root.btn_visualize.pack(side="bottom", padx=(0, 0), pady=(10, 25))

        if root.dose_layer_visible:
            root.button_dose_layer.pack_forget()
            root.button_dose_layer.pack(fill="x", padx=(0, 0), pady=(10, 0))
            dose_layer_frame.pack_forget()
            dose_layer_frame.pack(pady=(5, 0), fill="x")
            root.button_extra_computations.pack_forget()
            root.button_extra_computations.pack(fill="x", padx=(0, 0), pady=(10, 0))
            root.btn_visualize.pack_forget()
            root.btn_visualize.pack(side="bottom", padx=(0, 0), pady=(10, 25))
    else:
        root.button_parameters.configure(text=" ▼ Parameters")
        parameters_frame.pack_forget()

    if root.extra_computations_visible:
        extra_computations_frame.pack_forget()
        extra_computations_frame.pack(pady=(10, 0), fill="x")
        root.btn_visualize.pack_forget()
        root.btn_visualize.pack(side="bottom", padx=(0, 0), pady=(10, 25))

root.button_parameters = CTkButton(left_frame, text=" ▼ Parameters", text_color="#F0F0F0", fg_color="#3E3E3E",
                              anchor="w", corner_radius=0, command=toggle_parameters)
root.button_parameters.pack(fill="x", padx=(0, 0), pady=(10, 0))

parameters_frame = CTkFrame(left_frame, fg_color="#2E2E2E", corner_radius=0)

point_size_frame = CTkFrame(parameters_frame, fg_color="#2E2E2E", corner_radius=0)
point_size_frame.pack(fill="x", padx=(10, 10), pady=(0, 0))
label_point_size = CTkLabel(point_size_frame, text="Point Size:", text_color="#F0F0F0", font=("Arial", 12))
root.point_size_entry = CTkEntry(point_size_frame, width=50, font=("Arial", 12), state="disabled")
label_point_size.pack(side="left", padx=(10, 5))
root.point_size_entry.pack(side="left", padx=(0, 5))

voxelizer_frame = CTkFrame(parameters_frame, fg_color="#252525", corner_radius=0)
voxelizer_frame.pack(fill="x", padx=(10, 10), pady=(5, 0))
voxelizer_frame.grid_columnconfigure(0, weight=1)
voxelizer_frame.grid_columnconfigure(1, weight=1)
voxelizer_frame.grid_columnconfigure(2, weight=0)
voxelizer_frame.grid_columnconfigure(3, weight=1)
label_voxelizer = CTkLabel(voxelizer_frame, text="Voxelizer:", text_color="#F0F0F0", font=("Arial", 12))
label_voxelizer.grid(row=0, column=1, padx=(10, 5), pady=(5, 0), sticky="e")
root.voxelizer_var = BooleanVar()
root.voxelizer_switch = CTkSwitch(voxelizer_frame, variable=root.voxelizer_var, command=toggle_voxel_size, text="", state="disabled")
root.voxelizer_switch.grid(row=0, column=2, padx=(0, 5), pady=(5, 0), sticky="w")
voxelizerSize_frame = CTkFrame(parameters_frame, fg_color="#1E1E1E", corner_radius=0)
voxelizerSize_frame.pack(fill="x", padx=(10, 10), pady=(0, 0))
label_vox_size = CTkLabel(voxelizerSize_frame, text="Vox Size:", text_color="#F0F0F0", font=("Arial", 12))
label_vox_size.grid(row=1, column=0, padx=(10, 5), pady=(5, 5), sticky="w")
root.vox_size_entry = CTkEntry(voxelizerSize_frame, width=50, font=("Arial", 12), state="disabled")
root.vox_size_entry.grid(row=1, column=1, padx=(0, 5), pady=(5, 5), sticky="w")

root.dose_layer_visible = False

# Function to toggle the visibility of the dose layer
def toggle_dose_layer_b():
    root.dose_layer_visible = not root.dose_layer_visible

    if root.menu_visible:
        toggle_menu()

    if root.dose_layer_visible:
        root.button_dose_layer.configure(text=" ▲ Dose Layer")
        dose_layer_frame.pack(pady=(5, 0), fill="x")
        root.button_extra_computations.pack_forget()
        root.button_extra_computations.pack(fill="x", padx=(0, 0), pady=(10, 0))
        root.btn_visualize.pack_forget()
        root.btn_visualize.pack(side="bottom", padx=(0, 0), pady=(10, 25))
    else:
        root.button_dose_layer.configure(text=" ▼ Dose Layer")
        dose_layer_frame.pack_forget()

    if root.extra_computations_visible:
        extra_computations_frame.pack_forget()
        extra_computations_frame.pack(pady=(10, 0), fill="x")
        root.btn_visualize.pack_forget()
        root.btn_visualize.pack(side="bottom", padx=(0, 0), pady=(10, 25))

root.button_dose_layer = CTkButton(left_frame, text=" ▼ Dose Layer", text_color="#F0F0F0", fg_color="#3E3E3E",
                              anchor="w", corner_radius=0, command=toggle_dose_layer_b)
root.button_dose_layer.pack(fill="x", padx=(0, 0), pady=(10, 0))

dose_layer_frame = CTkFrame(left_frame, fg_color="#2E2E2E", corner_radius=0)

yes_no_frame = CTkFrame(dose_layer_frame, fg_color="#2E2E2E", corner_radius=0)
yes_no_frame.pack(pady=(5, 0), anchor="center")

label_no = CTkLabel(yes_no_frame, text="No", text_color="#F0F0F0", font=("Arial", 12))
label_no.pack(side="left", padx=(5, 5))  # Reducido el espacio después de "No"

root.dose_layer_switch = CTkSwitch(yes_no_frame, text="", command=lambda: toggle_dose_layer(source_location), state='disabled',
                                   width=36, height=18)
root.dose_layer_switch.pack(side="left", padx=(5, 2))  # Menos espacio entre switch y "Yes"

label_yes = CTkLabel(yes_no_frame, text="Yes", text_color="#F0F0F0", font=("Arial", 12))
label_yes.pack(side="left", padx=(2, 5))  # Reducido el espacio antes de "Yes"

dosis_elevation_frame = CTkFrame(dose_layer_frame, fg_color="#2E2E2E", corner_radius=0)
dosis_elevation_frame.pack(fill="x", pady=(5, 0), anchor="center")
label_dosis_elevation = CTkLabel(dosis_elevation_frame, text="Dose Elevation:", text_color="#F0F0F0",
                                 font=("Arial", 12))
label_dosis_elevation.pack(side="left", padx=(10, 5))

# Function to update the slider label
def update_slider_label(value):
    slider_label.configure(text=f"{value:.2f}", font=("Arial", 12))

root.dosis_slider = CTkSlider(dosis_elevation_frame, from_=-100, to=100, command=update_slider_label, state="disabled")
root.dosis_slider.set(1)
root.dosis_slider.pack(side="left", padx=(0, 5))
slider_label = CTkLabel(dosis_elevation_frame, text="1.00", text_color="#F0F0F0")
slider_label.pack(side="left", padx=(0, 5))

dose_sections_frame = CTkFrame(dose_layer_frame, fg_color="#2E2E2E", corner_radius=0)
dose_sections_frame.pack(fill="x", pady=(5, 0), anchor="center")

root.color_options = ["red", "yellow", "green", "blue", "purple", "orange", "cyan", "magenta", "pink", "white"]

root.high_min_medium_max = StringVar()
root.medium_min_low_max = StringVar()

label_high_dose = CTkLabel(dose_sections_frame, text="High Dose:", text_color="#F0F0F0", font=("Arial", 12))
label_high_dose.grid(row=0, column=0, padx=(10, 5), sticky="ew")
root.high_dose_cb = CTkComboBox(dose_sections_frame, values=root.color_options, font=("Arial", 12), width=90, state="disabled")
root.high_dose_cb.set("red")
root.high_dose_cb.grid(row=0, column=1, padx=(0, 5), sticky="ew")
high_dose_rgb = np.array(mcolors.to_rgb("red"))
label_min = CTkLabel(dose_sections_frame, text="Min:", text_color="#F0F0F0", font=("Arial", 12))
label_min.grid(row=0, column=2, padx=(0, 5), sticky="ew")
root.high_dose_min = CTkEntry(dose_sections_frame, width=50, font=("Arial", 11), textvariable=root.high_min_medium_max, state="disabled")
root.high_dose_min.grid(row=0, column=3, padx=(0, 5), sticky="ew")
label_max = CTkLabel(dose_sections_frame, text="Max:", text_color="#F0F0F0", font=("Arial", 12))
label_max.grid(row=0, column=4, padx=(0, 5), sticky="ew")
root.high_dose_max = CTkEntry(dose_sections_frame, width=50, font=("Arial", 11), state="disabled")
root.high_dose_max.grid(row=0, column=5, padx=(0, 5), sticky="ew")

label_medium_dose = CTkLabel(dose_sections_frame, text="Medium Dose:", text_color="#F0F0F0", font=("Arial", 12))
label_medium_dose.grid(row=1, column=0, padx=(10, 5), sticky="ew")
root.medium_dose_cb = CTkComboBox(dose_sections_frame, values=root.color_options, font=("Arial", 12), width=90, state="disabled")
root.medium_dose_cb.set("yellow")
root.medium_dose_cb.grid(row=1, column=1, padx=(0, 5), sticky="ew")
medium_dose_rgb = np.array(mcolors.to_rgb("yellow"))
label_min_medium = CTkLabel(dose_sections_frame, text="Min:", text_color="#F0F0F0", font=("Arial", 12))
label_min_medium.grid(row=1, column=2, padx=(0, 5), sticky="ew")
root.medium_dose_min = CTkEntry(dose_sections_frame, width=50, font=("Arial", 11), textvariable=root.medium_min_low_max, state="disabled")
root.medium_dose_min.grid(row=1, column=3, padx=(0, 5), sticky="ew")
label_max_medium = CTkLabel(dose_sections_frame, text="Max:", text_color="#F0F0F0", font=("Arial", 12))
label_max_medium.grid(row=1, column=4, padx=(0, 5), sticky="ew")
root.medium_dose_max = CTkEntry(dose_sections_frame, width=50, font=("Arial", 11), textvariable=root.high_min_medium_max, state="disabled")
root.medium_dose_max.grid(row=1, column=5, padx=(0, 5), sticky="ew")

label_low_dose = CTkLabel(dose_sections_frame, text="Low Dose:", text_color="#F0F0F0", font=("Arial", 12))
label_low_dose.grid(row=2, column=0, padx=(10, 5), sticky="ew")
root.low_dose_cb = CTkComboBox(dose_sections_frame, values=root.color_options, font=("Arial", 12), width=90, state="disabled")
root.low_dose_cb.set("green")
root.low_dose_cb.grid(row=2, column=1, padx=(0, 5), sticky="ew")
low_dose_rgb = np.array(mcolors.to_rgb("green"))
label_min_low = CTkLabel(dose_sections_frame, text="Min:", text_color="#F0F0F0", font=("Arial", 12))
label_min_low.grid(row=2, column=2, padx=(0, 5), sticky="ew")
root.low_dose_min = CTkEntry(dose_sections_frame, width=50, font=("Arial", 11), state="disabled")
root.low_dose_min.grid(row=2, column=3, padx=(0, 5), sticky="ew")
label_max_low = CTkLabel(dose_sections_frame, text="Max:", text_color="#F0F0F0", font=("Arial", 12))
label_max_low.grid(row=2, column=4, padx=(0, 5), sticky="ew")
root.low_dose_max = CTkEntry(dose_sections_frame, width=50, font=("Arial", 11), textvariable=root.medium_min_low_max, state="disabled")
root.low_dose_max.grid(row=2, column=5, padx=(0, 5), sticky="ew")

source_frame = CTkFrame(dose_layer_frame, fg_color="#2E2E2E", corner_radius=0)
source_frame.pack(fill="x", pady=(5, 0))
root.btn_find_source = CTkButton(source_frame, text="Find Radioactive Source", fg_color="#3E3E3E", text_color="#F0F0F0", font=("Arial", 12), command=lambda: find_radioactive_source(csv_filepath))
root.btn_find_source.grid(row=0, column=0, padx=(10, 5), pady=(5, 0), sticky="w")
show_source_label = CTkLabel(source_frame, text="Show Source on Map:", text_color="#F0F0F0", font=("Arial", 12))
show_source_label.grid(row=0, column=1, padx=(10, 5), pady=(5, 0), sticky="w")
root.show_source_switch = CTkSwitch(source_frame, text="", command=toggle_source, state='disabled')
root.show_source_switch.grid(row=0, column=2, padx=(10, 5), pady=(5, 0), sticky="w")

root.extra_computations_visible = False
root.obstacle_detection_visible = False

# Function to toggle the visibility of the extra computations frame
def toggle_extra_computations():
    root.extra_computations_visible = not root.extra_computations_visible

    if root.menu_visible:
        toggle_menu()

    if root.extra_computations_visible:
        root.button_extra_computations.configure(text=" ▲ Extra Computations")
        extra_computations_frame.pack(pady=(10, 0), fill="x")
        root.btn_visualize.pack_forget()
        root.btn_visualize.pack(side="bottom", padx=(0, 0), pady=(10, 25))
    else:
        if root.obstacle_detection_visible:
            root.obstacle_detection.configure(text=" ▼ Obstacle detection")
            frame_obstacle_detection.pack_forget()
            root.obstacle_detection_visible = False

        root.button_extra_computations.configure(text=" ▼ Extra Computations")
        extra_computations_frame.pack_forget()
        root.btn_visualize.pack_forget()
        root.btn_visualize.pack(side="bottom", padx=(0, 0), pady=(10, 25))

root.button_extra_computations = CTkButton(left_frame, text=" ▼ Extra Computations", text_color="#F0F0F0", fg_color="#3E3E3E", anchor="w", corner_radius=0, command=toggle_extra_computations)
root.button_extra_computations.pack(fill="x", padx=(0, 0), pady=(10, 0))

extra_computations_frame = CTkFrame(left_frame, fg_color="#2E2E2E", corner_radius=0)

root.btn_heatmap = CTkButton(extra_computations_frame, text="Heatmap H*(10) rate", fg_color="#3E3E3E", text_color="#F0F0F0", font=("Arial", 12), command=lambda: plot_heatmap(heatmap, xcenter, ycenter, Hcenter, lonmin, lonmax, latmin, latmax))
root.btn_heatmap.pack(fill="x", padx=(80, 80), pady=(5, 0))
root.btn_three_colors = CTkButton(extra_computations_frame, text="Heatmap with Three Color Range", fg_color="#3E3E3E", text_color="#F0F0F0", font=("Arial", 12), command=lambda: plot_three_color_heatmap(heatmap, xcenter, ycenter, Hcenter, lonmin, lonmax, latmin, latmax))
root.btn_three_colors.pack(fill="x", padx=(80, 80), pady=(5, 0))

segmentation_frame = CTkFrame(extra_computations_frame, fg_color="#2E2E2E", corner_radius=0)
segmentation_frame.pack(fill="x", padx=(80, 80), pady=(5, 0))
root.segmentation = CTkButton(segmentation_frame, text="Segmentation", fg_color="#3E3E3E", text_color="#F0F0F0", font=("Arial", 12), width=105, command=segmentation)
root.segmentation.pack(side="left", padx=(0, 2.5))
root.segmentation_with_trees = CTkButton(segmentation_frame, text="Segmentation\nwith trees", fg_color="#3E3E3E", text_color="#F0F0F0", font=("Arial", 12), command=set_trees)
root.segmentation_with_trees.pack(side="left", padx=(2.5, 0))

# # Function to toggle the visibility of the obstacle detection frame
def toggle_obstacle_detection():
    global las, min_x_las, max_x_las, min_y_las, max_y_las
    root.obstacle_detection_visible = not root.obstacle_detection_visible

    if root.obstacle_detection_visible:
        if las is None:
            fp = filedialog.askopenfilename(filetypes=[("LAS Files", "*.las")])
            if fp:
                las = fp
                root.obstacle_detection.configure(text=" ▲ Obstacle detection")
                frame_obstacle_detection.pack(fill="x", padx=(80, 80), pady=(0, 0))
            else:
                return
        else:
            root.obstacle_detection.configure(text=" ▲ Obstacle detection")
            frame_obstacle_detection.pack(fill="x", padx=(80, 80), pady=(0, 0))

        las_read = laspy.read(las)
        min_x_las, max_x_las = las_read.x.min(), las_read.x.max()
        min_y_las, max_y_las = las_read.y.min(), las_read.y.max()

    else:
        root.obstacle_detection.configure(text=" ▼ Obstacle detection")
        frame_obstacle_detection.pack_forget()

root.obstacle_detection = CTkButton(extra_computations_frame, text=" ▼ Obstacle detection", fg_color="#3E3E3E", text_color="#F0F0F0", font=("Arial", 12), command=toggle_obstacle_detection)
root.obstacle_detection.pack(fill="x", padx=(80, 80), pady=(5, 0))

frame_obstacle_detection = CTkFrame(extra_computations_frame, fg_color="#666666", corner_radius=0)

root.label_pixeles = CTkLabel(frame_obstacle_detection, text="Set the number of pixels to define the prism size of the mesh:", text_color="#2E2E2E", font=("Arial", 12), wraplength=200)
root.label_pixeles.pack(fill="x", padx=(5, 5), pady=(5, 0))

frame_pixels = CTkFrame(frame_obstacle_detection, fg_color="#666666", corner_radius=0)
frame_pixels.pack(fill="x", padx=(0, 0), pady=(0, 0))
for i in range(3):
    frame_pixels.rowconfigure(i, weight=1)
frame_pixels.columnconfigure(0, weight=1)
frame_pixels.columnconfigure(1, weight=1)

# Updates the prism size label based on the number of pixels
def update_prism_size_label(*args):
    global min_x_las, max_x_las, min_y_las, max_y_las
    if None in (min_x_las, max_x_las, min_y_las, max_y_las):
        root.prism_size_label_entry.configure(text="")
        return
    try:
        pixels_x = int(root.num_pixeles_x_entry.get())
        pixels_y = int(root.num_pixeles_y_entry.get())
        if pixels_x > 0 and pixels_y > 0:
            delta_x = (max_x_las - min_x_las) / pixels_x
            delta_y = (max_y_las - min_y_las) / pixels_y
            root.prism_size_label_entry.configure(text=f"{delta_x:.2f} x {delta_y:.2f} m")
        else:
            root.prism_size_label_entry.configure(text="")
    except ValueError:
        root.prism_size_label_entry.configure(text="")

root.num_pixeles_x_label = CTkLabel(frame_pixels, text="Num Pixels X:", text_color="#F0F0F0", font=("Arial", 12))
root.num_pixeles_x_label.grid(row=0, column=0, padx=(10, 5), pady=(5, 5), sticky="e")
root.num_pixeles_x_entry = CTkEntry(frame_pixels, width=50, font=("Arial", 12))
root.num_pixeles_x_entry.grid(row=0, column=1, padx=(0, 10), pady=(5, 5), sticky="w")
root.num_pixeles_x_entry.bind("<KeyRelease>", update_prism_size_label)

root.num_pixeles_y_label = CTkLabel(frame_pixels, text="Num Pixels Y:", text_color="#F0F0F0", font=("Arial", 12))
root.num_pixeles_y_label.grid(row=1, column=0, padx=(10, 5), pady=(5, 5), sticky="e")
root.num_pixeles_y_entry = CTkEntry(frame_pixels, width=50, font=("Arial", 12))
root.num_pixeles_y_entry.grid(row=1, column=1, padx=(0, 10), pady=(5, 5), sticky="w")
root.num_pixeles_y_entry.bind("<KeyRelease>", update_prism_size_label)

root.prism_size_label = CTkLabel(frame_pixels, text="Prism Size:", text_color="#F0F0F0", font=("Arial", 12))
root.prism_size_label.grid(row=2, column=0, padx=(10, 5), pady=(5, 5), sticky="e")
root.prism_size_label_entry = CTkLabel(frame_pixels, text="",text_color="#F0F0F0", font=("Arial", 12))
root.prism_size_label_entry.grid(row=2, column=1, padx=(0, 10), pady=(5, 5), sticky="w")

root.compute = CTkButton(frame_obstacle_detection, text="Compute", fg_color="#3E3E3E", text_color="#F0F0F0", font=("Arial", 12), command=lambda: set_flag(xcenter, ycenter, FAltcenter, root.num_pixeles_x_entry.get(), root.num_pixeles_y_entry.get()))
root.compute.pack(fill="x", padx=(40, 40), pady=(5, 5))

root.btn_visualize = CTkButton(left_frame, text="Visualize", text_color="#F0F0F0", fg_color="#1E3A5F",
                               hover_color="#2E4A7F",
                               anchor="center", corner_radius=0, border_color="#D3D3D3", border_width=2,
                               command=lambda: visualize(pc_filepath, csv_filepath, xml_filepath, show_dose_layer, dose_min_csv, dose_max_csv))
root.btn_visualize.pack(side="bottom", padx=(0, 0), pady=(10, 25))

# Genetic Algorithm for finding the best radioactive source location
class GeneticAlgorithm:
    def __init__(self, utm_coords, population_size=500, generations=100, mutation_rate=0.01):
        self.utm_coords = utm_coords
        self.population_size = population_size  # Define the size of the population
        self.generations = generations          # Define the number of generations to evolve
        self.mutation_rate = mutation_rate      # Define the mutation rate for genetic diversity
        self.bounds = self.get_bounds()         # Get the bounds of the UTM coordinates

    # Obtain the bounds of the UTM coordinates
    def get_bounds(self):
        x_min, y_min = np.min(self.utm_coords[:, :2], axis=0)
        x_max, y_max = np.max(self.utm_coords[:, :2], axis=0)
        return (x_min, x_max), (y_min, y_max)

    # Calculate the fitness of a candidate point based on the dose at that point
    def fitness(self, candidate):
        tree = cKDTree(self.utm_coords[:, :2])
        dist, idx = tree.query(candidate)       # Find the closest point in the point cloud to the candidate location, and return the distance and the index of the point
        return -self.utm_coords[idx, 2]         # Negative dose because we want to maximize (not minimize); the algorithm maximizes the dose by minimizing the negative value (we keep the most negative, which corresponds to the highest dose value with its sign changed)

    # Initialize the population with random candidates within the bounds
    def initialize_population(self):
        (x_min, x_max), (y_min, y_max) = self.bounds
        return np.array(
            [[random.uniform(x_min, x_max), random.uniform(y_min, y_max)] for _ in range(self.population_size)])

    # Select parents based on their fitness values using a weighted random choice
    def select_parents(self, population, fitnesses):
        idx = np.random.choice(np.arange(self.population_size), size=self.population_size, replace=True, p=fitnesses / fitnesses.sum())
        return population[idx]

    # Perform crossover between two parents to create a new candidate
    def crossover(self, parent1, parent2):
        alpha = random.random()
        return alpha * parent1 + (1 - alpha) * parent2

    # Mutate a candidate by randomly changing its coordinates within the bounds
    def mutate(self, candidate):
        (x_min, x_max), (y_min, y_max) = self.bounds
        if random.random() < self.mutation_rate:
            candidate[0] = random.uniform(x_min, x_max)
        if random.random() < self.mutation_rate:
            candidate[1] = random.uniform(y_min, y_max)
        return candidate

    # Run the genetic algorithm to find the best candidate location for the radioactive source
    def run(self):
        population = self.initialize_population()
        for generation in range(self.generations):
            fitnesses = np.array([self.fitness(candidate) for candidate in population])
            parents = self.select_parents(population, fitnesses)
            next_population = []
            for i in range(0, self.population_size, 2):
                parent1, parent2 = parents[i], parents[i + 1]
                child1, child2 = self.crossover(parent1, parent2), self.crossover(parent2, parent1)
                next_population.append(self.mutate(child1))
                next_population.append(self.mutate(child2))
            population = np.array(next_population)
        best_candidate = population[np.argmax(fitnesses)]
        return best_candidate

root.mainloop()