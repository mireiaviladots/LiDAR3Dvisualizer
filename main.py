import numpy as np
import open3d as o3d
import csv
from scipy.spatial import cKDTree
from pathlib import Path
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
from pyproj import Proj, transform
from shapely.geometry import LineString, box

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
panel_frame = None
height_frame = None
longitude_frame = None
posiciones = None
mi_set = False
selected_positions = None
combined_mesh = None
num_pixels_x = None
num_pixels_y = None
delta_x = None
delta_y = None
cell_stats = None

def mostrar_nube_no_vox(show_dose_layer, pc_filepath, downsample, xml_filepath, csv_filepath, high_dose_rgb, medium_dose_rgb,
                        low_dose_rgb, dose_min_csv, low_max, medium_min, medium_max, high_min, altura_extra, show_source, source_location, point_size, progress_bar):
    def run():
        global vis
        try:
            print(f"Show Dose Layer: {show_dose_layer}")
            # Cargar la nube de puntos PCD
            pcd = o3d.io.read_point_cloud(pc_filepath)

            # Downsamplear la nube de puntos si se ha especificado un porcentaje
            if downsample is not None:
                if not (1 <= downsample <= 100):
                    messagebox.showerror("Error", "El valor de downsample debe estar entre 1 y 100.")
                    return
                downsample_value = float(downsample) / 100.0
                if 0 < downsample_value <= 1:
                    if downsample_value == 1:
                        downsample_value = 0.99  # Evitar downsamplear a 0
                    voxel_size = 1 * downsample_value
                    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
                    pcd = downsampled_pcd

            # Obtener coordenadas XYZ
            nube_puntos = np.asarray(pcd.points)

            # Actualizar la barra de progreso
            update_progress_bar(progress_bar, 20)

            # Obtener colores si existen, de lo contrario, usar blanco
            if pcd.has_colors():
                rgb = np.asarray(pcd.colors)
            else:
                rgb = np.ones_like(nube_puntos)

            if show_dose_layer:
                origin = get_origin_from_xml(xml_filepath)

                # Sumar el origen a las coordenadas locales
                geo_points = nube_puntos + origin  # a utm

                utm_coords = np.genfromtxt(csv_filepath, delimiter=',', skip_header=1)
                utm_points = utm_coords[:, :2]  # Sólo coordenadas [easting, northing]
                dosis = utm_coords[:, 2]  # Dosis correspondiente

                # Actualizar la barra de progreso
                update_progress_bar(progress_bar, 30)

                # Construir el KD-Tree para los puntos UTM del CSV (BUSQUEDA EFICIENTE)
                tree = cKDTree(utm_points)

                # Actualizar la barra de progreso
                update_progress_bar(progress_bar, 40)

                # Determinar los límites del área del CSV con dosis
                x_min, y_min = np.min(utm_points, axis=0)  # Mínimo de cada columna (lat, long)
                x_max, y_max = np.max(utm_points, axis=0)  # Máximo de cada columna (lat, long)

                # Filtrar puntos de la nube dentro del área de dosis
                dentro_area = (
                    (geo_points[:, 0] >= x_min) & (geo_points[:, 0] <= x_max) &
                    (geo_points[:, 1] >= y_min) & (geo_points[:, 1] <= y_max)
                )

                # Actualizar la barra de progreso
                update_progress_bar(progress_bar, 60)

                # Solo los puntos dentro del área
                puntos_dentro = geo_points[dentro_area]

                # Crea vector de dosis como NaN
                dosis_nube = np.full(len(puntos_dentro), np.nan)

                # Encontrar el punto más cercano en el CSV para cada punto de la nube LAS (que está dentro)
                distancias, indices_mas_cercanos = tree.query(puntos_dentro[:, :2])

                # Asignar dosis correspondiente a los puntos dentro del área
                dosis_nube[:] = dosis[indices_mas_cercanos]  # Dosis para cada punto en la nube

                valid_points = ~np.isnan(dosis_nube)
                puntos_dosis_elevados = puntos_dentro[valid_points]
                dosis_filtrada = dosis_nube[valid_points]

                # Actualizar la barra de progreso
                update_progress_bar(progress_bar, 80)

                colores_dosis = get_dose_color(dosis_filtrada, high_dose_rgb, medium_dose_rgb, low_dose_rgb, dose_min_csv, low_max, medium_min, medium_max, high_min)

                puntos_dosis_elevados[:, 2] += altura_extra  # Aumentar Z

                # Crear nube de puntos Open3D
                pcd.points = o3d.utility.Vector3dVector(geo_points)
                pcd.colors = o3d.utility.Vector3dVector(rgb)  # Asignar colores

                update_progress_bar(progress_bar, 90)

                # Crear la nueva nube de puntos de dosis elevada
                pcd_dosis = o3d.geometry.PointCloud()
                pcd_dosis.points = o3d.utility.Vector3dVector(puntos_dosis_elevados)
                pcd_dosis.colors = o3d.utility.Vector3dVector(colores_dosis)  # Asignar colores según dosis

            else:
                update_progress_bar(progress_bar, 30)
                update_progress_bar(progress_bar, 40)
                update_progress_bar(progress_bar, 50)
                update_progress_bar(progress_bar, 60)
                update_progress_bar(progress_bar, 70)
                update_progress_bar(progress_bar, 80)

            # Actualizar la barra de progreso
            update_progress_bar(progress_bar, 100)

            # Eliminar la barra de progreso
            progress_bar.grid_forget()

            vis = o3d.visualization.Visualizer()

            # Obtener las dimensiones del right_frame
            right_frame.update_idletasks()
            right_frame_width = right_frame.winfo_width()
            right_frame_height = right_frame.winfo_height()

            # Obtener las dimensiones del left_frame
            left_frame.update_idletasks()
            left_frame_width = left_frame.winfo_width()

            # Calcular tittle bar
            title_bar_height = ctypes.windll.user32.GetSystemMetrics(4)

            vis.create_window(window_name='Open3D', width=right_frame_width, height=right_frame_height, left=left_frame_width, top=title_bar_height)

            vis.clear_geometries()  # Ahora estamos seguros de que self.vis no es None
            vis.add_geometry(pcd)
            if show_dose_layer:
                vis.add_geometry(pcd_dosis)

            if show_dose_layer and show_source and source_location is not None:
                source_point = [source_location[0], source_location[1], np.max(puntos_dosis_elevados[:, 2])]
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)  # Create a sphere with a radius of 5
                sphere.translate(source_point)  # Move the sphere to the source location
                sphere.paint_uniform_color([0, 0, 0])
                vis.add_geometry(sphere)

            # Cambiar el tamaño de los puntos (ajustar para evitar cuadrados)
            render_option = vis.get_render_option()
            render_option.point_size = point_size

            while True:
                vis.poll_events()
                vis.update_renderer()

                if not vis.poll_events():
                    print("Ventana Cerrada")
                    enable_left_frame()
                    break

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            enable_left_frame()

    threading.Thread(target=run, daemon=True).start()

def mostrar_nube_si_vox(show_dose_layer, pc_filepath, xml_filepath, csv_filepath, high_dose_rgb, medium_dose_rgb, low_dose_rgb, dose_min_csv, low_max,
            medium_min, medium_max, high_min, altura_extra, progress_bar):
    def run():
        global vis
        try:
            print(f"Show Dose Layer: {show_dose_layer}")
            pcd = o3d.io.read_point_cloud(pc_filepath)
            xyz = np.asarray(pcd.points)

            # Actualizar la barra de progreso
            update_progress_bar(progress_bar, 20)

            # Obtener colores si existen, de lo contrario usar blanco
            if pcd.has_colors():
                rgb = np.asarray(pcd.colors)
            else:
                rgb = np.ones_like(xyz)  # Blanco por defecto

            if not show_dose_layer:
                pcd.points = o3d.utility.Vector3dVector(xyz)
                pcd.colors = o3d.utility.Vector3dVector(rgb)

            if show_dose_layer:
                origin = get_origin_from_xml (xml_filepath)

                geo_points = xyz + origin

                pcd.points = o3d.utility.Vector3dVector(geo_points)
                pcd.colors = o3d.utility.Vector3dVector(rgb)

            # Defining the voxel size
            vsize = vox_size

            # Creating the voxel grid
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=vsize)

            # Extracting the bounds
            bounds = voxel_grid.get_max_bound() - voxel_grid.get_min_bound()
            # o3d.visualization.draw_geometries([voxel_grid])

            # Generating a single box entity
            cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
            cube.paint_uniform_color([1, 0, 0])  # Red
            cube.compute_vertex_normals()
            # o3d.visualization.draw_geometries([cube])

            # Automate and Loop to cerate one voxel Dataset (efficient)
            voxels = voxel_grid.get_voxels()  # Cada voxel con su grid index (posicion desde el centro, 0) y color, hay que hacer offset y translate
            vox_mesh = o3d.geometry.TriangleMesh()  # Creamos un mesh para ir colocando cada voxel

            for v in voxels:
                cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
                cube.paint_uniform_color(v.color)
                cube.translate(v.grid_index, relative=False)
                vox_mesh += cube

            # To align to the center of a cube of dimension 1
            vox_mesh.translate([0.5, 0.5, 0.5], relative=True)

            # To scale
            vox_mesh.scale(vsize, [0, 0, 0])

            # To translate
            vox_mesh.translate(voxel_grid.origin, relative=True)

            # Export
            output_file = Path("voxelize.ply")  # Puntos --> .las / Malla --> .obj, .ply
            o3d.io.write_triangle_mesh(str(output_file), vox_mesh)

            # Actualizar la barra de progreso
            update_progress_bar(progress_bar, 30)

            if show_dose_layer:
                utm_coords = np.genfromtxt(csv_filepath, delimiter=',', skip_header=1)
                utm_points = utm_coords[:, :2]  # Sólo coordenadas [easting, northing]
                dosis = utm_coords[:, 2]  # Dosis correspondiente

                # Construir el KD-Tree para los puntos UTM del CSV (BUSQUEDA EFICIENTE)
                tree = cKDTree(utm_points)

                # Actualizar la barra de progreso
                update_progress_bar(progress_bar, 40)

                # Determinar los límites del área del CSV con dosis
                x_min, y_min = np.min(utm_points, axis=0)  # Mínimo de cada columna (lat, long)
                x_max, y_max = np.max(utm_points, axis=0)  # Máximo de cada columna (lat, long)

                # Filtrar puntos de la nube dentro del área de dosis
                dentro_area = (
                        (geo_points[:, 0] >= x_min) & (geo_points[:, 0] <= x_max) &
                        (geo_points[:, 1] >= y_min) & (geo_points[:, 1] <= y_max)
                )

                # Solo los puntos dentro del área
                puntos_dentro = geo_points[dentro_area]

                # Crea vector de dosis como NaN
                dosis_nube = np.full(len(puntos_dentro), np.nan)

                # Encontrar el punto más cercano en el CSV para cada punto de la nube LAS (que está dentro)
                distancias, indices_mas_cercanos = tree.query(puntos_dentro[:,
                                                              :2])  # Devuelve distancia entre punto CSV y punto cloud; para cada nube_puntos[i] índice del punto del csv mas cercano

                # Actualizar la barra de progreso
                update_progress_bar(progress_bar, 60)

                # Asignar dosis correspondiente a los puntos dentro del área
                dosis_nube[:] = dosis[indices_mas_cercanos]  # Dosis para cada punto en la nube

                valid_points = ~np.isnan(dosis_nube)
                puntos_dosis_elevados = puntos_dentro[valid_points]
                dosis_filtrada = dosis_nube[valid_points]

                colores_dosis = get_dose_color(dosis_filtrada, high_dose_rgb, medium_dose_rgb,
                                                    low_dose_rgb, dose_min_csv, low_max,
                                                    medium_min, medium_max, high_min)

                puntos_dosis_elevados[:, 2] += altura_extra  # Aumentar Z

                pcd_dosis = o3d.geometry.PointCloud()
                pcd_dosis.points = o3d.utility.Vector3dVector(puntos_dosis_elevados)
                pcd_dosis.colors = o3d.utility.Vector3dVector(colores_dosis)

                voxel_grid_dosis = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_dosis, voxel_size=vsize)

                voxels_dosis = voxel_grid_dosis.get_voxels()
                vox_mesh_dosis = o3d.geometry.TriangleMesh()

                # Actualizar la barra de progreso
                update_progress_bar(progress_bar, 80)

                cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
                cube.paint_uniform_color([1, 0, 0])  # Red
                cube.compute_vertex_normals()

                for v in voxels_dosis:
                    cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
                    cube.paint_uniform_color(v.color)
                    cube.translate(v.grid_index, relative=False)
                    vox_mesh_dosis += cube

                update_progress_bar(progress_bar, 90)

                vox_mesh_dosis.translate([0.5, 0.5, 0.5], relative=True)

                vox_mesh_dosis.scale(vsize, [0, 0, 0])

                vox_mesh_dosis.translate(voxel_grid_dosis.origin, relative=True)

                output_file = Path("voxelize_dosis.ply")  # Puntos --> .las / Malla --> .obj, .ply
                o3d.io.write_triangle_mesh(str(output_file), vox_mesh_dosis)

            else:
                update_progress_bar(progress_bar, 40)
                update_progress_bar(progress_bar, 50)
                update_progress_bar(progress_bar, 60)
                update_progress_bar(progress_bar, 70)
                update_progress_bar(progress_bar, 80)
                update_progress_bar(progress_bar, 90)

            # Actualizar la barra de progreso
            update_progress_bar(progress_bar, 100)

            # Eliminar la barra de progreso
            progress_bar.grid_forget()

            vis = o3d.visualization.Visualizer()

            # Obtener las dimensiones del right_frame
            right_frame.update_idletasks()
            right_frame_width = right_frame.winfo_width()
            right_frame_height = right_frame.winfo_height()

            # Obtener las dimensiones del left_frame
            left_frame.update_idletasks()
            left_frame_width = left_frame.winfo_width()

            # Calcular tittle bar
            title_bar_height = ctypes.windll.user32.GetSystemMetrics(4)

            vis.create_window(window_name='Open3D', width=right_frame_width, height=right_frame_height,
                              left=left_frame_width, top=title_bar_height)

            vis.clear_geometries()
            vis.add_geometry(vox_mesh)
            if show_dose_layer:
                vis.add_geometry(vox_mesh_dosis)

            if show_dose_layer and show_source and source_location is not None:
                source_point = [[source_location[0], source_location[1], np.max(puntos_dosis_elevados[:, 2])]]
                source_pcd = o3d.geometry.PointCloud()
                source_pcd.points = o3d.utility.Vector3dVector(source_point)
                source_pcd.paint_uniform_color([0, 0, 0])  # Color negro para el punto de la fuente
                vis.add_geometry(source_pcd)

            while True:
                vis.poll_events()
                vis.update_renderer()

                if not vis.poll_events():
                    print("Ventana Cerrada")
                    enable_left_frame()
                    break

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            enable_left_frame()

    threading.Thread(target=run, daemon=True).start()

def grid(las_object):
    global combined_mesh, num_pixels_x, num_pixels_y, delta_x, delta_y, cell_stats

    progress_bar = create_progress_bar()

    # Actualizar la barra de progreso
    update_progress_bar(progress_bar, 10)

    las = las_object

    las_points = np.vstack((las.x, las.y, las.z)).T
    if hasattr(las, "red"):
        colors = np.vstack((las.red, las.green, las.blue)).T
        # Normalizar si vienen en 16-bit
        if colors.max() > 1.0:
            colors = colors / 65535.0
    else:
        colors = np.zeros_like(las_points)
    classifications = las.classification
    classificationtree = las['classificationtree']

    update_progress_bar(progress_bar, 20)

    # Get unique classifications
    unique_classifications = np.unique(classifications)
    unique_classificationtree = np.unique(classificationtree)

    # Print the unique classifications
    print("Unique classifications:", unique_classifications)
    print("Unique classificationtree values:", unique_classificationtree)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(las_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Extract point data
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(points)

    # Determine the bounds of the data
    min_x, min_y = np.min(points[:, :2], axis=0)
    max_x, max_y = np.max(points[:, :2], axis=0)

    update_progress_bar(progress_bar, 30)

    # Calculate pixel sizes
    num_pixels_x = 1000
    num_pixels_y = 1000
    delta_x = (max_x - min_x) / num_pixels_x
    delta_y = (max_y - min_y) / num_pixels_y

    # Initialize structures for statistics
    z_values = np.full((num_pixels_y, num_pixels_x), np.nan)
    cell_stats = np.empty((num_pixels_y, num_pixels_x), dtype=object)
    for i in range(num_pixels_y):
        for j in range(num_pixels_x):
            cell_stats[i, j] = {'z_values': [], 'colors': [], 'classes': [], 'tree_classes': []}

    update_progress_bar(progress_bar, 40)

    # Asignar puntos a celdas
    x_idx = ((points[:, 0] - min_x) / delta_x).astype(int)
    y_idx = ((points[:, 1] - min_y) / delta_y).astype(int)

    valid_mask = (
            (x_idx >= 0) & (x_idx < num_pixels_x) &
            (y_idx >= 0) & (y_idx < num_pixels_y)
    )

    for xi, yi, z, color in zip(x_idx[valid_mask], y_idx[valid_mask], points[valid_mask][:, 2],
                                colors[valid_mask]):
        cell_stats[yi, xi]['z_values'].append(z)
        cell_stats[yi, xi]['colors'].append(color)

    # Asignar puntos del LAS a celdas
    x_idx_las = ((las_points[:, 0] - min_x) / delta_x).astype(int)
    y_idx_las = ((las_points[:, 1] - min_y) / delta_y).astype(int)
    valid_mask_las = (
            (x_idx_las >= 0) & (x_idx_las < num_pixels_x) &
            (y_idx_las >= 0) & (y_idx_las < num_pixels_y)
    )

    update_progress_bar(progress_bar, 50)

    for xi, yi, cls, cls_tree in zip(
            x_idx_las[valid_mask_las],
            y_idx_las[valid_mask_las],
            classifications[valid_mask_las],
            classificationtree[valid_mask_las]
    ):
        cell_stats[yi, xi]['classes'].append(cls)
        cell_stats[yi, xi]['tree_classes'].append(cls_tree)

    # Calculate mean Z values and predominant colors for each cell
    for i in range(num_pixels_y):
        for j in range(num_pixels_x):
            z_vals = cell_stats[i][j]['z_values']
            if z_vals:
                z_vals = np.array(z_vals)
                z_mean = np.mean(z_vals)
                z_std = np.std(z_vals)

                # Filtrar valores atípicos por el criterio z_mean + 2*sigma
                mask = z_vals <= z_mean + 2 * z_std
                filtered_z_vals = z_vals[mask]
                z_values[i, j] = np.mean(filtered_z_vals) + 2 * np.std(filtered_z_vals)
                cell_stats[i][j]['color'] = np.mean(cell_stats[i][j]['colors'], axis=0)

    update_progress_bar(progress_bar, 60)

    # Create a list to hold all the prisms
    prisms = []

    # Draw horizontal cells and vertical surfaces
    for i in range(num_pixels_y):
        for j in range(num_pixels_x):
            if not np.isnan(z_values[i, j]):
                z_final = z_values[i, j]
                z_min = np.min(cell_stats[i][j]['z_values'])
                height = z_final - z_min
                if height > 0:
                    prism = o3d.geometry.TriangleMesh.create_box(width=delta_x, height=delta_y, depth=height)
                    prism.translate((min_x + j * delta_x, min_y + i * delta_y, z_min))
                    prism.paint_uniform_color(cell_stats[i][j]['color'])

                    # Obtener clase mayoritaria
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

                    # Guardar
                    cell_stats[i][j]['majority_class'] = majority_class
                    cell_stats[i][j]['majority_tree_class'] = majority_tree_class

                    prisms.append(prism)

    update_progress_bar(progress_bar, 80)

    # Combine all prisms into a single mesh
    combined_mesh = o3d.geometry.TriangleMesh()
    for prism in prisms:
        combined_mesh += prism

    update_progress_bar(progress_bar, 100)

    progress_bar.grid_forget()

    print ("acabado")

    panel_left_frame(xcenter, ycenter, FAltcenter, las_object)

def tree_obstacles(las_object, entry_latitude, entry_longitude, combined_mesh, num_pixels_x, num_pixels_y, delta_x, delta_y, cell_stats):
    def run():
        global vis
        try:
            if not selected_positions:
                messagebox.showwarning("Warning", "Please select one point before continuing.")
                return

            try:
                lat = float(entry_latitude.get())
            except (ValueError, AttributeError):
                messagebox.showerror("Error", "Latitude must be a number between -90 and 90.")
                return

            if not -90 <= lat <= 90:
                messagebox.showerror("Error", "Latitude must be between -90 and 90.")
                return

            try:
                lon = float(entry_longitude.get())
            except (ValueError, AttributeError):
                messagebox.showerror("Error", "Longitude must be a number between -180 and 180.")
                return

            if not -180 <= lon <= 180:
                messagebox.showerror("Error", "Longitude must be between -180 and 180.")
                return

            # Convert lat/lon to UTM 31N
            utm_x, utm_y = latlon_a_utm31(lat, lon)
            print(f"UTM coordinates: X={utm_x}, Y={utm_y}")

            # Load the LAS file
            las = las_object

            min_x, max_x = np.min(las.x), np.max(las.x)
            min_y, max_y = np.min(las.y), np.max(las.y)

            # Check if the input point is within bounds
            if not (min_x <= utm_x <= max_x and min_y <= utm_y <= max_y):
                messagebox.showerror("Error", "The entered position is outside the bounds of the point cloud.")
                return

            progress_bar = create_progress_bar()

            disable_left_frame()
            entry_latitude.configure(state='disabled')
            entry_longitude.configure(state='disabled')

            # Actualizar la barra de progreso
            update_progress_bar(progress_bar, 10)

            # Añadir altura posiciones dron
            posiciones_con_altura = []
            for (x, y, alt) in selected_positions:
                # Convertir (x, y) a índice de celda
                col = int((x - min_x) / delta_x)
                row = int((y - min_y) / delta_y)

                # Verificar si el índice es válido
                if 0 <= col < num_pixels_x and 0 <= row < num_pixels_y:
                    cell_z_vals = cell_stats[row][col]['z_values']
                    if cell_z_vals:
                        altitud_nivel_mar = np.mean(cell_z_vals)
                        print(f"Altitud nivel mar: {altitud_nivel_mar}")
                        z_dron = altitud_nivel_mar + alt
                        posiciones_con_altura.append((x, y, z_dron))
                    else:
                        print(f"No hay datos de elevación en la celda para ({x}, {y})")
                else:
                    print(f"Punto fuera de límites: ({x}, {y})")

            update_progress_bar(progress_bar, 40)

            #Añadir altura lat,lon
            posiciones_latlonh = []

            col = int((utm_x - min_x) / delta_x)
            row = int((utm_y - min_y) / delta_y)

            update_progress_bar(progress_bar, 50)

            # Verificar si el índice es válido
            if 0 <= col < num_pixels_x and 0 <= row < num_pixels_y:
                cell_z_vals = cell_stats[row][col]['z_values']
                if cell_z_vals:
                    altitud_nivel_mar = np.mean(cell_z_vals)
                    posiciones_latlonh.append((utm_x, utm_y, altitud_nivel_mar))
                else:
                    print(f"No hay datos de elevación en la celda para ({utm_x}, {utm_y})")
            else:
                print(f"Punto fuera de límites: ({utm_x}, {utm_y})")

            update_progress_bar(progress_bar, 70)

            print("Posicion introducida lat,lon:")
            for pos in posiciones_latlonh:
                print(pos)

            def generar_color_aleatorio_no_rosa():
                while True:
                    color = [random.random(), random.random(), random.random()]
                    if not (abs(color[0] - 1.0) < 0.1 and abs(color[1] - 0.0) < 0.1 and abs(
                            color[2] - 1.0) < 0.1):  # evitar rosa
                        return color

            update_progress_bar(progress_bar, 80)

            colores_puntos = []
            for _ in posiciones_con_altura:
                color = generar_color_aleatorio_no_rosa()
                colores_puntos.append(color)

            # Actualizar la barra de progreso
            update_progress_bar(progress_bar, 95)

            total_arboles_cruzados = []

            if posiciones_con_altura and posiciones_latlonh:
                # Junta todos los puntos: los seleccionados + el punto lat/lon (este será el último)
                puntos_linea = posiciones_con_altura + posiciones_latlonh

                # Crea un índice de línea para cada punto → conecta a la última posición (lat/lon)
                line_indices = [[i, len(puntos_linea) - 1] for i in range(len(posiciones_con_altura))]

                for idx, (start_point, _) in enumerate(zip(posiciones_con_altura, line_indices)):
                    end_point = posiciones_latlonh[0]  # Solo hay un punto final

                    linea = LineString([start_point[:2], end_point[:2]])
                    clases_cruzadas = []

                    for i in range(num_pixels_y):
                        for j in range(num_pixels_x):
                            # Coordenadas de la celda
                            x0 = min_x + j * delta_x
                            y0 = min_y + i * delta_y
                            x1 = x0 + delta_x
                            y1 = y0 + delta_y

                            celda = box(x0, y0, x1, y1)

                            if linea.intersects(celda):
                                tree_class = cell_stats[i][j].get('majority_tree_class')
                                if tree_class and tree_class != 0:
                                    clases_cruzadas.append(tree_class)

                    print(f"Línea desde punto {idx + 1} cruza clases de árbol: {clases_cruzadas}")

                    total_arboles_cruzados.append(len(set(clases_cruzadas)))

                print(f"Total de clases de árbol cruzadas por cada línea: {total_arboles_cruzados}")

            update_progress_bar(progress_bar, 100)

            # Eliminar la barra de progreso
            progress_bar.grid_forget()

            mostrar_resumen_lineas(colores_puntos, total_arboles_cruzados)

            vis = o3d.visualization.Visualizer()

            # Obtener las dimensiones del right_frame
            right_frame.update_idletasks()
            right_frame_width = right_frame.winfo_width()
            right_frame_height = right_frame.winfo_height()

            # Obtener las dimensiones del left_frame
            left_frame.update_idletasks()
            left_frame_width = left_frame.winfo_width()

            # Calcular tittle bar
            title_bar_height = ctypes.windll.user32.GetSystemMetrics(4)

            vis.create_window(window_name='Open3D', width=right_frame_width, height=right_frame_height,
                              left=left_frame_width, top=title_bar_height)

            vis.clear_geometries()

            vis.add_geometry(combined_mesh)

            if posiciones_con_altura and posiciones_latlonh:
                # Junta todos los puntos: los seleccionados + el punto lat/lon (este será el último)
                puntos_linea = posiciones_con_altura + posiciones_latlonh

                # Crea un índice de línea para cada punto → conecta a la última posición (lat/lon)
                line_indices = [[i, len(puntos_linea) - 1] for i in range(len(posiciones_con_altura))]

                # Crea las líneas con Open3D
                lines = o3d.geometry.LineSet()
                lines.points = o3d.utility.Vector3dVector(puntos_linea)
                lines.lines = o3d.utility.Vector2iVector(line_indices)

                # Color rosa para todas las líneas
                lines.colors = o3d.utility.Vector3dVector([[0, 0, 0]] * len(line_indices))

                # Añade al visor
                vis.add_geometry(lines)

            # Crear nubes de punto individuales para cada punto y agregar
            for punto, color in zip(posiciones_con_altura, colores_puntos):
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector([punto])
                pc.paint_uniform_color(color)
                vis.add_geometry(pc)

            if posiciones_latlonh:
                puntos_rosas = o3d.geometry.PointCloud()
                puntos_rosas.points = o3d.utility.Vector3dVector(posiciones_latlonh)
                puntos_rosas.paint_uniform_color([1.0, 0, 1.0])  # rosa
                vis.add_geometry(puntos_rosas)

            while True:
                vis.poll_events()
                vis.update_renderer()

                if not vis.poll_events():
                    print("Ventana Cerrada")

                    enable_left_frame()
                    break

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            enable_left_frame()

    threading.Thread(target=run, daemon=True).start()

def disable_left_frame():
    root.attributes('-disabled', True)

def enable_left_frame():
    root.attributes('-disabled', False)

def legend_left_frame(counts=None, color_map=None):
    global legend_frame, legend_canvas

    left_frame.update_idletasks()
    width = left_frame.winfo_width()
    height = left_frame.winfo_height()

    # Lienzo de fondo
    legend_canvas = CTkCanvas(left_frame, bg="#2E2E2E", highlightthickness=0, width=width, height=height)
    legend_canvas.place(x=0, y=0)
    legend_canvas.create_rectangle(0, 0, width, height, fill="#2E2E2E", outline="")

    # Frame de la leyenda
    legend_frame = CTkFrame(left_frame, fg_color="#2E2E2E", corner_radius=0, width=width, height=height, border_width=10, border_color="#2E2E2E")
    legend_frame.place(x=0, y=0)

    for widget in legend_frame.winfo_children():
        widget.destroy()

    # Si no se pasa color_map, cargar desde JSON
    if color_map is None:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(script_dir, "classification_colors_s.json")

            with open(json_path, "r") as f:
                color_map_raw = json.load(f)["classifications"]
                # Convertir claves a int y colores a lista de floats
                color_map = {int(k): v for k, v in color_map_raw.items()}
        except Exception as e:
            print("Error cargando clasificación desde JSON:", e)
            color_map = {}

    # Descripciones por clase LAS
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

    # Crear leyenda con los colores y etiquetas
    for class_id, rgb in color_map.items():
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )

        label = class_labels.get(class_id, f"Customized Class ({class_id})")
        count = counts.get(class_id, 0) if counts else 0

        item_frame = CTkFrame(legend_frame, fg_color="#2E2E2E")
        item_frame.pack(anchor="w", padx=10, pady=3)

        circle = CTkCanvas(item_frame, width=20, height=20, bg="#2E2E2E", highlightthickness=0)
        circle.create_oval(2, 2, 18, 18, fill=hex_color, outline=hex_color)
        circle.pack(side="left")

        text_label = CTkLabel(item_frame, text=label, text_color="#F0F0F0", font=("Arial", 12))
        text_label.pack(side="left", padx=(8, 2))

        count_label = CTkLabel(item_frame, text=f"({count})", text_color="#A0A0A0", font=("Arial", 12))
        count_label.pack(side="left")

def panel_left_frame (xcenter, ycenter, FAltcenter, las_object):
        global panel_canvas, panel_frame, height_frame, longitude_frame, progress_bar, posiciones, selected_positions

        enable_left_frame()

        botones = []
        posiciones = []
        selected_positions = []

        left_frame.update_idletasks()
        width = left_frame.winfo_width()
        height = left_frame.winfo_height()

        panel_canvas = CTkCanvas(left_frame, bg="#2E2E2E", highlightthickness=0, width=width, height=height)
        panel_canvas.place(x=0, y=0)

        # Fondo amarillo
        panel_canvas.create_rectangle(0, 0, width, height, fill="#2E2E2E", outline="")

        # Crear título (texto centrado en X, en la parte superior)
        title = "OBSTACLE DETECTION"
        panel_canvas.create_text(width//2, 40, text=title, font=("Arial", 18, "bold"), fill="white")
        panel_canvas.pack_propagate(False)

        build_seg = False

        def toggle_parameters():
            nonlocal build_seg, from_set, to_set, button_from, from_frame

            build_seg = not build_seg

            if build_seg:
                button_seg.configure(text=" ▲ Building Segmentation")
                seg_frame.pack(pady=(10, 0), fill="x")
                panel_building_frame.pack(pady=(0, 0), fill="x")
                button_from.pack_forget()
                button_from.pack(fill="x", padx=(0, 0), pady=(10, 0))
                button_to.pack_forget()
                button_to.pack(fill="x", padx=(0, 0), pady=(10, 0))

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

        # Parameters Button
        button_seg = CTkButton(panel_canvas, text=" ▼ Building Segmentation", text_color="#F0F0F0", fg_color="#3E3E3E",
                                           anchor="w", corner_radius=0, command=toggle_parameters)
        button_seg.pack(fill="x", padx=(0, 0), pady=(50, 0))

        seg_frame = CTkFrame(panel_canvas, fg_color="#2E2E2E", corner_radius=0)

        label_seg = CTkLabel(seg_frame, text="Selecciona X que formaran el edificio. Cada edificio clica OK.", text_color="white", font=("Arial", 12))
        label_seg.pack(fill="x", padx=(5, 5), pady=(0, 0))

        panel_building_frame = CTkFrame(seg_frame, fg_color="#2E2E2E", height=150, corner_radius=0)

        panel_building = CTkFrame(panel_building_frame, height=150, fg_color="white", corner_radius=10)
        panel_building.grid(row=0, column=0, rowspan=2, padx=(10, 10), pady=(10, 0), sticky="nsew")

        ok_button = CTkButton(panel_building_frame, text="OK", text_color="white", width=70, fg_color="#1E3A5F")
        ok_button.grid(row=0, column=1, padx=(0, 10), pady=(10, 5), sticky="s")

        reset_button = CTkButton(panel_building_frame, text="Reset", text_color="white", width=70, fg_color="#1E3A5F")
        reset_button.grid(row=1, column=1, padx=(0, 10), pady=(5, 10), sticky="n")

        # Configurar weights para que filas y columnas se distribuyan bien
        panel_building_frame.grid_rowconfigure(0, weight=1)
        panel_building_frame.grid_rowconfigure(1, weight=1)
        panel_building_frame.grid_columnconfigure(0, weight=1)
        panel_building_frame.grid_columnconfigure(1, weight=0)

        from_set = False

        def toggle_dose_layer_b():
            nonlocal from_set, build_seg, to_set, button_from, from_frame
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

        button_from = CTkButton(panel_canvas, text=" ▼ FROM", text_color="#F0F0F0", fg_color="#666666",
                                           anchor="w", corner_radius=0, command=toggle_dose_layer_b)
        button_from.pack(fill="x", padx=(0, 0), pady=(10, 0))

        # Dose Layer
        from_frame = CTkFrame(panel_canvas, fg_color="#2E2E2E", corner_radius=0)

        dronPos_label = CTkLabel(from_frame, text="Dron position:", text_color="white", font=("Arial", 12))
        dronPos_label.pack(padx=(5, 5), pady=(0, 0))

        panel_dronPos = CTkFrame(master=from_frame, width=300, height=200, fg_color="white", corner_radius=10)
        panel_dronPos.pack(padx=(10, 10), pady=(10, 0))

        to_set = False

        def toggle_extra_computations():
            nonlocal to_set, build_seg
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

        button_to = CTkButton(panel_canvas, text=" ▼ TO", text_color="#F0F0F0",
                                                   fg_color="#666666",
                                                   anchor="w", corner_radius=0, command=toggle_extra_computations)
        button_to.pack(fill="x", padx=(0, 0), pady=(10, 0))

        to_frame = CTkFrame(panel_canvas, fg_color="#2E2E2E", height=110, corner_radius=0)
        to_frame.grid_propagate(False)

        left_to_frame = CTkFrame(to_frame, fg_color="#999999", corner_radius=10)
        left_to_frame.grid(row=0, column=0, padx=(10, 5), pady=(10, 0))
        left_to_frame.grid_propagate(False)

        right_to_frame = CTkFrame(to_frame, fg_color="#999999", corner_radius=10)
        right_to_frame.grid(row=0, column=1, padx=(5, 10), pady=(10, 0))
        right_to_frame.grid_propagate(False)

        to_frame.grid_rowconfigure(0, weight=1)
        to_frame.grid_columnconfigure(0, weight=1)
        to_frame.grid_columnconfigure(1, weight=1)

        left_to_frame_container = CTkFrame(left_to_frame, fg_color="#999999", corner_radius=10, height=110)
        left_to_frame_container.pack(padx=(0, 0), pady=(0, 0))
        left_to_frame_container.pack_propagate(False)

        button_coordLatLng = CTkButton(left_to_frame_container, text="Geographic coordinates", text_color="#F0F0F0", fg_color="#3E3E3E",
                                       corner_radius=0, height=10)
        button_coordLatLng.pack(fill='x', padx=0, pady=0)

        lat_frame = CTkFrame(left_to_frame_container, fg_color="#999999", corner_radius=0)
        lat_frame.pack(pady=(5, 0), anchor='center')

        label_lat = CTkLabel(lat_frame, text="Lat: ", text_color="#F0F0F0", bg_color="#999999")
        label_lat.pack(side='left', padx=(0, 5))

        entry_lat = CTkEntry(lat_frame, width=50, text_color="black", state="normal", font=("Arial", 12))
        entry_lat.pack(side='left')

        lon_frame = CTkFrame(left_to_frame_container, fg_color="#999999", corner_radius=0)
        lon_frame.pack(pady=(5, 0), anchor='center')

        label_lon = CTkLabel(lon_frame, text="Lon: ", text_color="#F0F0F0", bg_color="#999999")
        label_lon.pack(side='left', padx=(0, 5))

        entry_lon = CTkEntry(lon_frame, width=50, text_color="black", state="normal", font=("Arial", 12))
        entry_lon.pack(side='left')

        right_to_frame_container = CTkFrame(right_to_frame, fg_color="#999999", corner_radius=10, height=110)
        right_to_frame_container.pack(padx=(0, 0), pady=(0, 0))
        right_to_frame_container.pack_propagate(False)

        button_coordUTM = CTkButton(right_to_frame_container, text="UTM coordinates", text_color="#F0F0F0",
                                       fg_color="#3E3E3E",
                                       corner_radius=0, height=10)
        button_coordUTM.pack(fill='x', padx=0, pady=0)

        easting_frame = CTkFrame(right_to_frame_container, fg_color="#999999", corner_radius=0)
        easting_frame.pack(pady=(5, 0), anchor='center')

        label_easting = CTkLabel(easting_frame, text="Easting: ", text_color="#F0F0F0", bg_color="#999999")
        label_easting.pack(side='left', padx=(0, 5))

        entry_easting = CTkEntry(easting_frame, width=50, text_color="black", state="normal", font=("Arial", 12))
        entry_easting.pack(side='left')

        northing_frame = CTkFrame(right_to_frame_container, fg_color="#999999", corner_radius=0)
        northing_frame.pack(pady=(5, 0), anchor='center')

        label_northing = CTkLabel(northing_frame, text="Northing: ", text_color="#F0F0F0", bg_color="#999999")
        label_northing.pack(side='left', padx=(0, 5))

        entry_northing = CTkEntry(northing_frame, width=50, text_color="black", state="normal", font=("Arial", 12))
        entry_northing.pack(side='left')

        button_return = CTkButton(left_frame, text = "Return", text_color = "#F0F0F0", fg_color = "#B71C1C",
            hover_color = "#C62828", corner_radius = 0, border_color = "#D3D3D3", border_width = 2)
        button_return.pack(side="bottom", padx=(0, 0), pady=(10, 0))

        button_visualize = CTkButton(left_frame, text="Visualize", text_color="#F0F0F0", fg_color="#1E3A5F",
            hover_color="#2E4A7F", corner_radius=0, border_color="#D3D3D3", border_width=2)
        button_visualize.pack(side="bottom", padx=(0, 0), pady=(5, 0))

        # Frame
        #panel_frame = CTkFrame(master=panel_canvas, width=300, height=300, fg_color="white", corner_radius=10)
        #panel_canvas.create_window(width // 2, height//2 - 100, window=panel_frame)

        ## Latitude Frame
        #latitude_frame = CTkFrame(panel_canvas, fg_color="#2E2E2E")
        #latitude_frame.place(relx=0.5, rely=0.65, anchor="n")  # Ajusta la posición según sea necesario
        #label_latitude = CTkLabel(latitude_frame, text="Latitude:", text_color="white", font=("Arial", 12))
        #label_latitude.pack(side="left", padx=(0, 5))
        #entry_latitude = CTkEntry(latitude_frame, width=50, font=("Arial", 12))
        #entry_latitude.pack(side="left")

        ## Longitude Frame
        #longitude_frame = CTkFrame(panel_canvas, fg_color="#2E2E2E")
        #longitude_frame.place(relx=0.5, rely=0.70, anchor="n")  # Ajusta la posición según sea necesario
        #label_longitude = CTkLabel(longitude_frame, text="Longitude:", text_color="white", font=("Arial", 12))
        #label_longitude.pack(side="left", padx=(0, 5))
        #entry_longitude = CTkEntry(longitude_frame, width=50, font=("Arial", 12))
        #entry_longitude.pack(side="left")

        #visualize_btn = CTkButton(panel_canvas, text="Visualize", text_color="#F0F0F0", fg_color="#1E3A5F",
          #hover_color="#2E4A7F", corner_radius=0, border_color="#D3D3D3", border_width=2, command=lambda: tree_obstacles(las_object, entry_latitude, entry_longitude, combined_mesh, num_pixels_x, num_pixels_y, delta_x, delta_y, cell_stats))
        #visualize_btn.place(relx=0.5, rely=0.80, anchor="n")

        #return_btn = CTkButton(panel_canvas, text="Return", text_color="#F0F0F0", fg_color="#B71C1C",
                                  #hover_color="#C62828", corner_radius=0, border_color="#D3D3D3", border_width=2, command=btn_return)
        #return_btn.place(relx=0.5, rely=0.85, anchor="n")

        # Normalizar coordenadas
        x_array = np.array(xcenter)
        y_array = np.array(ycenter)

        x_min, x_max = x_array.min(), x_array.max()
        y_min, y_max = y_array.min(), y_array.max()

        # Centro geométrico
        x_center_geom = (x_min + x_max) / 2
        y_center_geom = (y_min + y_max) / 2

        def escalar(val, min_val, max_val, new_min, new_max):
            return new_min + (val - min_val) / (max_val - min_val) * (new_max - new_min)

        for i in range(len(xcenter)):
            # Rotar los puntos 180° alrededor del centro
            x_rotado = 2 * x_center_geom - xcenter[i]
            y_rotado = 2 * y_center_geom - ycenter[i]

            # Reflejar horizontalmente (invertir en eje X)
            x_rotado = 2 * x_center_geom - x_rotado

            # Escalar después de rotar
            x = escalar(x_rotado, x_min, x_max, 10, 290)
            y = escalar(y_rotado, y_min, y_max, 10, 290)

            posiciones.append((xcenter[i], ycenter[i], FAltcenter[i]))
            print({posiciones[-1]})

            #btn = CTkButton(panel_frame, text="", width=6, height=6,
                            #fg_color="blue", hover_color="darkblue", corner_radius=3,
                            #command=lambda b=i: toggle_color(botones[b], b))
            #btn.place(x=x, y=y, anchor="center")
            #botones.append(btn)

def btn_return():
    if 'panel_canvas' in globals() and panel_canvas.winfo_exists():
        panel_canvas.place_forget()

    if 'panel_frame' in globals() and panel_frame.winfo_exists():
        panel_frame.place_forget()

    if 'height_frame' in globals() and height_frame.winfo_exists():
        height_frame.place_forget()

    if 'longitude_frame' in globals() and longitude_frame.winfo_exists():
        longitude_frame.place_forget()

def toggle_color(boton, index):
    global selected_positions, posiciones, botones

    x, y, alt = posiciones[index]

    if boton.cget("fg_color") == "blue":
        if not selected_positions:
            # No hay selección actual → permitir marcar
            boton.configure(fg_color="pink", hover_color="#ff69b4")
            selected_positions.append((x, y, alt))
            print(f"Seleccionado: ({x}, {y}, {alt})")
        else:
            print("Ya hay una posición seleccionada. Deselecciona antes de elegir otra.")
    else:
        # El botón ya estaba seleccionado → desmarcar
        boton.configure(fg_color="blue", hover_color="darkblue")
        selected_positions = [pos for pos in selected_positions if pos != (x, y, alt)]
        print(f"Eliminado: ({x}, {y}, {alt})")

def latlon_a_utm31(lat, lon):
    # Define projections
    wgs84 = Proj(proj='latlong', datum='WGS84')
    utm31 = Proj(proj='utm', zone=31, datum='WGS84', units='m')

    # Transform coordinates
    x, y = transform(wgs84, utm31, lon, lat)
    return x, y

def mostrar_resumen_lineas(colores_puntos, total_arboles_cruzados):
    def ventana():
        resumen = CTkToplevel()
        resumen.title("Resumen de líneas")
        resumen.geometry("300x400")
        resumen.configure(fg_color="#1E1E1E")

        scroll = CTkScrollableFrame(resumen, fg_color="#1E1E1E")
        scroll.pack(expand=True, fill="both", padx=10, pady=10)

        for idx, (color, num_arboles) in enumerate(zip(colores_puntos, total_arboles_cruzados)):
            item_frame = CTkFrame(scroll, fg_color="#1E1E1E")
            item_frame.pack(anchor="w", padx=10, pady=5)

            # Convertir color float a HEX
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(color[0]*255), int(color[1]*255), int(color[2]*255)
            )

            circle = CTkCanvas(item_frame, width=20, height=20, bg="#1E1E1E", highlightthickness=0)
            circle.create_oval(2, 2, 18, 18, fill=hex_color, outline=hex_color)
            circle.pack(side="left")

            label = CTkLabel(item_frame, text=f"{num_arboles} trees",
                                 text_color="#F0F0F0", font=("Arial", 12))
            label.pack(side="left", padx=8)

    threading.Thread(target=ventana, daemon=True).start()

# Crear la barra de progreso
def create_progress_bar():
    progress_bar = ttk.Progressbar(right_frame, orient="horizontal", length=300, mode="determinate", style="TProgressbar")
    right_frame.grid_rowconfigure(0, weight=1)
    right_frame.grid_rowconfigure(1, weight=1)
    right_frame.grid_rowconfigure(2, weight=1)
    right_frame.grid_columnconfigure(0, weight=1)
    progress_bar.grid(row=1, column=0, pady=10)
    return progress_bar

# Función para actualizar la barra de progreso
def update_progress_bar(progress_bar, value):
    progress_bar['maximum'] = 100
    progress_bar['value'] = value
    root.update_idletasks()

def load_point_cloud():
    global pc_filepath
    root.point_size_entry.delete(0, "end")
    filepath = filedialog.askopenfilename(filetypes=[("PCD Files", "*.pcd")])
    if filepath:
        pc_filepath = filepath
        print("Point Cloud Selected:", pc_filepath)
        root.point_size_entry.configure(state="normal")
        root.point_size_entry.insert(0, 2)
        root.voxelizer_switch.configure(state="normal")

def load_xml_metadata():
    global xml_filepath
    xml_filepath = filedialog.askopenfilename(filetypes=[("XML Files", "*.xml")])
    if xml_filepath:
        print("XML Selected:", xml_filepath)

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

    # cosmic dose measured by the instrument. This should be determined in a lake platform or in the sea after the internal background has already beenn determined
    # current values are obtained in Banyoles lake
    Dose_cosmic = 0.0
    H10_cosmic = 0.0

    # Influence of radon progeny. This values are really complicated. In outdoors, an estimation of 2 nSv/h per 10 Bq/m3 could be used (Vargas, Cornejo and Camp. 2017)
    Dose_Radon = 0.0
    H10_Radon = 0.0

    # low_ROI_counts/high_ROI_counts when no artifical source is present
    R = 13.5  # SiPM 50 mm

    pose = 0

    # define lists with none values with a maximum of 4096 bin in each spectrum
    En_ch = [None] * 4096
    Conv_coeff = [None] * 4096
    F = [None] * 4096

    x = [None] * 100000
    y = [None] * 100000

    xcenter = [None] * 100000
    ycenter = [None] * 100000

    intx = [None] * 100000
    inty = [None] * 100000
    Hmax = [None] * 100000
    Hcenter = [None] * 100000

    FAltcenter = [None] * 100000

    # Total_LTime = 0

    sys.path.insert(0, pathN42)

    os.chdir(
        pathN42)  # Change according to where  are the *.42 files for calculations, i.e., just rebinned or rebinned and summed
    listOfFiles = os.listdir(pathN42)

    # If the program is in the same directory than the data .n42 uncomment the following line and comment lines 80-82
    # listOfFiles = os.listdir()

    f_name = fnmatch.filter(listOfFiles, '*.n42')

    # print head of output.dat file
    # print('Meas_number ', 'Dose_(nGy/h) ', 'H*(10)_nSv/h ', 'H*(10)_1m_(nSv/h) ', 'MMGC ', 'uMMGC ')

    # loop for each *.n42 spectrum
    cont = 0
    for idx, file in enumerate(f_name):
        cont = cont + 1
        os.chdir(pathN42)
        f = open(file, "r")
        tree = ET.parse(file)
        roots = tree.getroot()

        # Read Start Date Time, LiveTime, DeadTime, ChannelData
        for each in roots.findall('.//RadMeasurement'):
            # read LiveTime
            rating = each.find('.//LiveTimeDuration')
            LiveTime = rating.text
            # print('LiveTime = ',LiveTime)

            # Find substring and convert to float
            LTime = LiveTime[LiveTime.find("T") + 1:LiveTime.find("S")]
            FLTime = float(LTime)

            # FLTime=float(LiveTime)
            # print('FLTime = ',FLTime)
            LTime = FLTime

            # Read counts in each energy bin
            rating = each.find('.//ChannelData')
            ChannelData = rating.text
            # print('ChannelData = ',ChannelData)

            # Convert string of counts in a list of integers
            Split_channels = ChannelData.split()

            # print('Split_channels: ', Split_channels)
            icounts = list(map(float, Split_channels))

            # The channel index starts with 0 up to n_channels-1
            n_channels = len(icounts)
        #         print('number of channels = ',n_channels)

        # Read Energy calibration
        for each in roots.findall('.//EnergyCalibration'):
            rating = each.find('.//CoefficientValues')
            Ecal = rating.text
            #    print('energy_CoefficientValues = ',Ecal)
            # Convert string of counts in a list of integers
            Split_coeff = Ecal.split()
            float_coeff = list(map(float, Split_coeff))
        # The first coefficient is 0
        #         print('Energy_coeff = ',float_coeff[0],float_coeff[1],float_coeff[2])
        #         float_coeff[1]=float_coeff[1]*1461./1480.

        # Read altitude a.g.l.
        for each in roots.findall('.//GeographicPoint'):
            rating = each.find('.//ElevationValue')
            Altitude = rating.text
            #         print('Altitude = ',Altitude)
            FAltitude = float(Altitude)

            rating = each.find('.//LongitudeValue')
            Longitude = rating.text
            #         print('Longitud = ',Longitud)
            FLongitude = float(Longitude)

            rating = each.find('.//LatitudeValue')
            Latitude = rating.text
            #         print('Latitude = ',Latitude)
            FLatitude = float(Latitude)

        # Calculation of absorbed Dose and H10 using band method function
        Dose_conv_meas = 0
        H10_conv_meas = 0
        low_ROI = 0
        high_ROI = 0

        for i in range(0, n_channels):
            En_ch[i] = float_coeff[0] + float_coeff[1] * (i + 1)  # HAT QYE VERIFUCAR SI ES i+1 o i
            #         print ('energia canal ',i,' es: ',En_ch[i])
            #         print ('cuentas canal ',i,' son: ',int_channel[i])

            # Calculate Man Made Gross Count MMGC
            if ((En_ch[i] > 200) and (En_ch[i] <= 1340)):
                low_ROI = low_ROI + int(icounts[i])
                # print('low_ROI: ',low_ROI)
            if ((En_ch[i] > 1340) and (En_ch[i] <= 2980)):
                high_ROI = high_ROI + int(icounts[i])
                # print('high_ROI: ',high_ROI)

            # Calculate de conversion coefficent for the energy nGy/h per cps WITH capsule and total surface 30 keV   50 mmm
            Conv_coeff[i] = 0
            if ((En_ch[i] >= 30) and (En_ch[i] <= 55)):
                Conv_coeff[i] = 5206.355632 * En_ch[i] ** (-2.853969336)

            if ((En_ch[i] > 55) and (En_ch[i] <= 350)):
                Conv_coeff[i] = -4.43804048E-13 * En_ch[i] ** 5 + 4.852144251E-10 * En_ch[
                    i] ** 4 - 1.997841663E-07 * En_ch[i] ** 3 + 4.123346655E-05 * En_ch[i] ** 2 - 0.0035372218034 * \
                                En_ch[i] + 0.1532763827
            if ((En_ch[i] > 350) and (En_ch[i] <= 3000)):
                #             Conv_coeff[i]=7.994448817E-20*En_ch[i]**6-8.688859196E-16*En_ch[i]**5+3.673134977E-12*En_ch[i]**4-7.579115501E-09*En_ch[i]**3+7.387567866E-06*En_ch[i]**2-0.0006714962472*En_ch[i]-0.0454215177
                Conv_coeff[i] = -4.434768244E-07 * En_ch[i] ** 2 + 0.003033563589 * En_ch[i] - 0.6528052087
                # print ('E = ',En_ch[i], 'w = ',Conv_coeff[i])

            # Calculate the conversion coefcient Gy --> Sv
            F[i] = 1
            if (En_ch[i] >= 3):
                a1 = math.log(En_ch[i] / 9.85)
                F[i] = a1 / (1.465 * a1 ** 2 - 4.414 * a1 + 4.789) + 0.7003 * math.atan(0.6519 * a1)

            Dose_conv_meas = Dose_conv_meas + icounts[i] / FLTime * Conv_coeff[i]
            H10_conv_meas = H10_conv_meas + (icounts[i] / FLTime * Conv_coeff[i]) * F[i]

            # print('Energy channel: ',En_ch[i], 'wi: ',Conv_coeff[i],'counts: ',icounts[i],'Live_time: ',FLTime, 'Count rate: ',icounts[i]/FLTime,'Dose rate contribution: ',icounts[i]/FLTime*Conv_coeff[i])
            # print('accumulatgerd dose rate: ',Dose_conv_meas)

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
        # print(cont, file, Dose_conv_meas, H10_conv_meas, H10_conv_1m, MMGC, u_MMGC)

        for dose in roots.iter('DoseRateValue'):
            # giving the value.
            dose.text = str(Dose_conv_meas)
            # dose.set('unit','nGy/h')

        for Ader in roots.iter('AmbientDoseEquivalentRateValue'):
            # giving the value.
            Ader.text = str(H10_conv_meas)
        #       Ader.set('unit','nSv/h')

        for Ader1m in roots.iter('AmbientDoseEquivalentRateValue_1m'):
            # giving the value.
            Ader1m.text = str(H10_conv_1m)
        #       dose.set('unit','nSv/h')

        for Man_Made in roots.iter('MMGC'):
            # giving the value.
            Man_Made.text = str(MMGC)

        for uMan_Made in roots.iter('uncertainty_MMGC'):
            # giving the value.
            uMan_Made.text = str(u_MMGC)

        # tranform x,y
        x0, y0, zone_number, zone_letter = utm.from_latlon(FLatitude, FLongitude, )
        # print('center projection in utm (meters): ', x0, y0)

        xcenter[cont] = x0
        ycenter[cont] = y0
        Hcenter[cont] = H10_conv_meas
        FAltcenter[cont] = FAltitude

        if cont == 1:
            latmin = y0
            latmax = y0
            lonmin = x0
            lonmax = x0

        # looking for max and min lat l

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

    # Verifica si hay NaN en los datos
    # print(type(xcenter))
    # print(type(ycenter))
    # print(type(Hcenter))

    xcenter = np.array(xcenter, dtype=float)
    ycenter = np.array(ycenter, dtype=float)
    Hcenter = np.array(Hcenter, dtype=float)

    # conversion to string and numbers to floats
    xcenter = np.array([float(i) for i in xcenter if str(i).replace('.', '', 1).isdigit()])
    ycenter = np.array([float(i) for i in ycenter if str(i).replace('.', '', 1).isdigit()])
    Hcenter = np.array([float(i) for i in Hcenter if str(i).replace('.', '', 1).isdigit()])
    FAltcenter = np.array([float(i) for i in FAltcenter if str(i).replace('.', '', 1).isdigit()])

    # print('latmin, latmax,lonmin, lonmax: ', latmin, latmax, lonmin, lonmax)
    # print('minx,maxx,miny,maxy', min(xcenter), max(xcenter), min(ycenter), max(ycenter))
    # print('minAlt,maxAlt: ', min(FAltcenter), max(FAltcenter))
    # print('minH*(10),maxH*(10): ', min(Hcenter), max(Hcenter))

    # print(np.isnan(xcenter).any())  # True si hay algún NaN en x
    # print(np.isnan(ycenter).any())  # True si hay algún NaN en y
    # print(np.isnan(Hcenter).any())  # True si hay algún NaN en Hmax

    # print(len(xcenter), len(ycenter), len(Hcenter))
    # print(xcenter.shape, ycenter.shape, Hcenter.shape)

    # Encuentra el valor máximo en Hcenter
    max_value = max(Hcenter)

    # Encuentra el índice correspondiente al valor máximo
    # cont_max = Hcenter.index(max_value)
    cont_max = np.argmax(Hcenter)

    # Calculo del maximo valor de H*(10) suponiento que es firnte puntual
    HmaxP = Hcenter[cont_max] * FAltcenter[cont_max] * FAltcenter[cont_max]

    # Define una cuadrícula para el área de interés
    Resolution = 50
    ygrid = np.linspace(latmin, latmax, Resolution)
    xgrid = np.linspace(lonmin, lonmax, Resolution)
    xmesh, ymesh = np.meshgrid(xgrid, ygrid)

    # Inicializar el mapa con valores muy bajos
    heatmap = np.full(xmesh.shape, -np.inf)

    # Iterar sobre cada circunferencia
    for xc, yc, radius, hval in zip(xcenter, ycenter, FAltcenter, Hcenter):
        # Distancia de cada punto de la cuadrícula al centro de la circunferencia
        distance = np.sqrt((xmesh - xc) ** 2 + (ymesh - yc) ** 2)
        # Máscara para identificar puntos dentro del círculo
        mask = distance <= radius
        # Actualizar el valor máximo en el mapa
        heatmap[mask] = np.maximum(heatmap[mask], hval)

    # Configurar los valores mínimos para que sean visibles (si es necesario)
    heatmap[heatmap == -np.inf] = np.nan

    # Write to CSV
    output_filename = "dose_data_pla_20m_2ms.csv"
    csv_filepath = output_filename
    with open(output_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Latitude", "Longitude", "Dose"])  # Header row

        # Flatten the arrays for iteration
        for i in range(xmesh.shape[0]):
            for j in range(xmesh.shape[1]):
                writer.writerow([xmesh[i, j], ymesh[i, j], heatmap[i, j]])

    # print('------------------------------------', '\n')
    # print('Total number of analysed spectra : ', cont, '\n')

    # Procesar el CSV existente
    dosis_values = np.genfromtxt(csv_filepath, delimiter=',', skip_header=1, usecols=2)
    dosis_values = dosis_values[~np.isnan(dosis_values)]  # Eliminar NaN
    dose_min_csv, dose_max_csv = np.min(dosis_values), np.max(dosis_values)
    # print(f"Dosis Range: Min={self.dose_min_csv}, Max={self.dose_max_csv}")

    # Asignar valores a los campos de Min y Max y deshabilitarlos
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

    # print('****END PROGRAM *****')

def get_dose_color(dosis_nube, high_dose_rgb, medium_dose_rgb, low_dose_rgb, dose_min_csv, low_max,
                   medium_min, medium_max, high_min):
    """ Asigna colores a los puntos según la dosis usando los valores actualizados. """

    colores_dosis = np.zeros((len(dosis_nube), 3))
    colores_dosis[(dosis_nube >= dose_min_csv) & (dosis_nube < low_max)] = low_dose_rgb
    colores_dosis[(dosis_nube >= medium_min) & (dosis_nube < medium_max)] = medium_dose_rgb
    colores_dosis[dosis_nube >= high_min] = high_dose_rgb

    return colores_dosis

def get_origin_from_xml(xml_filepath):
    """Extrae el origen georeferenciado del archivo metadatos.xml."""
    try:
        tree = ET.parse(xml_filepath)
        roots = tree.getroot()
        srs_origin = roots.find("SRSOrigin")

        if srs_origin is None or not srs_origin.text:
            print("Error: No se encontró la etiqueta <SRSOrigin> en el XML.")
            return None

        return np.array([float(coord) for coord in srs_origin.text.split(",")])

    except Exception as e:
        print(f"Error leyendo el archivo XML: {e}")
        return None

def toggle_voxel_size():
    global previous_point_value, previous_voxel_value, previous_downsample_value
    if root.voxelizer_var.get():
        previous_point_value = root.point_size_entry.get()
        previous_downsample_value = root.downsample_entry.get()
        root.vox_size_entry.delete(0, "end")
        root.point_size_entry.delete(0, "end")
        root.downsample_entry.delete(0, "end")
        root.point_size_entry.configure(state="disabled")
        root.downsample_entry.configure(state="disabled")
        root.vox_size_entry.configure(state="normal")
        if previous_voxel_value == "":
            root.vox_size_entry.insert(0, 2)
        else:
            root.vox_size_entry.insert(0, previous_voxel_value)
    else:
        previous_voxel_value = root.vox_size_entry.get()
        root.vox_size_entry.delete(0, "end")
        root.point_size_entry.delete(0, "end")
        root.vox_size_entry.configure(state="disabled")
        root.point_size_entry.configure(state="normal")
        root.downsample_entry.configure(state="normal")
        if previous_point_value == "":
            root.point_size_entry.insert(0, 2)
        else:
            root.point_size_entry.insert(0, previous_point_value)
        if previous_downsample_value != "":
            root.downsample_entry.insert(0, previous_downsample_value)

def toggle_dose_layer(source_location):
    global show_dose_layer
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
        if not root.low_dose_cb.get():
            root.low_dose_cb.set("green")
            root.high_dose_cb.set("red")
            root.medium_dose_cb.set("yellow")
        if source_location is not None:
            root.show_source_switch.configure(state="normal")
    else:
        show_dose_layer = False
        root.low_dose_max.configure(state="disabled")
        root.medium_dose_min.configure(state="disabled")
        root.medium_dose_max.configure(state="disabled")
        root.high_dose_min.configure(state="disabled")
        root.low_dose_cb.configure(state="disabled")
        root.medium_dose_cb.configure(state="disabled")
        root.high_dose_cb.configure(state="disabled")

def find_radioactive_source(csv_filepath):
    global source_location
    if not csv_filepath:
        messagebox.showwarning("Warning", "Please select a N42 file.")
        return

    utm_coords = np.genfromtxt(csv_filepath, delimiter=',', skip_header=1)
    # Filter out rows with NaN values in the dose column
    utm_coords = utm_coords[~np.isnan(utm_coords[:, 2])]
    ga = GeneticAlgorithm(utm_coords)
    source_location = ga.run()
    source_location = source_location
    print(f"Estimated source location: Easting = {source_location[0]}, Northing = {source_location[1]}")
    messagebox.showinfo("Source Location",
                        f"Estimated source location: Easting = {source_location[0]}, Northing = {source_location[1]}")
    root.show_source_switch.configure(state="normal")

def toggle_source():
    global show_source
    if root.show_source_switch.get() == 1:
        show_source = True
    else:
        show_source = False

def validate_dose_ranges(show_dose_layer, dose_min_csv, dose_max_csv):
    global low_max, medium_min, medium_max, high_min
    """
    Validates that the dose ranges have logical values.
    """
    if not show_dose_layer:
        return

    try:
        low_max = float(root.low_dose_max.get())
        medium_min = float(root.medium_dose_min.get())
        medium_max = float(root.medium_dose_max.get())
        high_min = float(root.high_dose_min.get())
    except ValueError:
        messagebox.showerror("Error", "Dose range values must be numeric.")
        raise ValueError("Dose range values must be numeric.")

    if not (dose_min_csv <= low_max <= medium_min <= medium_max <= high_min <= dose_max_csv):
        messagebox.showerror("Error","Dose ranges are not logical. Ensure: min < low_max < medium_min < medium_max < high_min < max.")
        raise ValueError("Dose ranges are not logical.")

def plot_heatmap(heatmap, xcenter, ycenter, Hcenter, lonmin, lonmax, latmin, latmax):
    if heatmap is None or xcenter is None or ycenter is None or Hcenter is None:
        messagebox.showerror("Error", "Please process the N42 files first.")
        return

    disable_left_frame()

    fig, ax = plt.subplots()

    cax = ax.imshow(
        heatmap,
        extent=(lonmin, lonmax, latmin, latmax),
        origin='lower',
        cmap='viridis',
        alpha=0.8
    )

    fig.colorbar(cax, label='H*(10) rate nSv/h', ax=ax)

    ax.set_title('Heatmap H*(10) rate')
    ax.set_xlabel('LONGITUDE')
    ax.set_ylabel('LATITUDE')

    scatter = ax.scatter(
        xcenter, ycenter,
        c=Hcenter, cmap='viridis',
        edgecolor='black', s=50, label='Measurement'
    )

    ax.grid(visible=True, color='black', linestyle='--', linewidth=0.5)
    ax.legend()

    # Conectar el evento de cierre
    def on_close(event):
        enable_left_frame()

    fig.canvas.mpl_connect('close_event', on_close)

    plt.show()

def plot_three_color_heatmap(heatmap, xcenter, ycenter, Hcenter, lonmin, lonmax, latmin, latmax):
    if heatmap is None or xcenter is None or ycenter is None or Hcenter is None:
        messagebox.showerror("Error", "Please process the N42 files first.")
        return

    disable_left_frame()  # Desactiva la interfaz al comenzar

    if root.dose_layer_switch.get() == 1:
        low_dose_color = root.low_dose_cb.get() if root.low_dose_cb.get() else 'green'
        medium_dose_color = root.medium_dose_cb.get() if root.medium_dose_cb.get() else 'yellow'
        high_dose_color = root.high_dose_cb.get() if root.high_dose_cb.get() else 'red'

        colors = [low_dose_color, medium_dose_color, high_dose_color]

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
        colors = ['green', 'yellow', 'red']
        R0, R1, R2 = 0, 80, 120

    R3 = max(Hcenter) * max(Hcenter)
    bounds = [R0, R1, R2, R3]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    # Crear figura para conectar evento de cierre
    fig, ax = plt.subplots()

    im = ax.imshow(
        heatmap,
        extent=(lonmin, lonmax, latmin, latmax),
        origin='lower',
        cmap=cmap,
        norm=norm,
        alpha=0.8
    )

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

    ax.set_title('Heatmap with Three Color Range')
    ax.set_xlabel('LONGITUDE')
    ax.set_ylabel('LATITUDE')

    ax.scatter(
        xcenter, ycenter,
        c=Hcenter, cmap=cmap, norm=norm,
        edgecolor='black', s=50, label='Measurement'
    )

    ax.grid(visible=True, color='black', linestyle='--', linewidth=0.5)
    ax.legend()

    def on_close(event):
        enable_left_frame()

    fig.canvas.mpl_connect('close_event', on_close)

    plt.show()

def set_run_prueba_flag(xcenter, ycenter, FAltcenter):
    # Check if xcenter or ycenter is None or empty
    global las_object, mi_set

    if xcenter is None or len(xcenter) == 0 or ycenter is None or len(ycenter) == 0:
        messagebox.showerror("Error", "Please process the N42 files first.")
        return

    if las_object is None:
        mi_set = False
        segmentationPlus(mi_set)

    else:
        if combined_mesh is None:
            grid(las_object)
        else:
            panel_left_frame(xcenter, ycenter, FAltcenter, las_object)

def set_trees():
    global mi_set
    mi_set = True
    segmentationPlus(mi_set)

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

    # Crear y mostrar la barra de progreso
    progress_bar = create_progress_bar()

    # Actualizar la barra de progreso
    update_progress_bar(progress_bar, 1)

    use_voxelization = root.voxelizer_switch.get() == 1

    point_size_str = root.point_size_entry.get().strip()
    vox_size_str = root.vox_size_entry.get().strip()
    altura_extra = root.dosis_slider.get()

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

    if root.downsample_entry.get().strip():
        downsample = float(root.downsample_entry.get().strip())
    else:
        downsample = None

    # Actualizar la barra de progreso
    update_progress_bar(progress_bar, 10)

    if use_voxelization:
        mostrar_nube_si_vox(show_dose_layer, pc_filepath, xml_filepath, csv_filepath, high_dose_rgb, medium_dose_rgb, low_dose_rgb, dose_min_csv, low_max,
            medium_min, medium_max, high_min, altura_extra, progress_bar)
    else:
        mostrar_nube_no_vox(show_dose_layer, pc_filepath, downsample, xml_filepath, csv_filepath, high_dose_rgb, medium_dose_rgb,
            low_dose_rgb, dose_min_csv, low_max, medium_min, medium_max, high_min, altura_extra, show_source, source_location, point_size, progress_bar)


def segmentation():
    def run():
        fp = filedialog.askopenfilename(filetypes=[("LAS Files", "*.las")])
        if fp:
            print("Point Cloud Selected:", fp)

            disable_left_frame()

            # Crear y mostrar la barra de progreso
            progress_bar = create_progress_bar()

            # Actualizar la barra de progreso
            update_progress_bar(progress_bar, 1)

            # Read the LAS file
            las = laspy.read(fp)

            # Extract points and classifications
            points = np.vstack((las.x, las.y, las.z)).transpose()
            classifications = np.array(las.classification)

            # Actualizar la barra de progreso
            update_progress_bar(progress_bar, 10)

            # Conteo de cada clasificación
            counts = dict(Counter(classifications))

            # Actualizar la barra de progreso
            update_progress_bar(progress_bar, 20)

            script_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(script_dir, "classification_colors_s.json")

            with open(json_path, "r") as f:
                color_map = json.load(f)["classifications"]

            # Convertir claves a int y valores a np.array
            color_map = {int(k): np.array(v) for k, v in color_map.items()}

            unique_classes = np.unique(classifications)
            for cls in unique_classes:
                if cls not in color_map:
                    color_map[cls] = np.random.rand(3)
                    print(f"Clase adicional detectada: {cls}, color asignado: {color_map[cls]}")

            # Actualizar la barra de progreso
            update_progress_bar(progress_bar, 40)

            # Create an Open3D PointCloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Actualizar la barra de progreso
            update_progress_bar(progress_bar, 50)

            colors = np.zeros((points.shape[0], 3))

            # Actualizar la barra de progreso
            update_progress_bar(progress_bar, 60)

            for classification, color in color_map.items():
                colors[classifications == classification] = color

            # Actualizar la barra de progreso
            update_progress_bar(progress_bar, 80)

            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Actualizar la barra de progreso
            update_progress_bar(progress_bar, 100)

            # Eliminar la barra de progreso
            progress_bar.grid_forget()

            legend_left_frame(counts, color_map)

            # Visualize the point cloud
            vis = o3d.visualization.Visualizer()

            # Obtener las dimensiones del right_frame
            right_frame.update_idletasks()
            right_frame_width = right_frame.winfo_width()
            right_frame_height = right_frame.winfo_height()

            # Obtener las dimensiones del left_frame
            left_frame.update_idletasks()
            left_frame_width = left_frame.winfo_width()

            # Calcular tittle bar
            title_bar_height = ctypes.windll.user32.GetSystemMetrics(4)

            vis.create_window(window_name='Open3D', width=right_frame_width, height=right_frame_height,
                              left=left_frame_width, top=title_bar_height)

            vis.clear_geometries()
            vis.add_geometry(pcd)

            while True:
                vis.poll_events()
                vis.update_renderer()

                if not vis.poll_events():
                    print("Ventana Cerrada")
                    #Elimina el legend
                    if 'legend_frame' in globals() and legend_frame.winfo_exists():
                        legend_frame.place_forget()

                    if 'legend_canvas' in globals() and legend_canvas.winfo_exists():
                        legend_canvas.place_forget()

                    enable_left_frame()
                    break

            # Verificar si la nube de puntos tiene atributos
            if not pcd.has_points():
                print("La nube de puntos no tiene puntos.")
                return

        else:
            print("No file selected.")

    threading.Thread(target=run, daemon=True).start()

def segmentationPlus(mi_set):
    def run():
        global las_object, progress_bar, classificationtree, labels_clean, chm, rows, cols, pcd, xmin, ymax, resolution

        if las_object is None:
            fp = filedialog.askopenfilename(filetypes=[("LAS Files", "*.las")])
            if fp:
                print("Point Cloud Selected:", fp)

                disable_left_frame()

                # Crear y mostrar la barra de progreso
                progress_bar = create_progress_bar()

                # Actualizar la barra de progreso
                update_progress_bar(progress_bar, 1)

                # Read the LAS file
                las = laspy.read(fp)

                # Extract points and classifications
                points = np.vstack((las.x, las.y, las.z)).transpose()
                classifications = np.array(las.classification)

                # Actualizar la barra de progreso
                update_progress_bar(progress_bar, 5)

                # Conteo de cada clasificación
                counts = dict(Counter(classifications))

                script_dir = os.path.dirname(os.path.abspath(__file__))
                json_path = os.path.join(script_dir, "classification_colors_sp.json")

                with open(json_path, "r") as f:
                    color_map = json.load(f)["classifications"]

                # Convertir claves a int y valores a np.array
                color_map = {int(k): np.array(v) for k, v in color_map.items()}

                unique_classes = np.unique(classifications)
                for cls in unique_classes:
                    if cls not in color_map:
                        gray = np.random.uniform(0.3, 0.8)
                        color_map[cls] = [gray, gray, gray]
                        print(f"Clase adicional detectada: {cls}, color asignado: {color_map[cls]}")

                # Actualizar la barra de progreso
                update_progress_bar(progress_bar, 10)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)

                colors = np.zeros((points.shape[0], 3))
                for classification, color in color_map.items():
                    colors[classifications == classification] = color

                # Actualizar la barra de progreso
                update_progress_bar(progress_bar, 15)

                pcd.colors = o3d.utility.Vector3dVector(colors)

                # === Filtrar clasificación 4 (Medium Vegetation) ===
                medium_veg_points = points[classifications == 4]
                if medium_veg_points.shape[0] == 0:
                    print("No hay puntos con clasificación 4 en esta nube.")
                    return

                # Actualizar la barra de progreso
                update_progress_bar(progress_bar, 20)

                # === Filtrar por altura mínima ===
                min_medium_veg_height = np.min(medium_veg_points[:, 2])
                max_medium_veg_height = np.max(medium_veg_points[:, 2])

                print(f"Altura mínima de la medium vegetation: {min_medium_veg_height}")
                print(f"Altura max de la medium vegetation: {max_medium_veg_height}")

                medium_veg_points = medium_veg_points[medium_veg_points[:, 2] >= min_medium_veg_height + 2.3]

                # Actualizar la barra de progreso
                update_progress_bar(progress_bar, 25)

                # Crear un Canopy Height Model (CHM) rasterizado
                resolution = 0.55  # tamaño de celda en metros
                xmin, ymin = medium_veg_points[:, 0].min(), medium_veg_points[:, 1].min() #Obtiene el área de la nube.
                xmax, ymax = medium_veg_points[:, 0].max(), medium_veg_points[:, 1].max()

                # Actualizar la barra de progreso
                update_progress_bar(progress_bar, 30)

                cols = int(np.ceil((xmax - xmin) / resolution))
                rows = int(np.ceil((ymax - ymin) / resolution))
                chm = np.full((rows, cols), -999.0) #Crea la matriz CHM vacía.

                for x, y, z in medium_veg_points:
                    col = int((x - xmin) / resolution)
                    row = int((ymax - y) / resolution)  # Y invertido
                    if 0 <= row < rows and 0 <= col < cols:
                        if z > chm[row, col]:   #Llena el CHM con la altura máxima en cada celda.
                            chm[row, col] = z

                # Actualizar la barra de progreso
                update_progress_bar(progress_bar, 40)

                chm[chm == -999.0] = np.nan  # Celda vacía = NaN
                chm_smooth = np.nan_to_num(chm)
                chm_smooth = gaussian_filter(chm_smooth, sigma=2) #Elimina ruido y micro-picos para mejorar la detección.

                # Detectar máximos locales: posibles copas de árboles
                coordinates = peak_local_max(chm_smooth, min_distance=2, exclude_border=False)

                # Actualizar la barra de progreso
                update_progress_bar(progress_bar, 50)

                # Crear marcadores para Watershed
                markers = np.zeros_like(chm_smooth, dtype=int)
                for i, (r, c) in enumerate(coordinates, 1): #Cada máximo local recibe un número.
                    markers[r, c] = i

                # Actualizar la barra de progreso
                update_progress_bar(progress_bar, 60)

                # Aplicar Watershed sobre CHM invertido
                elevation = -chm_smooth #Invierte la altura (Watershed trabaja buscando valles)
                labels = watershed(elevation, markers, mask=~np.isnan(chm)) #Segmenta el CHM en árboles. Asigna cada celda a un árbol específico.

                # Conteo de píxeles por árbol
                label_sizes = ndi.sum(~np.isnan(chm), labels, index=np.arange(1, labels.max() + 1))

                # Define un mínimo de celdas, por ejemplo 20 celdas
                min_size = 20
                mask = np.zeros_like(labels, dtype=bool)

                # Actualizar la barra de progreso
                update_progress_bar(progress_bar, 70)

                for i, size in enumerate(label_sizes, 1):
                  if size >= min_size:
                        mask |= labels == i

                # Filtrar etiquetas pequeñas
                labels_clean = labels * mask

                # Crear una nueva columna `classificationtree`
                classificationtree = np.zeros(len(points), dtype=int)
                for i, (x, y, z) in enumerate(points):
                    col = int((x - xmin) / resolution)
                    row = int((ymax - y) / resolution)
                    if 0 <= row < labels_clean.shape[0] and 0 <= col < labels_clean.shape[1]:
                        tree_id = labels_clean[row, col]
                        if tree_id > 0:
                            classificationtree[i] = tree_id

                # Agregar la nueva columna al archivo LAS
                las.add_extra_dim(
                    laspy.ExtraBytesParams(name="classificationtree", type=np.int32)
                )
                las["classificationtree"] = classificationtree

                las_object = las

                if mi_set == False:
                    update_progress_bar(progress_bar, 100)
                    progress_bar.grid_forget()
                    grid(las_object)

                if mi_set == True:
                    # Contar las ocurrencias de cada clasificación
                    classifications = np.array(las.classificationtree)
                    counts = Counter(classifications)
                    print("Clasificación de puntos en el archivo LAS:")
                    for classification, count in counts.items():
                        print(f"Categoría {classification}: {count} puntos")

                    unique_classificationtree = np.unique(classificationtree)
                    print("Unique classifications 1:", unique_classificationtree)

                    num_arboles = len(np.unique(labels_clean)) - 1  # Restar 1 para no contar el fondo
                    print(f"Número de árboles detectados: {num_arboles}")

                    # Filtrar el CHM y las etiquetas usando labels_clean
                    filtered_chm = np.where((labels_clean > 0), chm, np.nan)  # Mantén solo celdas válidas

                    # Actualizar la barra de progreso
                    update_progress_bar(progress_bar, 80)

                    # Get the maximum label value
                    max_label = np.max(labels_clean)

                    # Generate random colors for each label, including the background (label 0)
                    colors = np.random.rand(max_label + 1, 3)  # Ensure the size matches the maximum label

                    # Assign colors to each tree
                    tree_colors = np.zeros((labels_clean.shape[0], labels_clean.shape[1], 3))
                    for label in range(max_label + 1):
                        tree_colors[labels_clean == label] = colors[label]

                    # Actualizar la barra de progreso
                    update_progress_bar(progress_bar, 90)

                    # Crear una nube de puntos coloreada
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

                    # Actualizar la barra de progreso
                    update_progress_bar(progress_bar, 100)

                    # Eliminar la barra de progreso
                    progress_bar.grid_forget()

                    messagebox.showinfo("Segmentation Complete", f"Number of detected trees: {num_arboles}")

                    # Visualizar la nube de puntos
                    vis = o3d.visualization.Visualizer()

                    # Obtener las dimensiones del right_frame
                    right_frame.update_idletasks()
                    right_frame_width = right_frame.winfo_width()
                    right_frame_height = right_frame.winfo_height()

                    # Obtener las dimensiones del left_frame
                    left_frame.update_idletasks()
                    left_frame_width = left_frame.winfo_width()

                    # Calcular tittle bar
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
                            print("Ventana Cerrada")
                            enable_left_frame()
                            break

                    # Verificar si la nube de puntos tiene atributos
                    if not pcd.has_points():
                        print("La nube de puntos no tiene puntos.")
                        return

            else:
                print("No file selected.")
                return
        else:
            progress_bar = create_progress_bar()
            update_progress_bar(progress_bar, 10)

            if mi_set == False:
                update_progress_bar(progress_bar, 100)
                progress_bar.grid_forget()
                grid(las_object)

            if mi_set == True:
                # Contar las ocurrencias de cada clasificación
                classifications = np.array(las_object.classificationtree)
                counts = Counter(classifications)
                print("Clasificación de puntos en el archivo LAS:")
                for classification, count in counts.items():
                    print(f"Categoría {classification}: {count} puntos")

                update_progress_bar(progress_bar, 20)

                unique_classificationtree = np.unique(classificationtree)
                print("Unique classifications 1:", unique_classificationtree)

                num_arboles = len(np.unique(labels_clean)) - 1  # Restar 1 para no contar el fondo
                print(f"Número de árboles detectados: {num_arboles}")

                update_progress_bar(progress_bar, 40)

                # Filtrar el CHM y las etiquetas usando labels_clean
                filtered_chm = np.where((labels_clean > 0), chm, np.nan)  # Mantén solo celdas válidas

                # Actualizar la barra de progreso
                update_progress_bar(progress_bar, 80)

                # Get the maximum label value
                max_label = np.max(labels_clean)

                # Generate random colors for each label, including the background (label 0)
                colors = np.random.rand(max_label + 1, 3)  # Ensure the size matches the maximum label

                # Assign colors to each tree
                tree_colors = np.zeros((labels_clean.shape[0], labels_clean.shape[1], 3))
                for label in range(max_label + 1):
                    tree_colors[labels_clean == label] = colors[label]

                # Actualizar la barra de progreso
                update_progress_bar(progress_bar, 90)

                # Crear una nube de puntos coloreada
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

                # Actualizar la barra de progreso
                update_progress_bar(progress_bar, 100)

                # Eliminar la barra de progreso
                progress_bar.grid_forget()

                messagebox.showinfo("Segmentation Complete", f"Number of detected trees: {num_arboles}")

                # Visualizar la nube de puntos
                vis = o3d.visualization.Visualizer()

                # Obtener las dimensiones del right_frame
                right_frame.update_idletasks()
                right_frame_width = right_frame.winfo_width()
                right_frame_height = right_frame.winfo_height()

                # Obtener las dimensiones del left_frame
                left_frame.update_idletasks()
                left_frame_width = left_frame.winfo_width()

                # Calcular tittle bar
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
                        print("Ventana Cerrada")
                        enable_left_frame()
                        break

    threading.Thread(target=run, daemon=True).start()

# Crear la ventana de Tkinter
root = CTk()
root.title("Visor de Nube de Puntos")
root.title("Point Cloud Viewer")
root.configure(bg="#1E1E1E")

def maximize_window():
    root.state("zoomed")

root.after(0, maximize_window)

def disable_frame():
    root.attributes('-disabled', True)

def enable_frame():
    root.attributes('-disabled', False)

disable_frame()

def show_message():
    messagebox.showinfo("Information", "In this program, each time you want to edit the parameters, the Open3D window must be closed. Click the Accept button to start.")
    enable_frame()

root.after(1000, show_message)

# Configure the grid layout for the root window
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=4)

# Create the left frame
left_frame = CTkFrame(root, fg_color="#2E2E2E", corner_radius=0)
left_frame.grid(row=0, column=0, sticky="nsew")
left_frame.pack_propagate(False)

# Create the right frame
right_frame = CTkFrame(root, fg_color="white", corner_radius=0)
right_frame.grid(row=0, column=1, sticky="nsew")

# Frame para los botones del menú
menu_frame = CTkFrame(left_frame, fg_color="#2E2E2E", corner_radius=0)
menu_frame.pack(pady=(15, 0))

# Open...
root.menu_visible = False

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


def load_point_cloud_and_toggle():
    load_point_cloud()
    toggle_menu()


def process_n42_files_and_toggle():
    process_n42_files()
    toggle_menu()


def load_xml_metadata_and_toggle():
    load_xml_metadata()
    toggle_menu()


root.btn_open_pc = CTkButton(menu_frame, text="Point Cloud", text_color="#2E2E2E", fg_color="#F0F0F0",
                             border_color="#6E6E6E", border_width=1, font=("Arial", 12),
                             command=load_point_cloud_and_toggle)
root.btn_open_csv = CTkButton(menu_frame, text="N42 File", text_color="#2E2E2E", fg_color="#F0F0F0",
                              border_color="#6E6E6E", border_width=2, font=("Arial", 12),
                              command=process_n42_files_and_toggle)
root.btn_open_xml = CTkButton(menu_frame, text="XML", text_color="#2E2E2E", fg_color="#F0F0F0",
                              border_color="#6E6E6E", border_width=1, font=("Arial", 12),
                              command=load_xml_metadata_and_toggle)

# Downsample
downsample_frame = CTkFrame(left_frame, fg_color="#2E2E2E", corner_radius=0)
downsample_frame.pack(pady=(10, 0))
label_downsample = CTkLabel(downsample_frame, text="Downsample:", text_color="#F0F0F0", font=("Arial", 12))
root.downsample_entry = CTkEntry(downsample_frame, width=50, font=("Arial", 12))
label_percent = CTkLabel(downsample_frame, text="%", text_color="#F0F0F0", font=("Arial", 12))
label_downsample.pack(side="left", padx=(0, 5))
root.downsample_entry.pack(side="left", padx=(0, 5))
label_percent.pack(side="left")

# Parameters Button
root.parameters_visible = False


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


# Parameters Button
root.button_parameters = CTkButton(left_frame, text=" ▼ Parameters", text_color="#F0F0F0", fg_color="#3E3E3E",
                              anchor="w", corner_radius=0, command=toggle_parameters)
root.button_parameters.pack(fill="x", padx=(0, 0), pady=(10, 0))

parameters_frame = CTkFrame(left_frame, fg_color="#2E2E2E", corner_radius=0)

# Point Size
point_size_frame = CTkFrame(parameters_frame, fg_color="#2E2E2E", corner_radius=0)
point_size_frame.pack(fill="x", padx=(10, 10), pady=(0, 0))
label_point_size = CTkLabel(point_size_frame, text="Point Size:", text_color="#F0F0F0", font=("Arial", 12))
root.point_size_entry = CTkEntry(point_size_frame, width=50, font=("Arial", 12), state="disabled")
label_point_size.pack(side="left", padx=(10, 5))
root.point_size_entry.pack(side="left", padx=(0, 5))

# Voxelizer
voxelizer_frame = CTkFrame(parameters_frame, fg_color="#252525", corner_radius=0)
voxelizer_frame.pack(fill="x", padx=(10, 10), pady=(5, 0))
voxelizer_frame.grid_columnconfigure(0, weight=1)
voxelizer_frame.grid_columnconfigure(1, weight=1)
voxelizer_frame.grid_columnconfigure(2, weight=0)
voxelizer_frame.grid_columnconfigure(3, weight=1)
label_voxelizer = CTkLabel(voxelizer_frame, text="Voxelizer:", text_color="#F0F0F0", font=("Arial", 12))
label_voxelizer.grid(row=0, column=1, padx=(10, 5), pady=(5, 0), sticky="e")
root.voxelizer_var = BooleanVar()
root.voxelizer_switch = CTkSwitch(voxelizer_frame, variable=root.voxelizer_var, command=toggle_voxel_size,
                                  text="", state="disabled")
root.voxelizer_switch.grid(row=0, column=2, padx=(0, 5), pady=(5, 0), sticky="w")
voxelizerSize_frame = CTkFrame(parameters_frame, fg_color="#1E1E1E", corner_radius=0)
voxelizerSize_frame.pack(fill="x", padx=(10, 10), pady=(0, 0))
label_vox_size = CTkLabel(voxelizerSize_frame, text="Vox Size:", text_color="#F0F0F0", font=("Arial", 12))
label_vox_size.grid(row=1, column=0, padx=(10, 5), pady=(5, 5), sticky="w")
root.vox_size_entry = CTkEntry(voxelizerSize_frame, width=50, font=("Arial", 12), state="disabled")
root.vox_size_entry.grid(row=1, column=1, padx=(0, 5), pady=(5, 5), sticky="w")

# Dose Layer
root.dose_layer_visible = False


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

# Dose Layer
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

# Dosis Elevation
dosis_elevation_frame = CTkFrame(dose_layer_frame, fg_color="#2E2E2E", corner_radius=0)
dosis_elevation_frame.pack(fill="x", pady=(5, 0), anchor="center")
label_dosis_elevation = CTkLabel(dosis_elevation_frame, text="Dose Elevation:", text_color="#F0F0F0",
                                 font=("Arial", 12))
label_dosis_elevation.pack(side="left", padx=(10, 5))


def update_slider_label(value):
    slider_label.configure(text=f"{value:.2f}", font=("Arial", 12))


root.dosis_slider = CTkSlider(dosis_elevation_frame, from_=-100, to=100, command=update_slider_label,
                              state="disabled")
root.dosis_slider.set(1)
root.dosis_slider.pack(side="left", padx=(0, 5))
slider_label = CTkLabel(dosis_elevation_frame, text="1.00", text_color="#F0F0F0")
slider_label.pack(side="left", padx=(0, 5))

dose_sections_frame = CTkFrame(dose_layer_frame, fg_color="#2E2E2E", corner_radius=0)
dose_sections_frame.pack(fill="x", pady=(5, 0), anchor="center")

root.color_options = ["red", "yellow", "green", "blue", "purple", "orange", "cyan", "magenta", "pink", "white"]

root.high_min_medium_max = StringVar()
root.medium_min_low_max = StringVar()

# High Dose
label_high_dose = CTkLabel(dose_sections_frame, text="High Dose:", text_color="#F0F0F0", font=("Arial", 12))
label_high_dose.grid(row=0, column=0, padx=(10, 5), sticky="ew")
root.high_dose_cb = CTkComboBox(dose_sections_frame, values=root.color_options, font=("Arial", 12), width=90,
                                state="disabled")
root.high_dose_cb.set("red")
root.high_dose_cb.grid(row=0, column=1, padx=(0, 5), sticky="ew")
high_dose_rgb = np.array(mcolors.to_rgb("red"))
label_min = CTkLabel(dose_sections_frame, text="Min:", text_color="#F0F0F0", font=("Arial", 12))
label_min.grid(row=0, column=2, padx=(0, 5), sticky="ew")
root.high_dose_min = CTkEntry(dose_sections_frame, width=50, font=("Arial", 11),
                              textvariable=root.high_min_medium_max, state="disabled")
root.high_dose_min.grid(row=0, column=3, padx=(0, 5), sticky="ew")
label_max = CTkLabel(dose_sections_frame, text="Max:", text_color="#F0F0F0", font=("Arial", 12))
label_max.grid(row=0, column=4, padx=(0, 5), sticky="ew")
root.high_dose_max = CTkEntry(dose_sections_frame, width=50, font=("Arial", 11), state="disabled")
root.high_dose_max.grid(row=0, column=5, padx=(0, 5), sticky="ew")

# Medium Dose
label_medium_dose = CTkLabel(dose_sections_frame, text="Medium Dose:", text_color="#F0F0F0", font=("Arial", 12))
label_medium_dose.grid(row=1, column=0, padx=(10, 5), sticky="ew")
root.medium_dose_cb = CTkComboBox(dose_sections_frame, values=root.color_options, font=("Arial", 12), width=90,
                                  state="disabled")
root.medium_dose_cb.set("yellow")
root.medium_dose_cb.grid(row=1, column=1, padx=(0, 5), sticky="ew")
medium_dose_rgb = np.array(mcolors.to_rgb("yellow"))
label_min_medium = CTkLabel(dose_sections_frame, text="Min:", text_color="#F0F0F0", font=("Arial", 12))
label_min_medium.grid(row=1, column=2, padx=(0, 5), sticky="ew")
root.medium_dose_min = CTkEntry(dose_sections_frame, width=50, font=("Arial", 11),
                                textvariable=root.medium_min_low_max, state="disabled")
root.medium_dose_min.grid(row=1, column=3, padx=(0, 5), sticky="ew")
label_max_medium = CTkLabel(dose_sections_frame, text="Max:", text_color="#F0F0F0", font=("Arial", 12))
label_max_medium.grid(row=1, column=4, padx=(0, 5), sticky="ew")
root.medium_dose_max = CTkEntry(dose_sections_frame, width=50, font=("Arial", 11),
                                textvariable=root.high_min_medium_max, state="disabled")
root.medium_dose_max.grid(row=1, column=5, padx=(0, 5), sticky="ew")

# Low Dose
label_low_dose = CTkLabel(dose_sections_frame, text="Low Dose:", text_color="#F0F0F0", font=("Arial", 12))
label_low_dose.grid(row=2, column=0, padx=(10, 5), sticky="ew")
root.low_dose_cb = CTkComboBox(dose_sections_frame, values=root.color_options, font=("Arial", 12), width=90,
                               state="disabled")
root.low_dose_cb.set("green")
root.low_dose_cb.grid(row=2, column=1, padx=(0, 5), sticky="ew")
low_dose_rgb = np.array(mcolors.to_rgb("green"))
label_min_low = CTkLabel(dose_sections_frame, text="Min:", text_color="#F0F0F0", font=("Arial", 12))
label_min_low.grid(row=2, column=2, padx=(0, 5), sticky="ew")
root.low_dose_min = CTkEntry(dose_sections_frame, width=50, font=("Arial", 11), state="disabled")
root.low_dose_min.grid(row=2, column=3, padx=(0, 5), sticky="ew")
label_max_low = CTkLabel(dose_sections_frame, text="Max:", text_color="#F0F0F0", font=("Arial", 12))
label_max_low.grid(row=2, column=4, padx=(0, 5), sticky="ew")
root.low_dose_max = CTkEntry(dose_sections_frame, width=50, font=("Arial", 11),
                             textvariable=root.medium_min_low_max, state="disabled")
root.low_dose_max.grid(row=2, column=5, padx=(0, 5), sticky="ew")

# Source
source_frame = CTkFrame(dose_layer_frame, fg_color="#2E2E2E", corner_radius=0)
source_frame.pack(fill="x", pady=(5, 0))
root.btn_find_source = CTkButton(source_frame, text="Find Radioactive Source", fg_color="#3E3E3E",
                                 text_color="#F0F0F0", font=("Arial", 12), command=lambda: find_radioactive_source(csv_filepath))
root.btn_find_source.grid(row=0, column=0, padx=(10, 5), pady=(5, 0), sticky="w")
show_source_label = CTkLabel(source_frame, text="Show Source on Map:", text_color="#F0F0F0", font=("Arial", 12))
show_source_label.grid(row=0, column=1, padx=(10, 5), pady=(5, 0), sticky="w")
root.show_source_switch = CTkSwitch(source_frame, text="", command=toggle_source, state='disabled')
root.show_source_switch.grid(row=0, column=2, padx=(10, 5), pady=(5, 0), sticky="w")

# Extra Computations
root.extra_computations_visible = False


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
        root.button_extra_computations.configure(text=" ▼ Extra Computations")
        extra_computations_frame.pack_forget()
        root.btn_visualize.pack_forget()
        root.btn_visualize.pack(side="bottom", padx=(0, 0), pady=(10, 25))


root.button_extra_computations = CTkButton(left_frame, text=" ▼ Extra Computations", text_color="#F0F0F0",
                                      fg_color="#3E3E3E",
                                      anchor="w", corner_radius=0, command=toggle_extra_computations)
root.button_extra_computations.pack(fill="x", padx=(0, 0), pady=(10, 0))

extra_computations_frame = CTkFrame(left_frame, fg_color="#2E2E2E", corner_radius=0)

root.btn_heatmap = CTkButton(extra_computations_frame, text="Heatmap H*(10) rate", fg_color="#3E3E3E",
                             text_color="#F0F0F0", font=("Arial", 12), command=lambda: plot_heatmap(heatmap, xcenter, ycenter, Hcenter, lonmin, lonmax, latmin, latmax))
root.btn_heatmap.pack(fill="x", padx=(80, 80), pady=(5, 0))
root.btn_three_colors = CTkButton(extra_computations_frame, text="Heatmap with Three Color Range",
                                  fg_color="#3E3E3E",
                                  text_color="#F0F0F0", font=("Arial", 12),
                                  command=lambda: plot_three_color_heatmap(heatmap, xcenter, ycenter, Hcenter, lonmin, lonmax, latmin, latmax))
root.btn_three_colors.pack(fill="x", padx=(80, 80), pady=(5, 0))
root.btn_convert_pcd_to_dat = CTkButton(extra_computations_frame, text="3D grid from PCD", fg_color="#3E3E3E",
                                        text_color="#F0F0F0",
                                        font=("Arial", 12), command=lambda: panel_left_frame(xcenter, ycenter, FAltcenter, las_object))
root.btn_convert_pcd_to_dat.pack(fill="x", padx=(80, 80), pady=(5, 0))

segmentation_frame = CTkFrame(extra_computations_frame, fg_color="#2E2E2E", corner_radius=0)
segmentation_frame.pack(fill="x", padx=(80, 80), pady=(5, 0))
root.segmentation = CTkButton(segmentation_frame, text="Segmentation", fg_color="#3E3E3E",
                              text_color="#F0F0F0", font=("Arial", 12), width=105, command=segmentation)
root.segmentation.pack(side="left", padx=(0, 2.5))
root.segmentation_with_trees = CTkButton(segmentation_frame, text="Segmentation\nwith trees", fg_color="#3E3E3E",
                                         text_color="#F0F0F0", font=("Arial", 12), command=set_trees)
root.segmentation_with_trees.pack(side="left", padx=(2.5, 0))

# Visualize
root.btn_visualize = CTkButton(left_frame, text="Visualize", text_color="#F0F0F0", fg_color="#1E3A5F",
                               hover_color="#2E4A7F",
                               anchor="center", corner_radius=0, border_color="#D3D3D3", border_width=2,
                               command=lambda: visualize(pc_filepath, csv_filepath, xml_filepath, show_dose_layer, dose_min_csv, dose_max_csv))
root.btn_visualize.pack(side="bottom", padx=(0, 0), pady=(10, 25))

# Algoritmo genético para encontrar la ubicación de una fuente radiactiva
class GeneticAlgorithm:
    def __init__(self, utm_coords, population_size=500, generations=100, mutation_rate=0.01):
        self.utm_coords = utm_coords
        self.population_size = population_size  # Define el tamaño de la población
        self.generations = generations  # Define el número de generaciones
        self.mutation_rate = mutation_rate  # Define la tasa de mutación
        self.bounds = self.get_bounds()  # Obtiene los límites de las coordenadas UTM

    def get_bounds(self):  # Obtiene los límites de las coordenadas UTM
        x_min, y_min = np.min(self.utm_coords[:, :2], axis=0)
        x_max, y_max = np.max(self.utm_coords[:, :2], axis=0)
        return (x_min, x_max), (y_min, y_max)

    def fitness(self, candidate):  # Función de aptitud para evaluar la dosis en un punto candidato
        tree = cKDTree(self.utm_coords[:, :2])
        dist, idx = tree.query(
            candidate)  # Encuentra el punto más cercano en la nube de puntos a la ubicación candidata, devuelve la distancia y el índice del punto
        return -self.utm_coords[
            idx, 2]  # Dosis negativa porque queremos maximizar (no minimizar), el algoritmo maximiza la dosis porque minimiza el valor negativo (nos quedamos con el mas negativo que corresponde al valor de dosis mas alto cambiado de signo).

    def initialize_population(
            self):  # Genera la población inicial de posibles candidatos, tantos como el tamaño de la población establecido
        (x_min, x_max), (y_min, y_max) = self.bounds
        return np.array(
            [[random.uniform(x_min, x_max), random.uniform(y_min, y_max)] for _ in range(self.population_size)])

    def select_parents(self, population, fitnesses):
        idx = np.random.choice(np.arange(self.population_size), size=self.population_size, replace=True,
                               p=fitnesses / fitnesses.sum())  # candidates with higher fitness values have a higher chance of being selected
        return population[idx]

    def crossover(self, parent1, parent2):
        alpha = random.random()
        return alpha * parent1 + (1 - alpha) * parent2

    def mutate(self, candidate):
        (x_min, x_max), (y_min, y_max) = self.bounds
        if random.random() < self.mutation_rate:
            candidate[0] = random.uniform(x_min, x_max)
        if random.random() < self.mutation_rate:
            candidate[1] = random.uniform(y_min, y_max)
        return candidate

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

# Ejecutar la aplicación
root.mainloop()