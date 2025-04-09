import numpy as np
import open3d as o3d
import csv
from pyproj import Proj
from scipy.spatial import cKDTree
from pathlib import Path
from customtkinter import *
from tkinter import filedialog, messagebox
import multiprocessing
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

def gridfrompcd(pc_filepath, progress_bar, csv_filepath):
    def run():
        global vis
        try:
            # Actualizar la barra de progreso
            update_progress_bar(progress_bar, 10)

            print("Run prueba")
            # Load the PCD file
            pcd = o3d.io.read_point_cloud(pc_filepath)

            # Extract point data
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(points)

            # Actualizar la barra de progreso
            update_progress_bar(progress_bar, 20)

            # Determine the bounds of the data
            min_x, min_y = np.min(points[:, :2], axis=0)
            max_x, max_y = np.max(points[:, :2], axis=0)

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
                    cell_stats[i, j] = {'z_values': [], 'colors': []}

            # Actualizar la barra de progreso
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

            # Actualizar la barra de progreso
            update_progress_bar(progress_bar, 60)

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

            # Actualizar la barra de progreso
            update_progress_bar(progress_bar, 80)

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
                            prisms.append(prism)

            # Combine all prisms into a single mesh
            combined_mesh = o3d.geometry.TriangleMesh()
            for prism in prisms:
                combined_mesh += prism

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
            vis.add_geometry(combined_mesh)

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
    global dose_min_csv, dose_max_csv, csv_filepath, heatmap, xcenter, ycenter, Hcenter, lonmin, lonmax, latmin, latmax
    fp = filedialog.askdirectory(title="Select Folder with .n42 Files")
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

def set_run_prueba_flag(pc_filepath):
    global progress_bar
    if not pc_filepath:
        messagebox.showwarning("Warning", "Please select a Point Cloud.")
        return

    disable_left_frame()

    # Crear y mostrar la barra de progreso
    progress_bar = create_progress_bar()

    # Actualizar la barra de progreso
    update_progress_bar(progress_bar, 1)

    gridfrompcd(pc_filepath, progress_bar, csv_filepath)

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
    fp = filedialog.askopenfilename(filetypes=[("LAS Files", "*.las")])
    if fp:
        print("Point Cloud Selected:", fp)

        # Crear y mostrar la barra de progreso
        progress_bar = create_progress_bar()

        # Actualizar la barra de progreso
        update_progress_bar(progress_bar, 1)

        # Read the LAS file
        las = laspy.read(fp)

        # Extract points and classifications
        points = np.vstack((las.x, las.y, las.z)).transpose()
        classifications = np.array(las.classification)
        #unique_classifications = np.unique(classifications)

        # Actualizar la barra de progreso
        update_progress_bar(progress_bar, 10)

        # Define colors for specific classifications
        color_map = {
            0: [0.0, 0.0, 0.0],  # 0 - Created, never classified (Negro)
            1: [1.0, 1.0, 1.0],  # 1 - Unclassified (White)
            2: [0.55, 0.27, 0.07],  # 2 - Ground (Marrón)
            3: [0.0, 1.0, 0.0],  # 3 - Low Vegetation (Verde claro)
            4: [0.0, 0.6, 0.0],  # 4 - Medium Vegetation (Verde medio)
            5: [0.0, 0.39, 0.0],  # 5 - High Vegetation (Verde oscuro)
            6: [1.0, 0.0, 0.0],  # 6 - Building (Rojo)
            7: [1.0, 1.0, 0.0],  # 7 - Low Point (noise) (Amarillo)
            9: [0.0, 0.0, 1.0],  # 9 - Water (Azul)
            10: [1.0, 0.65, 0.0],  # 10 - Rail (Naranja claro)
            11: [0.5, 0.5, 0.0],  # 11 - Road Surface (Oliva)
            13: [0.8, 0.8, 0.0],  # 13 - Wire – Guard (Shield) (Amarillo pálido)
            14: [0.5, 0.5, 0.5],  # 14 - Wire – Conductor (Phase) (Gris)
            15: [0.8, 0.0, 0.8],  # 15 - Transmission Tower (Violeta)
            16: [0.0, 1.0, 1.0],  # 16 - Wire-structure Connector (Cian)
            17: [0.8, 0.5, 0.2],  # 17 - Bridge Deck (Marrón claro)
            18: [1.0, 0.0, 1.0],  # 18 - High Noise (Magenta)
        }

        # Actualizar la barra de progreso
        update_progress_bar(progress_bar, 20)

        # Create an Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Actualizar la barra de progreso
        update_progress_bar(progress_bar, 40)

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
                enable_left_frame()
                break

        # Verificar si la nube de puntos tiene atributos
        if not pcd.has_points():
            print("La nube de puntos no tiene puntos.")
            return

    else:
        print("No file selected.")

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
                                        font=("Arial", 12), command=lambda: set_run_prueba_flag(pc_filepath))
root.btn_convert_pcd_to_dat.pack(fill="x", padx=(80, 80), pady=(5, 0))
root.segmentation = CTkButton(extra_computations_frame, text="Segmentation", fg_color="#3E3E3E",
                                        text_color="#F0F0F0",
                                        font=("Arial", 12), command=segmentation)
root.segmentation.pack(fill="x", padx=(80, 80), pady=(5, 0))

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