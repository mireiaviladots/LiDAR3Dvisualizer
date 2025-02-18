import numpy as np
import threading
import open3d as o3d
import csv
from pyproj import Proj
from scipy.spatial import cKDTree
from pathlib import Path
from customtkinter import *
from PIL import Image # (imagenes en los botones)
from tkinter import filedialog, messagebox
import multiprocessing

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import open3d as o3d
import multiprocessing
import os


class PointCloudApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Point Cloud Processor")

        self.pc_filepath = None
        self.csv_filepath = None

        self.voxelizer_checkbox = tk.IntVar()
        self.visualizer_process = None
        self.data_queue = multiprocessing.Queue()

        self.create_widgets()

    def create_widgets(self):
        tk.Button(self.root, text="Select Point Cloud File", command=self.load_point_cloud).pack(pady=5)
        tk.Button(self.root, text="Select CSV File", command=self.load_csv).pack(pady=5)
        tk.Checkbutton(self.root, text="Use Voxelizer", variable=self.voxelizer_checkbox).pack(pady=5)
        tk.Button(self.root, text="Visualize", command=self.start_visualizer).pack(pady=10)

    def load_point_cloud(self):
        self.pc_filepath = filedialog.askopenfilename(filetypes=[("Point Cloud Files", "*.pcd *.ply *.xyz *.txt")])
        if self.pc_filepath:
            messagebox.showinfo("File Selected", f"Loaded: {os.path.basename(self.pc_filepath)}")

    def load_csv(self):
        self.csv_filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if self.csv_filepath:
            messagebox.showinfo("File Selected", f"Loaded: {os.path.basename(self.csv_filepath)}")

    def start_visualizer(self):
        """Inicia un proceso separado para Open3D y envía los datos a visualizar."""
        if not self.pc_filepath:
            messagebox.showwarning("Warning", "Please select a Point Cloud file.")
            return

        if not self.csv_filepath:
            messagebox.showwarning("Warning", "Please select a CSV file.")
            return

        use_voxelization = self.voxelizer_checkbox.get() == 1
        params = {"pc_filepath": self.pc_filepath, "use_voxelization": use_voxelization}

        if self.visualizer_process and self.visualizer_process.is_alive():
            self.data_queue.put(params)  # Enviar datos al proceso en ejecución
        else:
            self.visualizer_process = multiprocessing.Process(target=self.run_open3d, args=(self.data_queue,))
            self.visualizer_process.start()
            self.data_queue.put(params)  # Enviar datos iniciales

    @staticmethod
    def run_open3d(data_queue):
        """Proceso separado que maneja la visualización en Open3D."""
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        while True:
            if not data_queue.empty():
                params = data_queue.get()
                if params is None:
                    break  # Salir del bucle si recibimos un cierre

                pc_filepath = params["pc_filepath"]
                use_voxelization = params["use_voxelization"]

                pcd = o3d.io.read_point_cloud(pc_filepath)
                vis.clear_geometries()

                if use_voxelization:
                    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)
                    vis.add_geometry(voxel_grid)
                else:
                    vis.add_geometry(pcd)

                vis.update_renderer()
                vis.poll_events()

        vis.destroy_window()

    def on_closing(self):
        """Cierra el proceso de Open3D cuando la ventana Tkinter se cierra."""
        if self.visualizer_process and self.visualizer_process.is_alive():
            self.data_queue.put(None)  # Enviar señal de cierre
            self.visualizer_process.join()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PointCloudApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
