import tkinter as tk
from tkinter import ttk
import numpy as np
import open3d as o3d
import threading
import time

class PointCloudApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Controlador de Nube de Puntos")
        self.geometry("400x300")

        # Variables para modificar los parámetros en tiempo real
        self.noise_var = tk.DoubleVar(value=0.02)

        # Crear controles en la UI
        ttk.Label(self, text="Ruido:").pack()
        self.noise_slider = ttk.Scale(self, from_=0, to=0.1, variable=self.noise_var, command=self.update_noise)
        self.noise_slider.pack()

        # Botón para iniciar la visualización en otro hilo
        self.start_button = ttk.Button(self, text="Visualizar Nube", command=self.start_visualization)
        self.start_button.pack()

        # Crear la nube de puntos inicial
        self.pcd = self.generate_point_cloud()

        # Hilo para Open3D (inicialmente None)
        self.o3d_thread = None
        self.vis = None

    def generate_point_cloud(self):
        """Genera una nube de puntos inicial con ruido."""
        num_points = 500
        points = np.random.rand(num_points, 3)
        noise = np.random.normal(0, self.noise_var.get(), (num_points, 3))
        points += noise

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0, 0.6, 1])  # Azul

        return pcd

    def update_noise(self, event=None):
        """Actualiza la nube de puntos cuando se modifica el ruido en la UI."""
        if self.vis is not None:
            print("Actualizando nube de puntos con nuevo ruido...")
            self.pcd = self.generate_point_cloud()
            self.vis.clear_geometries()
            self.vis.add_geometry(self.pcd)
            self.vis.update_renderer()
            self.vis.poll_events()

    def visualize_point_cloud(self):
        """Ejecuta la ventana Open3D en un hilo separado."""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.add_geometry(self.pcd)

        while True:
            self.vis.update_geometry(self.pcd)
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.1)  # Pequeña pausa para no consumir 100% de CPU

    def start_visualization(self):
        """Lanza la visualización en un hilo separado."""
        if self.o3d_thread is None:
            self.o3d_thread = threading.Thread(target=self.visualize_point_cloud, daemon=True)
            self.o3d_thread.start()

# Iniciar la aplicación
if __name__ == "__main__":
    app = PointCloudApp()
    app.mainloop()

