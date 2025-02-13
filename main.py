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

class WelcomeScreen(CTk):
    def __init__(self):
        super().__init__()

        self.title("Welcome")
        self.geometry("900x600")

        bg_image = Image.open("welcome.png")
        self.bg_photo = CTkImage(light_image=bg_image, dark_image=bg_image, size=(900, 600))

        # Crear un Label para mostrar la imagen de fondo
        self.background_label = CTkLabel(self, image=self.bg_photo, text="")
        self.background_label.place(relwidth=1, relheight=1)  # Asegura que cubra toda la ventana

        self.start_button = CTkButton(self, text="Start", font=("Arial", 20), text_color="white",
                                 corner_radius=8, fg_color="#4B4B4B", width=300, height=100,
                                 hover_color="#6E6E6E", border_color="#2C2C2C", border_width=1,
                                 command=self.open_main_app)
        self.start_button.place(relx=0.5, rely=0.8, anchor="center")  # Centrado horizontalmente, más abajo

    def open_main_app(self):
        self.destroy()
        app = PointCloudApp()
        app.mainloop()

class PointCloudApp(CTk):
    def __init__(self):
        super().__init__()

        self.title("Point Cloud Viewer")
        self.geometry("900x600")
        self.configure(bg="#1E1E1E")  # Fondo oscuro

        self.pc_filepath = None
        self.csv_filepath = None
        self.vis = None

        frame = CTkFrame(self, fg_color="#2E2E2E", corner_radius=15)
        frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Botones de carga
        button_frame = CTkFrame(frame, fg_color="transparent")
        button_frame.pack(pady=10, padx=10, fill="x")

        self.btn_open_pc = CTkButton(master=button_frame, text="📂 Open Point Cloud", corner_radius=32, fg_color="#3A7EBF",
                                     hover_color="#C850C0", border_color="#FFCC70", border_width=2,
                                     font=("Arial", 14, "bold"), command=self.load_point_cloud)
        self.btn_open_pc.pack(side="left", padx=10, pady=5)

        self.btn_open_csv = CTkButton(master=button_frame, text="📊 Open CSV Dosis", corner_radius=32, fg_color="#3A7EBF",
                                      hover_color="#C850C0", border_color="#FFCC70", border_width=2,
                                      font=("Arial", 14, "bold"), command=self.load_csv_dosis)
        self.btn_open_csv.pack(side="left", padx=10, pady=5)

        # Parámetros
        parameters_frame = CTkFrame(master=frame, fg_color="#383838", corner_radius=10)
        parameters_frame.pack(pady=10, padx=10, fill="x")

        CTkLabel(master=parameters_frame, text="🔧 Parameters", font=("Arial", 16, "bold"), text_color="white").pack(pady=5)

        param_grid = CTkFrame(master=parameters_frame, fg_color="transparent")
        param_grid.pack(pady=5, padx=10)

        CTkLabel(master=param_grid, text="Point Size:", text_color="white").grid(row=0, column=0, pady=5, padx=5, sticky="w")
        self.point_size_entry = CTkEntry(param_grid, width=50)
        self.point_size_entry.grid(row=0, column=1, pady=5, padx=5)

        CTkLabel(master=param_grid, text="Dosis Elevation:", text_color="white").grid(row=1, column=0, pady=5, padx=5,
                                                                               sticky="w")
        self.dosis_slider = CTkSlider(master=param_grid, from_=0, to=100, fg_color="#FFCC70")
        self.dosis_slider.grid(row=1, column=1, pady=5, padx=5)

        self.voxelizer_checkbox = CTkCheckBox(master=parameters_frame, text="Voxelizer", text_color="white",
                                              fg_color="#FFCC70")
        self.voxelizer_checkbox.pack(pady=5)

        # Leyenda de dosis
        legend_frame = CTkFrame(master=frame, fg_color="#383838", corner_radius=10)
        legend_frame.pack(pady=10, padx=10, fill="x")

        CTkLabel(master=legend_frame, text="🎨 Dose Legend", font=("Arial", 16, "bold"), text_color="white").pack(pady=5)

        dose_colors = CTkFrame(master=legend_frame, fg_color="transparent")
        dose_colors.pack(pady=5, padx=10)

        self.high_dose = CTkLabel(master=dose_colors, text="🔴 High", text_color="red", font=("Arial", 12))
        self.high_dose.grid(row=0, column=0, pady=2, padx=5)

        self.medium_dose = CTkLabel(master=dose_colors, text="🟡 Medium", text_color="yellow", font=("Arial", 12))
        self.medium_dose.grid(row=1, column=0, pady=2, padx=5)

        self.low_dose = CTkLabel(master=dose_colors, text="🟢 Low", text_color="green", font=("Arial", 12))
        self.low_dose.grid(row=2, column=0, pady=2, padx=5)

        self.change_btn = CTkButton(master=legend_frame, text="Change", width=80, fg_color="#C850C0")
        self.change_btn.pack(pady=5)

        # Botón de visualización
        self.btn_visualize = CTkButton(master=frame, text="👁️ Visualize", corner_radius=32, fg_color="#4258D0",
                                       hover_color="#C850C0", border_color="#FFCC70", border_width=2,
                                       font=("Arial", 14, "bold"), command=self.visualize)
        self.btn_visualize.pack(pady=10)

    def load_point_cloud(self):
        filepath = filedialog.askopenfilename(filetypes=[("PCD Files", "*.pcd")])
        if filepath:
            self.pc_filepath = filepath
            print("Point Cloud Selected:", self.pc_filepath)

    def load_csv_dosis(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filepath:
            self.csv_filepath = filepath
            print("CSV Selected:", self.csv_filepath)

    def visualize(self):
        """Valida los archivos y procede con la visualización."""
        if not self.pc_filepath:
            messagebox.showwarning("Warning", "Please select a Point Cloud file.")
            return

        if not self.csv_filepath:
            messagebox.showwarning("Warning", "Please select a CSV file.")
            return

        if self.vis is None:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()

        # Verifica el estado del checkbox y ejecuta la función correspondiente.
        if self.voxelizer_checkbox.get():  # Si el checkbox está marcado
            self.voxelizer()
        else:
            self.process()

    def convert_to_utm(self, csv_filepath):
        # Convierte coordenadas lat/lon en el CSV a UTM y devuelve una matriz con easting, northing y dosis.
        utm_proj = Proj(proj='utm', zone=31, datum='WGS84')
        utm_coords = []

        output_file = Path(csv_filepath).with_name('sensorDataConverted.csv')

        with open(csv_filepath, mode='r', newline='', encoding='utf-8') as infile, \
                open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            writer.writerow(['Easting', 'Northing', 'Dosis'])  # Encabezados

            # Itera sobre cada fila en el archivo de entrada
            for row in reader:
                # Obtiene latitud, longitud y valor extra
                latitude = float(row[0])  # Primer elemento: latitud
                longitude = float(row[1])  # Segundo elemento: longitud
                valorextra = float(row[2])  # Tercer elemento: valor extra

                easting, northing = utm_proj(longitude, latitude)

                writer.writerow([easting, northing, valorextra])

                utm_coords.append([easting, northing, valorextra])

        utm_coords = np.array(utm_coords)  # Matriz de tamaño (N,3), cada fila representa easting, northing, y dosis

        return utm_coords

    def get_dose_color(self, dosis_nube):
        # Colorear nube de puntos según dosis
        colores_dosis = np.zeros((len(dosis_nube), 3))  # RGB por defecto negro

        # Verde (0-100)
        colores_dosis[(dosis_nube >= 0) & (dosis_nube < 100)] = [0, 1, 0]

        # Amarillo (100-200)
        colores_dosis[(dosis_nube >= 100) & (dosis_nube < 200)] = [1, 1, 0]

        # Rojo (200+)
        colores_dosis[dosis_nube >= 200] = [1, 0, 0]

        return colores_dosis

    def process(self):
        try:
            # Cargar la nube de puntos PCD
            pcd = o3d.io.read_point_cloud(self.pc_filepath)

            # Obtener coordenadas XYZ
            nube_puntos = np.asarray(pcd.points)

            # Obtener colores si existen, de lo contrario, usar blanco
            if pcd.has_colors():
                rgb = np.asarray(pcd.colors)
            else:
                rgb = np.ones_like(nube_puntos)

            utm_coords = self.convert_to_utm(self.csv_filepath)
            utm_points = utm_coords[:, :2]  # Sólo coordenadas [easting, northing]
            dosis = utm_coords[:, 2]  # Dosis correspondiente

            # Construir el KD-Tree para los puntos UTM del CSV (BUSQUEDA EFICIENTE)
            tree = cKDTree(utm_points)

            # Determinar los límites del área del CSV con dosis
            x_min, y_min = np.min(utm_points, axis=0)  # Mínimo de cada columna (lat, long)
            x_max, y_max = np.max(utm_points, axis=0)  # Máximo de cada columna (lat, long)

            # Filtrar puntos de la nube dentro del área de dosis
            dentro_area = (
                    (nube_puntos[:, 0] >= x_min) & (nube_puntos[:, 0] <= x_max) &
                    (nube_puntos[:, 1] >= y_min) & (nube_puntos[:, 1] <= y_max)
            )

            # Solo los puntos dentro del área
            puntos_dentro = nube_puntos[dentro_area]

            # Crea vector de dosis como NaN
            dosis_nube = np.full(len(puntos_dentro), np.nan)

            # Encontrar el punto más cercano en el CSV para cada punto de la nube LAS (que está dentro)
            distancias, indices_mas_cercanos = tree.query(puntos_dentro[:,
                                                          :2])  # Devuelve distancia entre punto CSV y punto cloud; para cada nube_puntos[i] índice del punto del csv mas cercano

            # Asignar dosis correspondiente a los puntos dentro del área
            dosis_nube[:] = dosis[indices_mas_cercanos]  # Dosis para cada punto en la nube

            colores_dosis = self.get_dose_color(dosis_nube)

            # Filtra las coordenadas y dosis de los puntos dentro del área
            nube_puntos_filtrada = puntos_dentro
            dosis_filtrada = dosis_nube

            altura_extra = 5  # Ajusta este valor según lo necesario
            puntos_dosis_elevados = np.copy(nube_puntos_filtrada)
            puntos_dosis_elevados[:, 2] += altura_extra  # Aumentar Z

            # Crear nube de puntos Open3D
            pcd.points = o3d.utility.Vector3dVector(nube_puntos)
            pcd.colors = o3d.utility.Vector3dVector(rgb)  # Asignar colores

            # Crear la nueva nube de puntos de dosis elevada
            pcd_dosis = o3d.geometry.PointCloud()
            pcd_dosis.points = o3d.utility.Vector3dVector(puntos_dosis_elevados)
            pcd_dosis.colors = o3d.utility.Vector3dVector(colores_dosis)  # Asignar colores según dosis

            # Limpiar y Agregar las nubes de puntos a la visualización
            self.vis.clear_geometries()
            self.vis.add_geometry(pcd)
            self.vis.add_geometry(pcd_dosis)

            # Cambiar el tamaño de los puntos (ajustar para evitar cuadrados)
            render_option = self.vis.get_render_option()
            render_option.point_size = 2

            self.vis.run()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def voxelizer(self):
        pcd = o3d.io.read_point_cloud(self.pc_filepath)
        xyz = np.asarray(pcd.points)

        # Obtener colores si existen, de lo contrario usar blanco
        if pcd.has_colors():
            rgb = np.asarray(pcd.colors)
        else:
            rgb = np.ones_like(xyz)  # Blanco por defecto

        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        # Defining the voxel size
        vsize = max(pcd.get_max_bound()-pcd.get_min_bound())*0.005
        vsize = round(vsize,4)

        # Creating the voxel grid
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=vsize)

        #Extracting the bounds
        bounds = voxel_grid.get_max_bound()-voxel_grid.get_min_bound()
                #o3d.visualization.draw_geometries([voxel_grid])

        # Generating a single box entity
        cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
        cube.paint_uniform_color([1,0,0])   # Red
        cube.compute_vertex_normals()
                #o3d.visualization.draw_geometries([cube])

        # Automate and Loop to cerate one voxel Dataset (efficient)
        voxels = voxel_grid.get_voxels() # Cada voxel con su grid index (posicion desde el centro, 0) y color, hay que hacer offset y translate
        vox_mesh = o3d.geometry.TriangleMesh() # Creamos un mesh para ir colocando cada voxel

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

        # DOSIS
        utm_coords = self.convert_to_utm(self.csv_filepath)
        utm_points = utm_coords[:, :2]  # Sólo coordenadas [easting, northing]
        dosis = utm_coords[:, 2]  # Dosis correspondiente

        # Construir el KD-Tree para los puntos UTM del CSV (BUSQUEDA EFICIENTE)
        tree = cKDTree(utm_points)

        # Determinar los límites del área del CSV con dosis
        x_min, y_min = np.min(utm_points, axis=0)  # Mínimo de cada columna (lat, long)
        x_max, y_max = np.max(utm_points, axis=0)  # Máximo de cada columna (lat, long)

        # Filtrar puntos de la nube dentro del área de dosis
        dentro_area = (
                (xyz[:, 0] >= x_min) & (xyz[:, 0] <= x_max) &
                (xyz[:, 1] >= y_min) & (xyz[:, 1] <= y_max)
        )

        # Solo los puntos dentro del área
        puntos_dentro = xyz[dentro_area]

        # Crea vector de dosis como NaN
        dosis_nube = np.full(len(puntos_dentro), np.nan)

        # Encontrar el punto más cercano en el CSV para cada punto de la nube LAS (que está dentro)
        distancias, indices_mas_cercanos = tree.query(puntos_dentro[:,
                                                      :2])  # Devuelve distancia entre punto CSV y punto cloud; para cada nube_puntos[i] índice del punto del csv mas cercano

        # Asignar dosis correspondiente a los puntos dentro del área
        dosis_nube[:] = dosis[indices_mas_cercanos]  # Dosis para cada punto en la nube

        colores_dosis = self.get_dose_color(dosis_nube)

        # Rojo (200+)
        colores_dosis[dosis_nube >= 200] = [1, 0, 0]

        # Filtra las coordenadas y dosis de los puntos dentro del área
        nube_puntos_filtrada = puntos_dentro
        dosis_filtrada = dosis_nube

        altura_extra = 5  # Ajusta este valor según lo necesario
        puntos_dosis_elevados = np.copy(nube_puntos_filtrada)
        puntos_dosis_elevados[:, 2] += altura_extra  # Aumentar Z

        pcd_dosis = o3d.geometry.PointCloud()
        pcd_dosis.points = o3d.utility.Vector3dVector(puntos_dosis_elevados)
        pcd_dosis.colors = o3d.utility.Vector3dVector(colores_dosis)

        voxel_grid_dosis = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_dosis, voxel_size=vsize)

        voxels_dosis = voxel_grid_dosis.get_voxels()
        vox_mesh_dosis = o3d.geometry.TriangleMesh()

        cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
        cube.paint_uniform_color([1, 0, 0])  # Red
        cube.compute_vertex_normals()

        for v in voxels_dosis:
            cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
            cube.paint_uniform_color(v.color)
            cube.translate(v.grid_index, relative=False)
            vox_mesh_dosis += cube

        vox_mesh_dosis.translate([0.5, 0.5, 0.5], relative=True)

        vox_mesh_dosis.scale(vsize, [0, 0, 0])

        vox_mesh_dosis.translate(voxel_grid_dosis.origin, relative=True)

        output_file = Path("voxelize_dosis.ply")  # Puntos --> .las / Malla --> .obj, .ply
        o3d.io.write_triangle_mesh(str(output_file), vox_mesh_dosis)

        self.vis.clear_geometries()
        self.vis.add_geometry(vox_mesh)
        self.vis.add_geometry(vox_mesh_dosis)

        self.vis.run()

if __name__ == "__main__":
    welcome_screen = WelcomeScreen()
    welcome_screen.mainloop()

## --------------- BOTON
#img = Image.open("emoticono.png")
#btn = CTkButton(master=app, text="Prueba", corner_radius=32, fg_color="#4258D0",
                #hover_color="#C850C0", border_color="#FFCC70", border_width=2,
                #image = CTkImage(dark_image=img, light_image=img))
#btn.place(relx=0.5, rely=0.5, anchor="center")

## --------------- LABEL (ya escrito)
#label = CTkLabel(master=app, text="Hola", font=("Arial",20), text_color="#FFCC70")
#label.place(relx=0.5, rely=0.5, anchor="center")

## --------------- COMBOBOX (desplegable)
#comboBox = CTkComboBox(master=app, values=["Option 1", "Option 2", "Option 3"],
                       #fg_color="#0093E9", border_color="#FBAB7E", dropdown_fg_color="#0093E9")
#comboBox.place(relx=0.5, rely=0.5, anchor="center")

## --------------- CHECKBOX (marcar varias opciones)
#checkBox = CTkCheckBox(master=app, text="Option 1",
                       #fg_color="#C850C0", checkbox_height=30, checkbox_width=30, corner_radius=36)
#checkBox.place(relx=0.5, rely=0.5, anchor="center")

## --------------- SWITCH (Mateix que checkbox, seleccionar pero amb una barra)
#switch = CTkSwitch(master=app, text="Option 1")
#switch.place(relx=0.5, rely=0.5, anchor="center")

## --------------- SLIDER (barrita)
#def change_handler (value):
    #print(f"Selected value {value}")

#slider = CTkSlider(master=app, from_=0, to=100, number_of_steps=5, button_color="#C850C0", orientation="vertical", command=change_handler) #command pasa el valor del slider a la def
#slider.place(relx=0.5, rely=0.5, anchor="center")

## --------------- ENTRY TEXT (escribir texto)
#entry = CTkEntry (master=app, placeholder_text="Start typing...", width=300, text_color="#FFCC70")
#entry.place(relx=0.5, rely=0.5, anchor="center")

## --------------- TEXTBOX (escribir texto)
#textbox = CTkTextbox (master=app, scrollbar_button_color="#FFCC70", corner_radius=16,
                      #border_color="#FFCC70", border_width=2)
#textbox.place(relx=0.5, rely=0.5, anchor="center")




