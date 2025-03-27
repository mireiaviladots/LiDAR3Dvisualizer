from customtkinter import *

class PointCloudApp(CTk):
    def __init__(self):
        super().__init__()

        self.title("Point Cloud Viewer")
        self.configure(bg="#1E1E1E")  # Dark background

        # Create a main frame that contains two subframes
        main_frame = CTkFrame(self, fg_color="#1E1E1E")
        main_frame.pack(fill="both", expand=True)

        # Configure grid layout for main_frame
        main_frame.grid_columnconfigure(0, weight=1)  # Left frame (1/3)
        main_frame.grid_columnconfigure(1, weight=4)  # Right frame (2/3)
        main_frame.grid_rowconfigure(0, weight=1)

        # Frame izquierdo
        left_frame = CTkFrame(main_frame, fg_color="#2E2E2E", corner_radius=0)
        left_frame.grid(row=0, column=0, sticky="nsew")
        left_frame.pack_propagate(False)

        # Frame derecho (blanco)
        right_frame = CTkFrame(main_frame, fg_color="white", corner_radius=0)
        right_frame.grid(row=0, column=1, sticky="nsew")

        # Frame para los botones del menú
        menu_frame = CTkFrame(left_frame, fg_color="#2E2E2E", corner_radius=0)
        menu_frame.pack(pady=(15, 0))

        # Open...
        self.menu_visible = False

        def toggle_menu():
            self.menu_visible = not self.menu_visible

            if self.menu_visible:
                btn_opcion1.pack(pady=0)
                btn_opcion2.pack(pady=0)
                btn_opcion3.pack(pady=0)
            else:
                btn_opcion1.pack_forget()
                btn_opcion2.pack_forget()
                btn_opcion3.pack_forget()

        btn_menu = CTkButton(menu_frame, text="Open ...", command=toggle_menu, fg_color="#6E6E6E")
        btn_menu.pack(pady=(5, 0))

        btn_opcion1 = CTkButton(menu_frame, text="Point Cloud", text_color="#2E2E2E", fg_color="#F0F0F0",
                                border_color="#6E6E6E", border_width=1)
        btn_opcion2 = CTkButton(menu_frame, text="N42 File", text_color="#2E2E2E", fg_color="#F0F0F0",
                                border_color="#6E6E6E", border_width=2)
        btn_opcion3 = CTkButton(menu_frame, text="XML", text_color="#2E2E2E", fg_color="#F0F0F0",
                                border_color="#6E6E6E", border_width=1)

        # Downsample
        downsample_frame = CTkFrame(left_frame, fg_color="#2E2E2E", corner_radius=0)
        downsample_frame.pack(pady=(10, 0))
        label_downsample = CTkLabel(downsample_frame, text="Downsample:", text_color="#F0F0F0")
        entry_downsample = CTkEntry(downsample_frame, width=50)
        label_percent = CTkLabel(downsample_frame, text="%", text_color="#F0F0F0")
        label_downsample.pack(side="left", padx=(0, 5))
        entry_downsample.pack(side="left", padx=(0, 5))
        label_percent.pack(side="left")

        # Parameters Button
        self.parameters_visible = False

        def toggle_parameters():
            self.parameters_visible = not self.parameters_visible

            if self.parameters_visible:
                button_parameters.configure(text=" ▲ Parameters")
                parameters_frame.pack(pady=(10, 0), fill="x")
            else:
                button_parameters.configure(text=" ▼ Parameters")
                parameters_frame.pack_forget()

        # Parameters Button
        button_parameters = CTkButton(left_frame, text=" ▼ Parameters", text_color="#F0F0F0", fg_color="#3E3E3E", anchor="w", corner_radius=0, command=toggle_parameters)
        button_parameters.pack(fill="x", padx=(0, 0), pady=(10, 0))

        parameters_frame = CTkFrame(left_frame, fg_color="#2E2E2E", corner_radius=0)

        # Point Size
        point_size_frame = CTkFrame(parameters_frame, fg_color="#2E2E2E", corner_radius=0)
        point_size_frame.pack(fill="x",padx=(10, 10), pady=(10, 0))
        label_point_size = CTkLabel(point_size_frame, text="Point Size:", text_color="#F0F0F0")
        entry_point_size = CTkEntry(point_size_frame, width=50)
        label_point_size.pack(side="left", padx=(10, 5))
        entry_point_size.pack(side="left", padx=(0, 5))

        # Dosis Elevation
        dosis_elevation_frame = CTkFrame(parameters_frame, fg_color="#2E2E2E", corner_radius=0)
        dosis_elevation_frame.pack(fill="x",padx=(10, 10), pady=(10, 0))
        label_dosis_elevation = CTkLabel(dosis_elevation_frame, text="Dosis Elevation:", text_color="#F0F0F0")
        label_dosis_elevation.pack(side="left", padx=(10, 5))

        def update_slider_label(value):
            slider_label.configure(text=f"{value:.2f}")

        slider = CTkSlider(dosis_elevation_frame, from_=-100, to=100, command=update_slider_label)
        slider.pack(side="left", padx=(0, 5))
        slider_label = CTkLabel(dosis_elevation_frame, text="0.00", text_color="#F0F0F0")
        slider_label.pack(side="left", padx=(0, 5))

        # Voxelizer
        voxelizer_frame = CTkFrame(parameters_frame, fg_color="#252525", corner_radius=0)
        voxelizer_frame.pack(fill="x", padx=(10, 10), pady=(10, 0))
        voxelizer_frame.grid_columnconfigure(0, weight=1)
        voxelizer_frame.grid_columnconfigure(1, weight=1)
        voxelizer_frame.grid_columnconfigure(2, weight=0)
        voxelizer_frame.grid_columnconfigure(3, weight=1)
        label_voxelizer = CTkLabel(voxelizer_frame, text="Voxelizer:", text_color="#F0F0F0")
        label_voxelizer.grid(row=0, column=1, padx=(10, 5), pady=(5, 0), sticky="e")
        voxelizer_switch = CTkSwitch(voxelizer_frame, text="")
        voxelizer_switch.grid(row=0, column=2, padx=(0, 5), pady=(5, 0), sticky="w")
        voxelizerSize_frame = CTkFrame(parameters_frame, fg_color="#1E1E1E", corner_radius=0)
        voxelizerSize_frame.pack(fill="x", padx=(10, 10), pady=(0, 0))
        label_vox_size = CTkLabel(voxelizerSize_frame, text="Vox Size:", text_color="#F0F0F0")
        label_vox_size.grid(row=1, column=0, padx=(10, 5), pady=(10, 10), sticky="w")
        entry_vox_size = CTkEntry(voxelizerSize_frame, width=50)
        entry_vox_size.grid(row=1, column=1, padx=(0, 5), pady=(10, 10), sticky="w")

        # Dose Layer

        # Ensure the window is maximized
        self.after(0, lambda: self.wm_state('zoomed'))

if __name__ == "__main__":
    pointCloud_app = PointCloudApp()
    pointCloud_app.mainloop()