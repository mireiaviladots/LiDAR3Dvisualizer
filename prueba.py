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

class PointCloudApp(CTk):
    def __init__(self):
        super().__init__()

        self.title("Point Cloud Viewer")
        self.configure(bg="#1E1E1E")

        self.source_location = None
        self.pc_filepath = None
        self.csv_filepath = None
        self.xml_filepath = None
        self.point_size = None
        self.vox_size = None
        self.altura_extra = None
        self.dose_min_csv = None
        self.dose_max_csv = None
        self.low_max = None
        self.medium_min = None
        self.medium_max = None
        self.high_min = None
        self.high_max = None
        self.previous_point_value = ""
        self.previous_voxel_value = ""
        self.show_dose_layer = False
        self.downsample = None
        self.show_source = False
        self.vis = None
        self.heatmap = None
        self.xcenter = None
        self.ycenter = None
        self.Hcenter = None
        self.lonmin = None
        self.lonmax = None
        self.latmin = None
        self.latmax = None
        self.run_prueba = False
        self.right_frame_width = False
        self.right_frame_height = False
        self.screen_width = False
        self.left_frame_width = False
        self.title_bar_height = False

        # Frame izquierdo
        self.left_frame_width = 370
        screen_height = self.winfo_screenheight()
        self.geometry(f"{self.left_frame_width}x{screen_height}+0+0")
        self.left_frame = CTkFrame(self, fg_color="#2E2E2E", corner_radius=0, width=self.left_frame_width,
                                   height=screen_height)
        self.left_frame.pack(side="left", fill="y", expand=True)
        self.left_frame.pack_propagate(False)

        # Frame para los botones del menú
        menu_frame = CTkFrame(self.left_frame, fg_color="#2E2E2E", corner_radius=0)
        menu_frame.pack(pady=(15, 0))

        # Open...
        self.menu_visible = False

        def toggle_menu():
            self.menu_visible = not self.menu_visible

            if self.menu_visible:
                self.btn_open_pc.pack(pady=0)
                self.btn_open_csv.pack(pady=0)
                self.btn_open_xml.pack(pady=0)
            else:
                self.btn_open_pc.pack_forget()
                self.btn_open_csv.pack_forget()
                self.btn_open_xml.pack_forget()

        btn_menu = CTkButton(menu_frame, text="Open ...", command=toggle_menu, fg_color="#3E3E3E")
        btn_menu.pack(pady=(5, 0))

        def load_point_cloud_and_toggle():
            self.load_point_cloud()
            toggle_menu()

        def process_n42_files_and_toggle():
            self.process_n42_files()
            toggle_menu()

        def load_xml_metadata_and_toggle():
            self.load_xml_metadata()
            toggle_menu()

        self.btn_open_pc = CTkButton(menu_frame, text="Point Cloud", text_color="#2E2E2E", fg_color="#F0F0F0",
                                     border_color="#6E6E6E", border_width=1, font=("Arial", 12),
                                     command=load_point_cloud_and_toggle)
        self.btn_open_csv = CTkButton(menu_frame, text="N42 File", text_color="#2E2E2E", fg_color="#F0F0F0",
                                      border_color="#6E6E6E", border_width=2, font=("Arial", 12),
                                      command=process_n42_files_and_toggle)
        self.btn_open_xml = CTkButton(menu_frame, text="XML", text_color="#2E2E2E", fg_color="#F0F0F0",
                                      border_color="#6E6E6E", border_width=1, font=("Arial", 12),
                                      command=load_xml_metadata_and_toggle)

        # Downsample
        downsample_frame = CTkFrame(self.left_frame, fg_color="#2E2E2E", corner_radius=0)
        downsample_frame.pack(pady=(10, 0))
        label_downsample = CTkLabel(downsample_frame, text="Downsample:", text_color="#F0F0F0", font=("Arial", 12))
        self.downsample_entry = CTkEntry(downsample_frame, width=50, font=("Arial", 12))
        label_percent = CTkLabel(downsample_frame, text="%", text_color="#F0F0F0", font=("Arial", 12))
        label_downsample.pack(side="left", padx=(0, 5))
        self.downsample_entry.pack(side="left", padx=(0, 5))
        label_percent.pack(side="left")

        # Parameters Button
        self.parameters_visible = False

        def toggle_parameters():
            self.parameters_visible = not self.parameters_visible

            if self.menu_visible:
                toggle_menu()

            if self.parameters_visible:
                button_parameters.configure(text=" ▲ Parameters")
                parameters_frame.pack(pady=(10, 0), fill="x")
                button_dose_layer.pack_forget()
                button_dose_layer.pack(fill="x", padx=(0, 0), pady=(10, 0))
                button_extra_computations.pack_forget()
                button_extra_computations.pack(fill="x", padx=(0, 0), pady=(10, 0))
                self.btn_visualize.pack_forget()
                self.btn_visualize.pack(side="bottom", padx=(0, 0), pady=(10, 25))

                if self.dose_layer_visible:
                    button_dose_layer.pack_forget()
                    button_dose_layer.pack(fill="x", padx=(0, 0), pady=(10, 0))
                    dose_layer_frame.pack_forget()
                    dose_layer_frame.pack(pady=(5, 0), fill="x")
                    button_extra_computations.pack_forget()
                    button_extra_computations.pack(fill="x", padx=(0, 0), pady=(10, 0))
                    self.btn_visualize.pack_forget()
                    self.btn_visualize.pack(side="bottom", padx=(0, 0), pady=(10, 25))
            else:
                button_parameters.configure(text=" ▼ Parameters")
                parameters_frame.pack_forget()

            if self.extra_computations_visible:
                extra_computations_frame.pack_forget()
                extra_computations_frame.pack(pady=(10, 0), fill="x")
                self.btn_visualize.pack_forget()
                self.btn_visualize.pack(side="bottom", padx=(0, 0), pady=(10, 25))

        # Parameters Button
        button_parameters = CTkButton(self.left_frame, text=" ▼ Parameters", text_color="#F0F0F0", fg_color="#3E3E3E",
                                      anchor="w", corner_radius=0, command=toggle_parameters)
        button_parameters.pack(fill="x", padx=(0, 0), pady=(10, 0))

        parameters_frame = CTkFrame(self.left_frame, fg_color="#2E2E2E", corner_radius=0)

        # Point Size
        point_size_frame = CTkFrame(parameters_frame, fg_color="#2E2E2E", corner_radius=0)
        point_size_frame.pack(fill="x", padx=(10, 10), pady=(0, 0))
        label_point_size = CTkLabel(point_size_frame, text="Point Size:", text_color="#F0F0F0", font=("Arial", 12))
        self.point_size_entry = CTkEntry(point_size_frame, width=50, font=("Arial", 12), state="disabled")
        label_point_size.pack(side="left", padx=(10, 5))
        self.point_size_entry.pack(side="left", padx=(0, 5))

        # Voxelizer
        voxelizer_frame = CTkFrame(parameters_frame, fg_color="#252525", corner_radius=0)
        voxelizer_frame.pack(fill="x", padx=(10, 10), pady=(5, 0))
        voxelizer_frame.grid_columnconfigure(0, weight=1)
        voxelizer_frame.grid_columnconfigure(1, weight=1)
        voxelizer_frame.grid_columnconfigure(2, weight=0)
        voxelizer_frame.grid_columnconfigure(3, weight=1)
        label_voxelizer = CTkLabel(voxelizer_frame, text="Voxelizer:", text_color="#F0F0F0", font=("Arial", 12))
        label_voxelizer.grid(row=0, column=1, padx=(10, 5), pady=(5, 0), sticky="e")
        self.voxelizer_var = BooleanVar()
        self.voxelizer_switch = CTkSwitch(voxelizer_frame, variable=self.voxelizer_var, command=self.toggle_voxel_size,
                                          text="", state="disabled")
        self.voxelizer_switch.grid(row=0, column=2, padx=(0, 5), pady=(5, 0), sticky="w")
        voxelizerSize_frame = CTkFrame(parameters_frame, fg_color="#1E1E1E", corner_radius=0)
        voxelizerSize_frame.pack(fill="x", padx=(10, 10), pady=(0, 0))
        label_vox_size = CTkLabel(voxelizerSize_frame, text="Vox Size:", text_color="#F0F0F0", font=("Arial", 12))
        label_vox_size.grid(row=1, column=0, padx=(10, 5), pady=(5, 5), sticky="w")
        self.vox_size_entry = CTkEntry(voxelizerSize_frame, width=50, font=("Arial", 12), state="disabled")
        self.vox_size_entry.grid(row=1, column=1, padx=(0, 5), pady=(5, 5), sticky="w")

        # Dose Layer
        self.dose_layer_visible = False

        def toggle_dose_layer():
            self.dose_layer_visible = not self.dose_layer_visible

            if self.menu_visible:
                toggle_menu()

            if self.dose_layer_visible:
                button_dose_layer.configure(text=" ▲ Dose Layer")
                dose_layer_frame.pack(pady=(5, 0), fill="x")
                button_extra_computations.pack_forget()
                button_extra_computations.pack(fill="x", padx=(0, 0), pady=(10, 0))
                self.btn_visualize.pack_forget()
                self.btn_visualize.pack(side="bottom", padx=(0, 0), pady=(10, 25))
            else:
                button_dose_layer.configure(text=" ▼ Dose Layer")
                dose_layer_frame.pack_forget()

            if self.extra_computations_visible:
                extra_computations_frame.pack_forget()
                extra_computations_frame.pack(pady=(10, 0), fill="x")
                self.btn_visualize.pack_forget()
                self.btn_visualize.pack(side="bottom", padx=(0, 0), pady=(10, 25))

        button_dose_layer = CTkButton(self.left_frame, text=" ▼ Dose Layer", text_color="#F0F0F0", fg_color="#3E3E3E",
                                      anchor="w", corner_radius=0, command=toggle_dose_layer)
        button_dose_layer.pack(fill="x", padx=(0, 0), pady=(10, 0))

        # Dose Layer
        dose_layer_frame = CTkFrame(self.left_frame, fg_color="#2E2E2E", corner_radius=0)

        yes_no_frame = CTkFrame(dose_layer_frame, fg_color="#2E2E2E", corner_radius=0)
        yes_no_frame.pack(pady=(5, 0), anchor="center")

        label_no = CTkLabel(yes_no_frame, text="No", text_color="#F0F0F0", font=("Arial", 12))
        label_no.pack(side="left", padx=(5, 5))  # Reducido el espacio después de "No"

        self.dose_layer_switch = CTkSwitch(yes_no_frame, text="", command=self.toggle_dose_layer, state='disabled',
                                           width=36, height=18)
        self.dose_layer_switch.pack(side="left", padx=(5, 2))  # Menos espacio entre switch y "Yes"

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

        self.dosis_slider = CTkSlider(dosis_elevation_frame, from_=-100, to=100, command=update_slider_label,
                                      state="disabled")
        self.dosis_slider.set(1)
        self.dosis_slider.pack(side="left", padx=(0, 5))
        slider_label = CTkLabel(dosis_elevation_frame, text="1.00", text_color="#F0F0F0")
        slider_label.pack(side="left", padx=(0, 5))

        dose_sections_frame = CTkFrame(dose_layer_frame, fg_color="#2E2E2E", corner_radius=0)
        dose_sections_frame.pack(fill="x", pady=(5, 0), anchor="center")

        self.color_options = ["red", "yellow", "green", "blue", "purple", "orange", "cyan", "magenta", "pink", "white"]

        self.high_min_medium_max = StringVar()
        self.medium_min_low_max = StringVar()

        # High Dose
        label_high_dose = CTkLabel(dose_sections_frame, text="High Dose:", text_color="#F0F0F0", font=("Arial", 12))
        label_high_dose.grid(row=0, column=0, padx=(10, 5), sticky="ew")
        self.high_dose_cb = CTkComboBox(dose_sections_frame, values=self.color_options, font=("Arial", 12), width=90,
                                        state="disabled")
        self.high_dose_cb.set("red")
        self.high_dose_cb.grid(row=0, column=1, padx=(0, 5), sticky="ew")
        self.high_dose_rgb = np.array(mcolors.to_rgb("red"))
        label_min = CTkLabel(dose_sections_frame, text="Min:", text_color="#F0F0F0", font=("Arial", 12))
        label_min.grid(row=0, column=2, padx=(0, 5), sticky="ew")
        self.high_dose_min = CTkEntry(dose_sections_frame, width=50, font=("Arial", 11),
                                      textvariable=self.high_min_medium_max, state="disabled")
        self.high_dose_min.grid(row=0, column=3, padx=(0, 5), sticky="ew")
        label_max = CTkLabel(dose_sections_frame, text="Max:", text_color="#F0F0F0", font=("Arial", 12))
        label_max.grid(row=0, column=4, padx=(0, 5), sticky="ew")
        self.high_dose_max = CTkEntry(dose_sections_frame, width=50, font=("Arial", 11), state="disabled")
        self.high_dose_max.grid(row=0, column=5, padx=(0, 5), sticky="ew")

        # Medium Dose
        label_medium_dose = CTkLabel(dose_sections_frame, text="Medium Dose:", text_color="#F0F0F0", font=("Arial", 12))
        label_medium_dose.grid(row=1, column=0, padx=(10, 5), sticky="ew")
        self.medium_dose_cb = CTkComboBox(dose_sections_frame, values=self.color_options, font=("Arial", 12), width=90,
                                          state="disabled")
        self.medium_dose_cb.set("yellow")
        self.medium_dose_cb.grid(row=1, column=1, padx=(0, 5), sticky="ew")
        self.medium_dose_rgb = np.array(mcolors.to_rgb("yellow"))
        label_min_medium = CTkLabel(dose_sections_frame, text="Min:", text_color="#F0F0F0", font=("Arial", 12))
        label_min_medium.grid(row=1, column=2, padx=(0, 5), sticky="ew")
        self.medium_dose_min = CTkEntry(dose_sections_frame, width=50, font=("Arial", 11),
                                        textvariable=self.medium_min_low_max, state="disabled")
        self.medium_dose_min.grid(row=1, column=3, padx=(0, 5), sticky="ew")
        label_max_medium = CTkLabel(dose_sections_frame, text="Max:", text_color="#F0F0F0", font=("Arial", 12))
        label_max_medium.grid(row=1, column=4, padx=(0, 5), sticky="ew")
        self.medium_dose_max = CTkEntry(dose_sections_frame, width=50, font=("Arial", 11),
                                        textvariable=self.high_min_medium_max, state="disabled")
        self.medium_dose_max.grid(row=1, column=5, padx=(0, 5), sticky="ew")

        # Low Dose
        label_low_dose = CTkLabel(dose_sections_frame, text="Low Dose:", text_color="#F0F0F0", font=("Arial", 12))
        label_low_dose.grid(row=2, column=0, padx=(10, 5), sticky="ew")
        self.low_dose_cb = CTkComboBox(dose_sections_frame, values=self.color_options, font=("Arial", 12), width=90,
                                       state="disabled")
        self.low_dose_cb.set("green")
        self.low_dose_cb.grid(row=2, column=1, padx=(0, 5), sticky="ew")
        self.low_dose_rgb = np.array(mcolors.to_rgb("green"))
        label_min_low = CTkLabel(dose_sections_frame, text="Min:", text_color="#F0F0F0", font=("Arial", 12))
        label_min_low.grid(row=2, column=2, padx=(0, 5), sticky="ew")
        self.low_dose_min = CTkEntry(dose_sections_frame, width=50, font=("Arial", 11), state="disabled")
        self.low_dose_min.grid(row=2, column=3, padx=(0, 5), sticky="ew")
        label_max_low = CTkLabel(dose_sections_frame, text="Max:", text_color="#F0F0F0", font=("Arial", 12))
        label_max_low.grid(row=2, column=4, padx=(0, 5), sticky="ew")
        self.low_dose_max = CTkEntry(dose_sections_frame, width=50, font=("Arial", 11),
                                     textvariable=self.medium_min_low_max, state="disabled")
        self.low_dose_max.grid(row=2, column=5, padx=(0, 5), sticky="ew")

        # Source
        source_frame = CTkFrame(dose_layer_frame, fg_color="#2E2E2E", corner_radius=0)
        source_frame.pack(fill="x", pady=(5, 0))
        self.btn_find_source = CTkButton(source_frame, text="Find Radioactive Source", fg_color="#3E3E3E",
                                         text_color="#F0F0F0", font=("Arial", 12), command=self.find_radioactive_source)
        self.btn_find_source.grid(row=0, column=0, padx=(10, 5), pady=(5, 0), sticky="w")
        show_source_label = CTkLabel(source_frame, text="Show Source on Map:", text_color="#F0F0F0", font=("Arial", 12))
        show_source_label.grid(row=0, column=1, padx=(10, 5), pady=(5, 0), sticky="w")
        self.show_source_switch = CTkSwitch(source_frame, text="", command=self.toggle_source, state='disabled')
        self.show_source_switch.grid(row=0, column=2, padx=(10, 5), pady=(5, 0), sticky="w")

        # Extra Computations
        self.extra_computations_visible = False

        def toggle_extra_computations():
            self.extra_computations_visible = not self.extra_computations_visible

            if self.menu_visible:
                toggle_menu()

            if self.extra_computations_visible:
                button_extra_computations.configure(text=" ▲ Extra Computations")
                extra_computations_frame.pack(pady=(10, 0), fill="x")
                self.btn_visualize.pack_forget()
                self.btn_visualize.pack(side="bottom", padx=(0, 0), pady=(10, 25))
            else:
                button_extra_computations.configure(text=" ▼ Extra Computations")
                extra_computations_frame.pack_forget()
                self.btn_visualize.pack_forget()
                self.btn_visualize.pack(side="bottom", padx=(0, 0), pady=(10, 25))

        button_extra_computations = CTkButton(self.left_frame, text=" ▼ Extra Computations", text_color="#F0F0F0",
                                              fg_color="#3E3E3E",
                                              anchor="w", corner_radius=0, command=toggle_extra_computations)
        button_extra_computations.pack(fill="x", padx=(0, 0), pady=(10, 0))

        extra_computations_frame = CTkFrame(self.left_frame, fg_color="#2E2E2E", corner_radius=0)

        self.btn_heatmap = CTkButton(extra_computations_frame, text="Heatmap H*(10) rate", fg_color="#3E3E3E",
                                     text_color="#F0F0F0", font=("Arial", 12), command=self.plot_heatmap)
        self.btn_heatmap.pack(fill="x", padx=(80, 80), pady=(5, 0))
        self.btn_three_colors = CTkButton(extra_computations_frame, text="Heatmap with Three Color Range",
                                          fg_color="#3E3E3E",
                                          text_color="#F0F0F0", font=("Arial", 12),
                                          command=self.plot_three_color_heatmap)
        self.btn_three_colors.pack(fill="x", padx=(80, 80), pady=(5, 0))
        self.btn_convert_pcd_to_dat = CTkButton(extra_computations_frame, text="3D grid from PCD", fg_color="#3E3E3E",
                                                text_color="#F0F0F0",
                                                font=("Arial", 12), command=self.set_run_prueba_flag)
        self.btn_convert_pcd_to_dat.pack(fill="x", padx=(80, 80), pady=(5, 0))

        # Visualize
        self.btn_visualize = CTkButton(self.left_frame, text="Visualize", text_color="#F0F0F0", fg_color="#1E3A5F",
                                       hover_color="#2E4A7F",
                                       anchor="center", corner_radius=0, border_color="#D3D3D3", border_width=2,
                                       command=self.visualize)
        self.btn_visualize.pack(side="bottom", padx=(0, 0), pady=(10, 25))

    def show_dose_layer_message(self):
        messagebox.showinfo("Information", "Please select a N42 file.")

    def toggle_dose_layer(self):
        if self.dose_layer_switch.get() == 1:
            self.show_dose_layer = True
            self.low_dose_max.configure(state="normal")
            self.medium_dose_min.configure(state="normal")
            self.medium_dose_max.configure(state="normal")
            self.high_dose_min.configure(state="normal")
            self.low_dose_cb.configure(state="normal")
            self.medium_dose_cb.configure(state="normal")
            self.high_dose_cb.configure(state="normal")
            self.dosis_slider.configure(state="normal")
            if not self.low_dose_cb.get():
                self.low_dose_cb.set("green")
                self.high_dose_cb.set("red")
                self.medium_dose_cb.set("yellow")
            if self.source_location is not None:
                self.show_source_switch.configure(state="normal")
        else:
            self.show_dose_layer = False
            self.low_dose_max.configure(state="disabled")
            self.medium_dose_min.configure(state="disabled")
            self.medium_dose_max.configure(state="disabled")
            self.high_dose_min.configure(state="disabled")
            self.low_dose_cb.configure(state="disabled")
            self.medium_dose_cb.configure(state="disabled")
            self.high_dose_cb.configure(state="disabled")

    def toggle_source(self):
        if self.show_source_switch.get() == 1:
            self.show_source = True
        else:
            self.show_source = False

    def toggle_voxel_size(self):
        if self.voxelizer_var.get():
            self.previous_point_value = self.point_size_entry.get()
            self.vox_size_entry.delete(0, "end")
            self.point_size_entry.delete(0, "end")
            self.point_size_entry.configure(state="disabled")
            self.vox_size_entry.configure(state="normal")
            if self.previous_voxel_value == "":
                self.vox_size_entry.insert(0, 2)
            else:
                self.vox_size_entry.insert(0, self.previous_voxel_value)
        else:
            self.previous_voxel_value = self.vox_size_entry.get()
            self.vox_size_entry.delete(0, "end")
            self.point_size_entry.delete(0, "end")
            self.vox_size_entry.configure(state="disabled")
            self.point_size_entry.configure(state="normal")
            if self.previous_point_value == "":
                self.point_size_entry.insert(0, 2)
            else:
                self.point_size_entry.insert(0, self.previous_point_value)

    def load_point_cloud(self):
        self.point_size_entry.delete(0, "end")
        filepath = filedialog.askopenfilename(filetypes=[("PCD Files", "*.pcd")])
        if filepath:
            self.pc_filepath = filepath
            print("Point Cloud Selected:", self.pc_filepath)
            self.point_size_entry.configure(state="normal")
            self.point_size_entry.insert(0, 2)
            self.voxelizer_switch.configure(state="normal")

    # def load_csv_dosis(self):
    # filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    # if filepath:
    # self.csv_filepath = filepath
    # print("CSV Selected:", self.csv_filepath)
    # dosis_values = np.genfromtxt(filepath, delimiter=',', skip_header=1)[:, 2]
    # self.dose_min_csv, self.dose_max_csv = np.min(dosis_values), np.max(dosis_values)
    # print(f"Dosis Range: Min={self.dose_min_csv}, Max={self.dose_max_csv}")

    # Asignar valores a los campos de Min y Max y deshabilitarlos
    # self.low_dose_min.configure(state="normal")
    # self.low_dose_min.delete(0, "end")
    # self.low_dose_min.insert(0, str(self.dose_min_csv))
    # self.low_dose_min.configure(state="disabled")

    # self.high_dose_max.configure(state="normal")
    # self.high_dose_max.delete(0, "end")
    # self.high_dose_max.insert(0, str(self.dose_max_csv))
    # self.high_dose_max.configure(state="disabled")

    # self.low_dose_max.configure(state="normal")
    # self.medium_dose_min.configure(state="normal")
    # self.medium_dose_max.configure(state="normal")
    # self.high_dose_min.configure(state="normal")

    def load_xml_metadata(self):
        filepath = filedialog.askopenfilename(filetypes=[("XML Files", "*.xml")])
        if filepath:
            self.xml_filepath = filepath
            print("XML Selected:", self.xml_filepath)

    def process_n42_files(self):
        folder_path = filedialog.askdirectory(title="Select Folder with .n42 Files")
        pathN42 = folder_path
        pathN42mod = os.path.join(folder_path)

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

        self.xcenter = [None] * 100000
        self.ycenter = [None] * 100000

        intx = [None] * 100000
        inty = [None] * 100000
        Hmax = [None] * 100000
        self.Hcenter = [None] * 100000

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
            root = tree.getroot()

            # Read Start Date Time, LiveTime, DeadTime, ChannelData
            for each in root.findall('.//RadMeasurement'):
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
            for each in root.findall('.//EnergyCalibration'):
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
            for each in root.findall('.//GeographicPoint'):
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

                # Calculate de conversion coefficent for the energy nGy/h per cps 38 mm
                # Conv_coeff[i]=0
                # if ((En_ch[i]>=30) and (En_ch[i]<=55)):
                # Conv_coeff[i]=  289.49*En_ch[i]**( -2.0699)

                # if ((En_ch[i]>55) and (En_ch[i] <=350)):
                # Conv_coeff[i]= -8.4577E-13*En_ch[i]**5+8.895E-10*En_ch[i]**4-3.45103E-07*En_ch[i]**3+6.8127E-05*En_ch[i]**2-5.5121E-03*En_ch[i]+2.321E-01
                # if ((En_ch[i]>350) and (En_ch[i] <=3000)):
                #             #Conv_coeff[i]=7.994448817E-20*En_ch[i]**6-8.688859196E-16*En_ch[i]**5+3.673134977E-12*En_ch[i]**4-7.579115501E-09*En_ch[i]**3+7.387567866E-06*En_ch[i]**2-0.0006714962472*En_ch[i]-0.0454215177
                # Conv_coeff[i]= 1.3604E-10*En_ch[i]**3-1.7289E-06*En_ch[i]**2+7.9658E-03*En_ch[i]-1.8388
                #         print ('E = ',En_ch[i], 'w = ',Conv_coeff[i])

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

            # u_eff de acuerod con los coeficientes másicos attenuacion del aire con densidad de 1.2E-03 g/cm3
            #     u_eff_conv = 0.1344*(662)**(-0.4227)
            #     u_eff_conv = 0.012

            # utilizando la integral exponencial de 2do orden
            #     u_eff_conv = 0.005
            #     H10_conv_1m=H10_conv_meas*sci.expn(2,u_eff_conv*1)/sci.expn(2,u_eff_conv*FAltitude)

            for dose in root.iter('DoseRateValue'):
                # giving the value.
                dose.text = str(Dose_conv_meas)
                # dose.set('unit','nGy/h')

            for Ader in root.iter('AmbientDoseEquivalentRateValue'):
                # giving the value.
                Ader.text = str(H10_conv_meas)
            #       Ader.set('unit','nSv/h')

            for Ader1m in root.iter('AmbientDoseEquivalentRateValue_1m'):
                # giving the value.
                Ader1m.text = str(H10_conv_1m)
            #       dose.set('unit','nSv/h')

            for Man_Made in root.iter('MMGC'):
                # giving the value.
                Man_Made.text = str(MMGC)

            for uMan_Made in root.iter('uncertainty_MMGC'):
                # giving the value.
                uMan_Made.text = str(u_MMGC)

            # tranform x,y
            x0, y0, zone_number, zone_letter = utm.from_latlon(FLatitude, FLongitude, )
            # print('center projection in utm (meters): ', x0, y0)

            self.xcenter[cont] = x0
            self.ycenter[cont] = y0
            self.Hcenter[cont] = H10_conv_meas
            FAltcenter[cont] = FAltitude

            if cont == 1:
                self.latmin = y0
                self.latmax = y0
                self.lonmin = x0
                self.lonmax = x0

            # looking for max and min lat l

            if x0 < self.lonmin:
                self.lonmin = x0
            if x0 > self.lonmax:
                self.lonmax = x0
            if y0 < self.latmin:
                self.latmin = y0
            if y0 > self.latmax:
                self.latmax = y0

            os.chdir(pathN42mod)
            tree.write(file)

        self.lonmin = self.lonmin - 50
        self.lonmax = self.lonmax + 50
        self.latmin = self.latmin - 50
        self.latmax = self.latmax + 50

        # Verifica si hay NaN en los datos
        # print(type(xcenter))
        # print(type(ycenter))
        # print(type(Hcenter))

        self.xcenter = np.array(self.xcenter, dtype=float)
        self.ycenter = np.array(self.ycenter, dtype=float)
        self.Hcenter = np.array(self.Hcenter, dtype=float)

        # conversion to string and numbers to floats
        self.xcenter = np.array([float(i) for i in self.xcenter if str(i).replace('.', '', 1).isdigit()])
        self.ycenter = np.array([float(i) for i in self.ycenter if str(i).replace('.', '', 1).isdigit()])
        self.Hcenter = np.array([float(i) for i in self.Hcenter if str(i).replace('.', '', 1).isdigit()])
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
        max_value = max(self.Hcenter)

        # Encuentra el índice correspondiente al valor máximo
        # cont_max = Hcenter.index(max_value)
        cont_max = np.argmax(self.Hcenter)

        # Calculo del maximo valor de H*(10) suponiento que es firnte puntual
        HmaxP = self.Hcenter[cont_max] * FAltcenter[cont_max] * FAltcenter[cont_max]

        # Define una cuadrícula para el área de interés
        Resolution = 50
        ygrid = np.linspace(self.latmin, self.latmax, Resolution)
        xgrid = np.linspace(self.lonmin, self.lonmax, Resolution)
        xmesh, ymesh = np.meshgrid(xgrid, ygrid)

        # Inicializar el mapa con valores muy bajos
        self.heatmap = np.full(xmesh.shape, -np.inf)

        # Iterar sobre cada circunferencia
        for xc, yc, radius, hval in zip(self.xcenter, self.ycenter, FAltcenter, self.Hcenter):
            # Distancia de cada punto de la cuadrícula al centro de la circunferencia
            distance = np.sqrt((xmesh - xc) ** 2 + (ymesh - yc) ** 2)
            # Máscara para identificar puntos dentro del círculo
            mask = distance <= radius
            # Actualizar el valor máximo en el mapa
            self.heatmap[mask] = np.maximum(self.heatmap[mask], hval)

        # Configurar los valores mínimos para que sean visibles (si es necesario)
        self.heatmap[self.heatmap == -np.inf] = np.nan

        # Write to CSV
        output_filename = "dose_data_pla_20m_2ms.csv"
        self.csv_filepath = output_filename
        with open(output_filename, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Latitude", "Longitude", "Dose"])  # Header row

            # Flatten the arrays for iteration
            for i in range(xmesh.shape[0]):
                for j in range(xmesh.shape[1]):
                    writer.writerow([xmesh[i, j], ymesh[i, j], self.heatmap[i, j]])

        # print('------------------------------------', '\n')
        # print('Total number of analysed spectra : ', cont, '\n')

        # Procesar el CSV existente
        dosis_values = np.genfromtxt(self.csv_filepath, delimiter=',', skip_header=1, usecols=2)
        dosis_values = dosis_values[~np.isnan(dosis_values)]  # Eliminar NaN
        self.dose_min_csv, self.dose_max_csv = np.min(dosis_values), np.max(dosis_values)
        # print(f"Dosis Range: Min={self.dose_min_csv}, Max={self.dose_max_csv}")

        # Asignar valores a los campos de Min y Max y deshabilitarlos
        self.low_dose_min.configure(state="normal")
        self.low_dose_min.delete(0, "end")
        self.low_dose_min.insert(0, str(self.dose_min_csv))
        self.low_dose_min.configure(state="disabled")

        self.high_dose_max.configure(state="normal")
        self.high_dose_max.delete(0, "end")
        self.high_dose_max.insert(0, str(self.dose_max_csv))
        self.high_dose_max.configure(state="disabled")

        resta = self.dose_max_csv - self.dose_min_csv
        ranges = resta/3

        self.low_dose_max.configure(state="normal")
        self.low_dose_max.delete(0, "end")
        self.low_dose_max.insert(0, f"{ranges + self.dose_min_csv:.2f}")
        self.low_dose_max.configure(state="disabled")

        self.medium_dose_max.configure(state="normal")
        self.medium_dose_max.delete(0, "end")
        self.medium_dose_max.insert(0, f"{(2 * ranges) + self.dose_min_csv:.2f}")
        self.medium_dose_max.configure(state="disabled")

        self.dose_layer_switch.configure(state="normal")

        # print('****END PROGRAM *****')

    def find_radioactive_source(self):
        if not self.csv_filepath:
            messagebox.showwarning("Warning", "Please select a N42 file.")
            return

        utm_coords = np.genfromtxt(self.csv_filepath, delimiter=',', skip_header=1)
        # Filter out rows with NaN values in the dose column
        utm_coords = utm_coords[~np.isnan(utm_coords[:, 2])]
        ga = GeneticAlgorithm(utm_coords)
        source_location = ga.run()
        self.source_location = source_location
        print(f"Estimated source location: Easting = {source_location[0]}, Northing = {source_location[1]}")
        messagebox.showinfo("Source Location",
                            f"Estimated source location: Easting = {source_location[0]}, Northing = {source_location[1]}")
        self.show_source_switch.configure(state="normal")

    def plot_heatmap(self):
        # Ensure the necessary data is available
        if not hasattr(self, 'heatmap') or self.heatmap is None or not hasattr(self,
                                                                               'xcenter') or self.xcenter is None or not hasattr(
            self, 'ycenter') or self.ycenter is None or not hasattr(self, 'Hcenter') or self.Hcenter is None:
            messagebox.showerror("Error", "Please process the N42 files first.")
            return

        # Visualize the heatmap
        plt.imshow(
            self.heatmap,
            extent=(self.lonmin, self.lonmax, self.latmin, self.latmax),
            origin='lower',
            cmap='viridis',
            alpha=0.8
        )
        plt.colorbar(label='H*(10) rate nSv/h')
        plt.title('Heatmap H*(10) rate')
        plt.xlabel('LONGITUDE')
        plt.ylabel('LATITUDE')

        # Add colored points
        plt.scatter(
            self.xcenter, self.ycenter,
            c=self.Hcenter, cmap='viridis',
            edgecolor='black', s=50, label='Measurement'
        )

        # Add grid
        plt.grid(visible=True, color='black', linestyle='--', linewidth=0.5)

        plt.legend()
        plt.show()

    def plot_three_color_heatmap(self):
        # Ensure the necessary data is available
        if not hasattr(self, 'heatmap') or self.heatmap is None or not hasattr(self,
                                                                               'xcenter') or self.xcenter is None or not hasattr(
            self, 'ycenter') or self.ycenter is None or not hasattr(self, 'Hcenter') or self.Hcenter is None:
            messagebox.showerror("Error", "Please process the N42 files first.")
            return

        if self.dose_layer_switch.get() == 1:
            # Define the color map and boundaries
            low_dose_color = self.low_dose_cb.get() if self.low_dose_cb.get() else 'green'
            medium_dose_color = self.medium_dose_cb.get() if self.medium_dose_cb.get() else 'yellow'
            high_dose_color = self.high_dose_cb.get() if self.high_dose_cb.get() else 'red'

            # Define the color map and boundaries
            colors = [low_dose_color, medium_dose_color, high_dose_color]

            R0 = 0

            try:
                R1 = float(self.low_dose_max.get()) if self.low_dose_max.get() else 80
            except ValueError:
                R1 = 80  # Default value

            try:
                R2 = float(self.medium_dose_max.get()) if self.medium_dose_max.get() else 120
            except ValueError:
                R2 = 120  # Default value

        else:
            colors = ['green', 'yellow', 'red']
            R0 = 0
            R1 = 80
            R2 = 120

        R3 = max(self.Hcenter) * max(self.Hcenter)  # Assuming HmaxP is calculated similarly
        bounds = [R0, R1, R2, R3]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(bounds, cmap.N)

        # Visualize the heatmap
        plt.imshow(
            self.heatmap,
            extent=(self.lonmin, self.lonmax, self.latmin, self.latmax),
            origin='lower',
            cmap=cmap,
            norm=norm,
            alpha=0.8
        )
        plt.colorbar(
            label='H*(10) rate (nSv/h)',
            boundaries=bounds,
            ticks=[R0, R0 + (R1 - R0) / 2, R1, R1 + (R2 - R1) / 2, R2, R2 + (R3 - R2) / 2, R3]
            # Intermediate values for the legend
        )
        plt.title('Heatmap with Three Color Range')
        plt.xlabel('LONGITUDE')
        plt.ylabel('LATITUDE')

        # Add colored points
        plt.scatter(
            self.xcenter, self.ycenter,
            c=self.Hcenter, cmap=cmap, norm=norm,
            edgecolor='black', s=50, label='Measurement'
        )

        # Add grid
        plt.grid(visible=True, color='black', linestyle='--', linewidth=0.5)

        plt.legend()
        plt.show()

    def visualize(self):
        if not self.pc_filepath:
            messagebox.showwarning("Warning", "Please select a Point Cloud.")
            return

        if self.dose_layer_switch.get() == 1 and not self.csv_filepath:
            messagebox.showerror("Error", "Please select a N42 file.")
            return

        if self.dose_layer_switch.get() == 1 and not self.xml_filepath:
            messagebox.showerror("Error", "Please select an XML.")
            return

        self.validate_dose_ranges()

        downsample_value = self.downsample_entry.get()
        if downsample_value:
            self.downsample = float(downsample_value)
        else:
            self.downsample = None

        use_voxelization = self.voxelizer_switch.get() == 1

        point_size_str = self.point_size_entry.get().strip()
        vox_size_str = self.vox_size_entry.get().strip()
        self.altura_extra = self.dosis_slider.get()

        if use_voxelization:
            if vox_size_str == "":
                self.vox_size_entry.insert(0, 2)
        else:
            if point_size_str == "":
                self.point_size_entry.insert(0, 2)

        if point_size_str == "":
            self.point_size = 2
        else:
            self.point_size = float(point_size_str)
            if self.point_size <= 0:
                raise ValueError("Point size must be positive.")

        if vox_size_str == "":
            self.vox_size = 2
        else:
            self.vox_size = float(vox_size_str)
            if self.vox_size <= 0:
                raise ValueError("Voxel size must be positive.")

        if self.show_dose_layer:
            high_dose_color = self.high_dose_cb.get()
            medium_dose_color = self.medium_dose_cb.get()
            low_dose_color = self.low_dose_cb.get()

            self.high_dose_rgb = np.array(mcolors.to_rgb(high_dose_color))
            self.medium_dose_rgb = np.array(mcolors.to_rgb(medium_dose_color))
            self.low_dose_rgb = np.array(mcolors.to_rgb(low_dose_color))
        else:
            self.high_dose_rgb = None
            self.medium_dose_rgb = None
            self.low_dose_rgb = None

        self.run_prueba = False

        self.right_frame_width = self.winfo_screenwidth() - self.left_frame_width
        self.right_frame_height = self.left_frame.winfo_height()

        self.left_frame_width = self.left_frame.winfo_width()

        user32 = ctypes.windll.user32
        self.title_bar_height = user32.GetSystemMetrics(4)

        # Create a separate thread for the visualization
        thread = threading.Thread(target=run_visualizer,
                                  args=(self.pc_filepath, self.csv_filepath, self.xml_filepath, use_voxelization,
                                        self.point_size, self.vox_size, self.altura_extra,
                                        self.high_dose_rgb, self.medium_dose_rgb, self.low_dose_rgb,
                                        self.dose_min_csv, self.low_max, self.medium_min, self.medium_max,
                                        self.high_min, self.high_max,
                                        self.show_dose_layer, self.downsample, self.source_location,
                                        self.show_source, self.run_prueba,
                                        self.right_frame_width, self.right_frame_height, self.left_frame_width,
                                        self.title_bar_height))
        thread.start()

    def validate_dose_ranges(self):
        """
        Validates that the dose ranges have logical values.
        """
        if not self.show_dose_layer:
            return

        try:
            self.low_max = float(self.low_dose_max.get())
            self.medium_min = float(self.medium_dose_min.get())
            self.medium_max = float(self.medium_dose_max.get())
            self.high_min = float(self.high_dose_min.get())
        except ValueError:
            messagebox.showerror("Error", "Dose range values must be numeric.")
            raise ValueError("Dose range values must be numeric.")

        if not (
                self.dose_min_csv <= self.low_max <= self.medium_min <= self.medium_max <= self.high_min <= self.dose_max_csv):
            messagebox.showerror("Error",
                                 "Dose ranges are not logical. Ensure: min < low_max < medium_min < medium_max < high_min < max.")
            raise ValueError("Dose ranges are not logical.")

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

    def get_dose_color(self, dosis_nube, high_dose_rgb, medium_dose_rgb, low_dose_rgb, dose_min_csv, low_max,
                       medium_min, medium_max, high_min):
        """ Asigna colores a los puntos según la dosis usando los valores actualizados. """

        colores_dosis = np.zeros((len(dosis_nube), 3))
        colores_dosis[(dosis_nube >= dose_min_csv) & (dosis_nube < low_max)] = low_dose_rgb
        colores_dosis[(dosis_nube >= medium_min) & (dosis_nube < medium_max)] = medium_dose_rgb
        colores_dosis[dosis_nube >= high_min] = high_dose_rgb

        return colores_dosis

    def get_origin_from_xml(self, xml_filepath):
        """Extrae el origen georeferenciado del archivo metadatos.xml."""
        try:
            tree = ET.parse(xml_filepath)
            root = tree.getroot()
            srs_origin = root.find("SRSOrigin")

            if srs_origin is None or not srs_origin.text:
                print("Error: No se encontró la etiqueta <SRSOrigin> en el XML.")
                return None

            return np.array([float(coord) for coord in srs_origin.text.split(",")])
        except Exception as e:
            print(f"Error leyendo el archivo XML: {e}")
            return None  # Devolver None si hay un problema

    def process(self):
        try:
            print(f"Show Dose Layer: {self.show_dose_layer}")
            # Cargar la nube de puntos PCD
            pcd = o3d.io.read_point_cloud(self.pc_filepath)

            # Downsamplear la nube de puntos si se ha especificado un porcentaje
            if self.downsample is not None:
                if not (1 <= self.downsample <= 100):
                    messagebox.showerror("Error", "El valor de downsample debe estar entre 1 y 100.")
                    return
                self.downsample = float(self.downsample) / 100.0
                if 0 < self.downsample <= 1:
                    if self.downsample == 1:
                        self.downsample = 0.99  # Evitar downsamplear a 0
                    voxel_size = 1 * self.downsample
                    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
                    pcd = downsampled_pcd

            # Obtener coordenadas XYZ
            nube_puntos = np.asarray(pcd.points)

            # Obtener colores si existen, de lo contrario, usar blanco
            if pcd.has_colors():
                rgb = np.asarray(pcd.colors)
            else:
                rgb = np.ones_like(nube_puntos)

            if self.show_dose_layer:
                origin = self.get_origin_from_xml(self.xml_filepath)

                # Sumar el origen a las coordenadas locales
                geo_points = nube_puntos + origin  # a utm

                utm_coords = np.genfromtxt(self.csv_filepath, delimiter=',', skip_header=1)
                utm_points = utm_coords[:, :2]  # Sólo coordenadas [easting, northing]
                dosis = utm_coords[:, 2]  # Dosis correspondiente

                # Construir el KD-Tree para los puntos UTM del CSV (BUSQUEDA EFICIENTE)
                tree = cKDTree(utm_points)

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

                # Asignar dosis correspondiente a los puntos dentro del área
                dosis_nube[:] = dosis[indices_mas_cercanos]  # Dosis para cada punto en la nube

                valid_points = ~np.isnan(dosis_nube)
                puntos_dosis_elevados = puntos_dentro[valid_points]
                dosis_filtrada = dosis_nube[valid_points]

                colores_dosis = self.get_dose_color(dosis_filtrada, self.high_dose_rgb, self.medium_dose_rgb,
                                                    self.low_dose_rgb, self.dose_min_csv, self.low_max,
                                                    self.medium_min, self.medium_max, self.high_min)

                puntos_dosis_elevados[:, 2] += self.altura_extra  # Aumentar Z

                # Crear nube de puntos Open3D
                pcd.points = o3d.utility.Vector3dVector(geo_points)
                pcd.colors = o3d.utility.Vector3dVector(rgb)  # Asignar colores

                # Crear la nueva nube de puntos de dosis elevada
                pcd_dosis = o3d.geometry.PointCloud()
                pcd_dosis.points = o3d.utility.Vector3dVector(puntos_dosis_elevados)
                pcd_dosis.colors = o3d.utility.Vector3dVector(colores_dosis)  # Asignar colores según dosis

            if self.vis is None:
                self.vis = o3d.visualization.Visualizer()

                self.vis.create_window(window_name='Open3D', width=self.winfo_screenwidth(),
                                       height=self.right_frame_height, left=self.left_frame_width,
                                       top=self.title_bar_height)

            self.vis.clear_geometries()  # Ahora estamos seguros de que self.vis no es None
            self.vis.add_geometry(pcd)
            if self.show_dose_layer:
                self.vis.add_geometry(pcd_dosis)

            if self.show_dose_layer and self.show_source and self.source_location is not None:
                source_point = np.array(
                    [self.source_location[0], self.source_location[1], np.max(puntos_dosis_elevados[:, 2])])
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)  # Create a sphere with a radius of 5
                sphere.translate(source_point)  # Move the sphere to the source location
                sphere.paint_uniform_color([0, 0, 0])
                self.vis.add_geometry(sphere)

            # Cambiar el tamaño de los puntos (ajustar para evitar cuadrados)
            render_option = self.vis.get_render_option()
            render_option.point_size = self.point_size

            self.vis.run()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def voxelizer(self):
        print(f"Show Dose Layer: {self.show_dose_layer}")
        pcd = o3d.io.read_point_cloud(self.pc_filepath)
        xyz = np.asarray(pcd.points)

        # Obtener colores si existen, de lo contrario usar blanco
        if pcd.has_colors():
            rgb = np.asarray(pcd.colors)
        else:
            rgb = np.ones_like(xyz)  # Blanco por defecto

        if not self.show_dose_layer:
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgb)

        if self.show_dose_layer:
            origin = self.get_origin_from_xml(self.xml_filepath)

            geo_points = xyz + origin

            pcd.points = o3d.utility.Vector3dVector(geo_points)
            pcd.colors = o3d.utility.Vector3dVector(rgb)

        # Defining the voxel size
        vsize = self.vox_size

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

        if self.show_dose_layer:
            utm_coords = np.genfromtxt(self.csv_filepath, delimiter=',', skip_header=1)
            utm_points = utm_coords[:, :2]  # Sólo coordenadas [easting, northing]
            dosis = utm_coords[:, 2]  # Dosis correspondiente

            # Construir el KD-Tree para los puntos UTM del CSV (BUSQUEDA EFICIENTE)
            tree = cKDTree(utm_points)

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

            # Asignar dosis correspondiente a los puntos dentro del área
            dosis_nube[:] = dosis[indices_mas_cercanos]  # Dosis para cada punto en la nube

            valid_points = ~np.isnan(dosis_nube)
            puntos_dosis_elevados = puntos_dentro[valid_points]
            dosis_filtrada = dosis_nube[valid_points]

            colores_dosis = self.get_dose_color(dosis_filtrada, self.high_dose_rgb, self.medium_dose_rgb,
                                                self.low_dose_rgb, self.dose_min_csv, self.low_max,
                                                self.medium_min, self.medium_max, self.high_min)

            puntos_dosis_elevados[:, 2] += self.altura_extra  # Aumentar Z

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

        if self.vis is None:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name='Open3D', width=self.winfo_screenwidth(),
                                   height=self.right_frame_height, left=self.left_frame_width,
                                   top=self.title_bar_height)

        self.vis.clear_geometries()
        self.vis.add_geometry(vox_mesh)
        if self.show_dose_layer:
            self.vis.add_geometry(vox_mesh_dosis)

        if self.show_dose_layer and self.show_source and self.source_location is not None:
            source_point = np.array(
                [[self.source_location[0], self.source_location[1], np.max(puntos_dosis_elevados[:, 2])]])
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(source_point)
            source_pcd.paint_uniform_color([0, 0, 0])  # Color negro para el punto de la fuente
            self.vis.add_geometry(source_pcd)

        self.vis.run()

    def grid_bcn(self):
        """
        Process a PCD file and save the point data to a .dat file.
        """
        # Load the PCD file
        pcd = o3d.io.read_point_cloud(self.pc_filepath)

        # Extract point data
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(points)

        # Combine points and colors
        nube_puntos = np.hstack((points, colors))

        # Procesar los datos sin necesidad de guardarlos en un archivo

        # output_dat_file = Path(self.pc_filepath).with_name('output_Guadalajara.dat')
        ## Open the output file in append mode
        # with open(output_dat_file, 'a') as f:
        ## Write header if the file does not exist
        # f.write("X Y Z Red Green Blue\n")

        ## Write the point data to the file
        # np.savetxt(f, nube_puntos, fmt="%f", delimiter=" ")

        # Determinar los límites de los datos
        min_x, min_y = np.min(points[:, :2], axis=0)
        max_x, max_y = np.max(points[:, :2], axis=0)

        # Calcular tamaños de píxel
        num_pixels_x = 100
        num_pixels_y = 100
        delta_x = (max_x - min_x) / num_pixels_x
        delta_y = (max_y - min_y) / num_pixels_y

        # Inicializar estructuras para estadísticas
        z_values = np.full((num_pixels_y, num_pixels_x), np.nan)
        cell_stats = [[{'z_values': []} for _ in range(num_pixels_x)] for _ in range(num_pixels_y)]

        total_points = 0

        # Process the point cloud data to fill Z values
        for point in points:
            x, y, z = point[:3]
            x_idx = int((x - min_x) // delta_x)
            y_idx = int((y - min_y) // delta_y)

            if 0 <= x_idx < num_pixels_x and 0 <= y_idx < num_pixels_y:
                cell_stats[y_idx][x_idx]['z_values'].append(z)
                if np.isnan(z_values[y_idx, x_idx]):
                    z_values[y_idx, x_idx] = z
                else:
                    z_values[y_idx, x_idx] = z  # You can change this to max(z_values[y_idx, x_idx], z) if needed

                # Count processed points
                total_points += 1

        # Identify cells with fewer points than the threshold and assign an average height based on adjacent cells
        for i in range(num_pixels_y):
            for j in range(num_pixels_x):
                cell = cell_stats[i][j]

                # Calculate the center of the cell
                center_x = min_x + (j + 0.5) * delta_x
                center_y = min_y + (i + 0.5) * delta_y

                # Calculate Zmax, Zmean, and Zmin
                Zmax = np.max(cell['z_values']) if cell['z_values'] else None
                Zmean = np.mean(cell['z_values']) if cell['z_values'] else None
                Zmin = np.min(cell['z_values']) if cell['z_values'] else None

                # Verify if the new criterion can be calculated
                if Zmax is not None and Zmin is not None and Zmean is not None:
                    criterion = (Zmax - Zmean) / (Zmax - Zmin) if (Zmax - Zmin) != 0 else 0

                    criterion_threshold = 0.95
                    if criterion > criterion_threshold:
                        z_values[i, j] = Zmean  # Assign Zmean
                        # Print debug information
                        print(f"Celda con valor maximo muy disperso: center_x={center_x}, center_y={center_y}")
                        print("Valor de Z maximo: ", Zmax)
                        print(f"Nuevo valor de Z asignado: {z_values[i, j]}")
                    else:
                        z_values[i, j] = Zmax  # Assign Zmax

        # Plot the results
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Calculate global Z min and max for color normalization
        z_min_global = np.nanmin(z_values)
        z_max_global = np.nanmax(z_values)

        # Create a colormap
        colormap = plt.colormaps.get_cmap("terrain")

        # Draw horizontal cells and vertical surfaces
        for i in range(num_pixels_y):
            for j in range(num_pixels_x):
                z_to_use = z_values[i, j]
                if np.isnan(z_to_use):
                    continue

                x_center = min_x + (j + 0.5) * delta_x
                y_center = min_y + (i + 0.5) * delta_y

                x_min = x_center - delta_x / 2
                x_max = x_center + delta_x / 2
                y_min = y_center - delta_y / 2
                y_max = y_center + delta_y / 2

                # Draw the top horizontal cell
                ax.plot_surface(
                    np.array([[x_min, x_max], [x_min, x_max]]),
                    np.array([[y_min, y_min], [y_max, y_max]]),
                    np.array([[z_to_use, z_to_use], [z_to_use, z_to_use]]),
                    color=colormap((z_to_use - z_min_global) / (z_max_global - z_min_global)),
                    alpha=0.9, shade=True
                )

                # Draw vertical surfaces connecting to adjacent cells
                neighbors = [
                    ((i - 1, j), [x_min, x_max], [y_min, y_min]),  # Bottom
                    ((i + 1, j), [x_min, x_max], [y_max, y_max]),  # Top
                    ((i, j - 1), [x_min, x_min], [y_min, y_max]),  # Left
                    ((i, j + 1), [x_max, x_max], [y_min, y_max])  # Right
                ]

                for (ni, nj), vx, vy in neighbors:
                    if 0 <= ni < num_pixels_y and 0 <= nj < num_pixels_x:
                        neighbor_z = z_values[ni, nj]
                        if np.isnan(neighbor_z):
                            continue

                        if vx[0] == vx[1]:  # Surface parallel to Y axis
                            vz = [[z_to_use, neighbor_z], [z_to_use, neighbor_z]]
                            X, Y = np.meshgrid(vx, vy)
                        elif vy[0] == vy[1]:  # Surface parallel to X axis
                            vz = [[z_to_use, z_to_use], [neighbor_z, neighbor_z]]
                            X, Y = np.meshgrid(vx, vy)

                        face_color = colormap(
                            (max(z_to_use, neighbor_z) - z_min_global) / (z_max_global - z_min_global))
                        ax.plot_surface(X, Y, np.array(vz), color=face_color, alpha=0.6, shade=True)

        # Add a color bar
        mappable = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=z_min_global, vmax=z_max_global))
        mappable.set_array([])
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, aspect=10, pad=0.1)
        cbar.set_label('Height (Z)', fontsize=12)

        # Labels and adjustments
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Grid from Point Cloud Data')
        plt.show()

    def set_run_prueba_flag(self):
        if not self.pc_filepath:
            messagebox.showwarning("Warning", "Please select a Point Cloud.")
            return

        self.right_frame_width = self.winfo_screenwidth() - self.left_frame_width
        self.right_frame_height = self.left_frame.winfo_height()

        self.left_frame_width = self.left_frame.winfo_width()

        user32 = ctypes.windll.user32
        self.title_bar_height = user32.GetSystemMetrics(4)

        self.run_prueba = True
        process = multiprocessing.Process(target=run_visualizer,
                                          args=(
                                              self.pc_filepath, self.csv_filepath, self.xml_filepath, None,
                                              self.point_size, self.vox_size, self.altura_extra,
                                              self.high_dose_rgb, self.medium_dose_rgb, self.low_dose_rgb,
                                              self.dose_min_csv, self.low_max, self.medium_min, self.medium_max,
                                              self.high_min, self.high_max,
                                              self.show_dose_layer, self.downsample, self.source_location,
                                              self.show_source, self.run_prueba,
                                              self.right_frame_width, self.right_frame_height, self.left_frame_width,
                                              self.title_bar_height))
        process.start()

    def prueba(self):
        try:
            print("Run prueba")
            # Load the PCD file
            pcd = o3d.io.read_point_cloud(self.pc_filepath)

            # Extract point data
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(points)

            # Determine the bounds of the data
            min_x, min_y = np.min(points[:, :2], axis=0)
            max_x, max_y = np.max(points[:, :2], axis=0)

            # Calculate pixel sizes
            num_pixels_x = 100
            num_pixels_y = 100
            delta_x = (max_x - min_x) / num_pixels_x
            delta_y = (max_y - min_y) / num_pixels_y

            # Initialize structures for statistics
            z_values = np.full((num_pixels_y, num_pixels_x), np.nan)
            cell_stats = [[{'z_values': [], 'colors': []} for _ in range(num_pixels_x)] for _ in range(num_pixels_y)]

            # Process the point cloud data to fill Z values and colors
            for point, color in zip(points, colors):
                x, y, z = point[:3]
                x_idx = int((x - min_x) // delta_x)
                y_idx = int((y - min_y) // delta_y)

                if 0 <= x_idx < num_pixels_x and 0 <= y_idx < num_pixels_y:
                    cell_stats[y_idx][x_idx]['z_values'].append(z)
                    cell_stats[y_idx][x_idx]['colors'].append(color)

            # Calculate mean Z values and predominant colors for each cell
            for i in range(num_pixels_y):
                for j in range(num_pixels_x):
                    z_vals = np.array(cell_stats[i][j]['z_values'])
                    if len(z_vals) > 0:
                        z_mean = np.mean(z_vals)
                        z_std = np.std(z_vals)

                        # Filtrar valores atípicos por el criterio z_mean + 2*sigma
                        filtered_z_vals = z_vals[z_vals <= z_mean + 2 * z_std]
                        z_values[i, j] = np.mean(filtered_z_vals) + 2 * np.std(filtered_z_vals)
                        cell_stats[i][j]['color'] = np.mean(cell_stats[i][j]['colors'], axis=0)

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

            # Visualize the combined mesh
            if self.vis is None:
                self.vis = o3d.visualization.Visualizer()
                self.vis.create_window(window_name='Open3D', width=self.winfo_screenwidth() + 100,
                                       height=self.right_frame_height, left=self.left_frame_width,
                                       top=self.title_bar_height)

            self.vis.clear_geometries()
            self.vis.add_geometry(combined_mesh)
            self.vis.run()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")


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


def run_visualizer(pc_filepath, csv_filepath, xml_filepath, use_voxelization, point_size, vox_size, altura_extra,
                   high_dose_rgb, medium_dose_rgb, low_dose_rgb, dose_min_csv, low_max, medium_min, medium_max,
                   high_min, high_max, show_dose_layer, downsample, source_location, show_source, run_prueba,
                   right_frame_width, right_frame_height, left_frame_width, title_bar_height):
    """Ejecuta Open3D Visualizer en un proceso separado con la opción de voxelizar o no."""
    app = PointCloudApp()  # Instanciar la clase principal para acceder a sus métodos
    app.pc_filepath = pc_filepath  # Asignar el archivo de la nube de puntos
    app.csv_filepath = csv_filepath
    app.xml_filepath = xml_filepath
    app.point_size = point_size
    app.vox_size = vox_size
    app.altura_extra = altura_extra
    app.high_dose_rgb = high_dose_rgb
    app.medium_dose_rgb = medium_dose_rgb
    app.low_dose_rgb = low_dose_rgb
    app.dose_min_csv = dose_min_csv
    app.low_max = low_max
    app.medium_min = medium_min
    app.medium_max = medium_max
    app.high_min = high_min
    app.high_max = high_max
    app.show_dose_layer = show_dose_layer
    app.downsample = downsample
    app.source_location = source_location
    app.show_source = show_source
    app.run_prueba = run_prueba
    app.right_frame_width = right_frame_width
    app.right_frame_height = right_frame_height
    app.left_frame_width = left_frame_width
    app.title_bar_height = title_bar_height

    if run_prueba:
        app.prueba()

    else:
        if use_voxelization:
            print("Voxelization applied")
            app.voxelizer()
        else:
            print("No voxelization applied")
            app.process()


if __name__ == "__main__":
    pointCloud_app = PointCloudApp()
    pointCloud_app.mainloop()
