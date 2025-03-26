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
        main_frame.grid_columnconfigure(1, weight=3)  # Right frame (2/3)
        main_frame.grid_rowconfigure(0, weight=1)

        # Frame izquierdo
        left_frame = CTkFrame(main_frame, fg_color="#2E2E2E", corner_radius=0)
        left_frame.grid(row=0, column=0, sticky="nsew")

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
        downsample_frame.place(relx=0.5, y=150, anchor="n")
        label_downsample = CTkLabel(downsample_frame, text="Downsample:", text_color="#F0F0F0")
        entry_downsample = CTkEntry(downsample_frame, width=50)
        label_percent = CTkLabel(downsample_frame, text="%", text_color="#F0F0F0")
        label_downsample.pack(side="left", padx=(0, 5))
        entry_downsample.pack(side="left", padx=(0, 5))
        label_percent.pack(side="left")

        # Ensure the window is maximized
        self.after(0, lambda: self.wm_state('zoomed'))

if __name__ == "__main__":
    pointCloud_app = PointCloudApp()
    pointCloud_app.mainloop()