import customtkinter as ctk

# Configuración de CustomTkinter
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

class DronePanel(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Panel de posiciones del dron")
        self.geometry("500x500")
        self.configure(bg="#a0a0a0")  # Fondo gris

        # Crear panel blanco (frame)
        self.panel = ctk.CTkFrame(self, width=300, height=300, fg_color="white", corner_radius=10)
        self.panel.place(relx=0.5, rely=0.5, anchor="center")

        # Canvas dentro del panel
        self.canvas = ctk.CTkCanvas(self.panel, width=300, height=300, bg="white", highlightthickness=0)
        self.canvas.pack()

        # Coordenadas dadas
        self.coordenadas = [
            (30, 40),
            (50, 25),
            (79, 49),
            (10, 5)
        ]

        # Ajustes visuales
        self.escala = 3
        self.offset_x = 20
        self.offset_y = 20

        # Guardar referencia a botones
        self.botones = []

        self.crear_botones()

    def crear_botones(self):
        for i, (x, y) in enumerate(self.coordenadas):
            x_scaled = x * self.escala + self.offset_x
            y_scaled = y * self.escala + self.offset_y
            self.crear_boton_posicion(x_scaled, y_scaled, i)

    def crear_boton_posicion(self, x, y, index):
        btn = ctk.CTkButton(self.canvas, text=f"{index + 1}", width=40, height=30,
                            fg_color="blue", hover_color="darkblue",
                            command=lambda i=index: self.cambiar_color(i))
        self.canvas.create_window(x, y, window=btn)
        self.botones.append(btn)

    def cambiar_color(self, index):
        boton = self.botones[index]
        boton.configure(fg_color="green", hover_color="darkgreen")

if __name__ == "__main__":
    app = DronePanel()
    app.mainloop()

