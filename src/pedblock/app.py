import customtkinter as ctk

from pedblock.ui.main_window import MainWindow


def main() -> None:
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = MainWindow()
    app.mainloop()


