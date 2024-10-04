import tkinter as tk
from tkinter import ttk

def main():
    root = tk.Tk()
    root.title("Drone Detection System")
    root.geometry("800x600")
    root.configure(bg="#FFFFFF")
    style = ttk.Style(root)
    style.theme_use("clam")

    def button_clicked(button_name):
        print(f"{button_name} clicked")

    btn_camera_feed = ttk.Button(
        root,
        text="Camera Feed",
        command=lambda: button_clicked("Camera Feed")
    )
    btn_camera_feed.place(x=36, y=38, width=155, height=155)

    btn_current_alerts = ttk.Button(
        root,
        text="Current Alerts",
        command=lambda: button_clicked("Current Alerts")
    )
    btn_current_alerts.place(x=223, y=38, width=155, height=155)

    btn_settings = ttk.Button(
        root,
        text="Settings",
        command=lambda: button_clicked("Settings")
    )
    btn_settings.place(x=36, y=220, width=155, height=155)

    btn_alert_history = ttk.Button(
        root,
        text="Alert History",
        command=lambda: button_clicked("Alert History")
    )
    btn_alert_history.place(x=223, y=220, width=155, height=155)

    btn_start = ttk.Button(
        root,
        text="Start",
        command=lambda: button_clicked("Start")
    )
    btn_start.place(x=620, y=16, width=150, height=40)

    btn_pause = ttk.Button(
        root,
        text="Pause",
        command=lambda: button_clicked("Pause")
    )
    btn_pause.place(x=620, y=75, width=150, height=40)

    btn_stop = ttk.Button(
        root,
        text="Stop",
        command=lambda: button_clicked("Stop")
    )
    btn_stop.place(x=620, y=134, width=150, height=40)

    root.mainloop()

if __name__ == '__main__':
    main()
