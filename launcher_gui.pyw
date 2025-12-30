"""VEILBREAKERS Rigger - Smart GUI Launcher"""
import tkinter as tk
import subprocess
import threading
import os
import sys
import webbrowser
import time

class RiggerLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VEILBREAKERS Monster Rigger")
        self.root.geometry("400x320")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(False, False)

        # Center window
        self.root.eval('tk::PlaceWindow . center')

        # Title
        title = tk.Label(
            self.root,
            text="VEILBREAKERS",
            font=("Arial Black", 24, "bold"),
            fg="#ff4444",
            bg="#1a1a2e"
        )
        title.pack(pady=(30, 5))

        subtitle = tk.Label(
            self.root,
            text="Monster Rigger v3.0",
            font=("Arial", 14),
            fg="#aaaaaa",
            bg="#1a1a2e"
        )
        subtitle.pack(pady=(0, 30))

        # Launch Button
        self.launch_btn = tk.Button(
            self.root,
            text="LAUNCH RIGGER",
            font=("Arial Black", 16),
            fg="white",
            bg="#cc0000",
            activebackground="#ff0000",
            activeforeground="white",
            width=20,
            height=2,
            cursor="hand2",
            command=self.launch_rigger
        )
        self.launch_btn.pack(pady=20)

        # Status
        self.status = tk.Label(
            self.root,
            text="Click to start the web UI",
            font=("Arial", 10),
            fg="#666666",
            bg="#1a1a2e"
        )
        self.status.pack(pady=10)

        # Open Browser button
        self.browser_btn = tk.Button(
            self.root,
            text="Open Browser",
            font=("Arial", 12),
            fg="white",
            bg="#333366",
            activebackground="#4444aa",
            cursor="hand2",
            command=lambda: webbrowser.open("http://localhost:7860")
        )

        # Stop button
        self.stop_btn = tk.Button(
            self.root,
            text="Stop Server",
            font=("Arial", 10),
            fg="white",
            bg="#663333",
            activebackground="#aa4444",
            cursor="hand2",
            command=self.stop_server
        )

        self.process = None
        self.running = False
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def launch_rigger(self):
        if self.running:
            return

        self.running = True
        self.launch_btn.configure(state="disabled", text="LAUNCHING...", bg="#666666")
        self.status.configure(text="Starting server...", fg="#ffaa00")

        def run():
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            self.process = subprocess.Popen(
                [sys.executable, "run.py"],
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            # Wait for server to start
            time.sleep(3)
            self.root.after(0, self.server_ready)

            # Monitor process
            self.process.wait()

            # Process ended - reset UI
            self.root.after(0, self.server_stopped)

        threading.Thread(target=run, daemon=True).start()

    def server_ready(self):
        self.launch_btn.configure(text="RUNNING", bg="#006600")
        self.status.configure(text="Server running at http://localhost:7860", fg="#00ff00")
        self.browser_btn.pack(pady=5)
        self.stop_btn.pack(pady=5)
        webbrowser.open("http://localhost:7860")

    def server_stopped(self):
        self.running = False
        self.process = None
        self.launch_btn.configure(
            state="normal",
            text="LAUNCH RIGGER",
            bg="#cc0000"
        )
        self.status.configure(text="Server stopped. Click to restart.", fg="#ff6666")
        self.browser_btn.pack_forget()
        self.stop_btn.pack_forget()

    def stop_server(self):
        if self.process:
            self.process.terminate()
            self.status.configure(text="Stopping server...", fg="#ffaa00")

    def on_close(self):
        if self.process:
            self.process.terminate()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = RiggerLauncher()
    app.run()
