import tkinter as tk
from tkinter import messagebox
import subprocess

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Final-Project")
        self.geometry("800x600")

        # Hyperparameters
        self.num_walks = tk.IntVar(value=200)
        self.workers = tk.IntVar(value=2)
        self.num_epochs = tk.IntVar(value=100)
        self.time_slices = tk.IntVar(value=61)

        # Create UI components
        
        self._create_hyperparameters_frame()
        self._create_buttons_frame()
        self._create_console_output()



    def _create_hyperparameters_frame(self):
        frame = tk.LabelFrame(self, text="Hyperparameters", padx=10, pady=10)
        frame.pack(fill="x", padx=5, pady=5)

        tk.Label(frame, text="Number of Walks:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(frame, textvariable=self.num_walks).grid(row=0, column=1, padx=5, pady=2)

        tk.Label(frame, text="Number of Workers:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(frame, textvariable=self.workers).grid(row=1, column=1, padx=5, pady=2)

        tk.Label(frame, text="Number of Epochs:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(frame, textvariable=self.num_epochs).grid(row=2, column=1, padx=5, pady=2)

        tk.Label(frame, text="Time Slices:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(frame, textvariable=self.time_slices).grid(row=3, column=1, padx=5, pady=2)

    def _create_buttons_frame(self):
        frame = tk.Frame(self, padx=10, pady=10)
        frame.pack(fill="x", padx=5, pady=5)

        tk.Button(frame, text="Generate Data", command=self.run_generate_data).pack(side="left", padx=5, pady=2)
        tk.Button(frame, text="Run Main", command=self.run_main).pack(side="left", padx=5, pady=2)

    def _create_console_output(self):
        frame = tk.LabelFrame(self, text="Console Output", padx=10, pady=10)
        frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.console = tk.Text(frame, wrap="word", state="disabled")
        self.console.pack(fill="both", expand=True)

    def run_generate_data(self):
        try:
            subprocess.run(["python", "generateData.py", str(self.num_walks.get()), str(self.workers.get())], check=True)
            subprocess.run(["python", "generateContentFile"], check=True)  # Corrected filename
            self._log_to_console("✅ Data generation completed.")
        except subprocess.CalledProcessError as e:
            self._log_to_console(f"❌ Error: {e}")

    def run_main(self):
        try:
            subprocess.run([
                "python", "main_run.py",
                "--max_epoch", str(self.num_epochs.get()),
                "--Time_stamps", str(self.time_slices.get())
            ], check=True)
            self._log_to_console("✅you can see the plots in the folder 'plots' in the pyton IDE.")
        except subprocess.CalledProcessError as e:
            self._log_to_console(f"❌ Error: {e}")

    def _log_to_console(self, message):
        self.console.configure(state="normal")
        self.console.insert(tk.END, message + "\n")
        self.console.configure(state="disabled")

if __name__ == "__main__":
    app = App()
    app.mainloop()