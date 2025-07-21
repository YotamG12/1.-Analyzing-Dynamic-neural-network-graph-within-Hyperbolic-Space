import tkinter as tk
from tkinter import messagebox
import subprocess

class ToolTip:
    """
    Tooltip widget for Tkinter elements. Displays a message when hovering over a widget.

    Args:
        widget (tk.Widget): The widget to attach the tooltip to.
        text (str): The tooltip text to display.
    """
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None

        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        """
        Show the tooltip window when mouse enters the widget.

        Args:
            event: Tkinter event (optional).
        """
        if self.tooltip_window:
            return
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{self.widget.winfo_rootx() + 20}+{self.widget.winfo_rooty() + 20}")
        label = tk.Label(
            self.tooltip_window,
            text=self.text,
            background="white",
            relief="solid",
            borderwidth=1,
            wraplength=300,
            anchor="w",
            justify="left"
        )
        label.pack()

    def hide_tooltip(self, event=None):
        """
        Hide the tooltip window when mouse leaves the widget.

        Args:
            event: Tkinter event (optional).
        """
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class App(tk.Tk):
    """
    Main application window for the citation network UI.
    Provides controls for data generation, hyperparameters, validation, and console output.
    """
    def __init__(self):
        """
        Initialize the main application window and all UI components.
        """
        super().__init__()
        self.title("Final-Project")
        self.geometry("800x600")

        # Hyperparameters
        self.num_walks = tk.IntVar(value=200)
        self.workers = tk.IntVar(value=2)
        self.num_epochs = tk.IntVar(value=100)
        self.time_slices = tk.IntVar(value=61)
        self.validation_iteration= tk.IntVar(value=30)
        self.graph_type = tk.StringVar(value="moving_window_histograms")

        self.graph_options = [
            "temporal_anomaly_distribution",
            "temporal_sharp_changes",
            "as_std_histogram",
            "moving_window_histograms",
            "All Graphs"
        ]

        # Create UI components
        self._create_Data_frame()
        self._create_hyperparameters_frame()
        self._create_validation_frame()
        self._create_console_output()

    def _create_Data_frame(self):
        """
        Create the data frame section of the UI for time slice input and data generation.
        """
        frame = tk.LabelFrame(self, text="DataFrame", padx=10, pady=10)
        frame.pack(fill="x", padx=5, pady=5)
        
        tk.Label(frame, text="Time Slices:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(frame, textvariable=self.time_slices).grid(row=0, column=1, padx=5, pady=2)
        time_slices_help = tk.Label(frame, text="?", font=("Arial", 10, "bold"), fg="blue", cursor="hand2")
        time_slices_help.grid(row=0, column=2, padx=5, pady=2)
        ToolTip(time_slices_help, text="The number of time slices...")

        # Button centered across all 3 columns
        tk.Button(frame, text="Generate Data", command=self.run_generate_data)\
            .grid(row=1, column=0, columnspan=3, pady=10, sticky="ew")

        # Make sure columns expand equally
        for i in range(3):
            frame.grid_columnconfigure(i, weight=1)

        # Dropdown for graph type selection
        tk.Label(frame, text="Graph Type:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        graph_menu = tk.OptionMenu(frame, self.graph_type, *self.graph_options, command=self._update_graph_explanation)
        graph_menu.grid(row=2, column=1, padx=5, pady=2, sticky="ew")
        self.graph_explanations = {
            "temporal_anomaly_distribution": "Shows the distribution of anomaly scores for each time slice.",
            "temporal_sharp_changes": "Highlights nodes with sharp changes in anomaly scores over time.",
            "as_std_histogram": "Displays a histogram of anomaly score standard deviations across all nodes.",
            "moving_window_histograms": "Plots moving window histograms and delta traces for top nodes.",
            "All Graphs": "Combines all graph types for comprehensive analysis."
        }
        self.graph_explanation_label = tk.Label(frame, text=self.graph_explanations[self.graph_type.get()], wraplength=400, fg="blue", justify="left", anchor="w")
        self.graph_explanation_label.grid(row=3, column=0, columnspan=3, sticky="ew", padx=5, pady=2)

    def _update_graph_explanation(self, selected):
        explanation = self.graph_explanations.get(selected, "")
        self.graph_explanation_label.config(text=explanation)

    def _create_hyperparameters_frame(self):
        """
        Create the hyperparameters frame section of the UI for model training parameters.
        """
        frame = tk.LabelFrame(self, text="Hyperparameters", padx=10, pady=10)
        frame.pack(fill="x", padx=5, pady=5)

        tk.Label(frame, text="Number of Walks:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(frame, textvariable=self.num_walks).grid(row=0, column=1, padx=5, pady=2)
        walks_help = tk.Label(frame, text="?", font=("Arial", 10, "bold"), fg="blue", cursor="hand2")
        walks_help.grid(row=0, column=2, padx=5, pady=2)
        ToolTip(walks_help, text="The number of walks per node. For example, if num_walks=10, then for each node in the graph, 10 random walks will be generated.")


        tk.Label(frame, text="Number of Workers:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(frame, textvariable=self.workers).grid(row=1, column=1, padx=5, pady=2)
        workers_help = tk.Label(frame, text="?", font=("Arial", 10, "bold"), fg="blue", cursor="hand2")
        workers_help.grid(row=1, column=2, padx=5, pady=2)
        ToolTip(workers_help, text="The number of parallel workers to use for generating random walks. More workers can speed up the process, especially for large graphs.")

        tk.Label(frame, text="Number of Epochs:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(frame, textvariable=self.num_epochs).grid(row=2, column=1, padx=5, pady=2)
        epochs_help = tk.Label(frame, text="?", font=("Arial", 10, "bold"), fg="blue", cursor="hand2")
        epochs_help.grid(row=2, column=2, padx=5, pady=2)
        ToolTip(epochs_help, text="The number of epochs for training the model. More epochs can lead to better performance but also increases training time. Adjust based on your dataset size and model complexity.")

        # Centered button
        tk.Button(frame, text="Run Main", command=self.run_main)\
            .grid(row=3, column=0, columnspan=3, pady=10, sticky="ew")

        for i in range(3):
          frame.grid_columnconfigure(i, weight=1)


  

    def _create_validation_frame(self):
        """
        Create the validation frame section of the UI for noise injection iterations.
        """
        frame = tk.LabelFrame(self, text="validation", padx=10, pady=10)
        frame.pack(fill="x", padx=5, pady=5)

        tk.Label(frame, text="Number of iteration:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(frame, textvariable=self.validation_iteration).grid(row=0, column=1, padx=5, pady=2)
        validation_iteration = tk.Label(frame, text="?", font=("Arial", 10, "bold"), fg="blue", cursor="hand2")
        validation_iteration.grid(row=0, column=2, padx=5, pady=2)
        ToolTip(validation_iteration, text="The number of iteration to insert random edges ('noises') to tje original graph.")

    def _create_console_output(self):
        """
        Create the console output section of the UI for displaying logs and messages.
        """
        frame = tk.LabelFrame(self, text="Console Output", padx=10, pady=10)
        frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.console = tk.Text(frame, wrap="word", state="disabled")
        self.console.pack(fill="both", expand=True)

    def run_generate_data(self):
        """
        Run the data generation scripts and log the result to the console.
        """
        try:
            subprocess.run(["python", "src/generateData.py", "--Time_stamps", str(self.time_slices.get())], check=True, capture_output=True, text=True)
            subprocess.run(["python", "src/generateContentFile"], check=True, capture_output=True, text=True)
            self._log_to_console("✅ Data generation completed.")
        except subprocess.CalledProcessError as e:
            self._log_to_console(f"❌ Error: {e}\n{e.stderr}")

    def run_main(self):
        """
        Run the main analysis script with selected hyperparameters and log the result.
        """
        try:
            subprocess.run([
                "python", "src/main_run.py",
                "--max_epoch", str(self.num_epochs.get()),
                "--num_walks", str(self.num_walks.get()), 
                "--workers",str(self.workers.get()),
                "--Time_stamps", str(self.time_slices.get()),
                "--validation_iteration",str(self.validation_iteration.get()),
                "--graph_type", self.graph_type.get()
            ], check=True, capture_output=True, text=True)
            self._log_to_console("✅you can see the plots in the folder 'plots' in the pyton IDE.")
        except subprocess.CalledProcessError as e:
            self._log_to_console(f"❌ Error: {e}\n{e.stderr}")

    def _log_to_console(self, message):
        """
        Log a message to the console output area.

        Args:
            message (str): Message to display.
        """
        self.console.configure(state="normal")
        self.console.insert(tk.END, message + "\n")
        self.console.configure(state="disabled")

if __name__ == "__main__":
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        print(f"❌ Failed to load UI: {e}")