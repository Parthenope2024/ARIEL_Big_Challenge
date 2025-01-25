import papermill as pm
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from threading import Thread
import os
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import io
import contextlib


NOTEBOOK_PATH = 'SpectralData-works.ipynb'
OUTPUT_NOTEBOOK_PATH = 'SpectralOutput.ipynb'
CSV_OUTPUT_PATH = 'outputs.csv'


def execute_notebook(status_text):
    """Executes the notebook and handles errors."""
    
    buffer = io.StringIO()
    
    try:
         with contextlib.redirect_stdout(buffer):
           pm.execute_notebook(
               NOTEBOOK_PATH,
               OUTPUT_NOTEBOOK_PATH,
             )
           
         output = buffer.getvalue()
         
         for line in output.splitlines():
             if line.startswith("Epoch:"):
                 update_loading_status(status_text, line)
         return True, None
    except pm.exceptions.PapermillExecutionError as e:
        return False, str(e)


def load_data():
    """Loads data from the generated CSV."""
    try:
        if not os.path.exists(CSV_OUTPUT_PATH):
            return None, f"File '{CSV_OUTPUT_PATH}' not found."
        data = pd.read_csv(CSV_OUTPUT_PATH)
        return data, None
    except FileNotFoundError:
        return None, f"File '{CSV_OUTPUT_PATH}' not found."
    except Exception as e:
        return None, f"Errore di lettura dati : {e}"


def create_main_window(data):
    """Creates the main application window with the desired layout."""
    if data is None or data.empty:
        messagebox.showerror("Errore", "No data.")
        return

    root = tk.Tk()
    root.title("ARIEL Big Challenge - Data Visualizer (GUI)")

    # --- Left Frame for Planet List ---
    left_frame = ttk.Frame(root, padding=10)
    left_frame.grid(row=0, column=0, sticky='ns')
    ttk.Label(left_frame, text="DATA", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky='w', pady=(0, 10))

    # Listbox to show planets
    planet_listbox = tk.Listbox(left_frame, width=20)
    planet_listbox.grid(row=1, column=0, sticky='nsew')

    # Scrollbar to make the listbox scrollable
    listbox_scroll_y = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=planet_listbox.yview)
    listbox_scroll_y.grid(row=1, column=1, sticky='ns')
    planet_listbox.config(yscrollcommand=listbox_scroll_y.set)


     # --- Middle Frame for Spectrum and Parameters ---
    middle_frame = ttk.Frame(root, padding=10)
    middle_frame.grid(row=0, column=1, sticky='nsew')

    # --- Right Frame for Planet Parameters ---
    planet_params_frame = ttk.Frame(middle_frame, borderwidth=2, relief='groove', padding=5)
    planet_params_frame.grid(row=0, column=1, sticky='nsew', rowspan = 2)
    
    # ---  Star Spectrum Frame ----
    star_spectrum_label = ttk.Label(middle_frame, text="Spettro stella : (<star_name>) ", font=("Arial", 12, "bold"))
    star_spectrum_label.grid(row=0, column=0, sticky='w', pady=(0, 10))

    # Frame to show star spectrum plot
    star_spectrum_frame = ttk.Frame(middle_frame, borderwidth=2, relief='groove', padding=5)
    star_spectrum_frame.grid(row=1, column=0, sticky='nsew')
    star_spectrum_frame.grid_columnconfigure(0, weight=1)
    star_spectrum_frame.grid_rowconfigure(0, weight=1)

    # --- Bottom Frame for Spectral Score ---
    spectral_score_label = ttk.Label(root, text="Spectral Score", font=("Arial", 12, "bold"))
    spectral_score_label.grid(row=1, column=0, columnspan = 2, sticky='w', pady=(10, 10))
    
    # Frame for the predicted and real spectrum plots
    spectral_score_frame = ttk.Frame(root, borderwidth=2, relief='groove', padding=5)
    spectral_score_frame.grid(row=2, column=0, sticky='nsew', columnspan = 2)
    spectral_score_frame.grid_columnconfigure(0, weight=1)
    spectral_score_frame.grid_rowconfigure(0, weight=1)
   
    # Configure grid weights for resizing
    root.grid_columnconfigure(1, weight=1)  # Middle frame takes remaining width
    root.grid_rowconfigure(0, weight=1) # Middle Frame expands vertically
    root.grid_rowconfigure(2, weight=1) # Spectral Score expands vertically
    middle_frame.grid_columnconfigure(0, weight=1) # Make the star spectrum plot expand
    middle_frame.grid_rowconfigure(1, weight = 1) # Make the star spectrum plot expand vertically
   
    # --- Function to Populate Planet Parameters ---
    def show_planet_parameters(planet_name):
       for widget in planet_params_frame.winfo_children():
         widget.destroy()
       if not planet_name:
          return
          
       selected_planet = data.loc[data['Planet Index']== float(planet_name.split(' ')[1])]
       if not selected_planet.empty:
          planet_index = selected_planet['Planet Index'].iloc[0]
          planet_radius = selected_planet['Planet Radius'].iloc[0]
          planet_temp = selected_planet['Planet Temp'].iloc[0]
          log_co2 = selected_planet['log_CO2'].iloc[0]
          log_h2o = selected_planet['log_H2O'].iloc[0]
          log_co = selected_planet['log_CO'].iloc[0]
          log_nh3 = selected_planet['log_NH3'].iloc[0]
          log_ch4 = selected_planet['log_CH4'].iloc[0]

          ttk.Label(planet_params_frame, text=f"Planet Index: {planet_index:.1f}",font=("Arial", 10, "bold")).grid(row=0, column=0, sticky='w',pady=(0, 5))
          ttk.Label(planet_params_frame, text=f"Radius: {planet_radius:.2f}").grid(row=1, column=0, sticky='w')
          ttk.Label(planet_params_frame, text=f"Temperature: {planet_temp:.2f}").grid(row=2, column=0, sticky='w')
          ttk.Label(planet_params_frame, text=f"log_CO2: {log_co2:.2f}").grid(row=3, column=0, sticky='w')
          ttk.Label(planet_params_frame, text=f"log_H2O: {log_h2o:.2f}").grid(row=4, column=0, sticky='w')
          ttk.Label(planet_params_frame, text=f"log_CO: {log_co:.2f}").grid(row=5, column=0, sticky='w')
          ttk.Label(planet_params_frame, text=f"log_NH3: {log_nh3:.2f}").grid(row=6, column=0, sticky='w')
          ttk.Label(planet_params_frame, text=f"log_CH4: {log_ch4:.2f}").grid(row=7, column=0, sticky='w')
    
    def plot_star_spectrum(planet_name):
       # Clear existing widgets
        for widget in star_spectrum_frame.winfo_children():
            widget.destroy()
        if not planet_name:
          return
       
        selected_planet = data.loc[data['Planet Index'] == float(planet_name.split(' ')[1])]
        if not selected_planet.empty:
           # Create dummy spectrum data (replace with actual values)
           x = np.linspace(0, 10, 100)
           y = np.sin(x)  # Dummy data for the star spectrum
           
           fig = Figure(figsize=(4, 3), dpi=100)
           ax = fig.add_subplot(111)
           ax.plot(x, y, label='Spettro stellare')
           ax.set_xlabel('Wavelength')
           ax.set_ylabel('Intensity')
           ax.set_title(f'Spettro stellare (Stella {int(selected_planet["Planet Index"].iloc[0])})')
           ax.legend()
           fig.tight_layout()

           canvas = FigureCanvasTkAgg(fig, master=star_spectrum_frame)
           canvas_widget = canvas.get_tk_widget()
           canvas_widget.pack(fill=tk.BOTH, expand=True)
           canvas.draw()


    def plot_predicted_and_real_spectrum(planet_name):
       # Clear existing widgets
        for widget in spectral_score_frame.winfo_children():
            widget.destroy()
        if not planet_name:
          return
        
        selected_planet = data.loc[data['Planet Index'] == float(planet_name.split(' ')[1])]
        if not selected_planet.empty:
            # Generate dummy predicted and real spectrum data (replace with actual values)
            x_values = np.linspace(0, 10, 100)
            predicted_y = np.cos(x_values)
            real_y = np.sin(x_values)
            # Create the figure and axes
            fig = Figure(figsize=(8, 3), dpi=100)
            ax = fig.add_subplot(111)
            # Plot the predicted and real spectra
            ax.plot(x_values, predicted_y, label='Spettro Predetto')
            ax.plot(x_values, real_y, label='Spettro Reale')
            # Add labels and titles
            ax.set_xlabel('Wavelength')
            ax.set_ylabel('Intensity')
            ax.set_title(f'Spettro predetto VS reale (Pianeta {int(selected_planet["Planet Index"].iloc[0])})')
            ax.legend()
            fig.tight_layout()
           # Integrate matplotlib figure in the Tk window
            canvas = FigureCanvasTkAgg(fig, master=spectral_score_frame)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)
            canvas.draw()

    # Populate planet listbox and set selection binding
    planet_names = []
    for index, row in data.iterrows():
        planet_name = f"Planet {int(row['Planet Index'])}"
        planet_names.append((int(row['Planet Index']), planet_name))

    planet_names.sort(key=lambda x: x[0])  # Sort by index

    for _, planet_name in planet_names:
        planet_listbox.insert(tk.END, planet_name)
    
    def on_planet_select(event):
        selected_planet_index = planet_listbox.curselection()
        if selected_planet_index:
            planet_name = planet_listbox.get(selected_planet_index)
            show_planet_parameters(planet_name)
            plot_star_spectrum(planet_name)
            plot_predicted_and_real_spectrum(planet_name)

    planet_listbox.bind("<<ListboxSelect>>", on_planet_select)
    
    # --- Quit Button ---
    quit_button = ttk.Button(root, text="Quit", command=root.destroy)
    quit_button.grid(row=3, column=0, columnspan=2, pady=10) # placed at the bottom
    
    root.mainloop()


def display_loading_screen(parent):
    """Creates and displays a loading screen window."""
    loading_window = tk.Toplevel(parent)
    loading_window.title("Elaborazione dati ARIEL")
    loading_window.geometry("300x150")
    loading_window.resizable(False, False)  # Disable resizing

    label = ttk.Label(loading_window, text="Esecuzione del Notebook...", font=("Arial", 12))
    label.pack(pady=20)

    # Create a text area to show status/messages
    status_text = scrolledtext.ScrolledText(loading_window, wrap=tk.WORD, height=3, width=40)
    status_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    status_text.insert(tk.END, "Avvio...\n")
    status_text.config(state=tk.DISABLED)  # make text area not editable

    return loading_window, status_text


def update_loading_status(status_text, message):
    """Updates the status message on the loading screen."""
    status_text.config(state=tk.NORMAL)
    status_text.insert(tk.END, f"{message}\n")
    status_text.config(state=tk.DISABLED)
    status_text.see(tk.END)
    status_text.update_idletasks()


def main():
    # Main application window
    main_root = tk.Tk()
    main_root.withdraw()  # Hide the main window
    loading_window, status_text = display_loading_screen(main_root)  # show a loading screen window

    def run_notebook_and_display_table():
        update_loading_status(status_text, "Eseguendo il Notebook...")
        notebook_success, notebook_error = execute_notebook(status_text)  # run the notebook

        if not notebook_success:
            loading_window.destroy()  # close the loading window if error occurs
            messagebox.showerror("Error", f"Errore durante l'esecuzione: {notebook_error}")
            return
        update_loading_status(status_text, "Esecuzione completata.")

        update_loading_status(status_text, "Caricamento dati...")
        data, data_error = load_data()  # load data

        if data_error:
            loading_window.destroy()
            messagebox.showerror("Errore", f"Errore caricamento dati : {data_error}")
            return

        loading_window.destroy()  # Close loading screen if everything is ok
        create_main_window(data)  # create main window

    # Run the processing in a separate thread to prevent the GUI from freezing
    processing_thread = Thread(target=run_notebook_and_display_table)
    processing_thread.start()

    main_root.mainloop()  # this line is necessary to keep the GUI running


if __name__ == "__main__":
    main()
