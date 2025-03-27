import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import os
import ast

class GMMAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GMM Analysis Viewer")
        self.root.geometry("1200x800")
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Caricamento dati
        self.spec_matrix = np.load('spectra.npy')
        self.test_ind = self.load_test_indices()
        self.score_data = self.parse_results_file()
        
        # Main container
        main_frame = ttk.Frame(root)
        main_frame.grid(row=0, column=0, sticky='nsew')
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # PanedWindow per sezioni ridimensionabili
        paned = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
        paned.grid(row=0, column=0, sticky='nsew')
        
        # Sezione superiore
        top_pane = ttk.PanedWindow(paned, orient=tk.HORIZONTAL)
        distrib_frame = self.create_distributions_frame(top_pane)
        spectra_frame = self.create_spectra_frame(top_pane)
        top_pane.add(distrib_frame, weight=3)
        top_pane.add(spectra_frame, weight=1)
        
        # Sezione inferiore
        bottom_pane = ttk.PanedWindow(paned, orient=tk.HORIZONTAL)
        scores_frame = self.create_scores_frame(bottom_pane)
        results_frame = self.create_results_frame(bottom_pane)
        bottom_pane.add(scores_frame, weight=2)
        bottom_pane.add(results_frame, weight=1)
        
        paned.add(top_pane)
        paned.add(bottom_pane)

        # Pulsante Exit in fondo al main_frame
        exit_button = ttk.Button(main_frame, text="Exit", command=self.exit_app)
        exit_button.grid(row=1, column=0, sticky='e', padx=10, pady=5)
    
    def exit_app(self):
        self.root.quit()
        self.root.destroy()

    def load_test_indices(self):
        validTraces = np.load('validTraces.npy').astype(np.int64)
        return np.sort(validTraces - 1)

    def parse_results_file(self):
        score_data = []
        try:
            with open("results.txt", "r") as f:
                content = f.read()
            
            blocks = [b.strip() for b in content.split('------------------------------') if b.strip()]
            
            for block in blocks:
                data = {}
                lines = block.split('\n')
                for line in lines:
                    if line.startswith('K1:'):
                        data['K1'] = ast.literal_eval(line.split(':')[1].strip())
                    elif line.startswith('K2:'):
                        data['K2'] = ast.literal_eval(line.split(':')[1].strip())
                    elif line.startswith('Posterior scores:'):
                        data['Posterior'] = ast.literal_eval(line.split(':')[1].strip())
                    elif line.startswith('Spectral scores:'):
                        data['Spectral'] = ast.literal_eval(line.split(':')[1].strip())
                    elif line.startswith('Final scores:'):
                        data['Final'] = ast.literal_eval(line.split(':')[1].strip())
                if data:
                    score_data.append(data)
        except Exception as e:
            print("Error parsing results file:", e)
        return score_data

    def create_distributions_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="Distribuzioni Posteriori")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)
        
        self.plot_files = [f for f in os.listdir('./GMM_plots/') if f.endswith('.png')]
        self.selected_plot = tk.StringVar()
        
        ttk.Label(frame, text="Seleziona Pianeta:").grid(row=0, column=0, padx=5, pady=2)
        self.plot_dropdown = ttk.Combobox(frame, textvariable=self.selected_plot, values=self.plot_files)
        self.plot_dropdown.grid(row=0, column=1, padx=5, pady=2)
        self.plot_dropdown.bind('<<ComboboxSelected>>', self.update_distribution_plot)
        
        self.distrib_canvas = tk.Canvas(frame, bg='white')
        self.distrib_canvas.grid(row=1, column=0, columnspan=2, sticky='nsew')
        
        return frame

    def update_distribution_plot(self, event=None):
        selection = self.selected_plot.get()
        if selection:
            img = Image.open(os.path.join('./GMM_plots/', selection))
            img = img.resize((800, 600), Image.Resampling.LANCZOS)
            self.distrib_img = ImageTk.PhotoImage(img)
            self.distrib_canvas.create_image(0, 0, image=self.distrib_img, anchor='nw')

    def create_spectra_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="Spettri Campione")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)
        
        # Dropdown per selezione spettri
        self.spectra_indices = [str(i) for i in self.test_ind]
        self.selected_spectra = tk.StringVar()
        
        ttk.Label(frame, text="Seleziona Indice:").grid(row=0, column=0, padx=5, pady=2)
        self.spectra_dropdown = ttk.Combobox(frame, textvariable=self.selected_spectra, values=self.spectra_indices)
        self.spectra_dropdown.grid(row=0, column=1, padx=5, pady=2)
        self.spectra_dropdown.bind('<<ComboboxSelected>>', self.update_spectra_plot)
        
        # Canvas per plot spettri
        self.spectra_fig = plt.figure(figsize=(5, 4))
        self.spectra_ax = self.spectra_fig.add_subplot(111)
        self.spectra_canvas = FigureCanvasTkAgg(self.spectra_fig, master=frame)
        self.spectra_canvas.get_tk_widget().grid(row=1, column=0, columnspan=2, sticky='nsew')
        
        return frame

    def update_spectra_plot(self, event=None):
        self.spectra_ax.clear()
        selection = self.selected_spectra.get()
        if selection:
            idx = int(selection)
            if idx in self.test_ind:
                spectrum = self.spec_matrix[idx]
                self.spectra_ax.plot(spectrum, color='blue', alpha=0.7)
                self.spectra_ax.set_title(f"Spettro {idx}")
                self.spectra_canvas.draw()

    def create_scores_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="Grafici degli Score")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)    
        # Carica lista grafici disponibili
        self.score_plots = self.load_score_plots()
        # Dropdown per selezione grafico
        self.selected_score_plot = tk.StringVar()
        ttk.Label(frame, text="Seleziona Grafico:").grid(row=0, column=0, padx=5, pady=2)
        self.score_dropdown = ttk.Combobox(frame, textvariable=self.selected_score_plot, values=list(self.score_plots.keys()))
        self.score_dropdown.grid(row=0, column=1, padx=5, pady=2)
        self.score_dropdown.bind('<<ComboboxSelected>>', self.update_score_plot)    
        # Canvas per visualizzazione immagine
        self.score_canvas = tk.Canvas(frame, bg='white')
        self.score_canvas.grid(row=1, column=0, columnspan=2, sticky='nsew')    
        return frame
    
    def load_score_plots(self):
        plot_files = {}
        try:
            for f in os.listdir('./Grafici_Clustering/'):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Estrai K1 e K2 dal nome del file
                    name_parts = f.split('_')
                    k1 = k2 = 'N/A'
                    for part in name_parts:
                        if part.startswith('K1'):
                            k1 = part[2:]
                        elif part.startswith('K2'):
                            k2 = part[2:]
                    plot_files[f] = {'path': os.path.join('./Grafici_Clustering/', f), 'K1': k1,'K2': k2}
        except FileNotFoundError:
            print("Cartella Grafici_Clustering non trovata")
        return plot_files
    
    def update_score_plot(self, event=None):
        selection = self.selected_score_plot.get()
        if selection and selection in self.score_plots:
            img_path = self.score_plots[selection]['path']
            try:
                img = Image.open(img_path)
                # Ridimensiona mantenendo aspect ratio
                canvas_width = self.score_canvas.winfo_width()
                canvas_height = self.score_canvas.winfo_height()            
                if canvas_width > 0 and canvas_height > 0:
                    img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)            
                self.score_img = ImageTk.PhotoImage(img)
                self.score_canvas.delete("all")
                self.score_canvas.create_image(canvas_width//2, canvas_height//2, image=self.score_img, anchor='center')
                # Aggiungi didascalia con parametri
                caption = f"K1: {self.score_plots[selection]['K1']} | K2: {self.score_plots[selection]['K2']}"
                self.score_canvas.create_text(canvas_width//2, canvas_height-10, text=caption, anchor='n', fill='black')
            except Exception as e:
                print(f"Errore caricamento immagine: {str(e)}")

    def create_results_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="Risultati CSV")
        tree = ttk.Treeview(frame, columns=('K1', 'K2', 'FinalScore'), show='headings')
        tree.heading('K1', text='K1')
        tree.heading('K2', text='K2')
        tree.heading('FinalScore', text='Score Finale')
        
        try:
            df = pd.read_csv('results.csv')
            for _, row in df.iterrows():
                tree.insert('', 'end', values=(row['K1'], row['K2'], row['FinalScore']))
        except Exception as e:
            print("Error loading CSV:", e)
        
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)
        
        return frame

if __name__ == "__main__":
    root = tk.Tk()
    app = GMMAnalyzerApp(root)
    root.mainloop()