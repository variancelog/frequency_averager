from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.axes.set_xscale('log')
        self.axes.set_xlabel('Frequency (Hz)')
        self.axes.set_ylabel('Magnitude (dB)')
        self.axes.grid(True, which='both', linestyle='--', alpha=0.5)

    def plot_data(self, freqs, raw_mags_2d, avg_mag):
        self.axes.clear()
        
        # Plot individual curves
        for mag in raw_mags_2d:
            self.axes.plot(freqs, mag, color='gray', alpha=0.3, linewidth=0.5)
            
        # Plot average curve
        if avg_mag is not None:
            self.axes.plot(freqs, avg_mag, color='blue', linewidth=2, label='Average')
            
        self.axes.set_xscale('log')
        self.axes.set_xlim(20, 20000)
        self.axes.set_xlabel('Frequency (Hz)')
        self.axes.set_ylabel('Magnitude (dB)')
        self.axes.grid(True, which='both', linestyle='--', alpha=0.5)
        self.axes.legend()
        self.canvas.draw()

    def clear_plot(self):
        self.axes.clear()
        self.axes.set_xscale('log')
        self.axes.grid(True, which='both', linestyle='--', alpha=0.5)
        self.canvas.draw()
