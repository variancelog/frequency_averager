from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QComboBox, QFileDialog, QListWidget,
                             QSlider, QDoubleSpinBox, QSpinBox, QProgressBar, QCheckBox)
from PyQt6.QtCore import Qt, QTimer
import pandas as pd
import numpy as np
import os
import multiprocessing
import queue

from src.gui.plot_widget import PlotWidget
from src.core.data_ingestion import DataIngestor
from src.core.interpolator import CommonAxisInterpolator
from src.core.averager import CurveAverager
from src.core.outlier_detector import OutlierDetector
from src.core.gpu_utils import AccelerationManager


def _compute_worker(method, filtered_mags_2d, norm_freqs, gamma, radius_percent, y_weight, result_queue):
    """
    Standalone function to run the math algorithm in a separate process.
    """
    try:
        if len(filtered_mags_2d) == 0:
            res = None
        elif method == "Arithmetic Mean":
            res = CurveAverager.arithmetic_mean(filtered_mags_2d)
        elif method == "Geometric Arc-Length Blend":
            res = CurveAverager.geometric_arc_length_blend(norm_freqs, filtered_mags_2d, y_weight=y_weight)
        elif method == "Hard-DTW Barycenter":
            # FIXED: Explicit keyword argument for radius_percent
            res = CurveAverager.hard_dtw_barycenter(norm_freqs, filtered_mags_2d, radius_percent=radius_percent)
        elif method == "Soft-DTW Barycenter":
            # FIXED: Explicit keyword arguments for gamma and progress_queue
            res = CurveAverager.soft_dtw_barycenter(filtered_mags_2d, gamma=gamma, progress_queue=result_queue)
        else:
            res = None
            
        result_queue.put(("success", res))
    except Exception as e:
        result_queue.put(("error", str(e)))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Frequency Response Averager")
        self.resize(1000, 700)
        self.setAcceptDrops(True)
        
        self.dataframes = []
        self.filepaths = []
        self.interp_data = {}
        self.avg_mag = None
        self.kept_indices = []
        self.excluded_indices = []
        
        self.calc_process = None
        self.calc_queue = None
        
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self._on_debounce_timeout)
        self._needs_interpolation = False
        
        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self._poll_process)
        
        self.fake_progress_timer = QTimer()
        self.fake_progress_timer.timeout.connect(self._on_fake_progress)
        
        self._init_ui()

    def _on_fake_progress(self):
        """Advances the progress bar up to 90% while waiting for processing to finish."""
        current_val = self.progress_bar.value()
        if current_val < 90:
            self.progress_bar.setValue(current_val + 1)

    def _update_outliers(self):
        """Synchronously updates the list of excluded outliers based on the UI state."""
        if not self.interp_data:
            return
            
        if self.outlier_checkbox.isChecked():
            norm_mags_2d = self.interp_data['norm_mags_2d']
            threshold_std = self.outlier_spinbox.value()
            self.kept_indices, self.excluded_indices = OutlierDetector.filter_outliers(
                norm_mags_2d, threshold_std
            )
        else:
            n = len(self.interp_data['norm_mags_2d'])
            self.kept_indices = list(range(n))
            self.excluded_indices = []
            
        self.refresh_file_list()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        layout = QHBoxLayout(main_widget)
        
        # Left Panel (Controls)
        left_panel = QVBoxLayout()
        
        # Algorithm Selection
        self.algo_label = QLabel("Averaging Method:")
        self.algo_combo = QComboBox()
        self.algo_combo.addItems([
            "Arithmetic Mean", 
            "Geometric Arc-Length Blend", 
            "Hard-DTW Barycenter",
            "Soft-DTW Barycenter"
        ])
        self.algo_combo.currentIndexChanged.connect(self.on_algo_changed)
        
        left_panel.addWidget(self.algo_label)
        left_panel.addWidget(self.algo_combo)
        
        # --- Gamma Control (Visible only for Soft-DTW) ---
        self.gamma_widget = QWidget()
        gamma_layout = QHBoxLayout(self.gamma_widget)
        gamma_layout.setContentsMargins(0, 0, 0, 0)
        
        self.gamma_label = QLabel("Smoothing (γ):")
        self.gamma_spinbox = QDoubleSpinBox()
        self.gamma_spinbox.setRange(0.01, 0.25)
        self.gamma_spinbox.setSingleStep(0.01)
        self.gamma_spinbox.setValue(0.05)
        self.gamma_spinbox.setKeyboardTracking(False)
        
        self.gamma_slider = QSlider(Qt.Orientation.Horizontal)
        self.gamma_slider.setRange(1, 25) # Maps to 0.01 - 0.50
        self.gamma_slider.setValue(5)
        
        # Connect slider and spinbox
        self.gamma_spinbox.valueChanged.connect(self.sync_slider_to_spinbox)
        self.gamma_slider.valueChanged.connect(self.sync_spinbox_to_slider)
        self.gamma_slider.sliderReleased.connect(self.recalculate_average)
        self.gamma_slider.sliderPressed.connect(self.stop_processing)
        
        gamma_layout.addWidget(self.gamma_label)
        gamma_layout.addWidget(self.gamma_slider)
        gamma_layout.addWidget(self.gamma_spinbox)
        
        left_panel.addWidget(self.gamma_widget)
        self.gamma_widget.hide() # Hidden initially
        # -------------------------------------------------

        # --- Y Weight Control (Visible only for Geometric Arc-Length) ---
        self.y_weight_widget = QWidget()
        y_weight_layout = QHBoxLayout(self.y_weight_widget)
        y_weight_layout.setContentsMargins(0, 0, 0, 0)
        
        self.y_weight_label = QLabel("Y Weight:")
        self.y_weight_spinbox = QDoubleSpinBox()
        self.y_weight_spinbox.setRange(0.01, 0.25)
        self.y_weight_spinbox.setSingleStep(0.01)
        self.y_weight_spinbox.setDecimals(2)
        self.y_weight_spinbox.setValue(0.05)
        self.y_weight_spinbox.setKeyboardTracking(False)
        
        self.y_weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.y_weight_slider.setRange(1, 25) # Maps to 0.01 - 0.25
        self.y_weight_slider.setValue(5)
        
        self.y_weight_spinbox.valueChanged.connect(self.sync_y_weight_slider_to_spinbox)
        self.y_weight_slider.valueChanged.connect(self.sync_y_weight_spinbox_to_slider)
        self.y_weight_slider.sliderReleased.connect(self.recalculate_average)
        self.y_weight_slider.sliderPressed.connect(self.stop_processing)
        
        y_weight_layout.addWidget(self.y_weight_label)
        y_weight_layout.addWidget(self.y_weight_slider)
        y_weight_layout.addWidget(self.y_weight_spinbox)
        
        left_panel.addWidget(self.y_weight_widget)
        self.y_weight_widget.hide() # Hidden initially
        # -------------------------------------------------

        # --- Hard-DTW Radius Control (Visible only for Hard-DTW) ---
        self.radius_widget = QWidget()
        radius_layout = QHBoxLayout(self.radius_widget)
        radius_layout.setContentsMargins(0, 0, 0, 0)
        
        self.radius_label = QLabel("Radius (%):")
        self.radius_spinbox = QDoubleSpinBox()
        self.radius_spinbox.setRange(1.0, 5.0)
        self.radius_spinbox.setSingleStep(0.1)
        self.radius_spinbox.setValue(1.5)
        self.radius_spinbox.setKeyboardTracking(False)
        
        self.radius_slider = QSlider(Qt.Orientation.Horizontal)
        self.radius_slider.setRange(10, 50) # Maps to 1.0% - 5.0%
        self.radius_slider.setValue(15)
        
        self.radius_spinbox.valueChanged.connect(self.sync_radius_slider)
        self.radius_slider.valueChanged.connect(self.sync_radius_spinbox)
        self.radius_slider.sliderReleased.connect(self.recalculate_average)
        self.radius_slider.sliderPressed.connect(self.stop_processing)
        
        radius_layout.addWidget(self.radius_label)
        radius_layout.addWidget(self.radius_slider)
        radius_layout.addWidget(self.radius_spinbox)
        
        left_panel.addWidget(self.radius_widget)
        self.radius_widget.hide() # Hidden initially
        # -------------------------------------------------

        # --- Interpolation Points Control ---
        self.points_widget = QWidget()
        points_layout = QHBoxLayout(self.points_widget)
        points_layout.setContentsMargins(0, 0, 0, 0)

        self.points_label = QLabel("Interp Points:")
        self.points_spinbox = QSpinBox()
        self.points_spinbox.setRange(128, 1024)
        self.points_spinbox.setSingleStep(128)
        self.points_spinbox.setValue(512)
        self.points_spinbox.setKeyboardTracking(False)

        self.points_slider = QSlider(Qt.Orientation.Horizontal)
        self.points_slider.setRange(1, 8) # Maps to 128 - 1024
        self.points_slider.setValue(4)

        self.points_spinbox.valueChanged.connect(self.sync_points_slider)
        self.points_slider.valueChanged.connect(self.sync_points_spinbox)
        self.points_slider.sliderReleased.connect(self.process_and_plot)
        self.points_slider.sliderPressed.connect(self.stop_processing)

        points_layout.addWidget(self.points_label)
        points_layout.addWidget(self.points_slider)
        points_layout.addWidget(self.points_spinbox)

        left_panel.addWidget(self.points_widget)
        # -------------------------------------------------

        # --- Hardware Acceleration ---
        status = AccelerationManager.check_gpu_support()
        self.gpu_checkbox = QCheckBox(f"Use {status['backend_to_use'].upper()} Acceleration")
        self.gpu_checkbox.setVisible(False)
        left_panel.addWidget(self.gpu_checkbox)
        # -------------------------------------------------

        # --- Outlier Control ---
        self.outlier_checkbox = QCheckBox("Exclude Outliers (Std Dev > Threshold)")
        self.outlier_checkbox.stateChanged.connect(self.on_outlier_toggled)
        left_panel.addWidget(self.outlier_checkbox)

        self.outlier_widget = QWidget()
        outlier_layout = QHBoxLayout(self.outlier_widget)
        outlier_layout.setContentsMargins(0, 0, 0, 0)
        
        self.outlier_label = QLabel("Threshold:")
        self.outlier_spinbox = QDoubleSpinBox()
        self.outlier_spinbox.setRange(0.1, 5.0)
        self.outlier_spinbox.setSingleStep(0.1)
        self.outlier_spinbox.setValue(1.0)
        self.outlier_spinbox.setKeyboardTracking(False)
        
        self.outlier_slider = QSlider(Qt.Orientation.Horizontal)
        self.outlier_slider.setRange(1, 50) # Maps to 0.1 - 5.0
        self.outlier_slider.setValue(10)
        
        self.outlier_spinbox.valueChanged.connect(self.sync_outlier_slider)
        self.outlier_slider.valueChanged.connect(self.sync_outlier_spinbox)
        self.outlier_slider.sliderReleased.connect(self._sync_outlier_and_recalc)
        self.outlier_slider.sliderPressed.connect(self.stop_processing)
        
        outlier_layout.addWidget(self.outlier_label)
        outlier_layout.addWidget(self.outlier_slider)
        outlier_layout.addWidget(self.outlier_spinbox)
        
        left_panel.addWidget(self.outlier_widget)
        self.outlier_widget.setEnabled(False) # Disabled initially
        # -------------------------------------------------
        
        # Progress Bar and Stop Button
        self.progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        
        self.stop_btn = QPushButton("Stop Processing")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_processing)
        
        self.progress_layout.addWidget(self.progress_bar)
        self.progress_layout.addWidget(self.stop_btn)
        left_panel.addLayout(self.progress_layout)
        
        # File List
        self.file_list_label = QLabel("Loaded Files:")
        self.file_list = QListWidget()
        left_panel.addWidget(self.file_list_label)
        left_panel.addWidget(self.file_list)
        
        # Clear Button
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_data)
        left_panel.addWidget(self.clear_btn)
        
        # Export Button
        self.export_btn = QPushButton("Export Average")
        self.export_btn.clicked.connect(self.export_csv)
        left_panel.addWidget(self.export_btn)
        
        left_panel.addStretch()
        
        # Right Panel (Plot)
        self.plot_widget = PlotWidget()
        
        layout.addLayout(left_panel, 1)
        layout.addWidget(self.plot_widget, 3)
        
        # Ensure UI state is synced with the default algorithm (Arithmetic Mean)
        self.on_algo_changed()

    def sync_slider_to_spinbox(self, value):
        self.gamma_slider.blockSignals(True)
        self.gamma_slider.setValue(int(value * 100))
        self.gamma_slider.blockSignals(False)
        self.recalculate_average()

    def sync_spinbox_to_slider(self, value):
        self.gamma_spinbox.blockSignals(True)
        self.gamma_spinbox.setValue(value / 100.0)
        self.gamma_spinbox.blockSignals(False)
        # Recalculate only if not currently dragging the slider
        if not self.gamma_slider.isSliderDown():
            self.recalculate_average()

    def sync_points_slider(self, value):
        self.points_slider.blockSignals(True)
        self.points_slider.setValue(int(value / 128))
        self.points_slider.blockSignals(False)
        self.process_and_plot()

    def sync_points_spinbox(self, value):
        self.points_spinbox.blockSignals(True)
        self.points_spinbox.setValue(value * 128)
        self.points_spinbox.blockSignals(False)
        # Recalculate only if not currently dragging the slider
        if not self.points_slider.isSliderDown():
            self.process_and_plot()

    def sync_radius_slider(self, value):
        self.radius_slider.blockSignals(True)
        self.radius_slider.setValue(int(value * 10))
        self.radius_slider.blockSignals(False)
        self.recalculate_average()

    def sync_radius_spinbox(self, value):
        self.radius_spinbox.blockSignals(True)
        self.radius_spinbox.setValue(value / 10.0)
        self.radius_spinbox.blockSignals(False)
        # Recalculate only if not currently dragging the slider
        if not self.radius_slider.isSliderDown():
            self.recalculate_average()

    def sync_y_weight_slider_to_spinbox(self, value):
        self.y_weight_slider.blockSignals(True)
        self.y_weight_slider.setValue(int(value * 100))
        self.y_weight_slider.blockSignals(False)
        self.recalculate_average()

    def sync_y_weight_spinbox_to_slider(self, value):
        self.y_weight_spinbox.blockSignals(True)
        self.y_weight_spinbox.setValue(value / 100.0)
        self.y_weight_spinbox.blockSignals(False)
        # Recalculate only if not currently dragging the slider
        if not self.y_weight_slider.isSliderDown():
            self.recalculate_average()

    def sync_outlier_slider(self, value):
        self.outlier_slider.blockSignals(True)
        self.outlier_slider.setValue(int(value * 10))
        self.outlier_slider.blockSignals(False)
        self._update_outliers()
        self.recalculate_average()

    def sync_outlier_spinbox(self, value):
        self.outlier_spinbox.blockSignals(True)
        self.outlier_spinbox.setValue(value / 10.0)
        self.outlier_spinbox.blockSignals(False)
        # Recalculate only if not currently dragging the slider
        if not self.outlier_slider.isSliderDown():
            self._update_outliers()
            self.recalculate_average()

    def _sync_outlier_and_recalc(self):
        self._update_outliers()
        self.recalculate_average()

    def on_outlier_toggled(self, state):
        self.outlier_widget.setEnabled(state == Qt.CheckState.Checked.value)
        self._update_outliers()
        self.recalculate_average()

    def on_algo_changed(self):
        method = self.algo_combo.currentText()
        self.gamma_widget.hide()
        self.radius_widget.hide()
        self.y_weight_widget.hide()
        
        # Interp points should always be visible since it dictates the 
        # common frequency axis resolution for ALL algorithms
        self.points_widget.show() 
        self.gpu_checkbox.setVisible(False)
        
        # Removed the 'if Arithmetic Mean -> hide' check
        
        if method == "Soft-DTW Barycenter":
            self.gamma_widget.show()
            if "CPU" not in self.gpu_checkbox.text():
                self.gpu_checkbox.setVisible(True)
        elif method == "Hard-DTW Barycenter":
            self.radius_widget.show()
        elif method == "Geometric Arc-Length Blend":
            self.y_weight_widget.show()
            
        self.recalculate_average()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        for url in urls:
            path = url.toLocalFile()
            if path.endswith('.csv') or path.endswith('.txt'):
                self.load_file(path)
        
        self.process_and_plot()

    def load_file(self, filepath):
        try:
            df = DataIngestor.load_csv(filepath)
            self.dataframes.append(df)
            self.filepaths.append(filepath)
            self.refresh_file_list()
        except Exception as e:
            print(f"Failed to load {filepath}: {e}")

    def refresh_file_list(self):
        self.file_list.clear()
        for i, filepath in enumerate(self.filepaths):
            filename = os.path.basename(filepath)
            if i in self.excluded_indices:
                self.file_list.addItem(f"[EXCLUDED] {filename}")
                item = self.file_list.item(i)
                item.setForeground(Qt.GlobalColor.red)
            else:
                self.file_list.addItem(filename)
                
    def process_and_plot(self):
        if not self.dataframes:
            return
            
        self._needs_interpolation = True
        
        if not self.debounce_timer.isActive():
            self.stop_processing()
        
        self.debounce_timer.start(750)

    def _on_debounce_timeout(self):
        if not self.dataframes:
            return
            
        try:
            if self._needs_interpolation:
                num_points = self.points_spinbox.value()
                self.interp_data = CommonAxisInterpolator.interpolate_to_common_axis(
                    self.dataframes, 
                    num_points=num_points
                )
                self._update_outliers()
                self._needs_interpolation = False
            
            self._do_recalculate()
        except Exception as e:
            print(f"Error during processing: {e}")

    def recalculate_average(self):
        if not self.interp_data:
            return
        
        if not self.debounce_timer.isActive():
            self.stop_processing()
        
        self.debounce_timer.start(750)

    def _do_recalculate(self):
        if not self.interp_data:
            return
            
        self.stop_processing() # ensure any running process is killed first
            
        method = self.algo_combo.currentText()
        norm_mags_2d = self.interp_data['norm_mags_2d']
        norm_freqs = self.interp_data['norm_freqs']
        gamma = self.gamma_spinbox.value()
        radius_percent = self.radius_spinbox.value()
        y_weight = self.y_weight_spinbox.value()
        
        # Slice data based on persistent outlier state
        filtered_mags_2d = norm_mags_2d[self.kept_indices]
        
        # UI updates for processing state
        self.progress_bar.setRange(0, 100) # Switch from indeterminate to 100% scale
        self.progress_bar.setValue(0)
        self.stop_btn.setEnabled(True)
        self.fake_progress_timer.start(50) # Ticks up to 90% over ~4.5 seconds
        
        self.calc_queue = multiprocessing.Queue()
        self.calc_process = multiprocessing.Process(
            target=_compute_worker,
            args=(method, filtered_mags_2d, norm_freqs, gamma, radius_percent, y_weight, self.calc_queue)
        )
        self.calc_process.start()
        self.poll_timer.start(50)
        
    def _poll_process(self):
        if self.calc_queue is None or self.calc_process is None:
            return
            
        while True:
            try:
                status, res = self.calc_queue.get_nowait()
                
                if status == "init_progress":
                    continue
                elif status == "sub_progress":
                    continue
                elif status == "progress":
                    continue  # Ignore deprecated generic progress message
                    
                self.poll_timer.stop()
                
                # Clean up the process
                self.calc_process.join(timeout=1.0)
                self.calc_process = None
                
                if status == "success":
                    self.on_calc_finished(res)
                else:
                    self.on_calc_error(res)
                return
                    
            except queue.Empty:
                if self.calc_process is not None and not self.calc_process.is_alive():
                    self.poll_timer.stop()
                    self.calc_process.join(timeout=1.0)
                    self.calc_process = None
                    self.on_calc_error("Background process died unexpectedly.")
                return

    def on_calc_finished(self, avg_mag_norm):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.stop_btn.setEnabled(False)
        
        if avg_mag_norm is not None:
            # SHAPE VALIDATION: Ensure results match current UI resolution
            # This prevents race conditions when changing 'Interp Points'
            current_points = len(self.interp_data['common_freqs'])
            if len(avg_mag_norm) != current_points:
                print(f"Ignoring stale calculation result (Shape {len(avg_mag_norm)} != UI {current_points})")
                return

            self.avg_mag = CommonAxisInterpolator.unnormalize_magnitude(
                avg_mag_norm, 
                self.interp_data['mag_min'], 
                self.interp_data['mag_max']
            )
            
            # Show raw curves but only the kept ones
            raw_mags_to_plot = self.interp_data['raw_mags_2d'][self.kept_indices]
            
            self.plot_widget.plot_data(self.interp_data['common_freqs'], raw_mags_to_plot, self.avg_mag)

    def on_calc_error(self, error_msg):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.stop_btn.setEnabled(False)
        print(f"Error during calculation: {error_msg}")

    def stop_processing(self):
        if hasattr(self, 'debounce_timer'):
            self.debounce_timer.stop()
        self.poll_timer.stop()
        if hasattr(self, 'fake_progress_timer'):
            self.fake_progress_timer.stop()
        
        if self.calc_process is not None:
            if self.calc_process.is_alive():
                self.calc_process.terminate()
                self.calc_process.join(timeout=1.0)
            self.calc_process = None
            
        if self.calc_queue is not None:
            self.calc_queue = None
            
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.stop_btn.setEnabled(False)

    def clear_data(self):
        self.stop_processing()
        self._needs_interpolation = False
        self.dataframes = []
        self.filepaths = []
        self.interp_data = {}
        self.avg_mag = None
        self.kept_indices = []
        self.excluded_indices = []
        self.file_list.clear()
        self.plot_widget.clear_plot()

    def export_csv(self):
        if not self.interp_data or self.avg_mag is None:
            return
            
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Average", "average_rew.txt", "Text Files (*.txt);;CSV Files (*.csv)")
        if filepath:
            if filepath.endswith('.txt'):
                orig_freqs = self.interp_data['common_freqs']
                orig_mags = self.avg_mag
                
                min_freq = max(20.0, orig_freqs.min())
                max_freq = min(22050.0, orig_freqs.max())
                
                # Use 48 PPO as requested by the user
                ppo = 48
                octaves = np.log2(max_freq / min_freq)
                num_points = int(np.round(octaves * ppo)) + 1
                target_freqs = min_freq * (2 ** (np.arange(num_points) / float(ppo)))
                target_freqs = target_freqs[target_freqs <= max_freq]
                
                target_mags = np.interp(target_freqs, orig_freqs, orig_mags)
                
                df = pd.DataFrame({
                    'frequency': target_freqs,
                    'magnitude': target_mags
                })
                
                # Format to match reference avg_rew.txt: 6 decimals for freq, 3 for mag, no headers
                df['frequency'] = df['frequency'].map(lambda x: f"{x:.6f}")
                df['magnitude'] = df['magnitude'].map(lambda x: f"{x:.3f}")
                
                df.to_csv(filepath, sep='\t', index=False, header=False, lineterminator='\n')
            else:
                df = pd.DataFrame({
                    'frequency': self.interp_data['common_freqs'],
                    'magnitude': self.avg_mag
                })
                df.to_csv(filepath, index=False)
