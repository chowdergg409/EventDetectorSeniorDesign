import os
import sys

os.environ["QT_API"] = "pyqt5"
os.environ["OMP_NUM_THREADS"] = '6'
os.environ["USE_SYSTEM_VTK"] = 'OFF'

import numpy as np
import time
import multiprocessing as mp
from queue import Empty
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication
from pyvistaqt import MainWindow
from GUIlayout import TabWidget
from config import config
from ripple_acquisition import RippleDataBuffer
from DataAquisition.dsp.hfo_detector import CSHFOdetector
from DataAquisition.dsp.ied_detector import IEDDetector
from DataAquisition.dsp.seizure_detector import LineLengthSeizureDetector
from PyQt5.QtWidgets import (QApplication, QMessageBox, QFileDialog, 
                           QDialog, QVBoxLayout, QPushButton)
from offline_buffer import NC3DataBuffer

from PyQt5.QtCore import QThread, pyqtSignal

class DataProcessor(QThread):
    data_ready = pyqtSignal(np.ndarray, list)  # (data, keys)

    def __init__(self, offline_mode, buffer, offline_buffer, window_sec):
        super().__init__()
        self.offline_mode = offline_mode
        self.buffer = buffer
        self.offline_buffer = offline_buffer
        self.window_sec = window_sec
        self.running = True

    def run(self):
        while self.running:
            try:
                if self.offline_mode:
                    data_dict = self.offline_buffer.read_new_data(self.window_sec)
                    full_data = np.stack([data_dict[ch] for ch in sorted(data_dict.keys())])
                    keys = sorted(data_dict.keys())
                else:
                    full_data, _ = self.buffer.get_data_window(self.window_sec)
                    keys = list(range(full_data.shape[0]))  # fallback channel names
                
                if full_data is not None:
                    self.data_ready.emit(full_data.copy(), keys)
                self.msleep(int(self.window_sec * 1000))  # control frequency
            except Exception as e:
                print(f"[DataProcessor] Error: {e}")
                continue

    def stop(self):
        self.running = False
        self.wait()


class ModeSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Operation Mode")
        layout = QVBoxLayout()
        
        self.online_btn = QPushButton("Online Mode (Real-time)")
        self.offline_btn = QPushButton("Offline Mode (File-based)")
        
        layout.addWidget(self.online_btn)
        layout.addWidget(self.offline_btn)
        self.setLayout(layout)
        
        self.online_btn.clicked.connect(lambda: self.done(1))
        self.offline_btn.clicked.connect(lambda: self.done(0))

class EDS(MainWindow):
    def __init__(self):
        super().__init__()
        self.config = config
        self.fs = config["detection_parameters"]["seizure"]['fs']
        self.window_sec = 1  # 10-second processing window
        self.offline_mode = False
        
        # Show mode selection dialog first
        self.select_operation_mode()
        
        # Initialize rest of the UI if not canceled
        if self.offline_mode is not None:
            self.init_ui()
            self.init_buffers()
            self.init_detectors()
            self.start_processing()
            self.process_data_window()
            self.show()

    def select_operation_mode(self):
        dialog = ModeSelectionDialog()
        result = dialog.exec_()
        
        
        if result == 1:  # Online
            self.offline_mode = False
        elif result == None:
            sys.exit(0)
        else:  # Offline
            self.offline_mode = True
            self.setup_offline_mode()

    def setup_offline_mode(self):
        # Get NSx.mat file
        nsx_path, _ = QFileDialog.getOpenFileName(
            self, "Select NSx.mat File", "", "MAT Files (*.mat)")
        
        if not nsx_path:
            QMessageBox.critical(self, "Error", "No NSx.mat file selected!")
            sys.exit(1)
            
        # Get NC3 files directory
        nc3_dir = QFileDialog.getExistingDirectory(
            self, "Select NC3 Files Directory")
        
        if not nc3_dir:
            QMessageBox.critical(self, "Error", "No NC3 directory selected!")
            sys.exit(1)

        # Initialize offline buffer
        self.offline_buffer = NC3DataBuffer(nsx_path, self.fs)
        self.offline_buffer.set_nc3_directory(nc3_dir)
        
        # Get channel selection
        channels = self.select_offline_channels()
        self.offline_buffer.start_acquisition(channels)

    def select_offline_channels(self):
       # Implement channel selection dialog if needed
       return None  # Select all channels



    def init_ui(self):
        self.tab_widget = TabWidget()
        self.setCentralWidget(self.tab_widget)
        self.setWindowTitle('Event Detector Suite')
    
    def init_buffers(self):
        if not self.offline_mode:
            self.buffer = RippleDataBuffer(self.config, 'electrode_map.map')
            self.buffer.initialize()
            self.buffer.start_acquisition()

    def init_detectors(self):
            self.data_queue = mp.Queue()
            self.results_queue = mp.Queue()
    
            self.detectors = [
                mp.Process(target=run_hfo_detector, args=(self.data_queue, self.results_queue, self.fs)),
                mp.Process(target=run_ied_detector, args=(self.data_queue, self.results_queue, self.fs)),
                mp.Process(target=run_seizure_detector, args=(self.data_queue, self.results_queue, self.fs))
            ]
    
            for detector in self.detectors:
                detector.daemon = True
                detector.start()

    def start_processing(self):
        self.data_thread = DataProcessor(self.offline_mode, getattr(self, 'buffer', None), getattr(self, 'offline_buffer', None), self.window_sec)
        self.data_thread.data_ready.connect(self.on_data_ready)
        self.data_thread.start()

        self.results_timer = QtCore.QTimer()
        self.results_timer.timeout.connect(self.check_results)
        self.results_timer.start(100)

    def on_data_ready(self, full_data, keys):
        self.data_queue.put(('full', full_data.copy(), "All"))
        for ch_idx in range(full_data.shape[0]):
            self.data_queue.put(('channel', full_data[ch_idx, :].copy(), keys[ch_idx]))

    def process_data_window(self):
        """Fetch and prepare data for detectors"""
        try:
            if self.offline_mode:
                data_dict = self.offline_buffer.read_new_data(self.window_sec)
                full_data = np.stack([data_dict[ch] for ch in sorted(data_dict.keys())])
            else:
                full_data, _ = self.buffer.get_data_window(self.window_sec)

            if full_data is not None:
                # Send full array for IED detector (tagged)
                self.data_queue.put(('full', full_data.copy(),"All"))

                # Send individual channels for HFO/seizure detectors (tagged)
                keys = list(sorted(data_dict.keys()))
                for ch_idx in range(full_data.shape[0]):
                    self.data_queue.put(('channel', full_data[ch_idx, :].copy(),keys[ch_idx]))
                
        except Exception as e:
            print(f"Data processing error: {str(e)}")

    def check_results(self):
        while not self.results_queue.empty():
            try:
                detector_type, results = self.results_queue.get_nowait()
                self.update_detection_results(detector_type, results)
            except Empty:
                break

    def update_detection_results(self, detector_type, results):
        # Update your GUI with the detection results
        if detector_type == 'hfo':
            self.tab_widget.update_hfo_results(results)
        elif detector_type == 'ied':
            self.tab_widget.update_ied_results(results)
        elif detector_type == 'seizure':
            self.tab_widget.update_seizure_results(results)

    def closeEvent(self, event):
        if self.offline_mode:
            self.offline_buffer.stop_acquisition()
        
        else:
            self.buffer.stop_acquisition()

        if hasattr(self, 'data_thread'):
            self.data_thread.stop()    
            
        for detector in self.detectors:
            detector.terminate()
        super().closeEvent(event)

def run_hfo_detector(data_queue, results_queue, fs):
    detector = CSHFOdetector(sampling_rate=fs)
    while True:
        try:
            data_type, data, label = data_queue.get(timeout=1)
            if data_type == 'channel':
                detections = detector.detect_hfos(data, channel_label=label)
                results_queue.put(('hfo', detections))
        except Empty:
            continue

def run_ied_detector(data_queue, results_queue, fs):
    detector = IEDDetector(fs=fs)
    while True:
        try:
            data_type, data, label = data_queue.get(timeout=1)
            if data_type == 'full':
                spikes = detector.process_recording(data)
                results_queue.put(('ied', spikes))
        except Empty:
            continue

def run_seizure_detector(data_queue, results_queue, fs):
    detector = LineLengthSeizureDetector(
        fs=fs,
        short_window_sec=2.0,
        long_window_sec=30.0,
        threshold_offset=3.0
    )
    while True:
        try:
            data_type, data, label = data_queue.get(timeout=1)
            if data_type == 'channel':
                seizure_data = detector.process_segment(data, label)
                results_queue.put(('seizure', seizure_data))
        except Empty:
            continue

if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseDesktopOpenGL)
    # Now create the application
    app = QApplication([])
    
    # Create and show window
    window = EDS()
    app.exec_()