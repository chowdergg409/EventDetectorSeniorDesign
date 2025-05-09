import sys
import os
import json
from typing import Tuple
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel,
    QComboBox, QPushButton, QHBoxLayout, QSpacerItem, QSizePolicy, QGroupBox,
    QListWidget, QTableWidget, QTableWidgetItem, QFileDialog, QListWidgetItem, QCheckBox,QMessageBox, QScrollArea,
    QDoubleSpinBox, QFormLayout, QSpinBox
)
from PyQt5.QtCore import Qt, QTimer, QEvent, QCoreApplication, QObject
import datetime
from PyQt5.QtGui import QColor, QPixmap, QVector3D
import numpy as np
import open3d as o3d
import pyqtgraph.opengl
from scipy.io import loadmat
import h5py
import pyqtgraph
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import trimesh
import nibabel as nib
from matplotlib.colors import Normalize
from collections import defaultdict
from PyQt5 import QtGui
import json
import csv
import zipfile


class ElectrodeUpdateEvent(QEvent):
    """Custom event type for electrode updates"""
    EVENT_TYPE = QEvent.Type(QEvent.registerEventType())
    
    def __init__(self, electrode_ids):
        super().__init__(self.EVENT_TYPE)
        self.electrode_ids = electrode_ids


class Visualization3d(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.electrode_labels = {}  # To store electrode labels
        self.current_label = None   # Currently displayed label
        self.event_handler_installed = False
        self.power_dict = defaultdict(list)  # Dictionary to store electrode power values
        self.view = None
        self.electrode_objects = {}
        self.electrodes = []
        self.electrode_states = defaultdict(set)
        self.default_color = (1.0, 1.0, 1.0, 1.0)  # White as default for no events
        self.color_map = {
            'none': self.default_color,         # No events
            'seizure': (1.0, 0.0, 0.0, 1.0),    # Red
            'hfo': (0.0, 0.0, 1.0, 1.0),        # Blue
            'ied': (1.0, 1.0, 0.0, 1.0)         # Yellow
        }
        self.priority = ['seizure', 'hfo', 'ied', 'none']

    def get_nii_path(self):
        self.nii_path, _ = QFileDialog.getOpenFileName(None, "Select the CT.nii file")
        
    def get_brain_surface_path(self):
        self.brain_surface_path, _ = QFileDialog.getOpenFileName(
            None, "Select the surfaces.mat file in Registered Dicom 10 (Meg) folder")
        
    def update_electrode_states(self, electrode_ids, state, active=True, duration=None):
        for eid in electrode_ids:
            if active:
                self.electrode_states[eid].add(state)
                # Automatically remove 'none' state
                self.electrode_states[eid].discard('none')
                if duration is not None:
                    QTimer.singleShot(duration, lambda eid=eid, state=state: self.remove_state(eid, state))
            else:
                self.electrode_states[eid].discard(state)
                # Add 'none' state if no other states
                if not self.electrode_states[eid]:
                    self.electrode_states[eid].add('none')
        self.update_electrode_colors()

    def remove_state(self, eid, state):
        self.electrode_states[eid].discard(state)
        self.update_electrode_colors()

    def clear_state(self, state):
        """Clear state with view check"""
        if not self.view:
            return  # Exit if view not initialized

        for eid in list(self.electrode_states.keys()):
            self.electrode_states[eid].discard(state)
        self.update_electrode_colors()

    def update_electrode_colors(self):
        if not self.view:
            return

        for item in self.view.items:
            if hasattr(item, 'electrode_data'):
                electrode = item.electrode_data
                eid = electrode['name']
                current_states = self.electrode_states.get(eid, set())
                
                # Always include 'none' state
                if not current_states:
                    current_states.add('none')
                
                # Find highest priority state
                color = self.default_color
                for state in self.priority:
                    if state in current_states:
                        color = self.color_map[state]
                        break
                        
                item.setColor(color)
        self.view.update()

    def load_brain_surface(self):
        img = nib.load(self.nii_path)
        header = img.header
        zooms = header.get_zooms()
        z_thicc = zooms[0]
        brain_surface_struct = loadmat(self.brain_surface_path)
        brain_surface_content = brain_surface_struct['BrainSurfRaw']
        brain_vertices = brain_surface_content['vertices'][0][0].astype(np.float32)
        brain_faces = brain_surface_content['faces'][0][0].astype(np.int32)
        brain_vertices[:,2] = brain_vertices[:,2]*(1+z_thicc)

        # Create mesh
        brain_faces = brain_faces - 1  # Convert from MATLAB to Python indexing
        mesh = trimesh.Trimesh(vertices=brain_vertices, faces=brain_faces)
        
        meshdata = pyqtgraph.opengl.MeshData(vertexes=brain_vertices, faces=brain_faces)
        meshitem = pyqtgraph.opengl.GLMeshItem(
            meshdata=meshdata,
            color=(1, 1, 1, 0.10),  # RGBA (blue with full opacity)
            drawEdges=True,
            edgeColor=(1, 1, 1, 0.1),
            drawFaces=False,
            glOptions='translucent'            # Proper depth handling
        )
        return meshitem
       
    def get_electrode_path(self):
        self.electrode_path, _ = QFileDialog.getOpenFileName(
            None, "Select the electrodes.mat file in Registered Dicom 10 (Meg) folder")

    def load_electrodes(self):
        with h5py.File(self.electrode_path, 'r') as mat_file:
            # Load electrode coordinates
            elec_coords = mat_file['ElecXYZRaw'][()]
            # Load electrode names if available
            try:
                elec_names = []
                for ref in mat_file['ElecMapRaw'][0,:]:
                    elec_name = ""
                    for c in mat_file[ref][()]: 
                        letter = chr(int(c))
                        elec_name += letter
                    elec_names.append(elec_name)
            except:
                elec_names = [f"Electrode_{i}" for i in range(elec_coords.shape[1])]
            
            # Store electrode data
            self.electrodes = []
            for i, (name, (x, y, z)) in enumerate(zip(elec_names, elec_coords.T)):
                self.electrodes.append({
                    'name': name,
                    'x': x,
                    'y': y,
                    'z': z,
                    'index': i
                })
                
        # Create electrode spheres
        electrode_items = []
        self.electrode_objects = {}
        
        for electrode in self.electrodes:
            md = pyqtgraph.opengl.MeshData.sphere(rows=10, cols=10, radius=3.0)
            item = pyqtgraph.opengl.GLMeshItem(
                meshdata=md,
                color=self.default_color,
                smooth=True,
                shader='shaded',
                glOptions='opaque'
            )
            item.translate(electrode['x'], electrode['y'], electrode['z'])
            
            # Store electrode reference in the item
            item.electrode_data = electrode
            
            # Make electrode pickable
            item.setGLOptions('additive')
            
            electrode_items.append(item)
            self.electrode_objects[id(item)] = electrode
            
        return electrode_items

    def setup_labels(self, view):
        """Initialize labels for all electrodes (hidden by default)"""
        for item_id, electrode in self.electrode_objects.items():
            label = QLabel(electrode['name'], view)
            label.setStyleSheet(
                "QLabel { background-color : white; color : black; border: 1px solid black; }"
            )
            label.setAlignment(Qt.AlignCenter)
            label.hide()
            self.electrode_labels[item_id] = label

    def on_mouse_click(self, event):
        """Handle mouse clicks to select electrodes"""
        if event.button() != Qt.LeftButton:
            return

        # Get the clicked position
        pos = event.pos()

        # Find the closest electrode
        closest_item = None
        min_distance = float('inf')

        view = self.view

        # Get viewport dimensions
        width = view.width()
        height = view.height()

        # Get camera matrices
        view_matrix = view.viewMatrix()
        projection_matrix = view.projectionMatrix()

        for item in view.items:
            if hasattr(item, 'electrode_data'):  # This is an electrode
                electrode = item.electrode_data
                x, y, z = electrode['x'], electrode['y'], electrode['z']

                # Convert 3D position to clip coordinates
                point = QtGui.QVector3D(x, y, z)
                clip_pos = projection_matrix * (view_matrix * point)

                if clip_pos.z() <= 0:  # Behind camera
                    continue

                # Perspective division to get normalized device coordinates
                ndc_x = clip_pos.x() / clip_pos.z()
                ndc_y = clip_pos.y() / clip_pos.z()

                # Convert to window coordinates
                win_x = (ndc_x + 1) * width / 2
                win_y = (1 - ndc_y) * height / 2  # Flip y-axis

                # Calculate distance from click to electrode
                distance = ((win_x - pos.x())**2 + 
                           (win_y - pos.y())**2)**0.5

                if distance < min_distance and distance < 20:  # 20 pixel threshold
                    min_distance = distance
                    closest_item = item

        # Hide previous label
        if self.current_label:
            self.current_label.hide()

        # Show new label if an electrode was clicked
        if closest_item:
            electrode = closest_item.electrode_data
            label = self.electrode_labels[id(closest_item)]

            # Calculate screen position again for the closest item
            point = QVector3D(electrode['x'], electrode['y'], electrode['z'])
            clip_pos = projection_matrix * (view_matrix * point)

            if clip_pos.z() > 0:  # Only if in front of camera
                ndc_x = clip_pos.x() / clip_pos.z()
                ndc_y = clip_pos.y() / clip_pos.z()
                win_x = (ndc_x + 1) * width / 2
                win_y = (1 - ndc_y) * height / 2

                label.move(int(win_x) + 20, int(win_y) + 20)
                label.show()
                label.associated_item = closest_item
                self.current_label = label
                     
    def normalize_power_data(self):
        # Normalize all power data to 0-1 range for consistent coloring
        if not self.power_dict:
            return
            
        # Find global max across all power values
        global_max = max(max(values) for values in self.power_dict.values())
        
        # Normalize each electrode's power values
        for key in self.power_dict.keys():
            norm = Normalize(vmin=0, vmax=global_max)
            self.power_dict[key] = norm(self.power_dict[key])

    def initialize_visualization(self):
        """Initialize the 3D visualization components"""
        self.view = pyqtgraph.opengl.GLViewWidget()
        self.view.setWindowTitle('Brain with Electrodes')
        
        # Add brain mesh
        brain_mesh = self.load_brain_surface()
        self.view.addItem(brain_mesh)
        
        # Add electrodes
        electrodes = self.load_electrodes()
        for electrode in electrodes:
            self.view.addItem(electrode)
        
        # Setup labels
        self.setup_labels(self.view)
        
        # Connect mouse click event
        self.view.mousePressEvent = self.on_mouse_click
        
        # Set camera position
        self.view.setCameraPosition(distance=300, elevation=-90, azimuth=180)
        
        if not self.event_handler_installed:
            QApplication.instance().installEventFilter(self)
            self.event_handler_installed = True

        self.update_electrode_colors()
        return self.view
      
    def highlight_electrodes(self, electrode_states):
        """
        Advanced version with multiple color states
        :param electrode_states: Dict of {electrode_id: state}
            where state can be: 'event', 'active', 'inactive', etc.
        """
        color_map = {
            'event': (0.0, 1.0, 0.0, 1.0),  # Green
            'active': (0.0, 0.0, 1.0, 1.0),  # Blue
            'inactive': (0.5, 0.5, 0.5, 1.0),  # Gray
            'default': (1.0, 0.0, 0.0, 1.0)  # Red
        }

        for item in self.view.items:
            if hasattr(item, 'electrode_data'):
                electrode = item.electrode_data
                state = None

                # Check both ID and name
                for id_type in ['index', 'name']:
                    if str(electrode[id_type]) in electrode_states:
                        state = electrode_states[str(electrode[id_type])]
                        break

                item.setColor(color_map.get(state, color_map['default']))

        self.view.update()
        
    def post_electrode_update(self, electrode_ids):
        """Thread-safe way to request electrode updates"""
        event = ElectrodeUpdateEvent(electrode_ids)
        QCoreApplication.postEvent(self, event)
    
    def eventFilter(self, obj, event):
        """Handle custom events"""
        if event.type() == ElectrodeUpdateEvent.EVENT_TYPE:
            self.handle_electrode_update(event)
            return True
        return super().eventFilter(obj, event)
    
    def handle_electrode_update(self, event):
        """Actual electrode highlighting implementation"""
        if not hasattr(self, 'view') or not self.view:
            return
            
        for item in self.view.items:
            if hasattr(item, 'electrode_data'):
                electrode = item.electrode_data
                should_highlight = (
                    str(electrode['index']) in event.electrode_ids or
                    electrode['name'] in event.electrode_ids
                )
                
                color = (0.0, 1.0, 0.0, 1.0) if should_highlight else (1.0, 0.0, 0.0, 1.0)
                item.setColor(color)
        
        self.view.update()


class TabWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.recording_start = datetime.datetime.now()

        self.hfo_timestamps = []
        self.ied_timestamps = []
        self.seizure_timestamps = []
        self.hfo_detections = []

        self.ignored_channels = set()

        self.visualization_3d = Visualization3d(self)

        self.seizure_status = False
        self.seizure_start_time = None

        self.setWindowTitle("GUI Layout")
        self.setGeometry(100, 100, 1000, 600)
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Initialize detection data structures
        self.ied_count = 0
        self.ied_timestamps = []
        self.hfo_count = 0
        self.hfo_timestamps = []
        self.seizure_status = False
        self.seizure_start_time = None
        self.top_channels = []  # Will store (channel_name, median_value) tuples
        self.bottom_channels = []  # Will store (channel_name, median_value) tuples

        # Initialize respiratory data
        self.resp_time = np.linspace(0, 10, 1000)  # 10 seconds of data
        self.resp_voltage = np.sin(self.resp_time * 2 * np.pi * 0.2)  # 0.2 Hz sine wave

        self.seizure_duration_label = QLabel("Duration: 00:00:00")
        self.channel_status_label = QLabel(f"Ignoring 0 of 0 channels. "
            f"Selected: {', '.join(sorted(self.ignored_channels)) or 'None'}")

        # Load configuration
        self.config = self.load_config()

        # Create tabs
        self.create_tab("Seizure detection", is_seizure_tab=True)
        self.create_tab("HFO/IEDs", is_hfo_tab=True)
        self.create_tab("Respiratory Signals", is_resp_tab=True)
        self.create_tab("3D Visualization", is_3d_tab=True)
        self.create_tab("Channel Selection", is_channel_selection=True)
        self.create_tab("Settings", is_settings=True)

        # Initialize 3D visualization
        self.visualization_3d = Visualization3d(self)

        # Setup update timers
        self.setup_update_timers()

        # Setup respiratory plot timer
        self.resp_plot_timer = QTimer()
        self.resp_plot_timer.timeout.connect(self.update_resp_plot)
        self.resp_plot_timer.start(100)  # Update every 100ms

    def get_total_duration(self):
        """Calculate total recording duration"""
        if self.recording_start:
            return (datetime.datetime.now() - self.recording_start).total_seconds()
        return 0.0

    def load_config(self):
        """Load configuration from file or use defaults"""
        config_file = "config.json"
        default_config = {
            "hfo": {
                "amplitude": {"k": 1.37, "theta": 1.73},
                "frequency_dominance": {"k": 1.26, "theta": 0.46},
                "product": {"k": 1.28, "theta": 34.4},
                "cycles": {"k": 1.76, "theta": 4.04},
                "combination": {"k": 2.24, "theta": 0.53}
            },
            "ied": {
                "int": 5000,
                "block_size_sec": 60,
                "spike_band": [20, 50],
                "display_band": [1, 35],
                "artifact_slope_thresh": 10.0,
                "spike_amplitude_thresh": 600.0,
                "spike_slope_thresh": 7.0,
                "spike_duration_thresh": 10.0,
                "target_median_amp": 70.0
            },
            "seizure": {
                "fs": 5000,
                "short_window_sec": 2.0,
                "long_window_sec": 30.0,
                "threshold_offset": 1.5
            }
        }
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    return {**default_config, **loaded_config}
            return default_config
        except Exception as e:
            print(f"Error loading config: {e}")
            return default_config

    def save_config(self):
        """Save current configuration to file"""
        try:
            with open("config.json", 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False

    def setup_update_timers(self):
        """Set up timers for periodic UI updates"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui_elements)
        self.update_timer.start(1000)  # Update every second

    def create_tab(self, title, text="", is_seizure_tab=False, is_hfo_tab=False, is_3d_tab=False, is_resp_tab=False, is_channel_selection=False, is_settings=False):
        tab = QWidget()
        layout = QVBoxLayout()

        if is_settings:
            tab = QWidget()
            main_layout = QVBoxLayout(tab)
            
            # Create scroll area for settings
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            content_widget = QWidget()
            scroll_layout = QVBoxLayout(content_widget)
    
            # Detection Parameters Sections
            self.create_hfo_settings_section(scroll_layout)
            self.create_ied_settings_section(scroll_layout)
            self.create_seizure_settings_section(scroll_layout)
    
            # Electrode Color Configuration
            color_group = QGroupBox("Electrode Colors Configuration")
            color_layout = QVBoxLayout()
            seizure_color = QLabel("Seizure: Red\nHFO: Blue\nIED: Yellow")
            color_layout.addWidget(seizure_color)
            color_group.setLayout(color_layout)
            scroll_layout.addWidget(color_group)
    
            # Data Export Configuration
            export_group = QGroupBox("Data Export")
            export_layout = QVBoxLayout()
            
            # Format selection and export button
            format_layout = QHBoxLayout()
            format_layout.addWidget(QLabel("Export Format:"))
            
            self.export_format = QComboBox()
            self.export_format.addItems(["JSON (Recommended)", "CSV (Zipped)"])
            format_layout.addWidget(self.export_format)
            
            # Add the Data Export button here
            self.save_button = QPushButton("Export Detection Data")
            self.save_button.clicked.connect(self.save_detection_data)
            format_layout.addWidget(self.save_button)
            
            export_layout.addLayout(format_layout)
            
            # Export options
            self.include_metadata_check = QCheckBox("Include metadata (timestamps, ignored channels)")
            self.include_metadata_check.setChecked(True)
            export_layout.addWidget(self.include_metadata_check)
            
            export_group.setLayout(export_layout)
            scroll_layout.addWidget(export_group)
    
            # Advanced Settings
            advanced_group = QGroupBox("Advanced Settings")
            advanced_layout = QVBoxLayout()
            reset_button = QPushButton("Reset All Settings to Defaults")
            reset_button.clicked.connect(self.reset_settings)
            advanced_layout.addWidget(reset_button)
            advanced_group.setLayout(advanced_layout)
            scroll_layout.addWidget(advanced_group)
    
            # Set up scroll area
            scroll.setWidget(content_widget)
            main_layout.addWidget(scroll)
    
            # Settings Save button (separate from data export)
            btn_save = QPushButton("Save Application Settings")
            btn_save.clicked.connect(self.save_config)
            main_layout.addWidget(btn_save, alignment=Qt.AlignRight)
    
            self.tab_widget.addTab(tab, title)

        if is_channel_selection:
            tab = QWidget()
            layout = QVBoxLayout()
            
            # Channel selection list
            self.channel_list_widget = QListWidget()
            self.channel_list_widget.setSelectionMode(QListWidget.MultiSelection)
            self.channel_list_widget.itemChanged.connect(self.update_ignored_channels)
            
            # Selection controls
            control_layout = QHBoxLayout()
            self.select_all_button = QPushButton("Select All")
            self.select_all_button.clicked.connect(self.select_all_channels)
            self.clear_all_button = QPushButton("Clear All")
            self.clear_all_button.clicked.connect(self.clear_all_channels)
            
            control_layout.addWidget(self.select_all_button)
            control_layout.addWidget(self.clear_all_button)
            
            layout.addWidget(QLabel("Channels to Ignore:"))
            layout.addWidget(self.channel_list_widget)
            layout.addLayout(control_layout)
            
            tab.setLayout(layout)
            self.tab_widget.addTab(tab, title)

        elif is_seizure_tab:
            # Seizure status indicator
            self.seizure_indicator = QLabel()
            self.seizure_indicator.setFixedSize(30, 30)
            self.update_seizure_status(False)
            layout.addWidget(self.seizure_indicator, alignment=Qt.AlignTop | Qt.AlignLeft)

            # Seizure duration label
            self.seizure_duration_label = QLabel("Duration: 00:00:00")
            layout.addWidget(self.seizure_duration_label)

            # Top Channels Table
            self.top_channels_table = QTableWidget()
            self.top_channels_table.setColumnCount(2)
            self.top_channels_table.setHorizontalHeaderLabels(["Channel", "Median Value"])
            self.top_channels_table.setRowCount(3)
            layout.addWidget(QLabel("Top 3 Channels:"))
            layout.addWidget(self.top_channels_table)

            # Bottom Channels Table
            self.bottom_channels_table = QTableWidget()
            self.bottom_channels_table.setColumnCount(2)
            self.bottom_channels_table.setHorizontalHeaderLabels(["Channel", "Median Value"])
            self.bottom_channels_table.setRowCount(3)
            layout.addWidget(QLabel("Bottom 3 Channels:"))
            layout.addWidget(self.bottom_channels_table)

            # Spacer
            layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        elif is_hfo_tab:
            # HFO Detection Section
            hfo_box = QGroupBox("HFO Detection")
            hfo_layout = QVBoxLayout()

            self.hfo_count_label = QLabel("Total HFOs Detected: 0")
            hfo_layout.addWidget(self.hfo_count_label)

            self.hfo_details_table = QTableWidget()
            self.hfo_details_table.setColumnCount(3)
            self.hfo_details_table.setHorizontalHeaderLabels(["Channel", "Time", "Amplitude"])
            hfo_layout.addWidget(self.hfo_details_table)

            hfo_box.setLayout(hfo_layout)
            layout.addWidget(hfo_box)

            # IED Detection Section
            ied_box = QGroupBox("IED Detection")
            ied_layout = QVBoxLayout()

            self.ied_count_label = QLabel("Total IEDs Detected: 0")
            ied_layout.addWidget(self.ied_count_label)

            self.ied_details_table = QTableWidget()
            self.ied_details_table.setColumnCount(3)
            self.ied_details_table.setHorizontalHeaderLabels(["Channel", "Time", "Duration (ms)"])
            ied_layout.addWidget(self.ied_details_table)

            ied_box.setLayout(ied_layout)
            layout.addWidget(ied_box)

        elif is_3d_tab:
            # 3D Visualization Tab
            self.init_3d_tab(layout)
            
        elif is_resp_tab:
            # Respiratory Signals Tab
            self.init_resp_tab(layout)
            
            
            
        else:
            layout.addWidget(QLabel(text))

        tab.setLayout(layout)
        self.tab_widget.addTab(tab, title)

    def update_default_color(self, state):
        """Update default electrode color based on checkbox state"""
        new_color = (1.0, 1.0, 1.0, 1.0) if state else (1.0, 0.0, 0.0, 1.0)
        if hasattr(self, 'visualization_3d'):
            self.visualization_3d.default_color = new_color
            self.visualization_3d.update_electrode_colors()

    def save_as_csv(self, data, path):
        """Save data to multiple CSV files in a zip archive"""
        with zipfile.ZipFile(path, 'w') as zipf:
            # Save metadata
            with zipf.open('metadata.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["Export Date", data['metadata']['export_date']])
                writer.writerow(["Ignored Channels", ','.join(data['metadata']['ignored_channels'])])
                
            # Save HFO data
            with zipf.open('hfo.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["Channel", "Timestamp", "Amplitude (µV)"])
                for d in data['hfo']['detections']:
                    writer.writerow([d['channel'], d['timestamp'], d['amplitude']])
                    
            # Save IED data
            with zipf.open('ied.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["Channel", "Timestamp", "Duration (ms)"])
                for d in data['ied']['detections']:
                    writer.writerow([d['channel'], d['timestamp'], d['duration']])
                    
            # Save seizure data
            with zipf.open('seizure.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["Start", "End", "Duration (s)", "Channels"])
                for d in data['seizure']['detections']:
                    writer.writerow([
                        d['start'], 
                        d['end'], 
                        d['duration'], 
                        ','.join(d['channels'])
                    ])

    def get_detection_data(self):
        """Collect all detection data into a structured dictionary"""
        return {
            "metadata": {
                "export_date": datetime.datetime.now().isoformat(),
                "ignored_channels": list(self.ignored_channels),
                "total_duration": self.get_total_duration(),
            },
            "hfo": {
                "count": self.hfo_count,
                "detections": [{
                    "channel": d['channel'],
                    "timestamp": d['timestamp'].isoformat(),
                    "amplitude": d['amplitude']
                } for d in self.hfo_timestamps]
            },
            "ied": {
                "count": self.ied_count,
                "detections": [{
                    "channel": d['channel'],
                    "timestamp": d['timestamp'].isoformat(),
                    "duration": d['duration']
                } for d in self.ied_timestamps]
            },
            "seizure": {
                "count": len(self.seizure_timestamps),
                "detections": [{
                    "start": s['start'].isoformat(),
                    "end": s['end'].isoformat(),
                    "channels": s['channels'],
                    "duration": s['duration'].total_seconds()
                } for s in self.seizure_timestamps]
            }
        }

    def reset_settings(self):
        """Reset all settings to default values"""
        # Reset color settings
        self.default_color_check.setChecked(True)
        self.update_default_color(Qt.Checked)
        
        # Reset export settings
        self.export_format.setCurrentIndex(0)
        self.include_metadata_check.setChecked(True)
        
        # Reset ignored channels
        self.ignored_channels.clear()
        self.update_channel_list()
        
        QMessageBox.information(self, "Settings Reset", "All settings have been reset to defaults.")

    def save_detection_data(self):
        """Handle data export based on selected format"""
        #try:
        if self.export_format.currentText().startswith("JSON"):
            self.save_json_data()
        else:
            self.save_csv_data()
            
        QMessageBox.information(self, "Export Successful", 
                                "Data saved successfully!")
            
        #except Exception as e:
        #    QMessageBox.critical(self, "Export Error", 
        #                       f"Failed to save data: {str(e)}")
            
    def save_json_data(self):
        """Save data to JSON format"""
        data = self.get_detection_data()
        if not self.include_metadata_check.isChecked():
            del data['metadata']
            
        path, _ = QFileDialog.getSaveFileName(
            self, "Save JSON Data", "", "JSON Files (*.json)")
        
        if path:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

    def save_csv_data(self):
        """Save data to zipped CSV format"""
        data = self.get_detection_data()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV Data", "", "ZIP Archives (*.zip)")
        
        if path:
            with zipfile.ZipFile(path, 'w') as zipf:
                # Save metadata
                if self.include_metadata_check.isChecked():
                    with zipf.open('metadata.csv', 'w') as f:
                        writer = csv.writer(f)
                        writer.writerow(["Export Date", datetime.datetime.now().isoformat()])
                        writer.writerow(["Ignored Channels", ','.join(self.ignored_channels)])
                
                # Save detection data
                self.save_csv_to_zip(zipf, 'hfo.csv', data['hfo']['detections'], 
                                   ["Channel", "Timestamp", "Amplitude (µV)"])
                self.save_csv_to_zip(zipf, 'ied.csv', data['ied']['detections'],
                                   ["Channel", "Timestamp", "Duration (ms)"])
                self.save_csv_to_zip(zipf, 'seizure.csv', data['seizure']['detections'],
                                   ["Start", "End", "Duration", "Channels"])
                
    def save_csv_to_zip(self, zipf, filename, data, headers):
        """Helper method to save CSV data to zip archive"""
        with zipf.open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for item in data:
                writer.writerow(list(item.values()))

    def update_channel_list(self):
        """Populate/update the channel list from loaded electrodes"""
        self.channel_list_widget.clear()
        
        # Check if electrodes are loaded
        if (hasattr(self.visualization_3d, 'electrodes') and 
            len(self.visualization_3d.electrodes) > 0):
            
            # Get sorted list of electrode names
            channels = sorted([e['name'] for e in self.visualization_3d.electrodes],
                             key=lambda x: (''.join(filter(str.isalpha, x)), 
                                          int(''.join(filter(str.isdigit, x)) or 0)))
            
            # Add channels to list widget
            for channel in channels:
                item = QListWidgetItem(channel)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                self.channel_list_widget.addItem(item)
        else:
            # Show instructional message
            item = QListWidgetItem("No channels available. Load electrodes in 3D Visualization tab first.")
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)  # Make non-selectable
            item.setForeground(QColor(255, 0, 0))  # Red text
            self.channel_list_widget.addItem(item)

    def select_all_channels(self):
        """Select all channels in the list"""
        for i in range(self.channel_list_widget.count()):
            item = self.channel_list_widget.item(i)
            item.setCheckState(Qt.Checked)

    def clear_all_channels(self):
        """Deselect all channels in the list"""
        for i in range(self.channel_list_widget.count()):
            item = self.channel_list_widget.item(i)
            item.setCheckState(Qt.Unchecked)

    def update_ignored_channels(self):
        """Update the set of ignored channels and show status"""
        self.ignored_channels.clear()
        checked_count = 0

        for i in range(self.channel_list_widget.count()):
            item = self.channel_list_widget.item(i)
            if item.checkState() == Qt.Checked:
                self.ignored_channels.add(item.text())
                checked_count += 1

        # Update status label
        total = self.channel_list_widget.count()
        self.channel_status_label.setText(
            f"Ignoring {checked_count} of {total} channels. "
            f"Selected: {', '.join(sorted(self.ignored_channels)) or 'None'}"
        )

    def create_settings_tab(self, layout):
        """Create the settings tab with editable parameters"""
        # Main container with scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        scroll.setWidget(content)
        main_layout = QVBoxLayout(content)
        
        # Add sections for each detection type
        self.create_hfo_settings_section(main_layout)
        self.create_ied_settings_section(main_layout)
        self.create_seizure_settings_section(main_layout)
        
        # Save button
        btn_save = QPushButton("Save Settings")
        btn_save.clicked.connect(self.save_config)
        main_layout.addWidget(btn_save, alignment=Qt.AlignRight)
        
        # Add stretch to push content up
        main_layout.addStretch()
        
        layout.addWidget(scroll)

    def create_hfo_settings_section(self, layout):
        """Create HFO settings section"""
        group = QGroupBox("HFO Detection Parameters")
        group_layout = QFormLayout()
        
        # Amplitude parameters
        group_layout.addRow(QLabel("<b>Amplitude</b>"))
        group_layout.addRow(QLabel("k:"), self.create_double_spinbox("hfo.amplitude.k", 0.1, 10.0, 0.01))
        group_layout.addRow(QLabel("θ:"), self.create_double_spinbox("hfo.amplitude.theta", 0.1, 10.0, 0.01))
        
        # Frequency dominance parameters
        group_layout.addRow(QLabel("<b>Frequency Dominance</b>"))
        group_layout.addRow(QLabel("k:"), self.create_double_spinbox("hfo.frequency_dominance.k", 0.1, 10.0, 0.01))
        group_layout.addRow(QLabel("θ:"), self.create_double_spinbox("hfo.frequency_dominance.theta", 0.1, 10.0, 0.01))
        
        # Product parameters
        group_layout.addRow(QLabel("<b>Product</b>"))
        group_layout.addRow(QLabel("k:"), self.create_double_spinbox("hfo.product.k", 0.1, 10.0, 0.01))
        group_layout.addRow(QLabel("θ:"), self.create_double_spinbox("hfo.product.theta", 1.0, 100.0, 0.1))
        
        # Cycles parameters
        group_layout.addRow(QLabel("<b>Cycles</b>"))
        group_layout.addRow(QLabel("k:"), self.create_double_spinbox("hfo.cycles.k", 0.1, 10.0, 0.01))
        group_layout.addRow(QLabel("θ:"), self.create_double_spinbox("hfo.cycles.theta", 0.1, 10.0, 0.01))
        
        # Combination parameters
        group_layout.addRow(QLabel("<b>Combination</b>"))
        group_layout.addRow(QLabel("k:"), self.create_double_spinbox("hfo.combination.k", 0.1, 10.0, 0.01))
        group_layout.addRow(QLabel("θ:"), self.create_double_spinbox("hfo.combination.theta", 0.1, 10.0, 0.01))
        
        group.setLayout(group_layout)
        layout.addWidget(group)

    def create_ied_settings_section(self, layout):
        """Create IED settings section"""
        group = QGroupBox("IED Detection Parameters")
        group_layout = QFormLayout()
        
        # Basic parameters
        group_layout.addRow(QLabel("Integration Interval (ms):"), self.create_spinbox("ied.int", 100, 10000, 100))
        group_layout.addRow(QLabel("Block Size (sec):"), self.create_spinbox("ied.block_size_sec", 1, 300, 1))
        
        # Band parameters
        group_layout.addRow(QLabel("<b>Frequency Bands</b>"))
        group_layout.addRow(QLabel("Spike Band (Hz):"), self.create_range_input("ied.spike_band", 1, 100))
        group_layout.addRow(QLabel("Display Band (Hz):"), self.create_range_input("ied.display_band", 1, 100))
        
        # Threshold parameters
        group_layout.addRow(QLabel("<b>Thresholds</b>"))
        group_layout.addRow(QLabel("Artifact Slope:"), self.create_double_spinbox("ied.artifact_slope_thresh", 0.1, 50.0, 0.1))
        group_layout.addRow(QLabel("Spike Amplitude (µV):"), self.create_double_spinbox("ied.spike_amplitude_thresh", 100.0, 1000.0, 10.0))
        group_layout.addRow(QLabel("Spike Slope:"), self.create_double_spinbox("ied.spike_slope_thresh", 0.1, 20.0, 0.1))
        group_layout.addRow(QLabel("Spike Duration (ms):"), self.create_double_spinbox("ied.spike_duration_thresh", 1.0, 50.0, 1.0))
        group_layout.addRow(QLabel("Target Median Amp (µV):"), self.create_double_spinbox("ied.target_median_amp", 10.0, 200.0, 5.0))
        
        group.setLayout(group_layout)
        layout.addWidget(group)

    def create_seizure_settings_section(self, layout):
        """Create seizure detection settings section"""
        group = QGroupBox("Seizure Detection Parameters")
        group_layout = QFormLayout()
        
        # Sampling parameters
        group_layout.addRow(QLabel("Sampling Frequency (Hz):"), self.create_spinbox("seizure.fs", 100, 10000, 100))
        
        # Window parameters
        group_layout.addRow(QLabel("<b>Window Sizes</b>"))
        group_layout.addRow(QLabel("Short Window (sec):"), self.create_double_spinbox("seizure.short_window_sec", 0.1, 10.0, 0.1))
        group_layout.addRow(QLabel("Long Window (sec):"), self.create_double_spinbox("seizure.long_window_sec", 5.0, 60.0, 1.0))
        
        # Threshold parameter
        group_layout.addRow(QLabel("Threshold Offset:"), self.create_double_spinbox("seizure.threshold_offset", 0.1, 5.0, 0.1))
        
        group.setLayout(group_layout)
        layout.addWidget(group)

    def create_double_spinbox(self, config_path, min_val, max_val, step):
        """Create a double spin box for float parameters"""
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setSingleStep(step)
        
        # Get current value from nested config
        keys = config_path.split('.')
        value = self.config
        for key in keys:
            value = value[key]
        spinbox.setValue(value)
        
        spinbox.valueChanged.connect(lambda val, path=config_path: self.update_nested_config(path, val))
        return spinbox

    def create_spinbox(self, config_path, min_val, max_val, step):
        """Create a spin box for integer parameters"""
        spinbox = QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setSingleStep(step)
        
        # Get current value from nested config
        keys = config_path.split('.')
        value = self.config
        for key in keys:
            value = value[key]
        spinbox.setValue(value)
        
        spinbox.valueChanged.connect(lambda val, path=config_path: self.update_nested_config(path, val))
        return spinbox

    def create_range_input(self, config_path, min_val, max_val):
        """Create a widget for editing range/tuple parameters"""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Get current values
        keys = config_path.split('.')
        current_values = self.config
        for key in keys:
            current_values = current_values[key]
        
        # Create spin boxes for min and max
        min_spin = QDoubleSpinBox()
        min_spin.setRange(min_val, max_val)
        min_spin.setValue(current_values[0])
        min_spin.valueChanged.connect(lambda val, path=config_path: self.update_range_config(path, 0, val))
        
        max_spin = QDoubleSpinBox()
        max_spin.setRange(min_val, max_val)
        max_spin.setValue(current_values[1])
        max_spin.valueChanged.connect(lambda val, path=config_path: self.update_range_config(path, 1, val))
        
        layout.addWidget(min_spin)
        layout.addWidget(QLabel("to"))
        layout.addWidget(max_spin)
        
        return container

    def update_nested_config(self, config_path, value):
        """Update a nested configuration value"""
        keys = config_path.split('.')
        config = self.config
        for key in keys[:-1]:
            config = config.setdefault(key, {})
        config[keys[-1]] = value

    def update_range_config(self, config_path, index, value):
        """Update a range/tuple configuration value"""
        keys = config_path.split('.')
        config = self.config
        for key in keys[:-1]:
            config = config.setdefault(key, {})
        current_range = config[keys[-1]]
        current_range[index] = value
        config[keys[-1]] = current_range

    def init_resp_tab(self, layout):
        """Initialize the Respiratory Signals tab with a plot"""
        # Create plot widget
        self.resp_plot_widget = pyqtgraph.PlotWidget()
        self.resp_plot_widget.setBackground('w')
        self.resp_plot_widget.setLabel('left', 'Voltage (V)')
        self.resp_plot_widget.setLabel('bottom', 'Time (s)')
        self.resp_plot_widget.addLegend()
        self.resp_plot_widget.showGrid(x=True, y=True)
        
        # Create plot curve
        self.resp_plot_curve = self.resp_plot_widget.plot(
            self.resp_time, 
            self.resp_voltage, 
            pen=pyqtgraph.mkPen(color='b', width=2),
            name='Respiratory Signal'
        )
        
        # Add to layout
        layout.addWidget(self.resp_plot_widget)
        
        # Add controls
        control_layout = QHBoxLayout()
        
        # Add a button to simulate new data
        self.simulate_button = QPushButton("Simulate New Data")
        self.simulate_button.clicked.connect(self.simulate_new_resp_data)
        control_layout.addWidget(self.simulate_button)
        
        # Add a combo box for signal type
        self.signal_type_combo = QComboBox()
        self.signal_type_combo.addItems(["Sine Wave", "Cosine Wave", "Random Noise"])
        control_layout.addWidget(self.signal_type_combo)
        
        layout.addLayout(control_layout)

    def simulate_new_resp_data(self):
        """Generate new simulated respiratory data based on selected type"""
        signal_type = self.signal_type_combo.currentText()
        
        if signal_type == "Sine Wave":
            self.resp_voltage = np.sin(self.resp_time * 2 * np.pi * 0.2)  # 0.2 Hz sine wave
        elif signal_type == "Cosine Wave":
            self.resp_voltage = np.cos(self.resp_time * 2 * np.pi * 0.2)  # 0.2 Hz cosine wave
        else:  # Random Noise
            self.resp_voltage = np.random.normal(0, 0.5, len(self.resp_time))
            
        # Add some drift to make it more realistic
        self.resp_voltage += np.linspace(0, 1, len(self.resp_time)) * 0.2

    def update_resp_plot(self):
        """Update the respiratory plot with new data"""
        # Shift time and add new point
        self.resp_time = np.roll(self.resp_time, -1)
        self.resp_time[-1] = self.resp_time[-2] + 0.01  # 10ms step
        
        # Update voltage data
        self.resp_voltage = np.roll(self.resp_voltage, -1)
        
        # Generate new point based on current pattern
        if "Sine" in self.signal_type_combo.currentText():
            self.resp_voltage[-1] = np.sin(self.resp_time[-1] * 2 * np.pi * 0.2)
        elif "Cosine" in self.signal_type_combo.currentText():
            self.resp_voltage[-1] = np.cos(self.resp_time[-1] * 2 * np.pi * 0.2)
        else:
            self.resp_voltage[-1] = np.random.normal(0, 0.5)
            
        # Add some drift
        self.resp_voltage[-1] += self.resp_time[-1] * 0.02
        
        # Update plot
        self.resp_plot_curve.setData(self.resp_time, self.resp_voltage)
        
        # Auto-range every 100 updates to prevent excessive computation
        if self.resp_plot_timer.interval() * 100 % 100 == 0:
            self.resp_plot_widget.autoRange()

    def init_3d_tab(self, layout):
        """Initialize the 3D visualization tab"""
        # Button to load files
        load_button = QPushButton("Load Brain Surface and Electrodes")
        load_button.clicked.connect(self.load_3d_data)
        layout.addWidget(load_button)

        # Container for the 3D view
        self.view_container = QWidget()
        self.view_layout = QVBoxLayout(self.view_container)
        layout.addWidget(self.view_container)

        # Button to highlight example electrodes
        example_button = QPushButton("Highlight Example Electrodes")
        example_button.clicked.connect(self.highlight_example_electrodes)
        layout.addWidget(example_button)

    def load_3d_data(self):
        """Load data for 3D visualization"""
        try:
            # Load paths and initialize visualization
            self.visualization_3d.get_brain_surface_path()
            self.visualization_3d.get_electrode_path()
            self.visualization_3d.get_nii_path()

            # Clear previous view
            for i in reversed(range(self.view_layout.count())):
                self.view_layout.itemAt(i).widget().setParent(None)

            # Initialize visualization
            view = self.visualization_3d.initialize_visualization()
            self.view_layout.addWidget(view)

            # Force channel list update
            self.update_channel_list()

            # Initial color update
            self.visualization_3d.update_electrode_colors()
        
        except Exception as e:
            print(f"Error loading 3D data: {str(e)}")
            self.show_error_message("3D Data Load Error", str(e))

    def highlight_example_electrodes(self):
        """Example method to highlight some electrodes"""
        if hasattr(self.visualization_3d, 'electrodes') and self.visualization_3d.electrodes:
            # Highlight first 3 electrodes as an example
            electrode_ids = [str(e['index']) for e in self.visualization_3d.electrodes[:3]]
            self.visualization_3d.post_electrode_update(electrode_ids)

    def update_ui_elements(self):
        """Periodically update UI elements that need refreshing"""
        if self.seizure_status and self.seizure_start_time:
            duration = datetime.datetime.now() - self.seizure_start_time
            self.seizure_duration_label.setText(f"Duration: {str(duration).split('.')[0]}")

    def update_hfo_results(self, hfo_detections):
        """Update GUI with new HFO detections"""
        filtered_detections = [
            d for d in hfo_detections 
            if d['channel'] not in self.ignored_channels
        ]

        self.hfo_detections.extend(filtered_detections)

        for detection in reversed(filtered_detections):  # Reverse to maintain temporal order
            self._insert_hfo_row(detection, 0)

        self.hfo_count += len(filtered_detections)
        self.hfo_count_label.setText(f"Total HFOs Detected: {self.hfo_count}")
        self.hfo_details_table.scrollToTop()

        hfo_electrodes = [d['channel'] for d in filtered_detections]
        self.visualization_3d.update_electrode_states(hfo_electrodes, 'hfo', duration=5000)
        
    def _insert_hfo_row(self, detection, position):
        """Insert a single HFO detection at specified position"""
        self.hfo_details_table.insertRow(position)
        
        # Create table items
        items = [
            QTableWidgetItem(detection['channel']),
            QTableWidgetItem(detection['timestamp'].strftime("%H:%M:%S.%f")[:-3]),
            QTableWidgetItem(f"{detection['amplitude']:.2f}"),
        ]
        
        # Highlight if above threshold
        if detection['amplitude'] > 100:
            for item in items:
                item.setBackground(QColor(255, 230, 230))
                
        # Add to table
        for col, item in enumerate(items):
            item.setTextAlignment(Qt.AlignCenter)
            self.hfo_details_table.setItem(position, col, item)

    def update_ied_results(self, ied_detections):
        """Update GUI with new IED detections"""
        filtered_detections = [
            d for d in ied_detections 
            if d['channel'] not in self.ignored_channels
        ]

        self.ied_count += len(filtered_detections)
        self.ied_count_label.setText(f"Total IEDs Detected: {self.ied_count}")

        # Update IED details and electrode colors
        ied_electrodes = [d['channel'] for d in filtered_detections]
        self.visualization_3d.update_electrode_states(ied_electrodes, 'ied', duration=5000)

        # Update IED details table
        self.ied_details_table.setRowCount(len(filtered_detections))
        for row, detection in enumerate(filtered_detections):
            channel_item = QTableWidgetItem(detection['channel'])
            time_item = QTableWidgetItem(datetime.datetime.now())
            dur_item = QTableWidgetItem(f"{detection['duration']:.1f}")

            # Color code based on duration
            if detection['duration'] > self.config["ied"]["spike_duration_thresh"]:
                dur_item.setBackground(QColor(200, 200, 255))  # Light blue

            self.ied_details_table.setItem(row, 0, channel_item)
            self.ied_details_table.setItem(row, 1, time_item)
            self.ied_details_table.setItem(row, 2, dur_item)
        self.ied_timestamps.extend(filtered_detections)

    def update_seizure_results(self, seizure_data):
        """Update GUI with seizure detection results"""

        detections = seizure_data[2]
        median = seizure_data[3]
        channel = seizure_data[4]
        if channel in self.ignored_channels:
            is_seizure = False
            current_electrodes = [ch[0] for ch in self.top_channels + self.bottom_channels]
            self.visualization_3d.update_electrode_states(current_electrodes, 'none')
        else:
            # Track channel activity metrics
            if not hasattr(self, 'channel_metrics'):
                self.channel_metrics = {}  # {channel: {'medians': [], 'detections': []}}

            # Update metrics for current channel
            if channel not in self.channel_metrics:
                self.channel_metrics[channel] = {'medians': [], 'detections': []}
            self.channel_metrics[channel]['medians'].append(median)
            self.channel_metrics[channel]['detections'].extend(detections)

            # Calculate seizure status (any detection in any channel)
            is_seizure = any(np.concatenate(
                [metrics['detections'] for metrics in self.channel_metrics.values()]
            ))

        self.update_seizure_status(is_seizure)

        if is_seizure:
            current_electrodes = [ch[0] for ch in self.top_channels + self.bottom_channels]
            self.visualization_3d.update_electrode_states(current_electrodes, 'seizure')
        
            if not self.seizure_status:  # Seizure just started
                self.seizure_start_time = datetime.datetime.now()
                self.top_channels = []
                self.bottom_channels = []

            # Calculate average medians for ranking
            avg_medians = {
                ch: np.mean(metrics['medians']) 
                for ch, metrics in self.channel_metrics.items()
            }

            # Sort channels by activity
            sorted_channels = sorted(avg_medians.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)

            # Keep top/bottom 3 channels updated
            self.top_channels = sorted_channels[:3]
            self.bottom_channels = sorted_channels[-3:]

            # Update tables
            self.update_channel_table(self.top_channels_table, self.top_channels)
            self.update_channel_table(self.bottom_channels_table, self.bottom_channels)
        else:
            self.visualization_3d.clear_state('seizure')
            if self.seizure_status:  # Seizure just ended
                # Clear temporary metrics
                self.channel_metrics = {}

            self.seizure_start_time = None
            self.top_channels = []
            self.bottom_channels = []
            return

        self.seizure_status = is_seizure
        self.seizure_timestamps.append({
        "start": self.seizure_start_time,
        "end": datetime.datetime.now(),
        "duration": datetime.datetime.now() - self.seizure_start_time,
        "channels": [ch[0] for ch in self.top_channels + self.bottom_channels]
    })

    def update_channel_table(self, table, channels):
        """Helper method to update channel tables"""
        table.setRowCount(len(channels))
        for row, (channel, median) in enumerate(channels):
            channel_item = QTableWidgetItem(channel)
            median_item = QTableWidgetItem(f"{median:.2f}")
            
            # Highlight extreme values
            if median > self.config["seizure"]["threshold_offset"] * 100:  # Example scaling
                median_item.setBackground(QColor(255, 220, 220))
            
            table.setItem(row, 0, channel_item)
            table.setItem(row, 1, median_item)

    def update_seizure_status(self, is_seizure):
        """Update seizure indicator"""
        self.seizure_status = is_seizure
        color = QColor(0, 255, 0) if is_seizure else QColor(255, 0, 0)
        
        # Create a colored pixmap for the indicator
        pixmap = QPixmap(30, 30)
        pixmap.fill(color)
        self.seizure_indicator.setPixmap(pixmap)
        
        if hasattr(self, 'visualization_3d') and self.visualization_3d:
            if is_seizure and not self.seizure_start_time:
                self.seizure_start_time = datetime.datetime.now()
            elif not is_seizure:
                self.visualization_3d.clear_state('seizure')
                self.seizure_start_time = None
                self.seizure_duration_label.setText("Duration: 00:00:00")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = TabWidget()
    main_window.show()
    sys.exit(app.exec_())