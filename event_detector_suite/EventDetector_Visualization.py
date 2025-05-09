from PyQt5.QtWidgets import QFileDialog, QApplication
import numpy as np
import open3d as o3d
import pyqtgraph.opengl
from scipy.io import loadmat
import h5py
import pyqtgraph
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import trimesh
from PyQt5.QtCore import QTimer
import nibabel as nib
from matplotlib.colors import Normalize
from collections import defaultdict
from PyQt5 import QtGui
from PyQt5.QtGui import QVector3D
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QEvent, QCoreApplication, QObject
from PyQt5.QtGui import QVector3D 

class ElectrodeUpdateEvent(QEvent):
    """Custom event type for electrode updates"""
    EVENT_TYPE = QEvent.Type(QEvent.registerEventType())
    
    def __init__(self, electrode_ids):
        super().__init__(self.EVENT_TYPE)
        self.electrode_ids = electrode_ids



class Visualization3d(QObject):
    def __init__(self):
        super().__init__() 
        self.electrode_labels = {}  # To store electrode labels
        self.current_label = None   # Currently displayed label
        
        self.event_handler_installed = False
        
        self.get_brain_surface_path()
        self.get_electrode_path()
        self.get_nii_path()
        self.power_dict = defaultdict(list)  # Dictionary to store electrode power values
        
    def get_nii_path(self):
        self.nii_path, _ = QFileDialog.getOpenFileName(None, "Select the CT.nii file")
        
    def get_brain_surface_path(self):
        self.brain_surface_path, _ = QFileDialog.getOpenFileName(
            None, "Select the surfaces.mat file in Registered Dicom 10 (Meg) folder")

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
        
        # Optional decimation (commented out as it may not be needed)
        # decimated_mesh = mesh.simplify_quadric_decimation(face_count=10000)
        # brain_vertices = np.array(decimated_mesh.vertices, dtype=np.float32)
        # brain_faces = np.array(decimated_mesh.faces, dtype=np.int32)
        
        meshdata = pyqtgraph.opengl.MeshData(vertexes=brain_vertices, faces=brain_faces)
        meshitem = pyqtgraph.opengl.GLMeshItem(
            meshdata=meshdata,
            color=(1, 1, 1, 0.10),  # RGBA (blue with full opacity)
            drawEdges = True,
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
                elec_names = [''.join(chr(c) for c in mat_file[ref]) 
                             for ref in mat_file['ElecNamesRaw'][:, 0]]
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
                color=(1.0, 0.0, 0.0, 1.0),
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
            # Reset previous electrode color
        

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

    def visualize(self, initial_electrodes=None):
        app = QApplication.instance() or QApplication([])
        self.view = pyqtgraph.opengl.GLViewWidget()  # Store as instance variable
        self.view.setWindowTitle('Brain with Electrodes')
        self.view.setGeometry(100, 100, 800, 600)
        
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
        
        self.view.show()
        if not self.event_handler_installed:
            app.installEventFilter(self)
            self.event_handler_installed = True
        
        # Post initial electrode update if provided
        if initial_electrodes:
            self.post_electrode_update(initial_electrodes)
        app.exec_()
      
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
        
if __name__ == "__main__":
    app = QApplication([])
    
    viz = Visualization3d()
    
    # First show the window
    QTimer.singleShot(0, lambda: viz.visualize())
    
    # Then after 2 seconds, highlight some electrodes
    QTimer.singleShot(2000, lambda: viz.post_electrode_update(['1', '3', 'LHD2']))
    
    QApplication.instance().exec_()