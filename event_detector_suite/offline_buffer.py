import os
import struct
import math
import scipy.io as sio
from typing import Dict, List, Optional, Tuple
import numpy as np
from threading import Lock
from datetime import datetime
import re

class NC3DataBuffer:
    def __init__(self, nsx_mat_path: str, sample_rate: float = 30000):
        """
        Simulates a data buffer for NC3 binary files with metadata from NSx.mat
        
        Args:
            nsx_mat_path: Path to the NSx.mat file
            sample_rate: Sampling rate in Hz (default: 30000)
        """
        self.nsx_mat_path = nsx_mat_path
        self.sample_rate = sample_rate
        self.lock = Lock()
        
        # Initialize data structures
        self.channels_info = []
        self.active_channels = []
        self.buffer = {}
        self.timestamps = []
        self.last_read_positions = {}
        
        # Load metadata
        self._load_metadata()

    def clean_electrode_name(self,name):
        """
        Converts electrode names by:
        1. Removing any suffixes after the main identifier
        2. Stripping leading zeros from numeric portion
        3. Preserving alphabetical prefix

        Example: "RO01 hi-res" -> "RO1"
        """
        # Split off suffix and take main identifier
        main_part = name.split()[0]

        # Separate alphabetical prefix and numeric portion
        match = re.match(r"^([A-Za-z]+)(\d+)(.*)$", main_part)

        if match:
            prefix = match.group(1).upper()
            numbers = match.group(2)
            # Strip leading zeros and combine
            return prefix + str(int(numbers))

        return main_part.upper()
        
    def _load_metadata(self):
        """Load and parse the NSx.mat file"""
        try:
            nsx_data = sio.loadmat(self.nsx_mat_path, simplify_cells=True)
            self.base_dir = os.path.dirname(self.nsx_mat_path)
            
            # Extract channel information
            for row in nsx_data['NSx']:
                self.channels_info.append({
                    'id': str(row['chan_ID']),
                    'conversion': float(row['conversion']),
                    'dc_offset': float(row['dc']),
                    'label': str(row['label']),
                    'extension': str(row['ext']),
                    'filepath': os.path.join(self.base_dir, str(row['filename']))
                })
                
        except Exception as e:
            raise RuntimeError(f"Failed to load NSx.mat file: {str(e)}")

    def get_channel_list(self) -> List[Dict]:
        """Return list of available channels with their metadata"""
        return self.channels_info.copy()

    def start_acquisition(self, channel_selection: Optional[List[str]] = None):
        """
        Initialize the buffer for selected channels
        
        Args:
            channel_selection: List of channel labels or IDs to acquire
                             If None, all channels will be selected
        """
        with self.lock:
            self.active_channels = []
            self.buffer = {}
            self.last_read_positions = {}
            
            # Determine which channels to activate
            for channel in self.channels_info:
                if "EKG" in channel['label']:
                    continue
                if channel_selection is None or any(
                    selector in channel['label'] or selector == channel['id']
                    for selector in channel_selection
                ):
                    self.active_channels.append(channel)
                    self.buffer[channel['label']] = []
                    self.last_read_positions[channel['label']] = 0

    def read_new_data(self, window_sec: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Read new data from all active channels
        
        Args:
            window_sec: Time window in seconds to read
            
        Returns:
            Dictionary of {channel_label: numpy_array} with new data
        """
        new_data = {}
        samples_to_read = int(window_sec * self.sample_rate)
        
        with self.lock:
            current_time = datetime.now()
            self.timestamps.append(current_time)
            
            for channel in self.active_channels:
                channel_label = channel['label']
                filepath = self._get_nc3_path(channel)
                
                try:
                    # Read binary data
                    raw_samples = self._read_nc3_segment(
                        filepath,
                        self.last_read_positions[channel_label],
                        samples_to_read,
                        channel['conversion'],
                        channel['dc_offset']
                    )
                    
                    # Update buffer and position
                    self.buffer[channel_label].extend(raw_samples)
                    self.last_read_positions[channel_label] += len(raw_samples)
                    dict_label = self.clean_electrode_name(channel_label)
                    new_data[dict_label] = np.array(raw_samples)
                    
                except Exception as e:
                    print(f"Error reading {channel_label}: {str(e)}")
                    new_data[channel_label] = np.array([])
        
        return new_data

    def _get_nc3_path(self, channel_info: Dict) -> str:
        """Generate the expected NC3 file path for a channel"""
        return os.path.join(
            self.base_dir,
            f"{channel_info['label']}_{channel_info['id']}{channel_info['extension']}"
        )

    def _read_nc3_segment(self, filepath: str, start_sample: int, 
                         num_samples: int, conversion: float, dc_offset: float) -> List[float]:
        """
        Read a segment from an NC3 file and apply conversion
        
        Args:
            filepath: Path to NC3 file
            start_sample: Starting sample index
            num_samples: Number of samples to read
            conversion: Conversion factor from raw to µV
            dc_offset: DC offset in µV
            
        Returns:
            List of converted samples in µV
        """
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"NC3 file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            # Seek to start position (2 bytes per sample)
            f.seek(start_sample * 2)
            
            # Read required number of samples
            bytes_to_read = num_samples * 2
            binary_data = f.read(bytes_to_read)
            
            # Handle incomplete reads
            if len(binary_data) < bytes_to_read:
                num_samples = len(binary_data) // 2
                if num_samples == 0:
                    return []
            
            # Unpack and convert data
            format_str = f'<{num_samples}h'  # little-endian 16-bit integers
            raw_data = struct.unpack(format_str, binary_data)
            
            return [x * conversion + dc_offset for x in raw_data]

    def get_latest_data(self, channel_label: str, num_samples: Optional[int] = None) -> np.ndarray:
        """
        Get the latest data from a specific channel
        
        Args:
            channel_label: Label of the channel to read
            num_samples: Number of samples to return (None for all available)
            
        Returns:
            Numpy array of the requested data
        """
        with self.lock:
            if channel_label not in self.buffer:
                raise ValueError(f"Channel {channel_label} not in active buffer")
            
            channel_data = self.buffer[channel_label]
            if num_samples is None:
                return np.array(channel_data)
            return np.array(channel_data[-num_samples:])

    def get_data_window(self, channel_label: str, window_sec: float) -> np.ndarray:
        """
        Get a time window of data from a specific channel
        
        Args:
            channel_label: Label of the channel to read
            window_sec: Time window in seconds
            
        Returns:
            Numpy array of the requested data
        """
        samples_needed = int(window_sec * self.sample_rate)
        return self.get_latest_data(channel_label, samples_needed)

    def stop_acquisition(self):
        """Stop data acquisition and clear buffers"""
        with self.lock:
            self.active_channels = []
            self.buffer = {}
            self.last_read_positions = {}
            self.timestamps = []

    def get_active_channels(self) -> List[str]:
        """Return list of currently active channel labels"""
        with self.lock:
            return list(self.buffer.keys())
        
    def set_nc3_directory(self, directory: str):
        self.nc3_dir = directory
        