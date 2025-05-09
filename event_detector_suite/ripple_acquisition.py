import os
import numpy as np
import threading
import time
import traceback
from threading import Lock
from ripple_map_file import RippleElectrodeInfo, RippleMapFile

try:
    import xipppy as xp
except ImportError:
    xp = None

class RippleDataBuffer:
    def __init__(self, config, ripple_map_filename):
        """
        Simplified data buffer for Ripple neural acquisition systems
        
        Args:
            config: Configuration dictionary
            ripple_map_filename: Path to electrode map file
        """
        self._data_lock = Lock()
        self.config = config
        self.ripple_map = RippleMapFile(filename=ripple_map_filename)
        self.th_stop_evt = threading.Event()
        
        # Acquisition parameters
        self.fs_clk = 5000  # Sampling rate (Hz)
        self.buffer_length = 10  # Buffer length in seconds
        self.n_points = self.buffer_length * self.fs_clk  # Total samples in buffer
        self.b_start_acq = False
        self.b_clr_buffer = True
        
        # Data structures
        self.disp_chs = []  # Channels to display
        self.data_buffer = None  # Circular buffer for data
        self.timestamps = []  # List of acquired timestamps
        self.latest_data = None  # Most recent data block
        
        # Threading
        self.buff_thread = None

    def initialize(self):
        """Initialize connection to Ripple hardware"""
        if xp is None:
            raise RuntimeError("xipppy not available - cannot connect to Ripple hardware")
            
        with xp.xipppy_open(use_tcp=True):
            # Get available electrodes
            self.micros = xp.list_elec(fe_type='micro', max_elecs=1024).tolist()
            self.analogs = xp.list_elec(fe_type='analog').tolist()
            
            # Initialize default display channels
            self._init_display_channels()
            
            # Pre-allocate buffer
            self.data_buffer = np.zeros((len(self.disp_chs), self.n_points), 
                                dtype=np.float32)

    def _init_display_channels(self):
        """Initialize default channels to acquire"""
        self.disp_chs = []
        
        # Add microelectrodes
        for ch in self.micros[:self.config['sEEG']['num_chs_to_plot']-1]:
            xp.signal_set(ch, 'raw', True)
            electrode_info = next(e for e in self.ripple_map.electrodes if e.id == ch)
            self.disp_chs.append(electrode_info)
            
        # Add analog channels (e.g., photodiode)
        for ch in self.analogs[:1]:
            electrode_info = next(e for e in self.ripple_map.electrodes if e.id == ch)
            self.disp_chs.append(electrode_info)

    def start_acquisition(self):
        """Start continuous data acquisition"""
        if self.b_start_acq:
            return True
            
        self.b_start_acq = True
        
        # Start acquisition thread
        self.buff_thread = threading.Thread(target=self._acquisition_thread)
        self.buff_thread.start()
        
        return True

    def stop_acquisition(self):
        """Stop continuous data acquisition"""
        self.b_start_acq = False
        if self.buff_thread:
            self.buff_thread.join()
            self.buff_thread = None
        return True

    def _acquisition_thread(self):
        """Main data acquisition thread"""
        try:
            with xp.xipppy_open(use_tcp=True):
                elecs = [int(ch.id) for ch in self.disp_chs]
                req_ts = xp.time()
                sample_idx = 0
                
                while self.b_start_acq:
                    # Read raw data from hardware
                    data, timestamp = xp.cont_raw(npoints=6000, elecs=elecs, 
                                                start_timestamp=req_ts)
                    
                    # Handle timestamp mismatches
                    if timestamp != req_ts:
                        print(f"Timestamp mismatch: requested {req_ts}, got {timestamp}")
                        data, timestamp = xp.cont_raw(npoints=6000, elecs=elecs,
                                                    start_timestamp=xp.time())
                    
                    # Store raw data
                    with self._data_lock:
                        self.latest_data = data
                        self.timestamps.append(timestamp)
                    
                    # Update circular buffer
                    samples_read = int(len(data)/len(self.disp_chs))
                    if sample_idx + samples_read >= self.n_points:
                        samples_to_write = self.n_points - sample_idx
                        sample_idx = 0  # Reset for circular buffer
                    else:
                        samples_to_write = samples_read
                    
                    # Write to buffer
                    for ch_idx in range(len(self.disp_chs)):
                        ch_start = ch_idx * samples_read
                        self.data_buffer[ch_idx, sample_idx:sample_idx+samples_to_write] = \
                            data[ch_start:ch_start+samples_to_write]
                    
                    sample_idx += samples_to_write
                    req_ts = timestamp + samples_read
                    
                    # Small delay to prevent CPU overload
                    time.sleep(0.01)
                    
        except Exception as e:
            print(f"Acquisition error: {traceback.format_exc()}")
            self.b_start_acq = False

    def get_latest_data(self, channel_idx=None, n_samples=None):
        """
        Get latest acquired data
        
        Args:
            channel_idx: Index of channel to retrieve (None for all)
            n_samples: Number of samples to retrieve (None for all)
            
        Returns:
            Requested data array and corresponding timestamps
        """
        if self.latest_data is None:
            return None, None
            
        if channel_idx is not None:
            samples_per_ch = len(self.latest_data)//len(self.disp_chs)
            start_idx = channel_idx * samples_per_ch
            data = self.latest_data[start_idx:start_idx+samples_per_ch]
        else:
            data = self.latest_data
            
        if n_samples is not None:
            data = data[:n_samples]

        with self._data_lock:
            data_copy = self.latest_data.__copy__() if self.latest_data is not None else None
            ts_copy = self.timestamps[-1] if self.timestamps else None
            
        return data_copy, ts_copy
    
    def get_data_window(self, channel_idx=0, window_sec=1.0):
        """Get a time window of data for a specific channel"""
        with self._data_lock:
            if self.data_buffer is None:
                return None

            n_samples = int(window_sec * self.fs_clk)
            start_idx = max(0, self.write_idx - n_samples)

            # Handle circular buffer wrapping
            if start_idx < self.write_idx:
                return self.data_buffer[channel_idx, start_idx:self.write_idx]
            else:
                return np.concatenate((
                    self.data_buffer[channel_idx, start_idx:],
                    self.data_buffer[channel_idx, :self.write_idx]
                ))

    def get_all_channels(self, n_samples=None):
        """Get data from all channels"""
        with self._data_lock:
            if n_samples is None:
                return self.data_buffer.copy()
            return self.data_buffer[:, -n_samples:].copy()

    def clear_buffer(self):
        """Clear the data buffer"""
        if self.data_buffer is not None:
            self.data_buffer.fill(0)
        self.timestamps = []
        self.latest_data = None

    def register_data_callback(self, callback_func, interval_ms=100):
        """Register a function to be called with new data periodically"""
        self.callback = callback_func
        self.callback_timer = threading.Timer(interval_ms/1000, self._invoke_callback)
        self.callback_timer.start()

    def _invoke_callback(self, interval_ms =100):
        if self.callback and self.b_start_acq:
            data, timestamp = self.get_latest_data()
            self.callback(data, timestamp)
        if self.b_start_acq:  # Continue only if still acquiring
            self.callback_timer = threading.Timer(interval_ms/1000, self._invoke_callback)
            self.callback_timer.start()

    def __iter__(self):
        """Enable iterator protocol for streaming access"""
        while self.b_start_acq:
            data, ts = self.get_latest_data()
            if data is not None:
                yield data, ts
            time.sleep(0.001)  # Prevent busy waiting

    def get_contiguous_blocks(self, channel_idx):
        """Yield contiguous memory blocks for efficient processing"""
        with self._data_lock:
            if self.write_idx == 0:
                yield self.data_buffer[channel_idx]
            else:
                yield self.data_buffer[channel_idx, :self.write_idx]
                yield self.data_buffer[channel_idx, self.write_idx:]