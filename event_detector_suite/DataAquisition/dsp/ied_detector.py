import numpy as np
from scipy.signal import butter, filtfilt
from typing import List, Tuple, Dict

class IEDDetector:
    def __init__(self, fs: int = 5000, 
                 block_size_sec: int = 60,
                 spike_band: Tuple[float, float] = (20, 50),
                 display_band: Tuple[float, float] = (1, 35),
                 artifact_slope_thresh: float = 10.0,
                 spike_amplitude_thresh: float = 600.0,
                 spike_slope_thresh: float = 7.0,
                 spike_duration_thresh: float = 10.0,
                 target_median_amp: float = 70.0):
        """
        Initialize the IED detector with processing parameters.
        
        Parameters match those described in the paper:
        - fs: sampling frequency (200Hz in paper)
        - block_size_sec: processing block size in seconds (60s in paper)
        - spike_band: bandpass filter for initial spike detection (20-50Hz in paper)
        - display_band: bandpass filter for display/scaling (1-35Hz in paper)
        - artifact_slope_thresh: slope threshold for artifact rejection (10 SD in paper)
        - spike_amplitude_thresh: amplitude threshold (600µV in paper)
        - spike_slope_thresh: slope threshold (7µV/ms in paper)
        - spike_duration_thresh: duration threshold (10ms in paper)
        - target_median_amp: target median amplitude for scaling (70µV in paper)
        """
        self.fs = fs
        self.block_size = block_size_sec * fs
        self.spike_band = spike_band
        self.display_band = display_band
        self.artifact_slope_thresh = artifact_slope_thresh
        self.spike_amp_thresh = spike_amplitude_thresh
        self.spike_slope_thresh = spike_slope_thresh
        self.spike_dur_thresh = spike_duration_thresh
        self.target_median_amp = target_median_amp
        
        # Initialize filters
        self.spike_b, self.spike_a = self._butter_bandpass(*spike_band, fs)
        self.display_b, self.display_a = self._butter_bandpass(*display_band, fs)
        
    def _butter_bandpass(self, lowcut: float, highcut: float, fs: int, order: int = 2):
        """Design Butterworth bandpass filter"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def _bandpass_filter(self, data: np.ndarray, filter_type: str = 'spike') -> np.ndarray:
        """Apply bandpass filter to data"""
        if filter_type == 'spike':
            return filtfilt(self.spike_b, self.spike_a, data)
        else:
            return filtfilt(self.display_b, self.display_a, data)
    
    def _remove_artifact_channels(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove artifact channels based on slope threshold.
        Returns filtered data and artifact mask.
        """
        # Calculate average slope for each channel
        slopes = np.mean(np.abs(np.diff(data, axis=0)), axis=0)
        
        # Calculate threshold (mean + N*SD)
        mean_slope = np.mean(slopes)
        std_slope = np.std(slopes)
        threshold = mean_slope + self.artifact_slope_thresh * std_slope
        
        # Create mask for non-artifact channels
        good_channels = slopes < threshold
        return data[:, good_channels], good_channels
    
    def _block_scale_data(self, data: np.ndarray) -> np.ndarray:
        """
        Scale all channels together to bring median amplitude to target.
        """
        # Calculate average rectified amplitude per channel
        avg_amps = np.mean(np.abs(data), axis=0)
        
        # Calculate scaling factor based on median
        median_amp = np.median(avg_amps)
        if median_amp == 0:
            return data  # Avoid division by zero
        
        scaling_factor = self.target_median_amp / median_amp
        return data * scaling_factor
    
    def _detect_spikes(self, raw_data: np.ndarray, filtered_data: np.ndarray) -> List[Dict]:
        """
        Detect spikes based on amplitude, slope, and duration criteria.
        Returns list of spike events with metadata.
        """
        spikes = []
        n_channels = raw_data.shape[1]
        
        for ch in range(n_channels):
            # Find local maxima in filtered data (potential spikes)
            diff_signal = np.diff(filtered_data[:, ch])
            peaks = np.where((diff_signal[:-1] > 0) & (diff_signal[1:] < 0))[0] + 1
            
            for peak in peaks:
                # Analyze half-waves around peak
                left_half = self._analyze_halfwave(raw_data[:, ch], peak, -1)  # Left half-wave
                right_half = self._analyze_halfwave(raw_data[:, ch], peak, 1)  # Right half-wave
                
                # Check if both half-waves meet criteria
                if (left_half['valid'] and right_half['valid'] and
                    (left_half['amp'] + right_half['amp']) > self.spike_amp_thresh):
                    spikes.append({
                        'channel': ch,
                        'sample': peak,
                        'time': peak / self.fs,
                        'amplitude': left_half['amp'] + right_half['amp'],
                        'left_slope': left_half['slope'],
                        'right_slope': right_half['slope'],
                        'left_duration': left_half['duration'],
                        'right_duration': right_half['duration']
                    })
        
        return spikes
    
    def _analyze_halfwave(self, signal: np.ndarray, peak: int, direction: int) -> Dict:
        """
        Analyze a half-wave (left or right of peak).
        Returns dictionary with amplitude, slope, duration, and validity.
        """
        n_samples = len(signal)
        start = peak
        end = peak
        
        # Find start/end of half-wave
        if direction == -1:  # Left half-wave
            while start > 0 and (signal[start-1] - signal[start]) * direction > 0:
                start -= 1
        else:  # Right half-wave
            while end < n_samples-1 and (signal[end+1] - signal[end]) * direction > 0:
                end += 1
        
        # Calculate characteristics
        amp = abs(signal[peak] - signal[start if direction == -1 else end])
        duration = (peak - start if direction == -1 else end - peak) * 1000 / self.fs  # in ms
        slope = amp / duration if duration > 0 else 0  # µV/ms
        
        # Check against thresholds
        valid = (duration > self.spike_dur_thresh and 
                slope > self.spike_slope_thresh)
        
        return {
            'amp': amp,
            'slope': slope,
            'duration': duration,
            'valid': valid
        }
    
    def process_block(self, data: np.ndarray) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
        """
        Process one block of multi-channel EEG data through the full pipeline.
        
        Parameters:
        - data: 2D array (samples × channels) of raw EEG data
        
        Returns:
        - good_channels: mask of non-artifact channels
        - spikes: list of detected spike events
        - scaled_data: block-scaled data for visualization
        """
        # Step 1: Remove artifact channels
        clean_data, good_channels = self._remove_artifact_channels(data)
        if clean_data.size == 0:
            return good_channels, None, None
        else:
            # Step 2: Bandpass filter (20-50Hz) for spike detection
            spike_filtered = self._bandpass_filter(clean_data, 'spike')

            # Step 3: Bandpass filter (1-35Hz) and block scaling
            display_filtered = self._bandpass_filter(clean_data, 'display')
            scaled_data = self._block_scale_data(display_filtered)

            # Step 4: Detect spikes using scaled data
            spikes = self._detect_spikes(scaled_data, spike_filtered)

            return good_channels, spikes, scaled_data
    
    def process_recording(self, eeg_data: np.ndarray) -> List[Dict]:
        """
        Process entire recording by breaking into blocks.
        
        Parameters:
        - eeg_data: 2D array (samples × channels) of raw EEG data
        
        Returns:
        - all_spikes: list of all detected spikes with metadata
        """
        try:
            n_samples, n_channels = eeg_data.shape
        except:
            n_samples = eeg_data.shape[0]
        all_spikes = []
        
        for start in range(0, n_samples, self.block_size):
            end = min(start + self.block_size, n_samples)
            block = eeg_data[start:end]
            
            _, spikes, _ = self.process_block(block)
            
            # Adjust spike timestamps for block position
            if spikes == None:
                return None
            else:
                for spike in spikes:
                    spike['sample'] += start
                    spike['time'] = spike['sample'] / self.fs

                all_spikes.extend(spikes)
        
                return all_spikes

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Simulate some multi-channel EEG data for demonstration
    fs = 200  # Sampling rate (Hz)
    duration = 120  # Seconds
    n_channels = 8
    t = np.arange(0, duration, 1/fs)
    
    # Create background activity
    eeg_data = np.zeros((len(t), n_channels))
    for ch in range(n_channels):
        # Vary background activity by channel
        eeg_data[:, ch] = (
            50 * np.sin(2 * np.pi * 1 * t) +  # Delta
            20 * np.sin(2 * np.pi * 10 * t) +  # Alpha
            10 * np.sin(2 * np.pi * 20 * t) +  # Beta
            5 * np.random.normal(size=len(t))  # Noise
        ) * (0.5 + 0.5 * ch/n_channels)  # Scale by channel
        
    # Add some spikes to channels 2 and 5
    spike_times = [10.2, 25.7, 45.1, 63.2, 88.5, 105.3]
    for st in spike_times:
        idx = int(st * fs)
        eeg_data[idx-5:idx+5, 2] += 300 * np.exp(-0.5 * ((np.arange(-5,5)/2)**2))
        eeg_data[idx-3:idx+3, 5] += 400 * np.exp(-0.5 * ((np.arange(-3,3)/1.5)**2))
    
    # Add an artifact to channel 7
    eeg_data[int(50*fs):int(51*fs), 7] += 1000 * np.random.normal(size=fs)
    
    # Initialize and run detector
    detector = IEDDetector(fs=fs)
    spikes = detector.process_recording(eeg_data)
    
    # Print summary
    print(f"Detected {len(spikes)} spikes")
    for spike in spikes[:5]:  # Print first 5 spikes
        print(f"Channel {spike['channel']} at {spike['time']:.2f}s "
              f"(amp: {spike['amplitude']:.1f}µV)")
    
    # Plot example channel with detections
    ch_to_plot = 2
    ch_spikes = [s for s in spikes if s['channel'] == ch_to_plot]
    
    plt.figure(figsize=(12, 6))
    plt.plot(t, eeg_data[:, ch_to_plot], label='EEG')
    for spike in ch_spikes:
        plt.axvline(spike['time'], color='r', alpha=0.3, label='Spike' if spike == ch_spikes[0] else None)
    plt.title(f'Channel {ch_to_plot} with detected spikes')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.legend()
    plt.show()