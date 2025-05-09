import numpy as np
from scipy import signal, stats
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import datetime

class CSHFOdetector:
    def __init__(self, sampling_rate=5000, and_threshold=0.0, or_threshold=0.2):
        """
        Initialize the CS HFO detector.
        
        Parameters:
        - sampling_rate: Sampling rate of the EEG data in Hz (default: 5000)
        - and_threshold: AND threshold for detection (default: 0.0)
        - or_threshold: OR threshold for detection (default: 0.2)
        """
        self.sampling_rate = sampling_rate
        self.and_threshold = and_threshold
        self.or_threshold = or_threshold
        
        # Define the 4 overlapping frequency bands (in Hz)
        self.frequency_bands = [
            (44, 120),   # Roughly gamma band
            (73, 197),   # Roughly ripple band
            (120, 326),  # Fast ripple band
            (197, 537)   # Fast ripple band
        ]
        
        # Minimum number of cycles for HFO detection
        self.min_cycles = 4
        
        # Parameters for Poisson normalization window
        self.norm_window_size = 10 * sampling_rate  # 10 second window
        self.norm_overlap = 1 * sampling_rate       # 1 second overlap
        
        # Store gamma distribution parameters for each metric
        self.gamma_params = {
            'amplitude': {'k': 1.37, 'theta': 1.73},
            'frequency_dominance': {'k': 1.26, 'theta': 0.46},
            'product': {'k': 1.28, 'theta': 34.4},
            'cycles': {'k': 1.76, 'theta': 4.04},
            'combination': {'k': 2.24, 'theta': 0.53}
        }
    
    def _butter_bandpass(self, lowcut, highcut, fs, order=3):
        """Design Butterworth bandpass filter."""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def _butter_lowpass(self, highcut, fs, order=3):
        """Design Butterworth lowpass filter."""
        nyq = 0.5 * fs
        high = highcut / nyq
        b, a = butter(order, high, btype='low')
        return b, a
    
    def _butter_highpass(self, lowcut, fs, order=3):
        """Design Butterworth highpass filter."""
        nyq = 0.5 * fs
        low = lowcut / nyq
        b, a = butter(order, low, btype='high')
        return b, a
    
    def _bandpass_filter(self, data, lowcut, highcut):
        """Apply bandpass filter to data."""
        b, a = self._butter_bandpass(lowcut, highcut, self.sampling_rate)
        return filtfilt(b, a, data)
    
    def _lowpass_filter(self, data, highcut):
        """Apply lowpass filter to data."""
        b, a = self._butter_lowpass(highcut, self.sampling_rate)
        return filtfilt(b, a, data)
    
    def _highpass_filter(self, data, lowcut):
        """Apply highpass filter to data."""
        b, a = self._butter_highpass(lowcut, self.sampling_rate)
        return filtfilt(b, a, data)
    
    def _poisson_normalize(self, data):
        """Apply Poisson normalization to data."""
        # Calculate mean for normalization
        mean_val = np.mean(data)
        # Poisson normalize
        normalized = (data - mean_val) / mean_val
        return normalized
    
    def _sliding_window_max(self, data, window_size):
        """Apply sliding window maximum filter."""
        # Pad the data to handle edges
        padded = np.pad(data, (window_size//2, window_size//2), mode='edge')
        # Apply maximum filter
        result = np.zeros_like(data)
        for i in range(len(data)):
            result[i] = np.max(padded[i:i+window_size])
        return result
    
    def _find_critical_points(self, data):
        """Find critical points (peaks and troughs) in the data."""
        # Find peaks and troughs
        peaks, _ = signal.find_peaks(data)
        troughs, _ = signal.find_peaks(-data)
        
        # Combine and sort
        critical_points = np.sort(np.concatenate((peaks, troughs)))
        
        # Ensure first and last points are included
        if critical_points[0] != 0:
            critical_points = np.insert(critical_points, 0, 0)
        if critical_points[-1] != len(data)-1:
            critical_points = np.append(critical_points, len(data)-1)
        
        return critical_points
    
    def _create_envelope(self, data):
        """Create envelope of the signal using critical points."""
        critical_points = self._find_critical_points(data)
        
        # Get absolute values at critical points
        abs_values = np.abs(data[critical_points])
        
        # Create interpolation function
        interp_func = interp1d(critical_points, abs_values, kind='linear', 
                              fill_value='extrapolate')
        
        # Interpolate to original length
        envelope = interp_func(np.arange(len(data)))
        
        return envelope
    
    def _compute_amplitude_trace(self, bandpass_data):
        """Compute amplitude trace for a frequency band."""
        # Step 1: Create envelope
        envelope = self._create_envelope(bandpass_data)
        
        # Step 2: Remove sharp peaks with sliding window max
        center_freq = np.sqrt(self.frequency_bands[0][0] * self.frequency_bands[0][1])  # Geometric mean
        window_size = int((self.min_cycles / center_freq) * self.sampling_rate)
        smoothed_envelope = self._sliding_window_max(envelope, window_size)
        
        # Step 3: Poisson normalize
        amplitude_trace = self._poisson_normalize(smoothed_envelope)
        
        return amplitude_trace
    
    def _compute_frequency_dominance_trace(self, raw_data, bandpass_data, lowcut, highcut):
        """Compute frequency dominance trace for a frequency band."""
        # Step 1: Create lowpass filtered version
        lowpass_data = self._lowpass_filter(raw_data, highcut)
        
        # Step 2: Differentiate both signals
        diff_bandpass = np.diff(bandpass_data, prepend=0)
        diff_lowpass = np.diff(lowpass_data, prepend=0)
        
        # Step 3: Clip values to [-1, 1]
        diff_bandpass = np.clip(diff_bandpass, -1, 1)
        diff_lowpass = np.clip(diff_lowpass, -1, 1)
        
        # Step 4: Create Local Oscillation Traces (LOT)
        lot_bandpass = np.cumsum(diff_bandpass)
        lot_lowpass = np.cumsum(diff_lowpass)
        
        # Step 5: Highpass filter the LOTs
        lot_bandpass_hp = self._highpass_filter(lot_bandpass, lowcut)
        lot_lowpass_hp = self._highpass_filter(lot_lowpass, lowcut)
        
        # Step 6: Calculate RMS signal-to-noise ratio
        center_freq = np.sqrt(lowcut * highcut)  # Geometric mean
        window_size = int((self.min_cycles / center_freq) * self.sampling_rate)
        
        # Calculate signal (RMS of bandpass LOT)
        signal_power = np.zeros_like(lot_bandpass_hp)
        for i in range(len(lot_bandpass_hp)):
            start = max(0, i - window_size//2)
            end = min(len(lot_bandpass_hp), i + window_size//2)
            signal_power[i] = np.sqrt(np.mean(lot_bandpass_hp[start:end]**2))
        
        # Calculate noise (RMS of difference between bandpass and lowpass LOTs)
        noise_power = np.zeros_like(lot_bandpass_hp)
        for i in range(len(lot_bandpass_hp)):
            start = max(0, i - window_size//2)
            end = min(len(lot_bandpass_hp), i + window_size//2)
            diff = lot_bandpass_hp[start:end] - lot_lowpass_hp[start:end]
            noise_power[i] = np.sqrt(np.mean(diff**2))
        
        # Avoid division by zero
        noise_power[noise_power == 0] = np.finfo(float).eps
        
        # Calculate SNR
        raw_fd_trace = signal_power / noise_power
        
        # Step 7: Apply sliding window max and Poisson normalize
        smoothed_fd_trace = self._sliding_window_max(raw_fd_trace, window_size)
        fd_trace = self._poisson_normalize(smoothed_fd_trace)
        
        return fd_trace
    
    def _compute_product_trace(self, amplitude_trace, frequency_dominance_trace):
        """Compute product trace from amplitude and frequency dominance traces."""
        # Clip negative values to 0
        amp_clipped = np.clip(amplitude_trace, 0, None)
        fd_clipped = np.clip(frequency_dominance_trace, 0, None)
        
        # Compute dot product
        product_trace = amp_clipped * fd_clipped
        
        # Poisson normalize
        normalized_product = self._poisson_normalize(product_trace)
        
        return normalized_product
    
    def _count_cycles(self, bandpass_data, detection_start, detection_end):
        """Count the number of cycles in a detection window."""
        # Extract the segment of interest
        segment = bandpass_data[detection_start:detection_end]
        
        # Find zero crossings
        zero_crossings = np.where(np.diff(np.sign(segment)))[0]
        
        # Count complete cycles (each zero crossing is half cycle)
        num_cycles = len(zero_crossings) / 2
        
        return num_cycles
    
    def _gamma_cdf(self, x, k, theta):
        """Gamma cumulative distribution function."""
        return stats.gamma.cdf(x, a=k, scale=theta)
    
    def _evaluate_detection(self, band_idx, amplitude, frequency_dominance, product, cycles):
        """Evaluate a detection using the cascade of thresholds."""
        # Get gamma parameters for this band
        params = self.gamma_params
        
        # Calculate CDF values for each metric
        amp_cdf = self._gamma_cdf(amplitude, 
                                 params['amplitude']['k'], 
                                 params['amplitude']['theta'])
        
        fd_cdf = self._gamma_cdf(frequency_dominance, 
                                params['frequency_dominance']['k'], 
                                params['frequency_dominance']['theta'])
        
        prod_cdf = self._gamma_cdf(product, 
                                  params['product']['k'], 
                                  params['product']['theta'])
        
        cycles_cdf = self._gamma_cdf(cycles, 
                                    params['cycles']['k'], 
                                    params['cycles']['theta'])
        
        # Apply AND threshold - all metrics must exceed their thresholds
        and_thresholds = {
            'amplitude': self._gamma_cdf(self.and_threshold, 
                                        params['amplitude']['k'], 
                                        params['amplitude']['theta']),
            'frequency_dominance': self._gamma_cdf(self.and_threshold, 
                                                  params['frequency_dominance']['k'], 
                                                  params['frequency_dominance']['theta']),
            'product': self._gamma_cdf(self.and_threshold, 
                                     params['product']['k'], 
                                     params['product']['theta']),
            'cycles': self._gamma_cdf(self.and_threshold, 
                                    params['cycles']['k'], 
                                    params['cycles']['theta'])
        }
        
        if (amp_cdf < and_thresholds['amplitude'] or 
            fd_cdf < and_thresholds['frequency_dominance'] or 
            prod_cdf < and_thresholds['product'] or 
            cycles_cdf < and_thresholds['cycles']):
            return False
        
        # Apply OR threshold - combination score must exceed threshold
        combination_score = amp_cdf + fd_cdf + prod_cdf + cycles_cdf
        combination_threshold = self._gamma_cdf(self.or_threshold, 
                                              params['combination']['k'], 
                                              params['combination']['theta']) * 4
        
        if combination_score < combination_threshold:
            return False
        
        return True
    
    def detect_hfos(self, raw_eeg, channel_label="Unkown"):
        """
        Detect HFOs in raw EEG data.
        
        Parameters:
        - raw_eeg: 1D numpy array of raw EEG data
        
        Returns:
        - List of detected HFOs, each represented as a dictionary with:
          - 'start': start sample index
          - 'end': end sample index
          - 'band': frequency band index
          - 'amplitude': amplitude metric
          - 'frequency_dominance': frequency dominance metric
          - 'product': product metric
          - 'cycles': number of cycles
          - 'channel': channel label
        """
        # Initialize list to store detections
        all_detections = []
        
        # Process each frequency band
        for band_idx, (lowcut, highcut) in enumerate(self.frequency_bands):
            # Step 1: Bandpass filter the raw data
            bandpass_data = self._bandpass_filter(raw_eeg, lowcut, highcut)
            
            # Step 2: Compute amplitude trace
            amplitude_trace = self._compute_amplitude_trace(bandpass_data)
            
            # Step 3: Compute frequency dominance trace
            fd_trace = self._compute_frequency_dominance_trace(raw_eeg, bandpass_data, lowcut, highcut)
            
            # Step 4: Compute product trace
            product_trace = self._compute_product_trace(amplitude_trace, fd_trace)
            
            # Step 5: Apply edge threshold (default = 1)
            edge_threshold = 1
            above_threshold = product_trace > edge_threshold
            
            # Find contiguous regions above threshold
            diff = np.diff(above_threshold.astype(int))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0] + 1
            timestamp = datetime.datetime.now()
            
            # Handle edge cases
            if above_threshold[0]:
                starts = np.insert(starts, 0, 0)
            if above_threshold[-1]:
                ends = np.append(ends, len(above_threshold))
            
            # Step 6: Merge detections that are too close
            min_gap = int((self.min_cycles / np.sqrt(lowcut * highcut)) * self.sampling_rate)
            merged_starts = []
            merged_ends = []
            
            if len(starts) > 0:
                current_start = starts[0]
                current_end = ends[0]
                
                for i in range(1, len(starts)):
                    if starts[i] - current_end < min_gap:
                        current_end = ends[i]
                    else:
                        merged_starts.append(current_start)
                        merged_ends.append(current_end)
                        current_start = starts[i]
                        current_end = ends[i]
                
                merged_starts.append(current_start)
                merged_ends.append(current_end)
            
            # Step 7: Evaluate each detection with cascade of thresholds
            for start, end in zip(merged_starts, merged_ends):
                # Calculate metrics for this detection
                amplitude = np.mean(amplitude_trace[start:end])
                frequency_dominance = np.mean(fd_trace[start:end])
                product = np.mean(product_trace[start:end])
                cycles = self._count_cycles(bandpass_data, start, end)
                
                # Evaluate detection
                if self._evaluate_detection(band_idx, amplitude, frequency_dominance, product, cycles):
                    detection = {
                        'start': start,
                        'end': end,
                        'band': band_idx,
                        'amplitude': amplitude,
                        'frequency_dominance': frequency_dominance,
                        'product': product,
                        'cycles': cycles,
                        'channel': channel_label
                    }
                    all_detections.append(detection)
        
        # Step 8: Create conglomerate detections (merge overlapping detections across bands)
        conglomerate_detections = []
        if all_detections:
            # Sort detections by start time
            sorted_detections = sorted(all_detections, key=lambda x: x['start'])
            
            current_start = sorted_detections[0]['start']
            current_end = sorted_detections[0]['end']
            bands_involved = [sorted_detections[0]['band']]
            
            for detection in sorted_detections[1:]:
                if detection['start'] <= current_end:  # Overlapping
                    current_end = max(current_end, detection['end'])
                    bands_involved.append(detection['band'])
                else:
                    conglomerate_detections.append({
                        'start': current_start,
                        'end': current_end,
                        'bands': list(set(bands_involved)),# Unique bands
                        'amplitude': amplitude,  
                        'channel': channel_label,
                        'timestamp': timestamp	
                    })
                    current_start = detection['start']
                    current_end = detection['end']
                    bands_involved = [detection['band']]
            
            # Add the last detection
            conglomerate_detections.append({
                'start': current_start,
                'end': current_end,
                'bands': list(set(bands_involved)),
                'amplitude': amplitude,
                'channel': channel_label,
                'timestamp': timestamp
            })
        
        return conglomerate_detections

if __name__ == "__main__":
    # Example usage
    import numpy as np
    import matplotlib.pyplot as plt

    # Generate some synthetic EEG data for testing
    sampling_rate = 5000  # 5 kHz
    duration = 10  # seconds
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # Create a synthetic signal with some HFO-like bursts
    raw_eeg = np.random.normal(0, 1, len(t))  # Background noise

    # Add some HFO-like bursts
    hfo1_start = 1.0  # seconds
    hfo1_duration = 0.05  # seconds
    hfo1_freq = 100  # Hz
    hfo1_samples = int(hfo1_duration * sampling_rate)
    hfo1_idx = int(hfo1_start * sampling_rate)
    raw_eeg[hfo1_idx:hfo1_idx+hfo1_samples] += 0.5 * np.sin(2 * np.pi * hfo1_freq * 
                                                            t[hfo1_idx:hfo1_idx+hfo1_samples])

    hfo2_start = 3.0  # seconds
    hfo2_duration = 0.08  # seconds
    hfo2_freq = 200  # Hz
    hfo2_samples = int(hfo2_duration * sampling_rate)
    hfo2_idx = int(hfo2_start * sampling_rate)
    raw_eeg[hfo2_idx:hfo2_idx+hfo2_samples] += 0.7 * np.sin(2 * np.pi * hfo2_freq * 
                                                            t[hfo2_idx:hfo2_idx+hfo2_samples])

    # Create and run the detector
    detector = CSHFOdetector(sampling_rate=sampling_rate)
    detections = detector.detect_hfos(raw_eeg)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(t, raw_eeg, label='EEG')
    for detection in detections:
        start_time = detection['start'] / sampling_rate
        end_time = detection['end'] / sampling_rate
        plt.axvspan(start_time, end_time, color='red', alpha=0.3, 
                    label='HFO' if detections.index(detection) == 0 else None)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('EEG with Detected HFOs')
    plt.legend()
    plt.show()

    print(f"Detected {len(detections)} HFO events:")
    for i, detection in enumerate(detections):
        print(f"HFO {i+1}: {detection['start']/sampling_rate:.3f}s to {detection['end']/sampling_rate:.3f}s")