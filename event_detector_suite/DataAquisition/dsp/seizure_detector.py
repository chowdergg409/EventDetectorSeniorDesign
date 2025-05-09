import numpy as np
from scipy.signal import butter, lfilter
from typing import Tuple, List

class LineLengthSeizureDetector:
    def __init__(self, fs: int = 5000, short_window_sec: float = 2.0, 
                 long_window_sec: float = 30.0, threshold_offset: float = 1.5):
        """
        Initialize the line length seizure detector with processing parameters.
        
        Parameters:
        - fs: int, sampling frequency in Hz (default 250Hz as in paper)
        - short_window_sec: float, short-term window length in seconds (default 2 sec)
        - long_window_sec: float, long-term window length in seconds (default 30 sec)
        - threshold_offset: float, fixed offset above trend for detection (default 1.5)
        """
        if short_window_sec <= 0 or long_window_sec <= 0:
            print("changing short_window_sec to 2.0")
            short_window_sec = 2.0
            long_window_sec = 30.0
            
        if long_window_sec <= short_window_sec:
            raise ValueError("Long window must be longer than short window")
        self.fs = fs
        self.short_window = int(short_window_sec * fs)  # Convert to samples
        self.long_window = int(long_window_sec * fs)    # Convert to samples
        self.threshold_offset = threshold_offset
        
        # Calculate how many short windows fit in the long window
        self.K = int(self.long_window / self.short_window)
        
        # Initialize buffers
        self.signal_buffer = np.zeros(self.long_window)
        self.ll_values = np.zeros(self.K)
        
    def compute_line_length(self, data_window: np.ndarray) -> float:
        """
        Compute the line length feature for a given data window.
        
        Parameters:
        - data_window: 1D array, EEG signal segment
        
        Returns:
        - float, normalized line length value
        """
        # Equation (4) from the paper: LL(n) = sum(abs(diff(x))) / K
        return np.sum(np.abs(np.diff(data_window))) / self.K
    
    def update_trend(self) -> float:
        """
        Update the long-term trend value by averaging line length values.
        
        Returns:
        - float, current trend value (average of line lengths in long window)
        """
        return np.mean(self.ll_values)
    
    def process_sample(self, new_sample: float) -> Tuple[float, float, bool]:
        """
        Process a new EEG sample through the detection system.
        
        Parameters:
        - new_sample: float, new EEG data point
        
        Returns:
        - tuple: (current_line_length, current_trend, detection_flag)
        """
        # Update the signal buffer (FIFO)
        self.signal_buffer = np.roll(self.signal_buffer, -1)
        if self.signal_buffer.size == 0:
            self.signal_buffer = np.array(new_sample)
        self.signal_buffer[-1] = new_sample
        
        # Only compute when we have enough samples
        if len(self.signal_buffer) >= self.short_window:
            # Get the most recent short window of data
            current_window = self.signal_buffer[-self.short_window:]
            # Compute line length for this window
            current_ll = self.compute_line_length(current_window)
            
            # Update line length buffer (FIFO)
            self.ll_values = np.roll(self.ll_values, -1)
            self.ll_values[-1] = current_ll
            
            # Compute trend if we have enough line length values
            if len(self.ll_values) >= self.K:
                current_trend = self.update_trend()
                threshold = current_trend + self.threshold_offset
                
                # Detection logic
                detection = current_ll >= threshold
                
                return current_ll, current_trend, detection
        
        return None, None, False, None
    
    def process_segment(self, eeg_segment: np.ndarray, channel) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Process a segment of EEG data through the detection system.
        
        Parameters:
        - eeg_segment: 1D array, EEG signal to process
        
        Returns:
        - tuple: (line_lengths, trends, detections, channel)
        """
        
        n_samples = len(eeg_segment)
        line_lengths = np.zeros(n_samples)
        trends = np.zeros(n_samples)
        detections = np.zeros(n_samples, dtype=bool)
        medians = np.zeros(n_samples)
        
        for i in range(n_samples):
            ll, trend, detection = self.process_sample(eeg_segment[i])
            line_lengths[i] = ll
            trends[i] = trend
            detections[i] = detection
            median = np.median(trends)
            
        
        return line_lengths, trends, detections,median, channel

    @staticmethod
    def preprocess_data(eeg_data: np.ndarray, fs: int, 
                        lowcut: float = 0.5, highcut: float = 70.0) -> np.ndarray:
        """
        Preprocess EEG data with bandpass filtering (optional).
        
        Parameters:
        - eeg_data: 1D array, raw EEG signal
        - fs: int, sampling frequency
        - lowcut: float, low cutoff frequency in Hz
        - highcut: float, high cutoff frequency in Hz
        
        Returns:
        - 1D array, filtered EEG signal
        """
        # Design bandpass filter
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(4, [low, high], btype='band')
        
        # Apply filter
        filtered_data = lfilter(b, a, eeg_data)
        return filtered_data

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.io import loadmat
    from PyQt5.QtWidgets import QFileDialog, QApplication
    import struct
    from pathlib import Path
    app = QApplication([])
    fs = 5000  # Sampling frequency (Hz)
   
    eeg_data_path, _ = QFileDialog.getOpenFileName(None,"eeg")
    nsx_path, _ = QFileDialog.getOpenFileName(None,"nsx")
    nsx = loadmat(nsx_path)['NSx']
    channel = Path(eeg_data_path).stem
    _, idx = np.nonzero(nsx['label'] == channel[:-4] )
    conversion = float(nsx['conversion'][0,idx][0][0][0])
    try:
        dc = float(nsx['dc'][0,idx][0][0][0])
    except:
        dc = 0
    
    f1 = open(eeg_data_path,'rb')
    binary_data = f1.read()
    samples = fs*60*2
    format_string = '<' + 'h' * (len(binary_data[0:samples]) // 2)
    unpacked_data = np.array(struct.unpack(format_string, binary_data[0:samples]))
    eeg_data = unpacked_data*conversion+dc
    f1.close()

    duration = np.size(eeg_data)/fs
    seizure_intervals = []
    t = np.arange(len(eeg_data)) / fs  # Time in seconds

    # Initialize detector - parameters may need tuning for your data
    detector = LineLengthSeizureDetector(
        fs=fs,
        short_window_sec=2.0,  # Start with 1-2 second window
        long_window_sec=30.0,  # 20-30 second window typical
        threshold_offset=3.0   # Start with 1.5-3.0, adjust based on FP/FN
    )

    # Process entire recording
    line_lengths, trends, detections = detector.process_segment(eeg_data)

    # Create ground truth array (1=seizure, 0=non-seizure)
    ground_truth = np.zeros_like(t, dtype=bool)
    for start, end in seizure_intervals:
        ground_truth[(t >= start) & (t <= end)] = True

    # Calculate performance metrics
    true_positives = np.sum(detections & ground_truth)
    false_positives = np.sum(detections & ~ground_truth)
    false_negatives = np.sum(~detections & ground_truth)

    sensitivity = true_positives / (true_positives + false_negatives)
    false_positive_rate = false_positives / np.sum(~ground_truth)

    print(f"Sensitivity: {sensitivity:.2%}")
    print(f"False positive rate: {false_positive_rate:.4f} per second")
    print(f"Total detections: {np.sum(detections)}")
    print(f"True positives: {true_positives}")
    print(f"False positives: {false_positives}")

    # Plot results (same as before but with actual seizure intervals)
    plt.figure(figsize=(14, 10))

    # Plot EEG data with seizure regions
    plt.subplot(3, 1, 1)
    plt.plot(t, eeg_data)
    plt.title('EEG Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    for start, end in seizure_intervals:
        plt.axvspan(start, end, color='red', alpha=0.2)
    plt.legend(['EEG', 'Clinical Seizures'])

    # Plot features
    plt.subplot(3, 1, 2)
    plt.plot(t, line_lengths, label='Line Length')
    plt.plot(t, trends, label='Trend')
    plt.plot(t, trends + detector.threshold_offset, '--', label='Threshold')
    plt.title('Line Length Features')
    plt.legend()

    # Plot detections vs ground truth
    plt.subplot(3, 1, 3)
    plt.plot(t, detections.astype(float), 'r', label='Detections')
    plt.plot(t, ground_truth.astype(float)*0.9, 'g', alpha=0.5, label='Ground Truth')
    plt.title('Detection Performance')
    plt.yticks([0, 1], ['Normal', 'Seizure'])
    plt.legend()

    plt.tight_layout()
    plt.show()