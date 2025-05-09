"""Test suite for dynamic behavior of the event detector system."""
# Standard library imports
import sys
import os

# Third-party imports
import pytest
import numpy as np
from PyQt5.QtWidgets import QApplication

# Local application imports
from event_detector_suite.main import EDS, DataProcessor
from event_detector_suite.DataAquisition.dsp.seizure_detector import LineLengthSeizureDetector
from event_detector_suite.DataAquisition.dsp.hfo_detector import CSHFOdetector
from event_detector_suite.DataAquisition.dsp.ied_detector import IEDDetector

# Fixture to handle QApplication
@pytest.fixture(scope="session")
def qapp():
    app = QApplication([])
    yield app
    app.quit()

# Test division safety in detectors

# Test data processor termination
def test_data_processor_termination(qapp):
    processor = DataProcessor(
        offline_mode=False,
        buffer=None,
        offline_buffer=None,
        window_sec=0.1
    )
    
    processor.start()
    processor.stop()
    
    assert not processor.isRunning()


# Test detector process startup
def test_detector_process_initialization():
    from event_detector_suite.main import run_hfo_detector
    import multiprocessing as mp
    
    data_queue = mp.Queue()
    results_queue = mp.Queue()
    
    process = mp.Process(
        target=run_hfo_detector,
        args=(data_queue, results_queue, 1000)
    )
    process.start()
    assert process.is_alive()
    
    process.terminate()
    process.join()

def test_division_handling():
    """Verify detectors handle division operations safely without crashing."""
    test_data = np.random.randn(2000)

    # This should not raise an error
    detector = LineLengthSeizureDetector(
        fs=1000,
        short_window_sec=2.0,
        long_window_sec=0.0,  # Force potential divide-by-zero
        threshold_offset=3.0
    )
    detector.process_segment(test_data, 0)

    hfo_detector = CSHFOdetector()
    hfo_detector.detect_hfos(test_data, "test")

    ied_detector = IEDDetector()
    ied_detector.process_recording(test_data)