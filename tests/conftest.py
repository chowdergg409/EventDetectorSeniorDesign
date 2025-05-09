# Standard library imports first
import sys
import os
import logging
import multiprocessing as mp
from unittest.mock import MagicMock

# Third-party imports
import pytest
import numpy as np
from PyQt5.QtWidgets import QApplication

# Local application imports
from event_detector_suite.main import EDS, DataProcessor

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add module directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../event_detector_suite')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

@pytest.fixture(scope="session")
def qapp():
    """Session-wide QApplication fixture for GUI tests"""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    yield app
    app.quit()

@pytest.fixture
def mock_hardware(mocker):
    """Mock hardware dependencies for offline testing"""
    # Mock RippleDataBuffer
    mock_buffer = MagicMock()
    mocker.patch('DataAquisition.ripple_acquisition.RippleDataBuffer', return_value=mock_buffer)
    
    # Mock NC3DataBuffer
    mock_nc3 = MagicMock()
    mocker.patch('DataAquisition.offline_buffer.NC3DataBuffer', return_value=mock_nc3)
    
    return mock_buffer, mock_nc3

@pytest.fixture
def detector_config():
    """Provide standard detector configuration"""
    return {
        'hfo': {'fs': 2000, 'threshold': 3.0},
        'ied': {'fs': 1000, 'window_size': 0.1},
        'seizure': {
            'fs': 500,
            'short_window_sec': 2.0,
            'long_window_sec': 30.0,
            'threshold_offset': 3.0
        }
    }

@pytest.fixture
def test_data_generator():
    """Generate test data with controlled properties"""
    def _generate_data(length=1000, fs=1000, channels=8):
        time = np.arange(length) / fs
        data = np.random.randn(channels, length) * 100  # μV scale
        # Add artificial events
        data[:, 500:550] += 500  # Spike
        data[:, 700:730] += 300  # Oscillation
        return time, data
    return _generate_data

@pytest.fixture(scope="module")
def shared_temp_dir(tmp_path_factory):
    """Shared temporary directory for file-based tests"""
    return tmp_path_factory.mktemp("shared_data")

@pytest.fixture(autouse=True)
def cleanup_processes():
    """Clean up any remaining multiprocessing processes after each test"""
    yield
    # Terminate any remaining processes
    for process in mp.active_children():
        process.terminate()
        process.join()

@pytest.fixture
def pylint_checker(request):
    """Pylint checker fixture for programmatic linting"""
    from pylint.lint import Run
    from pylint.reporters.text import TextReporter
    
    class WritableObject:
        def __init__(self):
            self.content = []
        
        def write(self, text):
            self.content.append(text)
        
        def read(self):
            return "".join(self.content)
    
    def _run_pylint(modules):
        writable = WritableObject()
        Run(
            modules + ['--disable=all', '--enable=E,F'],
            reporter=TextReporter(writable),
            exit=False
        )
        return writable.read()
    
    return _run_pylint

@pytest.fixture(autouse=True)
def add_np_errors(doctest_namespace):
    """Add numpy error constants to doctest namespace"""
    doctest_namespace['np'] = np
    doctest_namespace['NP_ERR_SETTINGS'] = np.seterr(all='warn')
    np.seterr(**doctest_namespace['NP_ERR_SETTINGS'])

@pytest.fixture
def capture_warnings():
    """Fixture to capture and inspect warnings"""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        yield w

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "skip_scipy: mark test as requiring scipy workaround"
    )

@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    for item in items:
        if "scipy" in item.nodeid:
            item.add_marker(pytest.mark.skip_scipy)