"""
Tests for the Config class.
"""

import pytest
import json
import tempfile
from pathlib import Path
from quantumml_fraud.utils import Config


def test_config_defaults():
    """Test that default configuration is loaded correctly."""
    config = Config()
    
    assert config.get('data.path') == 'data/raw/creditcard.csv'
    assert config.get('data.test_size') == 0.2
    assert config.get('preprocessing.scaler_type') == 'standard'


def test_config_get_nested():
    """Test getting nested configuration values."""
    config = Config()
    
    # Test nested access
    assert config.get('classical_model.type') == 'random_forest'
    assert config.get('quantum_model.n_qubits') == 4
    
    # Test default value
    assert config.get('nonexistent.key', 'default') == 'default'


def test_config_set():
    """Test setting configuration values."""
    config = Config()
    
    config.set('data.path', '/new/path/data.csv')
    assert config.get('data.path') == '/new/path/data.csv'
    
    config.set('new.nested.key', 'value')
    assert config.get('new.nested.key') == 'value'


def test_config_save_load():
    """Test saving and loading configuration from file."""
    config = Config()
    config.set('test.value', 'test')
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config_path = f.name
    
    try:
        # Save configuration
        config.save_to_file(config_path)
        
        # Load configuration in new instance
        config2 = Config(config_path)
        assert config2.get('test.value') == 'test'
        
    finally:
        Path(config_path).unlink()


def test_config_to_dict():
    """Test converting configuration to dictionary."""
    config = Config()
    config_dict = config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert 'data' in config_dict
    assert 'preprocessing' in config_dict
