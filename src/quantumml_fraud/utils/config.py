"""
Configuration management for the fraud detection project.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """
    Configuration manager for the fraud detection project.
    
    Handles loading and accessing configuration parameters
    from JSON files or dictionaries.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration.
        
        Args:
            config_path: Path to a JSON configuration file
        """
        self.config = {}
        
        if config_path:
            self.load_from_file(config_path)
        else:
            self._set_defaults()
    
    def _set_defaults(self):
        """Set default configuration values."""
        self.config = {
            'data': {
                'path': 'data/raw/creditcard.csv',
                'test_size': 0.2,
                'random_state': 42
            },
            'preprocessing': {
                'scaler_type': 'standard',
                'handle_imbalance': True,
                'imbalance_method': 'smote'
            },
            'classical_model': {
                'type': 'random_forest',
                'n_estimators': 100,
                'max_depth': None,
                'random_state': 42
            },
            'quantum_model': {
                'n_qubits': 4,
                'backend': 'qiskit',
                'circuit_type': 'variational',
                'epochs': 100,
                'learning_rate': 0.01
            },
            'evaluation': {
                'false_positive_cost': 1.0,
                'false_negative_cost': 10.0
            }
        }
    
    def load_from_file(self, config_path: str):
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the JSON configuration file
        """
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            self.config = json.load(f)
    
    def save_to_file(self, config_path: str):
        """
        Save configuration to a JSON file.
        
        Args:
            config_path: Path where to save the configuration
        """
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Supports nested keys using dot notation (e.g., 'data.path')
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set a configuration value.
        
        Supports nested keys using dot notation (e.g., 'data.path')
        
        Args:
            key: Configuration key
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Return the configuration as a dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
