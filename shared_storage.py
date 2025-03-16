import copy
import json
import pathlib

import ray
import torch


@ray.remote
class SharedStorage:
    """
    Class which run in a dedicated thread to store the network weights and some information.
    """

    def __init__(self, checkpoint, config):
        self.config = config
        self.current_checkpoint = copy.deepcopy(checkpoint)

    def save_checkpoint(self, path=None):
        if not path:
            path = self.config.results_path / "model.checkpoint"

        torch.save(self.current_checkpoint, path)
        
        # Save config as a JSON file alongside the checkpoint
        config_path = path.parent / "model.config.json"
        self.save_config(config_path)

    def save_config(self, path=None):
        """
        Save the configuration as a JSON file.
        
        Args:
            path: Path where to save the configuration. If None, uses results_path/model.config.json.
        """
        if not path:
            path = self.config.results_path / "model.config.json"
            
        # Convert config to a serializable dictionary
        config_dict = {k: v for k, v in self.config.__dict__.items() 
                      if not k.startswith('__') and not callable(v)}
        
        # Handle non-serializable types
        for key, value in config_dict.items():
            if isinstance(value, torch.Tensor):
                config_dict[key] = value.tolist()
            elif isinstance(value, (set, tuple)):
                config_dict[key] = list(value)
            elif isinstance(value, pathlib.Path):
                config_dict[key] = str(value)
                
        # Save config to JSON file
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
            
        print(f"Configuration saved to {path}")

    def get_checkpoint(self):
        return copy.deepcopy(self.current_checkpoint)

    def get_info(self, keys):
        if isinstance(keys, str):
            return self.current_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.current_checkpoint[key] for key in keys}
        else:
            raise TypeError

    def set_info(self, keys, values=None):
        if isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            raise TypeError
