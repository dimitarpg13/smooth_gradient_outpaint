import os
import yaml


class OutpainterConfig:
    """
    Singleton class to load configuration from config.yaml
    """
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(OutpainterConfig, cls).__new__(cls)
            cls.instance.SMOOTH_GRADIENT_OUTPAINTER_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
            cls.instance.config = cls.load_config(cls.instance.SMOOTH_GRADIENT_OUTPAINTER_HOME)
        return cls.instance

    @staticmethod
    def load_config(config_path):
        with open(f"{config_path}/config.yaml", "r") as file:
            config = yaml.safe_load(file)
        return config
