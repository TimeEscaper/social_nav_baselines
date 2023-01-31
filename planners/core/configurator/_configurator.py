import yaml

class Configurator():
    def __init__(self,
                 config_path: str) -> None:
        with open(r"config_path") as f:
            config = yaml.safe_load()
        