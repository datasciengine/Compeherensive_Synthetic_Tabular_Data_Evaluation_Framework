import yaml


class Config:
    def __init__(self, path):
        print("Config class is initializing...")
        self.cfg = self.load_config(path)

    def get_data_path(self, name):
        return self.cfg[name][0]["path"]

    def get_data_sep(self, name):
        return self.cfg[name][1]["sep"]

    def get_purpose(self):
        return self.cfg["purpose"]

    @staticmethod
    def load_config(path):
        with open(path) as stream:
            try:
                return yaml.safe_load(stream)
            except Exception as e:
                raise RuntimeError(f"There is a problem with {path} yaml file loading. It could be {e}")
