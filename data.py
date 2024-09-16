from config import Config
import pandas as pd
import json


class Data:
    def __init__(self, cfg_path):
        print("Data class is initializing...")
        self.cfg = Config(cfg_path)
        self.real_df = self.load_data("real")
        self.syn_df = self.load_data("synthetic")
        self.purpose = self.cfg.get_purpose()

        # self.col_dict = config["columns"]
        self._validate_data()

    def _validate_data(self):
        if self.real_df.shape[0] != self.syn_df.shape[0]:
            print("There are different shapes.")

    @staticmethod
    def load_config(path):
        with open(path, 'r') as file:
            return json.load(file)

    def load_data(self, name):
        path = self.cfg.get_data_path(name)
        try:
            if path.endswith("xlsx"):
                return pd.read_excel(path)
            else:
                sep = self.cfg.get_data_sep(name)
                return pd.read_csv(path, sep=sep)
        except Exception as e:
            print(f"Error occurred on load_data function for {path}, error looks like {e}.")
