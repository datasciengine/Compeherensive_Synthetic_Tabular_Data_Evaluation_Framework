import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from preprocess.process_data import Preprocessor


class ReconstructionError:
    def __init__(self, real_df, syn_df):
        self.pre = Preprocessor()
        self.real_processed = self.pre.fit_transform(real_df)
        self.syn_processed = self.pre.fit_transform(syn_df)

    def get_reconstruction_error(self):
        distance_matrix = pairwise_distances(self.real_processed, self.syn_processed, metric='euclidean')
        row_ind, col_ind = linear_sum_assignment(distance_matrix)

        matched_real_data = self.real_processed[row_ind]
        matched_synthetic_data = self.syn_processed[col_ind]
        return {
            "ReconstructionError": round(np.mean(np.linalg.norm(matched_real_data - matched_synthetic_data, axis=1)), 3),
            "message": "Low error means synthetic data really similar to real data."
                       "High error means synthetic data is not similar to real data."}
