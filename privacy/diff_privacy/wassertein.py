from scipy.stats import wasserstein_distance
import numpy as np


class WSNCalculator:
    def __init__(self, real_df, syn_df, cats, nums):
        self.real_df = real_df
        self.syn_df = syn_df
        self.cats = cats
        self.nums = nums

    def calculate_categorical_wsn(self, feature):
        real_prob = self.real_df[feature].value_counts(normalize=True)
        syn_prob = self.syn_df[feature].value_counts(normalize=True)

        combined_index = real_prob.index.union(syn_prob.index)
        real_prob = real_prob.reindex(combined_index, fill_value=0)
        syn_prob = syn_prob.reindex(combined_index, fill_value=0)

        real_norm_prob = real_prob / real_prob.sum()
        syn_norm_prob = syn_prob / syn_prob.sum()

        return round(wasserstein_distance(real_norm_prob, syn_norm_prob), 3)

    def calculate_numerical_wsn(self, feature):
        bins = int(len(self.real_df) / 5)
        real_hist, _ = np.histogram(self.real_df[feature], bins=bins, density=True)
        syn_hist, _ = np.histogram(self.syn_df[feature], bins=bins, density=True)

        real_hist /= np.sum(real_hist)
        syn_hist /= np.sum(syn_hist)

        return round(wasserstein_distance(real_hist, syn_hist), 3)

    def get_wsn_results(self):
        num_wsn, cat_wsn = {}, {}

        for num_feature in self.nums:
            num_wsn[num_feature] = self.calculate_numerical_wsn(num_feature)

        for cat_feature in self.cats:
            cat_wsn[cat_feature] = self.calculate_categorical_wsn(cat_feature)

        return {"Numericals": num_wsn,
                "Categoricals": cat_wsn,
                "message": "If results are close to 0, which means synthetic data is too similar to real data."
                           "There is no limit such as 0-1."}
