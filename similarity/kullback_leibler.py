import numpy as np
from scipy.stats import entropy


class KLDCalculator:
    def __init__(self):
        pass

    @staticmethod
    def kl_divergence(p, q):
        return entropy(p, q)

    def get_kld_result(self, real_df, syn_df, cats, nums):
        kld_results = {}
        bins = int(len(real_df) / 5)
        for num in nums:
            real_hist, _ = np.histogram(real_df[num], bins=bins, density=True)
            syn_hist, _ = np.histogram(syn_df[num], bins=bins, density=True)

            real_hist /= np.sum(real_hist)
            syn_hist /= np.sum(syn_hist)

            kld_results[num] = self.kl_divergence(real_hist, syn_hist)

        for cat in cats:
            real_prob = real_df[cat].value_counts(normalize=True)
            syn_prob = syn_df[cat].value_counts(normalize=True)

            combined_index = real_prob.index.union(syn_prob.index)
            real_prob = real_prob.reindex(combined_index, fill_value=0)
            syn_prob = syn_prob.reindex(combined_index, fill_value=0)

            real_norm_prob = real_prob / real_prob.sum()
            syn_norm_prob = syn_prob / syn_prob.sum()

            kld_results[cat] = self.kl_divergence(real_norm_prob, syn_norm_prob)

        return kld_results
