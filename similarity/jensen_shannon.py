import numpy as np
from scipy.spatial.distance import jensenshannon


class JSCalculator:
    def __init__(self):
        pass

    @staticmethod
    def get_js_result(real_df, syn_df, nums, cats):
        jsd_results = {}
        bins = int(len(real_df) / 10)
        for num in nums:
            real_hist = np.histogram(real_df[num], bins=bins, density=True)[0]
            syn_hist = np.histogram(syn_df[num], bins=bins, density=True)[0]

            real_hist /= np.sum(real_hist)
            syn_hist /= np.sum(syn_hist)

            jsd_results[num] = round(jensenshannon(real_hist, syn_hist), 3)

        for cat in cats:
            real_prob = real_df[cat].value_counts(normalize=True)
            syn_prob = syn_df[cat].value_counts(normalize=True)

            combined_index = real_prob.index.union(syn_prob.index)
            real_prob = real_prob.reindex(combined_index, fill_value=0)
            syn_prob = syn_prob.reindex(combined_index, fill_value=0)

            real_norm_prob = real_prob / real_prob.sum()
            syn_norm_prob = syn_prob / syn_prob.sum()
            jsd_results[cat] = round(jensenshannon(real_norm_prob, syn_norm_prob), 3)

        return jsd_results
