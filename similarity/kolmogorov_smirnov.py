from scipy import stats


class KSCalculator:
    def __init__(self):
        pass

    @staticmethod
    def get_ks_result(real_df, syn_df, nums, alpha=0.05):
        ks_results = {}
        for num in nums:
            ks_stats, p_value = stats.ks_2samp(real_df[num], syn_df[num])
            distribution_message = "Different distribution."
            if p_value > alpha:
                distribution_message = "Same distribution."

            ks_results[num] = {"ks_stats": round(ks_stats, 3),
                               "p_val": round(p_value, 3),
                               "message": distribution_message}
        return ks_results
