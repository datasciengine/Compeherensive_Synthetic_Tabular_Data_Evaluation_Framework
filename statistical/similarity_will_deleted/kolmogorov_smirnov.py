from scipy import stats


class KSTester:
    def __init__(self):
        pass

    @staticmethod
    def get_ks_result(real_df, syn_df, num_cols):
        result_dict = {}
        for col in num_cols:
            statistic, p_value = stats.ks_2samp(real_df[col], syn_df[col])
            print(f"For {col} Kolmogorov-Smirnov test: statistic={statistic}, p_value={p_value}")
            result_dict[f"{col}"] = {"statistic": statistic,
                                     "p_value": p_value}
        return result_dict
