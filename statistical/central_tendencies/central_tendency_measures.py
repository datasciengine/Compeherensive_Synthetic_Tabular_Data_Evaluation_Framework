class CentralTendencyCalculator:
    def __init__(self):
        pass

    @staticmethod
    def get_mean_results(real_df, syn_df, nums):
        mean_measurements = {}
        for num in nums:
            mean_measurements[num] = {"real": round(real_df[num].mean(), 3),
                                      "syn": round(syn_df[num].mean(), 3),
                                      "diff": round(real_df[num].mean() - syn_df[num].mean(), 3)}

        return mean_measurements

    @staticmethod
    def get_median_results(real_df, syn_df, nums):
        median_measurements = {}
        for num in nums:
            median_measurements[num] = {"real": round(real_df[num].median(), 3),
                                        "syn": round(syn_df[num].median(), 3),
                                        "diff": round(real_df[num].median() - syn_df[num].median(), 3)}
        return median_measurements

    @staticmethod
    def get_stddev_results(real_df, syn_df, nums):
        stddev_measurements = {}

        for num in nums:
            stddev_measurements[num] = {"real": round(real_df[num].std(), 3),
                                        "syn": round(syn_df[num].std(), 3),
                                        "diff": round(real_df[num].std() - syn_df[num].std(), 3)}
        return stddev_measurements

    @staticmethod
    def get_var_results(real_df, syn_df, nums):
        var_measurements = {}
        for num in nums:
            var_measurements[num] = {"real": round(real_df[num].var(), 3),
                                     "syn": round(syn_df[num].var(), 3),
                                     "diff": round(real_df[num].var() - syn_df[num].var(), 3)}
        return var_measurements

    @staticmethod
    def get_min_results(real_df, syn_df, nums):
        min_measurements = {}
        for num in nums:
            min_measurements[num] = {"real": round(real_df[num].min(), 3),
                                     "syn": round(syn_df[num].min(), 3),
                                     "diff": round(real_df[num].min() - syn_df[num].min(), 3)}
        return min_measurements

    @staticmethod
    def get_max_results(real_df, syn_df, nums):
        max_measurements = {}
        for num in nums:
            max_measurements[num] = {"real": round(real_df[num].max(), 3),
                                     "syn": round(syn_df[num].max(), 3),
                                     "diff": round(real_df[num].max() - syn_df[num].max(), 3)}
        return max_measurements

    @staticmethod
    def get_q1_results(real_df, syn_df, nums):
        q1_measurements = {}

        for num in nums:
            q1_measurements[num] = {"real": round(real_df[num].quantile(.25), 3),
                                    "syn": round(syn_df[num].quantile(.25), 3),
                                    "diff": round(real_df[num].quantile(.25) - syn_df[num].quantile(.25), 3)}
        return q1_measurements

    @staticmethod
    def get_q3_results(real_df, syn_df, nums):
        q3_measurements = {}

        for num in nums:
            q3_measurements[num] = {"real": round(real_df[num].quantile(.75), 3),
                                    "syn": round(syn_df[num].quantile(.75), 3),
                                    "diff": round(real_df[num].quantile(.75) - syn_df[num].quantile(.75), 3)}

        return q3_measurements
