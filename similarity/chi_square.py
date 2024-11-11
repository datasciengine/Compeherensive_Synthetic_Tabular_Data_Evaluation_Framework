import pandas as pd
from scipy.stats import chi2_contingency


class CSCalculator:
    def __init__(self):
        pass

    @staticmethod
    def get_cs_result(real_df, syn_df, cats, alpha=.05):
        cs_results = {}

        for cat in cats:
            real_freq = real_df[cat].value_counts()
            syn_freq = syn_df[cat].value_counts()
            cont_table = pd.DataFrame({"Real": real_freq, "Synthetic": syn_freq})
            chi2, p, dof, expected = chi2_contingency(cont_table)
            message = "There is no meaningful difference between real and synthetic categorical values."
            if p < alpha:
                message = "There is meaningful difference between real and synthetic categorical values."
            cs_results[cat] = {"chi2": round(chi2, 3), "p": round(p, 3), "dof": round(dof, 3),
                               "expected": expected, "message": message}
        return cs_results
