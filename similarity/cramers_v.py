import numpy as np
import pandas as pd
import scipy.stats as ss


class CramerCalculator:
    def __init__(self):
        pass

    @staticmethod
    def cramers_v(x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        r_corr = r - ((r - 1) ** 2) / (n - 1)
        k_corr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2_corr / min((k_corr - 1), (r_corr - 1)))

    def get_result(self, real_df, syn_df, cats):
        cramer_results = {}
        for cat in cats:
            result = self.cramers_v(real_df[cat], syn_df[cat])
            if result > .75:
                similarity = "high"
            elif result > .5:
                similarity = "up-medium"
            elif result > .25:
                similarity = "low-medium"
            else:
                similarity = "low"

            cramer_results[cat] = {"cv_result": round(result, 3),
                                   "message": f"Similarity is {similarity}, between real and synthetic {cat} feature."}
        return cramer_results
