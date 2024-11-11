import numpy as np


class KLTester:
    def __init__(self):
        pass

    @staticmethod
    def kl_divergence(p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    def get_kl_result(self, real_df, syn_df, num_cols):
        # her bir sayısal öznitelik için KL Divergence hesapla
        result_dict = {}
        for feature in num_cols:
            # histogramları hesapla
            p_hist, p_bins = np.histogram(syn_df[feature], bins=50, density=True)
            q_hist, q_bins = np.histogram(real_df[feature], bins=p_bins, density=True)

            # 0 değerlerini önlemek için her iki histograma da çok küçük bir değer ekleyin
            p_hist += 1e-8
            q_hist += 1e-8

            # KL Divergence'ı hesapla
            kl_div = self.kl_divergence(p_hist, q_hist)
            result_dict[f"{feature}"] = kl_div
            print(f"KL Divergence for {feature}: {kl_div}")
