from scipy.stats import entropy
import numpy as np


class TClosenessChecker:
    def __init__(self):
        pass

    @staticmethod
    def check_t_closeness(df, quasi_identifiers, sensitive_cols, t):
        total_distribution = df[sensitive_cols].value_counts(normalize=True)

        # Her bir Quasi Identifier grubunun dağılımını hesapla ve t değerinden fazla sapma olup olmadığını kontrol et
        for _, group in df.groupby(quasi_identifiers):
            group_distribution = group[sensitive_cols].value_counts(normalize=True)
            if np.abs(total_distribution - group_distribution).sum() > t:
                return False, "T-Closeness is Not OK."
        else:
            return True, "T-Closeness is OK."

    @staticmethod
    def check_t_closeness_old(df, quasi_identifiers, sensitive_cols, t):
        # Tüm veri kümesi için hassas özniteliklerin dağılımını hesapla
        total_distribution = df[sensitive_cols].value_counts(normalize=True)

        # Her bir ekvivalans sınıfı için hassas özniteliklerin dağılımını hesapla
        for _, group in df.groupby(quasi_identifiers):
            group_distribution = group[sensitive_cols].value_counts(normalize=True)

            # KL Divergence'ı hesapla
            kl_divergence = entropy(total_distribution, group_distribution)

            # Eğer KL Divergence T'den büyükse, T-Closeness geçerli değildir
            if kl_divergence > t:
                return False

        # Eğer tüm ekvivalans sınıfları için KL Divergence T'den küçükse, T-Closeness geçerlidir
        return True
