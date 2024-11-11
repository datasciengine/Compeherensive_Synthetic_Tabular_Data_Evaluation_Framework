import numpy as np


class PLDCalculator:
    def __init__(self, real_df, syn_df, cats, nums):
        self.cats, self.nums = cats, nums
        self.real_df = real_df
        self.syn_df = syn_df
        self.cats = cats
        self.nums = nums

    # Kategorik öznitelikler için PLD hesaplama
    def calculate_categorical_pld(self, feature):
        # real_counts = self.real_df[feature].value_counts(normalize=True)
        # synthetic_counts = self.syn_df[feature].value_counts(normalize=True)

        real_prob = self.real_df[feature].value_counts(normalize=True)
        syn_prob = self.syn_df[feature].value_counts(normalize=True)

        combined_index = real_prob.index.union(syn_prob.index)
        real_prob = real_prob.reindex(combined_index, fill_value=0)
        syn_prob = syn_prob.reindex(combined_index, fill_value=0)

        real_norm_prob = real_prob / real_prob.sum()
        syn_norm_prob = syn_prob / syn_prob.sum()

        return np.sum(np.abs(real_norm_prob - syn_norm_prob))

    def calculate_numerical_pld(self, feature):
        real_mean = self.real_df[feature].mean()
        synthetic_mean = self.syn_df[feature].mean()
        real_std = self.real_df[feature].std()
        synthetic_std = self.syn_df[feature].std()
        return np.abs(real_mean - synthetic_mean) / (real_std + synthetic_std)

    def get_pld_results(self):
        num_plds, cat_plds = {}, {}

        for num_feature in self.nums:
            num_plds[num_feature] = round(self.calculate_numerical_pld(num_feature), 3)

        for cat_feature in self.cats:
            cat_plds[cat_feature] = round(self.calculate_categorical_pld(cat_feature), 3)

        return {"Numericals": num_plds,
                "Categoricals": cat_plds,
                "message": "Low PLD value means too similar synthetic data. High PLD value means synthetic data is different from real data."}


""" 
Bu kod, gerçek ve sentetik veri setleri arasındaki farkları analiz ederek her bir öznitelik için PLD'yi hesaplar. Kategorik öznitelikler için, her bir kategorinin oranlarını karşılaştırır ve farkların toplamını hesaplar. Sayısal öznitelikler için, ortalama ve standart sapma farklarını kullanarak PLD'yi hesaplar.

Bu yöntem, temel bir PLD hesaplama yaklaşımıdır ve daha karmaşık yöntemler de kullanılabilir. Ancak, bu kod size PLD hesaplamanın temel bir örneğini sunar.
"""
