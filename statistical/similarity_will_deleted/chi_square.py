# Chi-Square testi, iki kategorik değişken arasındaki bağımsızlığı test etmek için kullanılır. Yani, bir değişkenin değerlerinin, diğer değişkenin değerlerinden bağımsız olduğunu test eder.
#
# Bu durumda, sentetik ve gerçek veri setlerindeki kategorik özniteliklerin dağılımlarını karşılaştırmak için Chi-Square testini kullanabiliriz.
#
# Python'da bu işlemi gerçekleştirmek için öncelikle pandas ve scipy kütüphanelerini import etmemiz gerekiyor.
#
# Aşağıda, bu işlemi gerçekleştiren bir Python kodu örneği bulunmaktadır:

import pandas as pd
from scipy.stats import chi2_contingency


class CSTester:
    def __init__(self):
        pass

    def get_cs_result(self, real_df, syn_df, cat_cols):
        for cat_col in combs(cat_cols):
            # burada 1den fazla cat val olmalı ki kombinasyonları ortaya çıksın.
            # Kategorik öznitelikler için frekans tablolarını oluştur
            real_freq = pd.crosstab(index=real_df[cat_col], columns=real_df[cat_col])
            synthetic_freq = pd.crosstab(index=syn_df[cat_col], columns=syn_df[cat_col])

            # Chi-Square testini uygula
            chi2, p, dof, expected = chi2_contingency(synthetic_freq.values, real_freq.values)

            print("Chi-Square Statistic: ", chi2)
            print("P Value: ", p)

        # Bu kod, öncelikle veri setlerini okur ve kategorik öznitelikler için frekans tablolarını oluşturur. Ardından, `chi2_contingency` fonksiyonunu kullanarak Chi-Square testini uygular. Bu fonksiyon, Chi-Square istatistiğini, p değerini, serbestlik derecesini ve beklenen frekansları döndürür.
        #
        # Sonuç olarak, eğer p değeri belirlediğiniz anlamlılık düzeyinden (genellikle 0.05) daha küçükse, iki kategorik değişken arasında anlamlı bir ilişki olduğunu söyleyebiliriz. Yani, sentetik veri setindeki kategorik özniteliklerin dağılımı, gerçek veri setindekine benzerdir. Eğer p değeri belirlediğiniz anlamlılık düzeyinden daha büyükse, iki kategorik değişken arasında anlamlı bir ilişki olmadığını söyleyebiliriz. Yani, sentetik veri setindeki kategorik özniteliklerin dağılımı, gerçek veri setindekine benzememektedir.
