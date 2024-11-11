import numpy as np
from scipy.spatial.distance import jensenshannon

"""
Diferansiyel mahremiyetin sağlanıp sağlanmadığını kontrol etmek için bazı istatistiksel testler uygulayabiliriz. Örneğin, `pandas` ve `numpy` kullanarak Jensen-Shannon Divergence (JSD) hesaplayabiliriz.
"""


def calculate_jsd(real_data, synthetic_data, feature):
    real_dist = real_data[feature].value_counts(normalize=True).sort_index()
    synthetic_dist = synthetic_data[feature].value_counts(normalize=True).sort_index()

    # Align the distributions
    real_dist, synthetic_dist = real_dist.align(synthetic_dist, fill_value=0)

    return jensenshannon(real_dist, synthetic_dist)


# Kategorik özellikler için JSD hesaplama
for feature in categorical_features:
    jsd = calculate_jsd(real_df, synthetic_df, feature)
    print(f'Jensen-Shannon Divergence for {feature}: {jsd}')

# Numerik özellikler için JSD hesaplama
for feature in numeric_features:
    real_dist = np.histogram(real_df[feature], bins=10, density=True)[0]
    synthetic_dist = np.histogram(synthetic_df[feature], bins=10, density=True)[0]
    jsd = jensenshannon(real_dist, synthetic_dist)
    print(f'Jensen-Shannon Divergence for {feature}: {jsd}')

"""
- **Özellik Dağılımları**: Orijinal ve sentetik veri setlerindeki özelliklerin dağılımlarının benzer olup olmadığını kontrol edin. Eğer dağılımlar çok farklıysa, sentetik veri seti orijinal veri setini iyi temsil etmiyor olabilir.
- **Jensen-Shannon Divergence (JSD)**: JSD değeri 0 ile 1 arasında değişir. 0'a yakın değerler, iki dağılımın birbirine çok benzediğini gösterir. 1'e yakın değerler ise dağılımların çok farklı olduğunu gösterir. JSD değerlerinin düşük olması, sentetik verinin orijinal veriye benzer olduğunu ve diferansiyel mahremiyetin sağlandığını gösterebilir.

Bu adımları izleyerek, sentetik verinizin diferansiyel mahremiyeti sağlayıp sağlamadığını değerlendirebilirsiniz. Ancak, diferansiyel mahremiyetin tam olarak sağlandığını garanti etmek için daha ileri düzeyde analizler ve testler gerekebilir.
"""
