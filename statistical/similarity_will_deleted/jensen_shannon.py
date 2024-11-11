# Jensen-Shannon Divergence (JSD), iki olasılık dağılımı arasındaki benzerliği ölçmek için kullanılan bir metriktir. İki dağılımın ne kadar benzer olduğunu ölçer. Eğer iki dağılım tamamen aynıysa, JSD değeri 0 olacaktır. Eğer iki dağılım tamamen farklıysa, JSD değeri 1 olacaktır.

# Öncelikle, JSD'yi hesaplamak için, her bir öznitelik için ayrı ayrı hesaplama yapmamız gerekiyor.
# Sayısal öznitelikler için, öncelikle verileri normalleştirmeli ve ardından bir histogram oluşturmalıyız.
# Kategorik öznitelikler için, her bir kategori için frekansları hesaplamalıyız.

# Aşağıda, Python'da JSD'yi hesaplamanın bir örneğini bulabilirsiniz:


import numpy as np
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler


class JSTester:
    def __init__(self):
        pass

    def jsd(self, p, q, base=np.e):
        '''
        Method to compute the Jenson-Shannon Distance
        between two probability distributions
        '''

        p = np.asarray(p)
        q = np.asarray(q)

        p = p / p.sum()
        q = q / q.sum()

        m = (p + q) / 2

        return (entropy(p, m, base=base) / 2. + entropy(q, m, base=base) / 2.)

    def get_jsd_result(self, real_df, syn_df, num_col):
        # Normalizing numerical features
        scaler = MinMaxScaler()
        num1_real = scaler.fit_transform(real_df[num_col].values.reshape(-1, 1))
        num1_synthetic = scaler.transform(syn_df[num_col].values.reshape(-1, 1))

        # Computing histogram for real data
        hist_real, _ = np.histogram(num1_real, bins='auto', density=True)

        # Computing histogram for synthetic data
        hist_synthetic, _ = np.histogram(num1_synthetic, bins='auto', density=True)

        # Computing JSD for num1 feature
        return self.jsd(hist_real, hist_synthetic)

# Repeat the same process for num2, num3, cat1, cat2 features

# Bu kod, 'num1' özniteliği için JSD'yi hesaplar. Aynı işlemi diğer öznitelikler için de tekrarlamalısınız. Kategorik öznitelikler için, 'np.histogram' yerine 'np.bincount' veya 'collections.Counter' kullanabilirsiniz.
#
# Sonuç olarak, her bir öznitelik için ayrı bir JSD değeri elde edersiniz. Bu değerler, sentetik verinin gerçek veriye ne kadar benzediğini gösterir. Bu değerlerin ortalamasını alarak, tüm veri seti için genel bir JSD değeri elde edebilirsiniz.
