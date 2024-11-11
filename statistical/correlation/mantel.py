from skbio.stats.distance import mantel
from scipy.spatial import distance
import numpy as np


class MantelTest:
    def __init__(self):
        pass

    @staticmethod
    def get_mantel_result(real_df, syn_df, nums):
        # Gerçek ve sentetik veri setlerinizi numpy dizilerine dönüştürün
        real_data = np.array(real_df[nums])
        synthetic_data = np.array(syn_df[nums])

        # Her iki veri seti için de mesafe matrislerini hesaplayın
        real_dist = distance.pdist(real_data, 'euclidean')
        synthetic_dist = distance.pdist(synthetic_data, 'euclidean')

        # Mesafe matrislerini kare matrislere dönüştürün
        real_dist_square = distance.squareform(real_dist)
        synthetic_dist_square = distance.squareform(synthetic_dist)

        # Mantel testini gerçekleştirin
        r, p_value, n = mantel(real_dist_square, synthetic_dist_square,
                               method='pearson', permutations=9999)
        return {"mantel_result": round(r, 3),
                "p_value": p_value,
                "message": """ 
                - r, iki mesafe matrisi arasındaki korelasyon katsayısını temsil eder. Bu değer -1 ile 1 arasında değişir. 1'e yakın bir değer, iki matris arasında güçlü bir pozitif korelasyon olduğunu gösterir. Yani, bir matristeki değerler arttıkça, diğer matristeki değerlerin de arttığını gösterir. -1'e yakın bir değer, güçlü bir negatif korelasyon olduğunu gösterir. Yani, bir matristeki değerler arttıkça, diğer matristeki değerlerin azaldığını gösterir. 0'a yakın bir değer, iki matris arasında zayıf bir korelasyon olduğunu gösterir.
                - p_value, testin istatistiksel anlamlılığını gösterir. Genellikle, p_value 0.05'ten küçükse, sonuçlar istatistiksel olarak anlamlı kabul edilir. Bu, r değerinin sıfırdan farklı olduğunu ve bu farklılığın rastgele bir sonuç olmadığını gösterir.
                Bu durumda, eğer r değeri yüksek (yaklaşık 1) ve p_value değeri düşük (0.05'ten küçük) ise, bu, sentetik veri setinizin gerçek veri setinize oldukça benzediğini gösterir. Eğer r değeri düşük ve/veya p_value değeri yüksekse, bu, sentetik veri setinizin gerçek veri setine benzemediğini gösterir.
                """
                }
