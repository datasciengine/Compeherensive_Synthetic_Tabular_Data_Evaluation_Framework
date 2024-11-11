# K Anonymity, bir veri setinin belirli bir K değeri için anonim olup olmadığını kontrol etmek için kullanılan bir yöntemdir.
# Bir veri setinin K Anonymity 'ye sahip olması, her bir kombinasyonun en az K kez tekrarlanması anlamına gelir.
# Aşağıdaki kod, verilen quasi_identifiers ve K değeri için K Anonymity 'nin geçerli olup olmadığını kontrol eder.


# Bu kod, verilen quasi_identifiers'a göre veri setini gruplar ve her bir grup için sayıyı hesaplar. Eğer her bir grup en az K kez tekrarlanıyorsa, K Anonymity geçerlidir ve fonksiyon True döner. Aksi halde, K Anonymity geçersizdir ve fonksiyon False döner.

class KAnonymizeChecker:
    def __init__(self):
        pass

    @staticmethod
    def is_k_anonymous(df, quasi_identifiers, k_range):
        result_dict = {}
        for k in range(k_range[0], k_range[1]):
            # Quasi-identifierlara göre gruplama yap ve her bir grup için sayıyı hesapla.
            groups = df.groupby(quasi_identifiers).size().reset_index(name='counts')
            # Eğer her bir grup en az K kez tekrarlanıyorsa, K Anonymity geçerlidir
            result_dict[f"{k}"] = (groups['counts'].min() >= k)
        return result_dict
