import pandas as pd


class LinkageAttacker:
    def __init__(self, real_df, syn_df, external_df, common_cols):
        """
        Linkage Attack (Bağlantı Saldırısı), anonimleştirilmiş veya sentetik verilerin, ek dış kaynak verilerle birleştirilerek orijinal
        bireylerin kimliklerinin tespit edilmesi sürecidir. Bu tür saldırılar, veri gizliliği ve mahremiyetini tehdit eder.
        Linkage Attack'ı adım adım nasıl uygulayabileceğinizi aşağıda açıklıyorum:

        ### Adım 1: Veri Hazırlığı
        1. **Gerçek Veri (real_df)**: Orijinal, anonimleştirilmiş veya kısmen anonimleştirilmiş veri setiniz.
        2. **Sentetik Veri (synthetic_df)**: GAN (Generative Adversarial Network) kullanarak ürettiğiniz sentetik veri setiniz.
        3. **Ek Dış Kaynak Veri (external_df)**: Gerçek dünyadan elde edilen ve bazı ortak özelliklere sahip olabilecek ek veri seti.
        Bu veri seti, saldırıyı gerçekleştirmek için kullanılacak.

        ### Adım 2: Ortak Özelliklerin Belirlenmesi
        Her üç veri setinde de ortak olan özellikleri (örneğin, yaş, cinsiyet, posta kodu gibi) belirleyin. Bu özellikler, veri setlerini birleştirmek için kullanılacak.

        ### Adım 3: Veri Setlerinin Birleştirilmesi
        1. **Ortak Özellikler Üzerinden Eşleştirme**: Gerçek veri seti (real_df) ve dış kaynak veri seti (external_df) üzerinde ortak özellikler kullanarak eşleştirme yapın. Bu eşleştirme, her iki veri setindeki bireylerin kimliklerini tespit etmeye yönelik olacaktır.
        2. **Eşleştirme Algoritması**: Eşleştirme işlemi için çeşitli algoritmalar kullanabilirsiniz. Örneğin, basit bir inner join veya daha karmaşık makine öğrenimi tabanlı eşleştirme algoritmaları kullanılabilir.

        ### Adım 4: Saldırının Gerçekleştirilmesi
        1. **Eşleştirilmiş Kayıtların Analizi**: Eşleştirilmiş kayıtları analiz ederek, gerçek veri setindeki bireylerin kimliklerini tespit etmeye çalışın.
        2. **Sentetik Veri ile Karşılaştırma**: Elde edilen eşleştirilmiş kayıtları sentetik veri seti (synthetic_df) ile karşılaştırarak, sentetik verinin ne kadarının gerçek veriyi doğru bir şekilde temsil ettiğini ve ne kadarının kimlik tespitine yol açabileceğini değerlendirin.

        ### Adım 5: Sonuçların Değerlendirilmesi
        1. **Başarı Oranı**: Linkage Attack'ın ne kadar başarılı olduğunu değerlendirin. Bu, eşleştirilen kayıtların sayısı ve doğruluğu ile ölçülebilir.
        2. **Gizlilik Riski**: Elde edilen sonuçlara göre, sentetik verinin gizlilik riskini değerlendirin. Eğer çok sayıda doğru eşleştirme yapılabiliyorsa, sentetik veri seti gizlilik açısından riskli olabilir.

        ### Örnek Kod
        Aşağıda, basit bir Python kodu ile Linkage Attack'ın nasıl gerçekleştirilebileceğine dair bir örnek verilmiştir:
        """
        self.real_df = real_df
        self.syn_df = syn_df
        self.external_df = external_df
        self.common_cols = common_cols

    def get_linkage_attack_eval(self):
        # Ortak özellikler üzerinden eşleştirme
        # Gerçek veri ve dış kaynak veri setini birleştirme
        real_merged_df = pd.merge(self.real_df, self.external_df,
                                  on=self.common_cols, how='inner')
        # Sentetik veri ile karşılaştırma
        synthetic_merged_df = pd.merge(self.syn_df, self.external_df,
                                       on=self.common_cols, how='inner')
        return {"Matched_Real_External": len(real_merged_df),
                "Matched_Syn_External": len(synthetic_merged_df),
                "message": "Counts are matched observation in with common cols between real/syn and external datasets."}
