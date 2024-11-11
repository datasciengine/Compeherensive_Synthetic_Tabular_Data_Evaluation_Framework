"""
Attribute inference attack (özellik çıkarım saldırısı), bir saldırganın, bir makine öğrenimi modeline veya veri setine erişerek,
belirli bir bireyin gizli veya eksik özelliklerini tahmin etmeye çalıştığı bir saldırı türüdür.

Bu tür saldırılar, özellikle hassas bilgilerin korunması gereken durumlarda ciddi güvenlik ve gizlilik riskleri oluşturabilir.
Bu saldırıyı simüle etmek için aşağıdaki adımları izleyebilirsiniz:
"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from preprocess.process_data import Preprocessor
from sklearn.metrics import accuracy_score


class AttributeInferenceAttacker:
    """
    Kendime Not :

    Sentetik datada secret target için model eğit, adına da atak modeli de.
    Bu modele gerçek datadan bir kısım predict al,
    Eğer sentetik data'dan eğitilmiş atak modeli gerçek datayı ifşa edebiliyorsa atak modeli kazanıyor, risk var demek.

    as you know, Train on Syn, Test on Real.
    """

    # TODO : Regression desteklenecek, preprocess kısmını da ayrıca düşünmek gerekir ama en son detay o.
    def __init__(self, real_df, syn_df, nums, cats, secret_attr, test_size=.5):
        self.real_df = real_df
        self.syn_df = syn_df
        self.nums = nums
        self.cats = cats
        self.secret_attr = secret_attr
        self.test_size = test_size
        self.pre = Preprocessor()

    def _get_attack_model(self):

        if self.secret_attr not in (self.cats + self.nums):
            raise TypeError(
                f"Secret attribute could be in categorical or numerical features. {self.secret_attr}, cats are {self.cats}, nums are {self.nums}")

        if self.secret_attr in self.cats:
            # which means classification.
            # Sentetik veri setini kullanarak bir model eğitin. Bu model, saldırganın kullanacağı model olacak.
            X_synthetic = self.syn_df.drop(columns=[self.secret_attr])
            y_synthetic = self.syn_df[self.secret_attr]
            attack_model = RandomForestClassifier()
            X_scaled_synthetic = self.pre.fit_transform(X_synthetic)
            attack_model.fit(X_scaled_synthetic, y_synthetic)
            return attack_model

        else:
            # which means regression.
            X_synthetic = self.syn_df.drop(columns=[self.secret_attr])
            y_synthetic = self.syn_df[self.secret_attr]
            attack_model = RandomForestRegressor()
            X_scaled_synthetic = self.pre.fit_transform(X_synthetic)
            attack_model.fit(X_scaled_synthetic, y_synthetic)
            return attack_model

    def get_eval_attack_model(self):
        ### Adım 6: Saldırıyı Simüle Edin
        # Eğitilen saldırı modelini kullanarak, gerçek veri setindeki gizli özellikleri tahmin edin.
        X_train, X_test, y_train, y_test = self.pre.get_train_test_split(df=self.real_df,
                                                                         target_col=self.secret_attr,
                                                                         test_size=self.test_size)
        # pay attention! -> here target is actually secret attr.
        attack_model = self._get_attack_model()
        X_scaled_test = self.pre.transform(X_test)
        y_pred = attack_model.predict(X_scaled_test)

        ### Adım 7: Performansı Değerlendirin
        # Saldırının başarısını değerlendirmek için tahminlerin doğruluğunu ölçün.
        accuracy = accuracy_score(y_test, y_pred)
        return {"accuracy": round(accuracy, 3),
                "message": "Successful model means attribute inference attacker is successful."}
