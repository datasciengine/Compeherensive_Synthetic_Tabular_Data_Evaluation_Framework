import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocess.process_data import Preprocessor


class MembershipInferenceAttacker:
    def __init__(self, real_data, syn_data):
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_split_data(real_data, syn_data)
        self.pre = Preprocessor()

    def get_split_data(self, real_data, syn_data):
        real_train, real_test = train_test_split(real_data, test_size=0.2, random_state=42)
        synthetic_train, synthetic_test = train_test_split(syn_data, test_size=0.2, random_state=42)

        real_train['membership_inference_label'] = 1
        real_test['membership_inference_label'] = 1

        synthetic_train['membership_inference_label'] = 0
        synthetic_test['membership_inference_label'] = 0

        attack_train_data = pd.concat([real_train, synthetic_train])
        attack_test_data = pd.concat([real_test, synthetic_test])

        X_train = attack_train_data.drop('membership_inference_label', axis=1)
        y_train = attack_train_data['membership_inference_label']
        X_test = attack_test_data.drop('membership_inference_label', axis=1)
        y_test = attack_test_data['membership_inference_label']

        return X_train, X_test, y_train, y_test

    def _get_attack_model(self):
        attack_model = RandomForestClassifier(n_estimators=100, random_state=42)
        X_scaled_train = self.pre.fit_transform(self.X_train)
        attack_model.fit(X_scaled_train, self.y_train)
        return attack_model

    def get_eval_attack_model(self):
        attack_model = self._get_attack_model()
        X_scaled_test = self.pre.transform(self.X_test)
        y_pred = attack_model.predict(X_scaled_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return {"accuracy": round(accuracy, 3),
                "message": "Successful model means membership inference attacker is successful. "
                           "Because synthetic and real observations are distinguishable."}
