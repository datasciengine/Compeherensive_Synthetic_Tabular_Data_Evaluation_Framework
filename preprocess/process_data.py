from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd


class Preprocessor:
    def __init__(self, nums=["sepal_l", "sepal_w", "petal_l", "petal_w"],
                 noms=["nominal"], ords=["edu", "gender"]):
        self.numeric_features = nums
        self.nominal_features = noms
        self.ordinal_features = ords
        self.preprocessor = None

    def fit_transform(self, X_train):
        # Define the transformers for each type of feature
        numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
        nominal_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        ordinal_transformer = Pipeline(steps=[('ordinal', OrdinalEncoder(
            categories=[["ilkokul", "ortaokul", "lise", "üniversite", "yüksek", "doktora"],
                        ["erkek", "kadın"]]))])

        # Combine all transformers into a single ColumnTransformer
        self.preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, self.numeric_features),
                                                            ('nom', nominal_transformer, self.nominal_features),
                                                            ('ord', ordinal_transformer, self.ordinal_features)])

        # Fit and transform the training data
        return self.preprocessor.fit_transform(X_train)

    def transform(self, X_test):
        # Transform the test data using the already fitted preprocessor
        return self.preprocessor.transform(X_test)

    def get_train_test_split(self, df, target_col, test_size=.2):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        # X_train, X_test, y_train, y_test
        return train_test_split(X, y,
                                test_size=test_size, random_state=42)
