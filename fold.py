
import math

class Fold():
    def __init__(self, dataset, proportion, target):
        self.data = dataset
        self.target = target

        self.feature_cols = list(self.data.columns)
        self.feature_cols.remove(self.target)

        df_length = len(dataset)

        self.data_train = dataset[0: math.floor(df_length*(proportion["train"]-proportion["train-validation"]))]

        df_validation = dataset.drop(self.data_train.index)
        self.data_validation = df_validation[0: math.floor(df_length*(proportion["train-validation"]))]

        self.data_test = df_validation.drop(self.data_validation.index)

    def values_data_training(self):
        return self.data_train_scaled.loc[:, self.feature_cols].values, self.data_train_scaled.loc[:, self.target].values

    def values_data_validation(self):
        return self.data_validation_scaled.loc[:, self.feature_cols].values, self.data_validation_scaled.loc[:, self.target].values

    def values_data_test(self):
        return self.data_test_scaled.loc[:, self.feature_cols].values, self.data_test_scaled.loc[:, self.target].values
    
    def expected_data(self):
        return self.data_test.loc[:, self.target]

    def __str__(self):
        return self.data.to_string()