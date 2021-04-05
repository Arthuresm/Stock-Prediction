from window_generator import WindowGenerator
from data import WindowData
import math

class DataModeling():
    DATA_SET_PROPORTION_KEY = "data-set-proportion"

    def __init__(self, dataset, features, target, rules, input_width, label_width, shift):
        all_feature_columns = features + [target]
        dataset = dataset[all_feature_columns]
        self.rules = rules

        self.window_data = WindowData(dataset, rules[self.DATA_SET_PROPORTION_KEY], target)

        self.process_rules()

        self.window = WindowGenerator(
            input_width=input_width, label_width=label_width, shift=shift,
            label_columns=[target], train_df=self.window_data.train_scaled, val_df=self.window_data.validation_scaled, test_df=self.window_data.test_scaled)
    
    def process_rules(self):
        process_dict = {
            "zscore" : self.zscore,
            "norm" : self.norm
        }

        for type_process in self.rules["global"]:
            process_dict[type_process](self.window_data)
        

    def zscore(self, folds):
        print("Under development")
               
    def norm(self, window_data):
    
        train_stats = window_data.data_train.describe().transpose()
        window_data.train_stats = train_stats

        window_data.train_scaled = (window_data.data_train - train_stats['mean']) / train_stats['std']

        window_data.validation_scaled = (window_data.data_validation - train_stats['mean']) / train_stats['std']

        window_data.test_scaled = (window_data.data_test - train_stats['mean']) / train_stats['std']

    def inverse_norm(train_stats, scaled_data):
        return (scaled_data*train_stats['std'] + train_stats['mean'])