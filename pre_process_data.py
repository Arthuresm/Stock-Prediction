from fold import Fold

class DataModeling():
    DATA_SET_PROPORTION_KEY = "data-set-proportion"

    def __init__(self, dataset, features, target, rules, interval_fold, with_overlap=False):
        all_feature_columns = features + [target]
        dataset = dataset[all_feature_columns]

        self.folds = self.create_folds(dataset, rules[self.DATA_SET_PROPORTION_KEY], interval_fold, with_overlap, target)

        self.rules = rules
        self.process_rules()
    
    def create_folds(self, dataset, proportion, interval_fold, with_overlap, target):
        dataset_length = len(dataset)
        num_instances_per_partition = interval_fold*5
        arr_folds = []
        
        start_index = 0
        end_index = num_instances_per_partition

        if with_overlap:
            while end_index <= dataset_length:
                arr_folds.append(Fold(dataset[start_index:end_index], proportion, target))
                
                end_index += interval_fold
                start_index += interval_fold
            arr_folds.append(Fold(dataset[start_index:(dataset_length-1)], proportion, target))
        else:
            num_folds = dataset_length/num_instances_per_partition
            for num_fold in range(0, num_folds):
                arr_folds.append(Fold(dataset[start_index:end_index], proportion, target))

                end_index += num_instances_per_partition
                start_index += num_instances_per_partition
        return arr_folds
    
    def process_rules(self):
        process_dict = {
            "zscore" : self.zscore,
            "norm" : self.norm
        }

        for type_process in self.rules["global"]:
            process_dict[type_process](self.folds)
        

    def zscore(self, folds):
        print("Under development")
               
    def norm(self, folds):
        for fold in folds:
            train_stats = fold.data_train.describe().transpose()
            fold.train_stats = train_stats

            fold.data_train_scaled = (fold.data_train - train_stats['mean']) / train_stats['std']

            fold.data_validation_scaled = (fold.data_validation - train_stats['mean']) / train_stats['std']

            fold.data_test_scaled = (fold.data_test - train_stats['mean']) / train_stats['std']

    def inverse_norm(train_stats, scaled_data, target):
        return (scaled_data*train_stats['std'][target] + train_stats['mean'][target])