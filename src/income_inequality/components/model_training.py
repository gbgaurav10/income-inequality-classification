
import pandas as pd
import os 
from income_inequality.logging import logger 
from sklearn.ensemble import ExtraTreesClassifier
import joblib
from income_inequality.entity.config_entity import ModelTrainingConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config 

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        X_train = train_data.drop([self.config.target_column], axis=1)
        X_test = test_data.drop([self.config.target_column], axis=1)
        y_train = train_data[[self.config.target_column]]
        y_test = test_data[[self.config.target_column]]


        extree = ExtraTreesClassifier(n_estimators=self.config.n_estimators, max_depth=self.config.max_depth,
                                            criterion=self.config.criterion)
        extree.fit(X_train, y_train)

        joblib.dump(extree, os.path.join(self.config.root_dir, self.config.model_name))