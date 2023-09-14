
import os, sys 
import pandas as pd 
from sklearn.metrics import f1_score, accuracy_score
from urllib.parse import urlparse
import json 
import numpy as np 
import joblib 
from income_inequality.constants import * 
from income_inequality.utils.common import read_yaml, create_directories, save_json
from income_inequality.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config


    def evaluation_metrics(self, actual, predicted):
        f1 = f1_score(actual, predicted, average="weighted")
        acc = accuracy_score(actual, predicted)

        return f1, acc


    def log_metrics(self, f1, acc):
        scores = {
            "f1_score": f1,
            "accuracy_score": acc
        }


        with open(self.config.metric_file_name, "w") as f:
            json.dump(scores, f)


    def evaluate_model(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        X_test = test_data.drop(self.config.target_column, axis=1)
        y_test = test_data[[self.config.target_column]]

        predictions = model.predict(X_test)

        score_f1, accuracy = self.evaluation_metrics(y_test, predictions)

        self.log_metrics(score_f1, accuracy)