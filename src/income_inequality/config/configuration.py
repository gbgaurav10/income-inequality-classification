
import os, sys
from income_inequality.constants import *
from income_inequality.utils.common import read_yaml, create_directories
from income_inequality.entity.config_entity import (DataIngestionConfig,
                                                    DataTransformationConfig,
                                                    ModelTrainingConfig,)


class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH,
            schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion 

        create_directories([config.root_dir])


        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_url = config.source_url,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )

        return data_ingestion_config


    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation 

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir = config.root_dir,
            data_path = config.data_path,
            preprocessor_path = config.preprocessor_path
        )


        return data_transformation_config


    def get_model_trainer_config(self) -> ModelTrainingConfig:
        config = self.config.model_trainer
        params = self.params.ExtraTreesClassifier
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

    
        model_trainer_config = ModelTrainingConfig(
            root_dir = config.root_dir,
            train_data_path = config.train_data_path,
            test_data_path = config.test_data_path,
            model_name = config.model_name,
            n_estimators = params.n_estimators,
            criterion = params.criterion,
            max_depth = params.max_depth,
            target_column = schema.name 
        )

        return model_trainer_config