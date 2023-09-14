

from income_inequality.config.configuration import ConfigurationManager
from income_inequality.components.model_training import ModelTrainer
from income_inequality.logging import logger


STAGE_NAME = "Model Training Stage"


class ModelTrainingPipeline:
    def __init__(self):
        pass 

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train()


if __name__ == "__main__":
    try:
        logger.info(f">>> Stage {STAGE_NAME} Started <<<")
        obj = ModelTrainingPipeline()
        obj.main() 
        logger.info(f">>> Stage {STAGE_NAME} Completed <<<")
    except Exception as e:
        logger.exception(e)
        raise e