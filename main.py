from src.components.data_ingestion import DataIngestion
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
import sys
from src.entity.config_entity import DataIngestionConfig
from src.entity.config_entity import TrainingPipelineConfig

if __name__=="__main__":
    try:
       training_pipeline_config=TrainingPipelineConfig()
       data_ingestion_config=DataIngestionConfig(training_pipeline_config) 
       data_ingestion=DataIngestion(data_ingestion_config)
       logging.info("Initiated Data Ingestion Stage")
       data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
       print(data_ingestion_artifact)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
