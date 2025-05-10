from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
import sys
from src.entity.config_entity import DataIngestionConfig
from src.entity.config_entity import TrainingPipelineConfig
from src.entity.config_entity import DataValidationConfig
from src.entity.config_entity import DataTransformationConfig

if __name__=="__main__":
    try:
       training_pipeline_config=TrainingPipelineConfig()
       data_ingestion_config=DataIngestionConfig(training_pipeline_config) 
       data_ingestion=DataIngestion(data_ingestion_config)
       logging.info("Initiated Data Ingestion Stage")
       data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
       logging.info("Data Ingestion Completed")
       print(data_ingestion_artifact)
       data_validation_config=DataValidationConfig(training_pipeline_config)
       data_validation=DataValidation(data_ingestion_artifact,data_validation_config)
       logging.info("Initiated Data Validation Stage")
       data_validation_artifact=data_validation.initiate_data_validation()
       logging.info("Data Validation Completed")
       print(data_validation_artifact)
       data_transformation_config=DataTransformationConfig(training_pipeline_config)
       data_transformation=DataTransformation(data_validation_artifact,data_transformation_config)
       logging.info("Initiated Data Transformation Stage")
       data_transformation_artifact=data_transformation.initiate_data_transformation()
       logging.info("Data Transformation Completed")
       print(data_transformation_artifact)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
