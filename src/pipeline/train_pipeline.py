import os
import sys

from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.cloud.s3_syncer import S3Sync

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.data_validation import DataValidation
from src.constant.training_pipeline import TRAINING_BUCKET_NAME
from src.constant.training_pipeline import SAVED_MODEL_DIR

from src.entity.config_entity import ModelTrainerConfig, DataIngestionConfig, DataTransformationConfig, DataValidationConfig, TrainingPipelineConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, DataIngestionArtifact, DataValidationArtifact


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync=S3Sync()

    def start_data_ingestion(self):
        try:
            data_ingestion_config=DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start Data Ingestion")
            data_ingestion=DataIngestion(dataingestionconfig=data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            logging.info("Data Ingestion Completed")
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def start_data_validation(self, data_ingestion_artifact= DataIngestionArtifact):
        try:
            data_validation_config=DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start Data Validation")
            data_validation=DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
            data_validation_artifact=data_validation.initiate_data_validation()
            logging.info("Data Validation Completed")
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact):
        try:
            data_transformation_config=DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start Data Transformation")
            data_transformation=DataTransformation(data_transformation_config=data_transformation_config, data_validation_artifact=data_validation_artifact)
            data_transformation_artifact=data_transformation.initiate_data_transformation()
            logging.info("Data Transformation Completed")
            return data_transformation_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact):
        try:
            model_trainer_config=ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start Model Trainer")
            model_trainer=ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact=model_trainer.initiate_model_trainer()
            logging.info("Model Trainer Completed")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url=f"s3://{TRAINING_BUCKET_NAME}/Artifacts/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(self.training_pipeline_config.artifact_dir, aws_bucket_url)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def saved_model_dir_to_s3(self):
        try:
            aws_bucket_url=f"s3://{TRAINING_BUCKET_NAME}/final_models/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(self.training_pipeline_config.model_dir, aws_bucket_url)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    
    def run_pipeline(self):
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact=self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact=self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            logging.info("Training Pipeline Completed")

            self.sync_artifact_dir_to_s3()
            self.saved_model_dir_to_s3()
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)