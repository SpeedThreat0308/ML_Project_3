import os
import sys
import pandas as pd 
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging

from src.constant.training_pipeline import TARGET_COLUMN
from src.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from src.entity.artifact_entity import DataTransformationArtifact,DataValidationArtifact

from src.entity.config_entity import DataTransformationConfig
from src.utils.main_utils.utils import save_numpy_array,save_object

class DataTransformation:
    def __init__(self,data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact=data_validation_artifact
            self.data_transformation_config=data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            df=pd.read_csv(file_path)
            return df
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e
        
    def get_data_transformation_object(cls)->Pipeline:
        logging.info("Entered get_data_transformation_object method")
        try:
            imputer=KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            processor: Pipeline=Pipeline(
                [
                    ("imputer", imputer)
                ]
            )
            return processor
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info("Entered Data Transformation method")
            train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            ##training dataframe
            input_feature_train_df=train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df=train_df[TARGET_COLUMN]
            target_feature_train_df=target_feature_train_df.replace(-1,0)

            ##testing dataframe
            input_feature_test_df=test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df=test_df[TARGET_COLUMN]
            target_feature_test_df=target_feature_test_df.replace(-1,0)

            preprocessor_obj=self.get_data_transformation_object()

            preprocessor_obj.fit(input_feature_train_df)
            input_feature_train_array=preprocessor_obj.transform(input_feature_train_df)

            preprocessor_obj.fit(input_feature_test_df)
            input_feature_test_array=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_array, np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_array, np.array(target_feature_test_df)]

            ##save numpy array data
            save_numpy_array(self.data_transformation_config.data_transform_train_file_path, array=train_arr)
            save_numpy_array(self.data_transformation_config.data_transform_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_obj)


            ##preparing artifacts
            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.data_transform_train_file_path,
                transformed_test_file_path=self.data_transformation_config.data_transform_test_file_path
            )

            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e,sys) from e