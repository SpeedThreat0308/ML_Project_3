import os
import sys
import pandas as pd
import numpy as np
import mlflow
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact

from src.utils.main_utils.utils import load_object, save_object, load_numpy_array
from src.utils.ml_utils.model.estimator import NetworkModel
from src.utils.ml_utils.metric.classification_metric import get_classification_score


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
from src.utils.main_utils.utils import evaluate_models
import dagshub
dagshub.init(repo_owner='SpeedThreat0308', repo_name='ML_Project_3', mlflow=True)


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
             raise NetworkSecurityException(e,sys)
    
    def track_mlflow(self,model,classification_score):
        with mlflow.start_run():    
            f1_score=classification_score.f1_score
            precision_score=classification_score.precision_score
            recall_score=classification_score.recall_score

            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precision_score",precision_score)
            mlflow.log_metric("recall_score",recall_score)
            mlflow.sklearn.log_model(model,"model")

    def train_model(self,x_train,y_train,x_test,y_test):
        try:
            models={
                "RandomForestClassifier":RandomForestClassifier(verbose=1),
                "LogisticRegression":LogisticRegression(verbose=1),
                "GradientBoostingClassifier":GradientBoostingClassifier(verbose=1),
                "KNeighborsClassifier":KNeighborsClassifier(),
                "DecisionTreeClassifier":DecisionTreeClassifier(),
                "AdaBoostClassifier":AdaBoostClassifier()
            }

            params={
                "DecisionTreeClassifier":{
                    "criterion":["gini","entropy","log_loss"],
                },
                "RandomForestClassifier":{
                    "criterion": ['gini','entropy','log_loss'],
                    "n_estimators": [8,16,32,64,128,256]
                },
                "LogisticRegression":{},
                "KNeighborsClassifier":{},
                "GradientBoostingClassifier":{
                    "learning_rate": [0.1,.01,.05,.001],
                    "subsample": [0.6,0.7,0.75,0.8,0.85,0.9],
                    "n_estimators": [8,16,32,64,128,256]
                },
                "AdaBoostClassifier":{
                    "learning_rate": [0.1,.01,.05,.001],
                    "n_estimators": [8,16,32,64,128,256]
                }
            }
            model_report:dict=evaluate_models(X_train=x_train,X_test=x_test,Y_train=y_train,Y_test=y_test,params=params,models=models)

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            y_train_pred=best_model.predict(x_train)

            classification_score=get_classification_score(y_true=y_train,y_pred=y_train_pred)

            ##Track the MLFLOW
            self.track_mlflow(best_model,classification_score)

            y_test_pred=best_model.predict(x_test)
            classification_score_test=get_classification_score(y_true=y_test,y_pred=y_test_pred)
            self.track_mlflow(best_model,classification_score)

            preprocessor=load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            model_dir_path=os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)

            Network_models=NetworkModel(
                preprocessor=preprocessor,
                model=best_model,
            )
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=Network_models)

            ##Model trainer artifact

            model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                 train_metric_artifact=classification_score,
                                 test_metric_artifact=classification_score_test
                                 )
            logging.info("Model trainer artifact created")
            return model_trainer_artifact


        except Exception as e:
            raise NetworkSecurityException(e,sys) from e

        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path=self.data_transformation_artifact.transformed_train_file_path
            test_file_path=self.data_transformation_artifact.transformed_test_file_path

            train_arr=load_numpy_array(file_path=train_file_path)
            test_arr=load_numpy_array(file_path=test_file_path)

            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            model=self.train_model(x_train,y_train,x_test,y_test)
            return model
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e
