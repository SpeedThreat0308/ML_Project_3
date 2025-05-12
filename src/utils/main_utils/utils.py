import os
import sys
import yaml
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
import pandas as ps
import numpy as np
import dill
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def read_yaml_file(file_path: str)->dict:
    try:
        with open(file_path,"rb") as file_obj:
            return yaml.safe_load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e,sys)

def write_yaml_file(file_path: str, content: object, replace: bool = False)->None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file_obj:
            yaml.dump(content, file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def save_numpy_array(file_path: str, array: np.array):
    '''
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    '''

    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def load_numpy_array(file_path: str)->np.array:
    '''
    Load numpy array data from file
    file_path: str location of file to load
    return: np.array loaded data
    '''
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def load_object(file_path: str)->object:
    try:
        logging.info(f"Loading object from {file_path}")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def save_object(file_path: str, obj: object)->None:
    try:
        logging.info(f"Saving object to {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
        logging.info("Exited save_object method")
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def evaluate_models(X_train,X_test,Y_train,Y_test,models,params):
    try:
        report={}

        for i in range(len(list(models))):
            model_name=list(models.values())[i]
            para=params[list(models.keys())[i]]

            gs=GridSearchCV(estimator=model_name,param_grid=para,cv=3,n_jobs=-1,verbose=2)
            gs.fit(X_train,Y_train)

            model_name.set_params(**gs.best_params_)
            model_name.fit(X_train,Y_train)
            
            y_train_pred=model_name.predict(X_train)

            y_test_pred=model_name.predict(X_test)

            train_model_score=r2_score(Y_train,y_train_pred)

            test_model_score=r2_score(Y_test,y_test_pred)

            report[list(models.keys())[i]]=test_model_score

        return report

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e