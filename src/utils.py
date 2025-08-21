import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path: str, obj: object):
    """
    Save an object to a file using dill.
    
    :param file_path: Path where the object will be saved.
    :param obj: The object to save.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e


def evaluate_model(x_train, y_train, x_test, y_test, models, params):
    try:
        print("Parameters for hyperparameter tuning:", params)
        if not isinstance(params, dict):
            params = {}

        model_report = {}
        print("Evaluating models...")

        for model_name, model in models.items():
            para = params.get(model_name)

            if para is not None and len(para) > 0:
                # Run tuning to find best params
                gs = GridSearchCV(model, para, cv=3)
                gs.fit(x_train, y_train)
                
                # Set the best params on the original model and fit it
                model.set_params(**gs.best_params_)
                model.fit(x_train, y_train)
                
                print(f"Best params for {model_name}: {gs.best_params_}")
                
                y_train_pred = model.predict(x_train)
                y_test_pred = model.predict(x_test)
            else:
                # No tuning
                model.fit(x_train, y_train)
                y_train_pred = model.predict(x_train)
                y_test_pred = model.predict(x_test)

            # Score calculation
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            model_report[model_name] = test_model_score

            logging.info(
                f"{model_name} - Train R2 Score: {train_model_score}, Test R2 Score: {test_model_score}"
            )

        return model_report

    except Exception as e:
        raise CustomException(e, sys) from e
    
def load_object(file_path: str):
    """
    Load an object from a file using dill.
    
    :param file_path: Path from where the object will be loaded.
    :return: The loaded object.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            obj = dill.load(file_obj)
        logging.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys) from e
