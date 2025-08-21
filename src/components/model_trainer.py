import os
import sys
from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass
from catboost import CatBoostRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor  
from xgboost import XGBRegressor
from src.utils import save_object, evaluate_model
from src.components.Hypertuning.params import params

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        logging.info("Initiating model training")
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "LinearRegression": LinearRegression(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=0),
                "XGBRegressor": XGBRegressor()
            }
            logging.info("Models initialized for training")
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models, params)

            logging.info(f"Model evaluation report: {model_report}")
            best_model_name = max(model_report, key=model_report.get)
            best_score = model_report[best_model_name]

            best_model = models[best_model_name]

            if best_score < 0.6:
                raise CustomException("No suitable model found", sys)
            
            logging.info(f"Best model found: {best_model_name} with R2 score: {best_score}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f"R2 score of the best model: {r2_square}")
            
            return r2_square

        except Exception as e:
            raise CustomException(e, sys) from e


