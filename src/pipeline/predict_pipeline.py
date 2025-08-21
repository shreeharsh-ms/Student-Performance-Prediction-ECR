import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info("Loading preprocessor and model for prediction")
            preprocessor = load_object('artifacts/preprocessor.pkl')
            model = load_object('artifacts/model.pkl')

            logging.info("Transforming features using preprocessor")
            transformed_features = preprocessor.transform(features)

            logging.info("Making predictions using the model")
            predictions = model.predict(transformed_features)

            return predictions
        except Exception as e:
            raise CustomException(e, sys) from e

class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: float,
                 writing_score: float,
               ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            logging.info(f"Custom data input dictionary: {custom_data_input_dict}")
            df = pd.DataFrame(custom_data_input_dict)
            logging.info(f"Custom data DataFrame: {df}")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e
                 

