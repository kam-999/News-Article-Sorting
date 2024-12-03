import os
import sys

import numpy as np
import pandas as pd
from BBC_News.entity.config_entity import BBCPredictorConfig
from BBC_News.entity.s3_estimator import BBCEstimator
from BBC_News.exceptions import BBCException
from BBC_News.logger import logging
from BBC_News.utils.main_utils import read_yaml_file
from pandas import DataFrame


class BBCNewsData:
    def __init__(self,
                text
                ):
        """
        BBCNews Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.text = text


        except Exception as e:
            raise BBCException(e, sys) from e

    def get_bbc_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from BBCNewsData class input
        """
        try:
            
            bbc_input_dict = self.get_bbc_data_as_dict()
            return DataFrame(bbc_input_dict)
        
        except Exception as e:
            raise BBCException(e, sys) from e


    def get_bbc_data_as_dict(self):
        """
        This function returns a dictionary from BBCNewsData class input 
        """
        logging.info("Entered get_bbc_data_as_dict method as BBCData class")

        try:
            input_data = {
                "Text": [self.text],
                
            }

            logging.info("Created bbc data dict")

            logging.info("Exited get_bbc_data_as_dict method as BBCNewsData class")

            return input_data

        except Exception as e:
            raise BBCException(e, sys) from e

class BBCNewsClassifier:
    def __init__(self,prediction_pipeline_config: BBCPredictorConfig = BBCPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            # self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise BBCException(e, sys)

    def predict(self, dataframe) -> str:
        """
        This is the method of BBCNewsClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of BBCNewsClassifier class")
            model = BBCEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise BBCException(e, sys)