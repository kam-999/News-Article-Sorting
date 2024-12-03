import json
import sys

import pandas as pd
from pandas import DataFrame

from BBC_News.exceptions import BBCException
from BBC_News.logger import logging
from BBC_News.utils.main_utils import read_yaml_file, write_yaml_file
from BBC_News.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from BBC_News.entity.config_entity import DataValidationConfig
from BBC_News.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_validation_config: configuration for data validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise BBCException(e, sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            raise BBCException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Method Name :   is_column_exist
        Description :   This method validates the existence of numerical and categorical columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)
            
            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")

            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns) > 0:
                logging.info(f"Missing categorical columns: {missing_categorical_columns}")

            return False if missing_numerical_columns or missing_categorical_columns else True
        except Exception as e:
            raise BBCException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise BBCException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline
        
        Output      :   Returns DataValidationArtifact based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            validation_error_msg = ""
            logging.info("Starting data validation")

            # Read the training and testing data
            train_df, test_df = (
                DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path),
                DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path),
            )

            # Validate number of columns for training data
            status = self.validate_number_of_columns(dataframe=train_df)
            logging.info(f"All required columns present in training dataframe: {status}")
            if not status:
                validation_error_msg += "Columns are missing in training dataframe."

            # Validate number of columns for testing data
            status = self.validate_number_of_columns(dataframe=test_df)
            logging.info(f"All required columns present in testing dataframe: {status}")
            if not status:
                validation_error_msg += " Columns are missing in test dataframe."

            # Validate column existence for training data
            status = self.is_column_exist(df=train_df)
            if not status:
                validation_error_msg += " Columns are missing in training dataframe."

            # Validate column existence for testing data
            status = self.is_column_exist(df=test_df)
            if not status:
                validation_error_msg += " Columns are missing in test dataframe."

            # Determine if validation passed
            validation_status = len(validation_error_msg) == 0

            # Create a DataValidationArtifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message="Validation Passed" if validation_status else validation_error_msg,
                #drift_report_file_path=None  # Removed data drift component
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise BBCException(e, sys) from e

