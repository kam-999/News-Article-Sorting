import sys
from typing import Tuple

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from BBC_News.exceptions import BBCException
from BBC_News.logger import logging
from BBC_News.utils.main_utils import load_numpy_array_data, load_object, save_object
from BBC_News.entity.config_entity import ModelTrainerConfig
from BBC_News.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from BBC_News.entity.estimator import BBCModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def evaluate_models(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Evaluate models using cross-validation and select the best one based on accuracy.
        """
        try:
            logging.info("Evaluating models using cross-validation.")
            models = {
                "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
                "SVC": SVC(kernel="linear", random_state=42),
                "MultinomialNB": MultinomialNB(),
            }
            
            best_model = None
            best_score = 0.0
            best_model_name = None
            
            for name, model in models.items():
                logging.info(f"Evaluating {name}")
                scores = cross_val_score(model, x_train, y_train, cv=5, scoring="accuracy")
                mean_score = scores.mean()
                logging.info(f"{name} Accuracy: {mean_score:.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    best_model_name = name
            
            logging.info(f"Best model: {best_model_name} with accuracy: {best_score:.4f}")
            return best_model, best_score

        except Exception as e:
            raise BBCException(e, sys) from e

    def train_and_evaluate(self, train_arr: np.ndarray, test_arr: np.ndarray) -> ClassificationMetricArtifact:
        """
        Train the best model and evaluate it on the test dataset.
        """
        try:
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            best_model, best_score = self.evaluate_models(x_train, y_train)
            
            if best_score < self.model_trainer_config.expected_accuracy:
                raise Exception("No model meets the expected accuracy threshold.")

            logging.info("Training the best model on the entire training dataset.")
            best_model.fit(x_train, y_train)

            logging.info("Predicting on the test dataset.")
            y_pred = best_model.predict(x_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")

            logging.info(f"Test Metrics - Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}")
            metric_artifact = ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall)

            return best_model, metric_artifact

        except Exception as e:
            raise BBCException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Initiates the model training process.
        """
        try:
            logging.info("Loading transformed train and test arrays.")
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            best_model, metric_artifact = self.train_and_evaluate(train_arr, test_arr)

            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            #if best_model.best_score < self.model_trainer_config.expected_accuracy:
                #logging.info("No best model found with score more than base score")
                #raise Exception("No best model found with score more than base score")

            #best_model, metric_artifact = self.train_and_evaluate(train_arr, test_arr)
            BBC_Model = BBCModel(preprocessing_object=preprocessing_obj,
                                       trained_model_object=best_model)

            logging.info("Saving the best trained model.")
            save_object(self.model_trainer_config.trained_model_file_path, BBC_Model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise BBCException(e, sys) from e