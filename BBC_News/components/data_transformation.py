import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
#from imblearn.combine import SMOTEENN
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from BBC_News.constants import TARGET_COLUMN
from BBC_News.entity.config_entity import DataTransformationConfig
from BBC_News.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from BBC_News.exceptions import BBCException
from BBC_News.logger import logging
from BBC_News.utils.main_utils import save_object, save_numpy_array_data
from BBC_News.entity.estimator import TargetValueMapping

from sklearn.base import BaseEstimator, TransformerMixin

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    A custom transformer for text preprocessing.
    This removes stopwords, applies stemming, and lemmatization to the input text.
    """
    def __init__(self, use_stemming=False, use_lemmatization=True, stop_words=None):
        """
        :param use_stemming: Boolean flag to apply stemming
        :param use_lemmatization: Boolean flag to apply lemmatization
        :param stop_words: List of stopwords to use (default is NLTK's English stopwords)
        """
        try:
            
            nltk_data_dir = "/app/nltk_data"
            os.makedirs(nltk_data_dir, exist_ok=True)
            os.environ["NLTK_DATA"] = nltk_data_dir
            nltk.data.path.append(nltk_data_dir)
            
            # Download required NLTK data
            nltk.download("stopwords", download_dir=nltk_data_dir)
            nltk.download("wordnet", download_dir=nltk_data_dir)
            nltk.download("omw-1.4", download_dir=nltk_data_dir)
            nltk.download("averaged_perceptron_tagger_eng", download_dir=nltk_data_dir)
            '''
            nltk.download('stopwords')  # Ensure stopwords are available
            nltk.download('wordnet')    # Ensure WordNet is available for lemmatization
            nltk.download('omw-1.4')    # Download for WordNet lemmatizer support
            nltk.download('averaged_perceptron_tagger')
            '''
            self.stop_words = stop_words or set(stopwords.words('english'))
            self.use_stemming = use_stemming
            self.use_lemmatization = use_lemmatization
            
            if use_stemming:
                self.stemmer = PorterStemmer()
            if use_lemmatization:
                self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            raise BBCException(e, sys)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            # Apply preprocessing to each text entry in the series
            return X.apply(self._preprocess_text)
        except Exception as e:
            raise BBCException(e, sys)

    def _preprocess_text(self, text):
        # Tokenize, remove stopwords, and apply stemming or lemmatization
        tokens = text.split()
        filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]

        if self.use_stemming:
            processed_tokens = [self.stemmer.stem(word) for word in filtered_tokens]
        elif self.use_lemmatization:
            processed_tokens = [self.lemmatizer.lemmatize(word, self._get_wordnet_pos(word)) for word in filtered_tokens]
        else:
            processed_tokens = filtered_tokens

        return ' '.join(processed_tokens)

    def _get_wordnet_pos(self, word):
        """
        Map POS tag to the format accepted by WordNetLemmatizer
        """
        from nltk import pos_tag
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)  # Default to noun if POS tag is unavailable


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact

            logging.info("Downloading NLTK stopwords if not already present.")
            nltk.download('stopwords')

            
        except Exception as e:
            raise BBCException(e, sys)



    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer pipeline object.
        """
        try:
            logging.info("Initializing text preprocessing and TF-IDF Vectorizer pipeline")
            
            # Instantiate the components of the pipeline
            text_preprocessor = TextPreprocessor(use_stemming=False, use_lemmatization=True)
            tfidf_vectorizer = TfidfVectorizer(
                sublinear_tf=True, max_features=5000, min_df=5,
                norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english'
            )
            
            # Create the pipeline
            preprocessor = Pipeline(steps=[
                ('text_preprocessor', text_preprocessor),
                ('tfidf', tfidf_vectorizer)
            ])
            
            return preprocessor
        except Exception as e:
            raise BBCException(e, sys)

    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline
        
        Output      :   Data transformer steps are performed and preprocessor object is created
        On Failure  :   Raise an exception
        """
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")
                preprocessor = self.get_data_transformer_object()

                train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
                test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

                #train_df['Text'] = train_df['Text'].apply(self.preprocess_text)
                #test_df['Text'] = test_df['Text'].apply(self.preprocess_text)

                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]

                logging.info("Encoding train target variables")
                target_feature_train_df = target_feature_train_df.replace(
                    TargetValueMapping()._asdict()
                )

                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]

                logging.info("Encoding test target variables")
                target_feature_test_df = target_feature_test_df.replace(
                TargetValueMapping()._asdict()
                )

                #target_feature_train_df = self.encode_target(target_feature_train_df)
                #target_feature_test_df = self.encode_target(target_feature_test_df)

                logging.info("Transforming input features using preprocessor")
                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df['Text'])
                input_feature_test_arr = preprocessor.transform(input_feature_test_df['Text'])

                '''logging.info("Applying SMOTEENN for handling class imbalance")
                smt = SMOTEENN(sampling_strategy="minority")

                input_feature_train_final, target_feature_train_final = smt.fit_resample(
                    input_feature_train_arr, target_feature_train_df
                )
                input_feature_test_final, target_feature_test_final = smt.fit_resample(
                    input_feature_test_arr, target_feature_test_df
                )'''

                input_feature_train_final = input_feature_train_arr.toarray()
                input_feature_test_final = input_feature_test_arr.toarray()

               

                train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_df)]
                test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_df)]

                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
                return data_transformation_artifact

            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise BBCException(e, sys) from e
