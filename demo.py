#from BBC_News.configuration.mongodb_connection import MongoDBClient

#ins = MongoDBClient()

#from BBC_News.configuration.aws_connection import S3Client

#ins = S3Client()

#from BBC_News.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig,ModelPusherConfig
#DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig

#from BBC_News.components.data_ingestion import DataIngestion
#from BBC_News.components.data_validation import DataValidation
#from BBC_News.components.data_transformation import DataTransformation
#from BBC_News.components.model_trainer import ModelTrainer
#from BBC_News.components.model_evaluation import ModelEvaluation
#from BBC_News.components.model_pusher import ModelPusher

#di_ins = DataIngestion(DataIngestionConfig)

#di_art = di_ins.initiate_data_ingestion()

#dv_ins = DataValidation(data_ingestion_artifact=di_art, data_validation_config=DataValidationConfig)

#dv_art = dv_ins.initiate_data_validation()

#dt_ins = DataTransformation(data_ingestion_artifact=di_art, data_transformation_config=DataTransformationConfig, data_validation_artifact=dv_art)

#dt_art = dt_ins.initiate_data_transformation()

#mt_ins = ModelTrainer(data_transformation_artifact=dt_art, model_trainer_config=ModelTrainerConfig)

#mt_art = mt_ins.initiate_model_trainer()

#me_ins = ModelEvaluation(model_eval_config=ModelEvaluationConfig, data_ingestion_artifact=di_art, model_trainer_artifact=mt_art)

#me_art = me_ins.initiate_model_evaluation()

#mp_ins = ModelPusher(me_art, ModelPusherConfig)

#mp_art = mp_ins.initiate_model_pusher()

#from BBC_News.pipeline.training_pipeline import TrainPipeline

#tr_pp = TrainPipeline()

#tr_pp.run_pipeline()



from BBC_News.pipeline.prediction_pipeline import BBCNewsData, BBCNewsClassifier

news_data = BBCNewsData(
                         text ="Arsenal manager Mikel Arteta has warned Liverpool that their lead at the top of the Premier League will be 'extremely difficult' to sustain - and believes his side can make up the nine-point gap to the leaders.The Gunners travel to West Ham, live on Sky Sports Saturday Night Football, knowing they could temporarily cut that deficit to six points, before Liverpool host reigning champions Manchester City on Super Sunday."
                         )
'''
news_data = BBCNewsData(
                        text= "Young people often spend long hours on social media despite mounting evidence that its not good for them. Health experts list harms as varied as lost sleep, eating disorders and suicide.In one of the boldest steps yet to address the problem, Australian lawmakers passed a bill on Nov. 29 to bar children under 16 from setting up accounts on social media sites including Facebook, Instagram, Snapchat and TikTok.In France, Norway and the UK, similar bans have either been proposed or discussed. In the US, a growing stack of lawsuits accuse the social media giants of knowingly hooking kids before they reach their teens, taking a cue from behavioral techniques used in the gambling and cigarette industries. If enough of them succeed, it could force the companies to change how they engage with younger audiences. "
                        )
'''
news_data_df = news_data.get_bbc_input_data_frame()

model_predictor = BBCNewsClassifier()

value = model_predictor.predict(dataframe=news_data_df)[0]

print(value)

