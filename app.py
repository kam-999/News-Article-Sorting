from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse
from uvicorn import run as app_run

from typing import Optional
import os

from BBC_News.constants import APP_HOST, APP_PORT
from BBC_News.pipeline.prediction_pipeline import BBCNewsData, BBCNewsClassifier
from BBC_News.pipeline.training_pipeline import TrainPipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

log_dir = 'logs'

class ArticleForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.text_article: Optional[str] = None

    async def get_article_data(self):
        form = await self.request.form()
        self.text_article = form.get("text_article")

@app.get("/", tags=["authentication"])
async def index(request: Request):
    return templates.TemplateResponse(
        "BBCNews.html", {"request": request, "context": "Rendering", "logs": ""}
    )

@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()

        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = ArticleForm(request)
        await form.get_article_data()

        text_data = BBCNewsData(
            text=form.text_article
        )

        text_df = text_data.get_bbc_input_data_frame()

        model_predictor = BBCNewsClassifier()

        value = model_predictor.predict(dataframe=text_df)[0]

        # Category mapping
        category_mapping = {
            0: "Sport",
            1: "Business",
            2: "Entertainment",
            3: "Politics",
            4: "Tech"
        }

        status = category_mapping.get(value, "Unknown")

        # Get latest log content
        log_file = max(
            (os.path.join(log_dir, f) for f in os.listdir(log_dir)), 
            key=os.path.getctime
        )
        with open(log_file, 'r') as file:
            logs = file.read()

        return templates.TemplateResponse(
            "BBCNews.html",
            {"request": request, "context": status, "logs": logs},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}

@app.get("/logs", tags=["logs"])
async def get_logs():
    try:
        log_file = max(
            (os.path.join(log_dir, f) for f in os.listdir(log_dir)), 
            key=os.path.getctime
        )  # Get the latest log file

        with open(log_file, 'r') as file:
            log_content = file.read()

        return {"status": True, "logs": log_content}

    except Exception as e:
        return {"status": False, "error": str(e)}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
