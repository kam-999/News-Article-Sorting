from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

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
            "BBCNews.html", {"request": request, "context": "Rendering"})

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

        return templates.TemplateResponse(
            "BBCNews.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
