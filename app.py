import os
import sys
import certifi
from dotenv import load_dotenv
import pymongo
from src.exception.exception import NetworkSecurityException
from src.utils.main_utils.utils import load_object
from src.logging.logger import logging
from src.pipeline.train_pipeline import TrainingPipeline
from src.utils.ml_utils.model.estimator import NetworkModel
from src.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME, DATA_INGESTION_COLLECTION_NAME


from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd
import numpy as np  
from fastapi.templating import Jinja2Templates
templates=Jinja2Templates(directory="./templates")

load_dotenv()
ca=certifi.where()
mongo_db_url=os.getenv("MONGO_DB_URL")

client=pymongo.MongoClient(mongo_db_url,tlsCAFile=ca)


database=client[DATA_INGESTION_DATABASE_NAME]
collection=database[DATA_INGESTION_COLLECTION_NAME]

app=FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/",tags=["authentication"])
async def root():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e
    
@app.post("/predict")
async def predict(request:Request, file: UploadFile=File(...)):
    try:
        df=pd.read_csv(file.file)
        preprocessor=load_object("final_models/preprocessor.pkl")
        model=load_object("final_models/model.pkl")
        network_model=NetworkModel(preprocessor=preprocessor,model=model)
        print(df.iloc[0])
        y_pred=network_model.predict(df)
        print(y_pred)
        df["predicted_column"]=y_pred
        print(df["predicted_column"])

        df.to_csv("./prediction_output/output.csv")
        table_html=df.to_html(classes="table table-striped")
        return templates.TemplateResponse("data.html", {"request": request, "table_html": table_html})
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e
    

if __name__=="__main__":
    app_run(app,host="0.0.0.0",port=8000,debug=True)





