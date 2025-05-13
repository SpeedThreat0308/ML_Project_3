import os
import sys
import certifi
from dotenv import load_dotenv
import pymongo
from src.exception.exception import NetworkSecurityException
from src.utils.main_utils.utils import load_object
from src.logging.logger import logging
from src.pipeline.train_pipeline import TrainingPipeline
from src.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME, DATA_INGESTION_COLLECTION_NAME


from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd
import numpy as np  

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
    

if __name__=="__main__":
    app_run(app,host="localhost",port=8000,debug=True)





