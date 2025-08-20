from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Annotated
from server.model_service import ModelService
import traceback
import time

async def get_model_service():
    model_service = ModelService('model/final/pretrained_model.pt')
    try:
        yield model_service
    except Exception  as e:
        print(f'An exception occured: {e}')

app = FastAPI(dependencies=[Depends(get_model_service)])

class GraphInput(BaseModel):
    particle_types : list
    positions : list

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.post('/predict')
def predict(data: GraphInput, model_service: Annotated[ModelService, Depends(get_model_service)]):
    try:
        particle_data, positions = model_service.preprocess_raw_data(data.particle_types, data.positions)
        t0 = time.perf_counter()
        pred = model_service.predict(particle_data, positions)
        t1 = time.perf_counter()
        print(f"Execution time: {t1 - t0}")
        return {"predicted_position" : pred.tolist()} 

    except Exception as e:
        print(f"While predicting exception occured:{e}")
        traceback.print_exc()