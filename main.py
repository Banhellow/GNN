from fastapi import FastAPI, HTTPException, Depends, WebSocket
from pydantic import BaseModel
from typing import Annotated
from server.model_service import ModelService
import traceback
import time
import json, asyncio
import torch

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

class WebSocketGraphInput(BaseModel):
    type : str
    particle_types : list
    positions : list

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


@app.websocket("/ws")
async def ws_endpoint(ws : WebSocket, model_service: Annotated[ModelService, Depends(get_model_service)]):
    try:
        print("websocket started")
        await ws.accept()
        raw = await ws.receive_text()
        init_message = json.loads(raw)
        await handle_init(init_message, model_service)
        while True:
            print("websocket executed")
            await send_predict_message(ws, model_service)
    except Exception as e:
        print(f"While predicting exception occured:{e}")
        traceback.print_exc()


@app.post('/predict')
def predict(data: GraphInput, model_service: Annotated[ModelService, Depends(get_model_service)]):
    try:
        particle_data, positions = model_service.preprocess_raw_data(data.particle_types, data.positions)
        print("Masses shape:", particle_data.shape)
        print("Positions shape:", positions.shape)
        t0 = time.perf_counter()
        pred = model_service.predict(particle_data, positions)
        t1 = time.perf_counter()
        print(f"Execution time: {t1 - t0}")
        return {"predicted_position" : pred.tolist()} 

    except Exception as e:
        print(f"While predicting exception occured:{e}")
        traceback.print_exc()


async def send(ws : WebSocket, obj: dict):
    await ws.send_text(json.dumps(obj, separators=(',', ':')))

async def handle_init(msg: dict, model_service: ModelService):
    particle_data, positions = model_service.preprocess_raw_data(msg['particle_types'], msg['positions'])
    model_service.masses = particle_data
    model_service.working_window = positions.permute(1,0,2)
    print(f"Masses:{model_service.masses}, positions: {model_service.working_window}")

async def send_predict_message(ws: WebSocket, model_service: ModelService):
        pred = model_service.predict(model_service.masses, model_service.working_window)
        model_service.working_window = torch.cat([model_service.working_window[:, 1:], pred.unsqueeze(1)], dim = 1)
        print(pred)
        await send(ws, {"predicted_position" : pred.tolist()})