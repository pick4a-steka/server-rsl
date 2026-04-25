from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
import uvicorn

from app.common.gru_model import GRUModel

app = FastAPI()

LABELS = {
    0: "НЕТ",
    1: "ДА",
    2: "Я"
}

model_gru = GRUModel(
    input_size=132,
    hidden_size=128,
    num_layers=2,
    num_classes=3
)

# Загрузка модели GRU
state_dict = torch.load(
    "/home/mihal/projects/server-rsl/app/models/model_18042026_210405_10",
    map_location=torch.device('cpu'))

model_gru.load_state_dict(state_dict)
model_gru.eval()

# Структура запроса от клиента
class Item(BaseModel):
    tensor: List[List[float]]  # [12 x 132]

# Структура ответа от сервера
class Response(BaseModel):
    label: str
    confidence: float

@app.post("/recognize")
async def recognize(item: Item):
    tensor =  np.array(item.tensor, dtype=np.float32)
    print("Полученный тензор имеет форму:", tensor.shape)

    x = torch.tensor(tensor).unsqueeze(0)

    with torch.no_grad():
        output = model_gru(x)
        probabolities = F.softmax(output, dim=1)

        confidence, predicted_class = torch.max(probabolities, dim=1)

    label_id = predicted_class.item()
    confidence_value = confidence.item()

    label = LABELS.get(label_id, f"НЕИЗВЕСТНО: {label_id}")
    print(f"Предсказанный класс: {label} (ID: {label_id}), уверенность: {confidence_value:.4f}")
    
    return Response(
        label=label,
        confidence=confidence_value
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)