import torch
import torch.nn.functional as F
import numpy as np
from typing import List
from pathlib import Path

from app.common.gru_model import GRUModel
from app.common.config import Request, Response, LABELS

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "model_18042026_210405_10"

 # загружаем модель распознавания
model_gru = GRUModel(
    input_size=132,
    hidden_size=128,
    num_layers=2,
    num_classes=3
)
state_dict = torch.load(
    MODEL_PATH,
    map_location=torch.device('cpu'))
model_gru.load_state_dict(state_dict)
model_gru.eval()

def run_inference(item: List[List[float]]) -> dict:
    '''
    Функция для запуска инференса на входящем тензоре
    '''
    tensor = np.array(item, dtype=np.float32)
    input_tensor = torch.tensor(tensor).unsqueeze(0)

    with torch.no_grad():
        output = model_gru(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
    
    label_id = predicted_class.item()
    confidence_value = confidence.item()
    label = LABELS.get(label_id, f"НЕИЗВЕСТНО: {label_id}")
    status = True

    if (confidence_value < 0.5):
        label = "ЖЕСТ НЕ РАСПОЗНАН"
        status = False
    
    print(f"Предсказанный класс: {label} (ID: {label_id}), уверенность: {confidence_value:.4f}")

    return {
        "status": status,
        "label": label
    }
