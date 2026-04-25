import torch
import torch.nn.functional as F
import numpy as np

from app.common.gru_model import GRUModel
from app.api.config import Request, Response, LABELS

def run_inference(item: Request) -> Response:
    '''
    Функция для запуска инференса на входящем тензоре
    '''
    # загружаем модель распознавания
    model_gru = GRUModel(
        input_size=132,
        hidden_size=128,
        num_layers=2,
        num_classes=3
    )
    state_dict = torch.load(
        "/home/mihal/projects/server-rsl/app/models/model_18042026_210405_10",
        map_location=torch.device('cpu'))
    model_gru.load_state_dict(state_dict)
    model_gru.eval()


    tensor = np.array(item.tensor, dtype=np.float32)
    input_tensor = torch.tensor(tensor).unsqueeze(0)

    with torch.no_grad():
        output = model_gru(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
    
    label_id = predicted_class.item()
    confidence_value = confidence.item()
    label = LABELS.get(label_id, f"НЕИЗВЕСТНО: {label_id}")

    if (confidence_value < 0.5):
        label = "ЖЕСТ НЕ РАСПОЗНАН"
    
    print(f"Предсказанный класс: {label} (ID: {label_id}), уверенность: {confidence_value:.4f}")

    return Response(
        label=label,
        confidence=confidence_value
    )
