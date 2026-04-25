from pydantic import BaseModel
from typing import List

# Структура запроса от клиента
class Request(BaseModel):
    tensor: List[List[float]]  # [12 x 132]

# Структура ответа от сервера
class Response(BaseModel):
    label: str
    confidence: float

LABELS = {
    0: "НЕТ",
    1: "ДА",
    2: "ПОКА"
}