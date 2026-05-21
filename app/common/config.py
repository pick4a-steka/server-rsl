from pydantic import BaseModel
from typing import List
import os

# Структура запроса от клиента
class Request(BaseModel):
    tensor: List[List[float]]  # [18 x 132]

# Структура ответа от сервера
class Response(BaseModel):
    status: bool
    label: str | None

LABELS = {
    0: "ДА",
    1: "НЕТ",
    2: "Я",
    3: "ХОРОШО",
    4: "СПАСИБО",
    5: "ВОДА",
    6: "ДОМ",
    7: "ЕДА",
    8: "ИДТИ",
    9: "ЛЮБИТЬ",
    10: "ПЛОХО",
    11: "ПОНИМАТЬ",
    12: "ПРИВЕТ",
    13: "СЛЫШАТЬ",
    14: "СПАТЬ",
    15: "ХОТЕТЬ",
    16: "ЧЕЛОВЕК"
}

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

TASK_STREAM_NAME = "recognition_tasks"
TASK_GROUP_NAME = "recognition_workers"

RESULT_TTL_SECONDS = 60
TASK_WAIT_TIMEOUT_MS = 5000