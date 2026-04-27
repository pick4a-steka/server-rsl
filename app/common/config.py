from pydantic import BaseModel
from typing import List
import os

# Структура запроса от клиента
class Request(BaseModel):
    tensor: List[List[float]]  # [12 x 132]

# Структура ответа от сервера
class Response(BaseModel):
    status: bool
    label: str | None

LABELS = {
    0: "НЕТ",
    1: "ДА",
    2: "Я"
}

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

TASK_STREAM_NAME = "recognition_tasks"
TASK_GROUP_NAME = "recognition_workers"

RESULT_TTL_SECONDS = 60
TASK_WAIT_TIMEOUT_MS = 5000